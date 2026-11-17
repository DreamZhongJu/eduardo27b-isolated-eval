"""Conservative per-process GPU monitor. No terminal output of raw task data.

sample(roots, gpu_apps) accepts controller PIDs and nvidia-smi GPU apps plus total used. A
sample is 'verified' only if every NVML process on that GPU can be mapped
unambiguously to a visible /proc process and all relevant root processes
still exist. Unmappable processes or PID-namespace ambiguity yield unknown.
This intentionally errs on the side of unknown on shared GPUs.
"""
from __future__ import annotations
import json, os, time, threading
from pathlib import Path


def read_proc(root=Path('/proc')):
    rows = {}
    for path in root.iterdir():
        if not path.name.isdigit(): continue
        try:
            content = (path/'status').read_text()
            lines = dict(x.split(':',1) for x in content.splitlines() if ':' in x)
            pid = int(path.name)
            nspids = [int(x) for x in lines.get('NSpid','').split()]
            # Mapping namespace IDs requires proof that this process has nested
            # namespace IDs and that its visible ID is the final entry.
            hostpid = nspids[0] if len(nspids)>=2 and nspids[-1]==pid else None
            stat = (path/'stat').read_text()
            start = int(stat[stat.rfind(')')+2:].split()[19])
            rows[pid] = {'ppid':int(lines['PPid'].strip()), 'hostpid':hostpid, 'start':start}
        except (ValueError, OSError, KeyError):
            continue
    return rows


def root_identity(pid):
    row = read_proc().get(pid)
    if row is None: raise RuntimeError('根进程不可核验')
    return {'pid':pid, 'start':row['start']}


def sample(roots, gpu_apps, proc_rows):
    """roots={role:[root_identity(pid)]}; gpu_apps={gpu:{apps:[(NVML PID,MiB)],used_mib:int}}."""
    identities = [identity for values in roots.values() for identity in values]
    all_roots = {x['pid'] for x in identities}
    roots_valid = bool(identities) and all(x['pid'] in proc_rows and proc_rows[x['pid']].get('start') == x['start'] for x in identities)
    descendants = set(all_roots)
    while True:
        new = {pid for pid,row in proc_rows.items() if row['ppid'] in descendants}
        if new <= descendants: break
        descendants.update(new)
    mapping = {}
    for pid,row in proc_rows.items():
        hp = row['hostpid']
        if hp is not None: mapping.setdefault(hp, []).append(pid)
    out = {}
    for gpu,info in gpu_apps.items():
        apps = info.get('apps', [])
        used_total = info.get('used_mib')
        accounted = sum(int(used) for _, used in apps)
        # Large unaccounted allocations imply invisible processes or stale NVML.
        all_mapped = roots_valid and bool(apps) and isinstance(used_total, int) and 0 <= used_total-accounted <= 2048
        known = []
        for hostpid, used in apps:
            matches = mapping.get(hostpid, [])
            if len(matches)!=1:
                all_mapped = False
            elif matches[0] in descendants:
                known.append({'pid':matches[0], 'mib':int(used)})
            else:
                # It could be an orphaned Ray/vLLM child. Do not assume foreign.
                all_mapped = False
        out[str(gpu)] = {'verified': all_mapped,
                         'mib': sum(x['mib'] for x in known) if all_mapped else None,
                         'known_lower_bound_mib':sum(x['mib'] for x in known),
                         'own_processes':known,
                         'reason': None if all_mapped else 'PID命名空间或进程归属无法完整核验'}
    return out


class Sampler:
    """Starts a daemon sampler; get_roots/get_gpu_apps must not start workloads.

    A missing/unverifiable sample makes the corresponding peak unknown rather
    than silently producing a lower bound as a peak. path must be a NEW file.
    """
    def __init__(self, path, get_roots, get_gpu_apps, interval=10):
        self.file = open(path, 'x', encoding='utf-8')
        self.get_roots, self.get_gpu_apps, self.interval = get_roots, get_gpu_apps, interval
        self.stop_event = threading.Event(); self.peaks = {}; self.unknown = set()
        self.thread = threading.Thread(target=self._run, daemon=True)
    def start(self): self.thread.start()
    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = sample(self.get_roots(), self.get_gpu_apps(), read_proc())
                for gpu, x in result.items():
                    if x['verified']: self.peaks[gpu] = max(self.peaks.get(gpu, 0), x['mib'])
                    else: self.unknown.add(gpu)
                record={'time':time.time(),'gpus':result}
            except Exception as e:
                self.unknown.add('*'); record={'time':time.time(),'error':type(e).__name__}
            self.file.write(json.dumps(record,ensure_ascii=False)+'\n'); self.file.flush()
            self.stop_event.wait(self.interval)
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=self.interval+5)
        if self.thread.is_alive(): raise RuntimeError('监测线程无法确认已退出')
        self.file.close()
    def report(self):
        return {gpu: (None if '*' in self.unknown or gpu in self.unknown else peak)
                for gpu, peak in self.peaks.items()} | {gpu:None for gpu in self.unknown if gpu!='*'}
