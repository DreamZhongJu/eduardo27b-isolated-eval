"""Collect only numeric host/Tutor samples and aggregate by native training phase."""

from collections import Counter, defaultdict
from collections.abc import Mapping
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


PHASES = ('gen', 'old_log_prob', 'update_actor')
GPU_FIELDS = ('util_pct', 'used_mib', 'sm_clock_mhz', 'temp_c', 'power_w')
HOST_FIELDS = ('load1', 'nproc', 'mem_free_mib', 'mem_available_mib', 'swap_total_mib',
               'swap_used_mib', 'task_cpu_pct', 'task_procs')
QUERY_PREFIX = ('index', 'utilization.gpu', 'memory.used', 'clocks.current.sm')
QUERY_SUFFIX = ('temperature.gpu', 'power.draw')
THROTTLE_FIELDS = ('clocks_event_reasons.active', 'clocks_throttle_reasons.active', None)
PROC = Path('/proc')


def numeric(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def parse_number(value):
    try:
        guess = float(value)
    except (TypeError, ValueError):
        return None
    return guess if numeric(guess) and guess >= 0 else None


def parse_gpu(text, gpus, throttle=True):
    result = {}
    for raw in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(raw) != 6 + throttle:
            continue
        index, util, used, clock, *extra = (field.strip() for field in raw)
        if not index.isascii() or not index.isdigit() or int(index) not in gpus:
            continue
        temp, power = extra[-2:]
        values = {key: parse_number(value) for key, value in
                  zip(GPU_FIELDS, (util, used, clock, temp, power))}
        if values['util_pct'] is not None and values['util_pct'] > 100:
            values['util_pct'] = None
        values['throttle_mask'] = None
        if throttle:
            try:
                bits = int(extra[0], 0)
                if 0 <= bits < 1 << 64:
                    values['throttle_mask'] = bits
            except ValueError:
                pass
        result[str(int(index))] = values
    return result


class GpuQuery:
    """Cache the name supported by this driver, keeping base fields on old drivers."""

    def __init__(self):
        self.field = ...  # driver alias has not yet been probed

    def __call__(self, gpus):
        if not gpus:
            return {}
        choices = THROTTLE_FIELDS if self.field is ... else (self.field,)
        for throttle in choices:
            extra = (throttle,) if throttle else ()
            columns = ','.join((*QUERY_PREFIX, *extra, *QUERY_SUFFIX))
            args = ('nvidia-smi', f'--query-gpu={columns}', '--format=csv,noheader,nounits')
            try:
                done = subprocess.run(args, env=dict(os.environ, LC_ALL='C'), capture_output=True,
                                      encoding='utf-8', errors='replace', timeout=3, check=False)
            except (OSError, subprocess.TimeoutExpired):
                return {}
            if done.returncode == 0:
                self.field = throttle
                return parse_gpu(done.stdout, gpus, throttle is not None)
        return {}


def memory(proc=PROC):
    try:
        fields = (proc / 'meminfo').read_text().splitlines()
    except OSError:
        return {}
    values = {}
    for row in fields:
        head, sep, tail = row.partition(':')
        columns = tail.split()
        if (sep and head in ('MemFree', 'MemAvailable', 'SwapTotal', 'SwapFree')
                and len(columns) == 2 and columns[1] == 'kB' and columns[0].isdigit()):
            values[head] = int(columns[0]) / 1024
    mem = {label: values[key] for label, key in (('mem_free_mib', 'MemFree'),
           ('mem_available_mib', 'MemAvailable'), ('swap_total_mib', 'SwapTotal')) if key in values}
    if 'SwapTotal' in values and 0 <= values.get('SwapFree', -1) <= values['SwapTotal']:
        mem['swap_used_mib'] = values['SwapTotal'] - values['SwapFree']
    return mem


def read_processes(proc=PROC):
    rows = {}
    for child in proc.iterdir():
        if not child.name.isascii() or not child.name.isdigit():
            continue
        try:
            # The command name can contain spaces and right parentheses.
            fields = (child / 'stat').read_text(errors='replace').rpartition(') ')[2].split()
            if len(fields) >= 20 and fields[0] not in ('Z', 'X'):
                rows[int(child.name)] = (int(fields[1]), int(fields[19]),
                                         int(fields[11]) + int(fields[12]))
        except (OSError, ValueError):
            continue
    return rows


def identities(roots):
    items = (item for group in roots.values() for item in group) if isinstance(roots, Mapping) else iter(roots)
    result = set()
    for row in items:
        pid, start = (row.get('pid'), row.get('start')) if isinstance(row, Mapping) else row
        if type(pid) is not int or type(start) is not int or pid <= 0 or start < 0:
            return None
        result.add((pid, start))
    return result or None


def owned_processes(roots, rows):
    """Only verified roots and descendants that still have the same parent identity."""
    roots = identities(roots)
    if not roots or any(rows.get(pid, (None, None))[1] != start for pid, start in roots):
        return None
    children = defaultdict(list)
    for pid, (parent, start, _ticks) in rows.items():
        # If a parent start is newer, its PID was reused during the snapshot.
        if parent in rows and parent != pid and rows[parent][1] <= start:
            children[parent].append(pid)
    pending = [pid for pid, _ in roots]
    owned = {}
    while pending:
        pid = pending.pop()
        _parent, start, ticks = rows[pid]
        key = (pid, start)
        if key not in owned:
            owned[key] = ticks
            pending.extend(children[pid])
    return owned


def cpu_percent(before, after, elapsed, ticks_per_second):
    if before is None or after is None or elapsed <= 0:
        return None
    stable = before.keys() & after.keys()
    deltas = [after[key] - before[key] for key in stable]
    if not deltas or any(value < 0 for value in deltas):
        return None
    return 100 * sum(deltas) / ticks_per_second / elapsed


def affinity_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


class Sampler:
    """Write a new numeric JSONL file. Does not allocate GPUs or read task payloads."""

    def __init__(self, path, tutor_gpus, task_roots, interval=10, append=False):
        if not numeric(interval) or interval <= 0:
            raise ValueError('采样间隔不合法')
        self.path = Path(path)
        self.gpus = tutor_gpus
        self.roots = task_roots
        self.interval = interval
        self.append = append
        self.stop_event = threading.Event()
        self.thread = None
        self.failed = False
        self.query = GpuQuery()

    def start(self):
        if self.thread is not None:
            raise RuntimeError('采样器已启动')
        destination = self.path.open('a' if self.append else 'x', encoding='utf-8')
        self.thread = threading.Thread(target=self._run, args=(destination,), daemon=True)
        try:
            self.thread.start()
        except BaseException:
            destination.close()
            raise

    def _run(self, destination):
        ticks_per_second = os.sysconf('SC_CLK_TCK')
        previous, previous_clock, previous_wall = None, 0, 0
        deadline = time.monotonic()
        with destination:
            while not self.stop_event.is_set():
                try:
                    gpu_ids = self.gpus() if callable(self.gpus) else self.gpus
                    gpus = self.query(set(gpu_ids or ()))
                except (OSError, RuntimeError, TypeError, ValueError, csv.Error):
                    gpus = {}
                host = memory()
                host.update(load1=os.getloadavg()[0], nproc=affinity_count())
                try:
                    roots = self.roots() if callable(self.roots) else self.roots
                    current = owned_processes(roots, read_processes()) if roots else None
                except (OSError, RuntimeError, TypeError, ValueError):
                    current = None
                wall, clock = time.time(), time.monotonic()
                host['task_procs'] = len(current) if current is not None else None
                host['task_cpu_pct'] = cpu_percent(previous, current, clock - previous_clock,
                                                   ticks_per_second)
                row = {'time': wall, 'cpu_from': previous_wall, 'gpus': gpus, 'host': host}
                previous, previous_clock, previous_wall = current, clock, wall
                try:
                    destination.write(json.dumps(row, allow_nan=False) + '\n')
                    destination.flush()
                except OSError:
                    self.failed = True
                    return
                deadline += self.interval
                if deadline < clock:
                    deadline = clock + self.interval
                self.stop_event.wait(max(0, deadline - clock))

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
            return not self.thread.is_alive() and not self.failed
        return True


def windows_for(value):
    if not isinstance(value, (tuple, list)):
        return []
    if len(value) == 2 and all(numeric(x) for x in value):
        value = [value]
    valid = sorted((item[0], item[1]) for item in value
                   if isinstance(item, (tuple, list)) and len(item) == 2
                   and all(numeric(x) for x in item) and 0 < item[0] < item[1])
    merged = []
    for start, end in valid:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def aggregate(values):
    return {
        'samples': len(values),
        'min': min(values) if values else None,
        'mean': round(sum(values) / len(values), 4) if values else None,
        'max': max(values) if values else None,
    }


def throttle_summary(values):
    bits, mask = Counter(), 0
    for value in values:
        mask |= value
        remaining = value
        while remaining:
            bit = remaining & -remaining
            bits[hex(bit)] += 1
            remaining ^= bit
    return {'samples': len(values), 'active_samples': sum(value != 0 for value in values),
            'or_mask': mask if values else None, 'by_bit': dict(bits)}


def summarize(path, windows, gpu_indices=()):
    """Only return known numeric fields inside named wall-clock phase windows."""
    sections = {phase: windows_for(windows.get(phase, ())) for phase in PHASES}
    counts = Counter()
    by_host = defaultdict(lambda: defaultdict(list))
    by_gpu = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    known = {str(index) for index in gpu_indices if type(index) is int and index >= 0}
    readable = False
    invalid = 0
    try:
        with Path(path).open(encoding='utf-8', errors='replace') as stream:
            readable = True
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    if line.endswith('\n'):
                        invalid += 1
                    continue
                if not isinstance(row, dict) or not numeric(row.get('time')):
                    invalid += 1
                    continue
                gpus, host = row.get('gpus'), row.get('host')
                if not isinstance(gpus, dict) or not isinstance(host, dict):
                    invalid += 1
                    continue
                valid_gpus = {key: value for key, value in gpus.items()
                              if key.isascii() and key.isdigit() and isinstance(value, dict)}
                known.update(valid_gpus)
                for phase, spans in sections.items():
                    match = next((begin for begin, end in spans if begin <= row['time'] < end), None)
                    if match is None:
                        continue
                    counts[phase] += 1
                    for field in HOST_FIELDS:
                        value = host.get(field)
                        if field == 'task_cpu_pct' and not (
                                numeric(row.get('cpu_from')) and match <= row['cpu_from'] < row['time']):
                            continue
                        if numeric(value) and value >= 0:
                            by_host[phase][field].append(value)
                    for card, measurements in valid_gpus.items():
                        for field in GPU_FIELDS:
                            value = measurements.get(field)
                            if numeric(value) and value >= 0:
                                by_gpu[phase][card][field].append(value)
                        mask = measurements.get('throttle_mask')
                        if type(mask) is int and 0 <= mask < 1 << 64:
                            by_gpu[phase][card]['throttle_mask'].append(mask)
    except OSError:
        readable = False

    return {
        phase: {
            'samples': counts[phase],
            'duration_s': round(sum(end - start for start, end in spans), 4) if spans else None,
            'sample_file_readable': readable,
            'invalid_sample_rows': invalid,
            'host': {field: aggregate(by_host[phase][field]) for field in HOST_FIELDS},
            'gpus': {card: {
                **{field: aggregate(by_gpu[phase][card][field]) for field in GPU_FIELDS},
                'throttle': throttle_summary(by_gpu[phase][card]['throttle_mask']),
            } for card in sorted(known, key=int)},
        } for phase, spans in sections.items()
    }
