#!/usr/bin/env python3
"""Read-only aggregate progress. Never emits paths, source IDs or contents."""
import argparse
import hashlib
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/results/own-test-499-v1')
KEY_RE = re.compile(r'^[0-9a-f]{24}$')
HEX_RE = re.compile(r'^[0-9a-f]{64}$')
COUNTERS = ('sources_started', 'generation_complete', 'generation_held', 'generation_partial',
            'generation_invalid', 'grading_complete', 'grading_held', 'grading_pending', 'grading_invalid',
            'grading_orphan', 'correct', 'incorrect', 'unknown', 'read_errors')


def wait_phase(root):
    """Read fixed, aggregate supervisor events; never return arbitrary log text."""
    allowed = {'STARTING', 'VERIFYING_FROZEN_INPUTS', 'VERIFYING_MODEL_HASH', 'WAITING_FOR_FREE_GPUS',
               'GPU_READY', 'STARTING_SERVICES', 'PILOT_RUNNING', 'PILOT_GRADED',
               'PILOT_GRADING_PENDING', 'BULK_RUNNING', 'GENERATION_DONE_GRADING_PENDING',
               'SCANNED_COMPLETE', 'SCANNED_WITH_UNRESOLVED', 'ERROR'}
    allowed_steps = {'ROOT_LOCK', 'STAGE_LOCK', 'WORKER_CONFLICT', 'KEY', 'PIDFD',
                     'V4_CONTROLLER', 'V4_LAUNCHER', 'CODE_BINDING', 'DATA_BINDING',
                     'PILOT', 'ENVIRONMENT', 'MODEL_HASH', 'MODEL_METADATA', 'GPU_QUERY',
                     'GPU_VALIDATION', 'CUDA_DEVICE', 'PORTS'}
    path = root / 'shared-supervisor-elastic.log'
    try:
        if path.is_symlink():
            raise OSError('invalid log')
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 1000000))
            lines = f.read().decode('utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return {'state': 'not_recorded'}
    except OSError:
        return {'state': 'unreadable'}
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict) or value.get('state') not in allowed:
            continue
        events.append(value)
    if events:
        value = events[-1]
        result = {'state': value['state']}
        if value.get('step') in allowed_steps:
            result['step'] = value['step']
        # Selection is reported only after a GPU_READY in this invocation and
        # after the last wait. Later re-selection replaces the previous count.
        selected = None
        for event in events:
            if event['state'] in ('STARTING', 'VERIFYING_FROZEN_INPUTS', 'WAITING_FOR_FREE_GPUS'):
                selected = None
            elif (event['state'] == 'GPU_READY' and type(event.get('available_gpus')) is int and
                  event['available_gpus'] in (2, 3, 4, 5)):
                selected = event['available_gpus']
        if selected is not None and value['state'] in ('GPU_READY', 'STARTING_SERVICES', 'PILOT_RUNNING',
                                                         'PILOT_GRADED', 'BULK_RUNNING'):
            result['selected_gpus'] = selected
        return result
    return {'state': 'not_recorded'}


def process_status(script, proc=Path('/proc')):
    wanted = os.fsencode(str(script))
    errors = 0
    try:
        candidates = list(proc.iterdir())
    except OSError:
        return {'state': 'unknown', 'unreadable_processes': 1}
    running = False
    for p in candidates:
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            errors += 1
            continue
        if wanted in args and b'--run' in args:
            running = True
    return {'state': 'running' if running else ('unknown' if errors else 'not_running'),
            'unreadable_processes': errors}


def folders(path):
    try:
        if path.is_symlink():
            raise OSError('symlink')
        return list(path.iterdir()), 0
    except FileNotFoundError:
        return [], 0
    except OSError:
        return [], 1


def safe_exists(path):
    try:
        if path.is_symlink():
            raise OSError('symlink')
        path.stat()
        return True
    except FileNotFoundError:
        return False


def read_json(path):
    if path.is_symlink():
        raise OSError('symlink')
    return json.loads(path.read_bytes())


def validate_grade(doc, source_id):
    if not isinstance(doc, dict) or doc.get('source_id') != source_id:
        return False
    rows = doc.get('results')
    if (not isinstance(doc.get('input_sha256'), str) or not HEX_RE.fullmatch(doc['input_sha256']) or
            not isinstance(rows, list) or len(rows) != 96):
        return False
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get('status') not in ('correct', 'incorrect', 'unknown'):
            return False
        ident = row.get('identity')
        if (not isinstance(ident, dict) or ident.get('source_id') != source_id or
                not isinstance(ident.get('target'), str) or not ident['target'] or
                ident.get('phase') not in ('pre', 'control', 'post') or
                type(ident.get('index')) is not int or not 0 <= ident['index'] < 16):
            return False
        key = (ident['target'], ident['phase'], ident['index'])
        if key in seen:
            return False
        seen.add(key)
    targets = {x[0] for x in seen}
    return len(targets) == 2 and 'source' in targets and len(seen) == 96


def stage_snapshot(root, stage):
    counts = {name: 0 for name in COUNTERS}
    files, errors = folders(root / stage / 'sources')
    counts['read_errors'] += errors
    source_keys = set()
    for folder in files:
        try:
            if not KEY_RE.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
                counts['generation_invalid'] += 1
                continue
            source_keys.add(folder.name)
            counts['sources_started'] += 1
            marker = folder / 'RAW_COMPLETE.json'
            held = folder / 'SOURCE_HELD.json'
            marker_exists = safe_exists(marker)
            held_exists = safe_exists(held)
            if marker_exists and held_exists:
                counts['generation_invalid'] += 1
                continue
            if held_exists:
                counts['generation_held'] += 1
                continue
            if not marker_exists:
                counts['generation_partial'] += 1
                continue
            try:
                raw = read_json(marker)
                if (not isinstance(raw, dict) or raw.get('samples') != 96 or
                        not isinstance(raw.get('source_id'), str) or not raw['source_id'] or
                        hashlib.sha256(raw['source_id'].encode()).hexdigest()[:24] != folder.name):
                    counts['generation_invalid'] += 1
                    continue
            except (OSError, ValueError, TypeError):
                counts['generation_invalid'] += 1
                continue
            counts['generation_complete'] += 1
            grade = root / 'AGNES' / stage / 'sources' / folder.name
            result, held_grade = grade / 'RESULTS.json', grade / 'HELD.json'
            has_result, has_held = safe_exists(result), safe_exists(held_grade)
            if has_result and has_held:
                counts['grading_invalid'] += 1
            elif has_held:
                counts['grading_held'] += 1
            elif not has_result:
                counts['grading_pending'] += 1
            else:
                try:
                    doc = read_json(result)
                    if not validate_grade(doc, raw['source_id']):
                        counts['grading_invalid'] += 1
                        continue
                    counts['grading_complete'] += 1
                    for item in doc['results']:
                        counts[item['status']] += 1
                except (OSError, ValueError, TypeError):
                    counts['grading_invalid'] += 1
        except OSError:
            counts['read_errors'] += 1
    grade_entries, errors = folders(root / 'AGNES' / stage / 'sources')
    counts['read_errors'] += errors
    for folder in grade_entries:
        if KEY_RE.fullmatch(folder.name) and folder.name not in source_keys:
            counts['grading_orphan'] += 1
    return counts


def snapshot(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError('results root unavailable')
    groups = {'easy': ('PILOT', 'EASY'), 'medium': ('MEDIUM',), 'hard': ('HARD',)}
    out = {}
    for name, stages in groups.items():
        parts = [stage_snapshot(root, stage) for stage in stages]
        out[name] = {key: sum(part[key] for part in parts) for key in COUNTERS}
    return out


SUPERVISOR = Path('/workspace/eduardo27b-dual-2687b5f/supervisor.py')
SUPERVISOR_SHA = 'f02e1cae71f67f7ba6be3bbc771e5b22245d89969e91e7ceaa844eb9f7c81300'
STAGE_ZH = {'easy': '简单', 'medium': '中等', 'hard': '困难'}
PHASE_ZH = {
    'STARTING': '准备启动', 'VERIFYING_FROZEN_INPUTS': '核对冻结配置和数据',
    'VERIFYING_MODEL_HASH': '核对模型文件', 'WAITING_FOR_FREE_GPUS': '等待空闲显卡',
    'GPU_READY': '已选卡', 'STARTING_SERVICES': '加载教师和学生模型',
    'PILOT_RUNNING': '试点中', 'PILOT_GRADED': '试点判分完成',
    'PILOT_GRADING_PENDING': '试点等待判分', 'BULK_RUNNING': '批量评测中',
    'GENERATION_DONE_GRADING_PENDING': '生成结束，判分待处理',
    'SCANNED_COMPLETE': '已扫描完成', 'SCANNED_WITH_UNRESOLVED': '结束，仍有未解决项',
    'ERROR': '异常', 'not_recorded': '无阶段记录', 'unreadable': '无法读取阶段',
}


def running_pids(script, proc=Path('/proc')):
    """Return PIDs or None if anything prevents reliable inspection."""
    found = []
    try:
        paths = list(proc.iterdir())
    except OSError:
        return None
    for path in paths:
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return None
        if os.fsencode(str(script)) in argv and b'--run' in argv:
            found.append(int(path.name))
    return found


def process_start(pid, proc=Path('/proc'), now=None):
    now = time.time() if now is None else now
    uptime = float((proc / 'uptime').read_text().split()[0])
    raw = (proc / str(pid) / 'stat').read_text()
    end = raw.rfind(')')
    if end < 0:
        raise ValueError('invalid process stat')
    return now - uptime + int(raw[end + 2:].split()[19]) / os.sysconf('SC_CLK_TCK')


def active_receipt(root, pid, started):
    matches = []
    for path in (root / 'FIVEGPU_EXECUTIONS').glob('*.json'):
        match = re.fullmatch(r'(\d+)-(\d+)\.json', path.name)
        if not match or int(match.group(2)) != pid or path.is_symlink():
            continue
        stamp = int(match.group(1)) / 1_000_000_000
        if stamp < started:
            continue
        try:
            data = json.loads(path.read_bytes())
            if isinstance(data, dict) and data.get('coordinator_sha256') == SUPERVISOR_SHA:
                matches.append((stamp, data))
        except (OSError, ValueError, TypeError):
            continue
    return max(matches, key=lambda pair: pair[0])[1] if matches else None


def query_gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                   '--format=csv,noheader,nounits'], text=True,
                                  stderr=subprocess.DEVNULL, timeout=10)
    devices = {}
    for line in raw.splitlines():
        values = [x.strip() for x in line.split(',')]
        if len(values) != 2:
            raise ValueError('invalid GPU response')
        index, uuid = values
        index = int(index)
        if index in devices or not uuid.startswith('GPU-'):
            raise ValueError('invalid GPU identity')
        devices[index] = uuid
    return devices


def gpu_roles(record, devices):
    """Validate receipt and physical identities; return no UUID or other raw content."""
    if not isinstance(record, dict):
        return None
    gpus, uuids, ports, routing = (record.get(k) for k in ('gpus', 'gpu_uuids', 'ports', 'routing'))
    if (not all(isinstance(x, list) and len(x) == 5 for x in (gpus, uuids, ports)) or
            any(type(x) is not int or x < 0 for x in gpus) or len(set(gpus)) != 5 or
            any(not isinstance(x, str) or not x.startswith('GPU-') for x in uuids) or len(set(uuids)) != 5 or
            any(type(x) is not int or not 1024 <= x <= 65535 for x in ports) or len(set(ports)) != 5 or
            not isinstance(routing, dict) or not isinstance(routing.get('teacher'), dict) or
            not isinstance(routing.get('student'), dict)):
        return None
    if any(devices.get(gpu) != uuid for gpu, uuid in zip(gpus, uuids)):
        return None
    expected_teacher = dict(zip(('EASY', 'MEDIUM', 'HARD'), ports[:3]))
    if routing['teacher'] != expected_teacher or routing['student'] != {
            stage: ports[3:] for stage in ('EASY', 'MEDIUM', 'HARD')}:
        return None
    return [(gpus[0], '教师：简单', ports[0]), (gpus[1], '教师：中等', ports[1]),
            (gpus[2], '教师：困难', ports[2]), (gpus[3], '学生：主服务', ports[3]),
            (gpus[4], '学生：第二服务', ports[4])]


def verified_services(pid, roles, root, proc=Path('/proc')):
    """Inspect direct service children; return ports, or None if unreadable.

    A matching process proves only that it exists with the expected settings;
    it does not establish readiness, active GPU kernels or exclusivity.
    """
    try:
        paths = list(proc.iterdir())
    except OSError:
        return None
    expected = {port: str(gpu).encode() for gpu, _, port in roles}
    found = {}
    for path in paths:
        if not path.name.isdigit():
            continue
        try:
            stat_text = (path / 'stat').read_text()
            end = stat_text.rfind(')')
            if end < 0:
                return None
            fields = stat_text[end + 2:].split()
            if int(fields[1]) != pid:
                continue
            if fields[0] in ('Z', 'X', 'x'):
                continue
            argv = (path / 'cmdline').read_bytes().split(b'\0')
            if b'vllm.entrypoints.openai.api_server' not in argv or b'--port' not in argv:
                continue
            port = int(argv[argv.index(b'--port') + 1])
            if port not in expected or port in found:
                return None
            env = (path / 'environ').read_bytes().split(b'\0')
            if (b'CUDA_VISIBLE_DEVICES=' + expected[port] not in env or
                    b'FIVEGPU_RUN_ROOT=' + os.fsencode(str(root)) not in env):
                return None
            tags = [v for v in env if v.startswith(b'FIVEGPU_RUN_ID=')]
            if len(tags) != 1 or not re.fullmatch(rb'FIVEGPU_RUN_ID=[0-9a-f]{32}', tags[0]):
                return None
            found[port] = tags[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            return None
    if len(set(found.values())) > 1:
        return None
    return set(found)


def get_gpu_roles(root, supervisor, proc=Path('/proc'), get_devices=query_gpu):
    """Bind to the active process, executable hash, PID start time and receipt."""
    try:
        if supervisor.is_symlink() or hashlib.sha256(supervisor.read_bytes()).hexdigest() != SUPERVISOR_SHA:
            return None
        pids = running_pids(supervisor, proc)
        if pids is None or len(pids) != 1:
            return None
        pid = pids[0]
        started = process_start(pid, proc)
        record = active_receipt(root, pid, started)
        if record is None:
            return None
        roles = gpu_roles(record, get_devices())
        # Recheck that PID wasn't recycled or controller ended during query.
        pids_after = running_pids(supervisor, proc)
        if pids_after != [pid] or abs(process_start(pid, proc) - started) > 2:
            return None
        services = verified_services(pid, roles, root, proc) if roles is not None else None
        return (roles, services) if roles is not None else None
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        return None


def render(root, supervisor, proc=Path('/proc'), get_devices=query_gpu):
    pids = running_pids(supervisor, proc)
    alive = '运行中' if pids is not None and len(pids) == 1 else ('未运行' if pids == [] else '无法确认')
    phase = wait_phase(root)
    phase_name = PHASE_ZH.get(phase.get('state'), '未知')
    devices = get_gpu_roles(root, supervisor, proc, get_devices) if alive == '运行中' else None
    roles, services = devices if devices is not None else (None, None)
    stats = snapshot(root)
    lines = [f'控制器：{alive}', f'日志最近阶段：{phase_name}（可能滞后）',
             f'已核验的配置显卡数：{len(roles) if roles is not None else "未知"}',
             f'已确认服务进程的显卡数：{len(services) if services is not None else "未知"}', '']
    if roles is not None:
        lines += ['| 显卡编号 | 本次配置角色 | 服务进程 |', '| --- | --- | --- |']
        lines += [f'| GPU {gpu} | {role} | '
                  f'{"运行中" if services is not None and port in services else "未确认"} |'
                  for gpu, role, port in roles]
        lines += ['']
    else:
        lines += ['显卡角色：暂未核验；请勿按编号猜测。', '']
    lines += ['| 难度 | 已完成生成 / 499 | 已有判分记录 | 未决作答 | 暂停来源 | 不完整来源 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in ('easy', 'medium', 'hard'):
        v = stats[key]
        lines.append(f"| {STAGE_ZH[key]} | {v['generation_complete']} / 499 | {v['grading_complete']} | "
                     f"{v['unknown']} | {v['generation_held']} | {v['generation_partial']} |")
    read_errors = sum(stats[key]['read_errors'] for key in ('easy', 'medium', 'hard'))
    invalid = sum(stats[key]['generation_invalid'] + stats[key]['grading_invalid'] for key in ('easy', 'medium', 'hard'))
    orphan = sum(stats[key]['grading_orphan'] for key in ('easy', 'medium', 'hard'))
    pending = sum(stats[key]['grading_pending'] for key in ('easy', 'medium', 'hard'))
    held_grading = sum(stats[key]['grading_held'] for key in ('easy', 'medium', 'hard'))
    lines += ['', f'判分待处理来源：{pending}；判分暂停来源：{held_grading}；'
                  f'异常标记：{invalid}；孤立判分：{orphan}；读取错误：{read_errors}。']
    lines += ['', '说明：完成数依据文件标记和结构，不代表完整数据复核；不完整来源可能仍在运行。',
              '未决是尚无明确对错的作答数；判分记录完成也可能包含未决。',
              '显卡角色来自当前运行回执和设备身份核对；服务进程存在不代表服务就绪或显卡独占。']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--supervisor', type=Path, default=SUPERVISOR)
    args = parser.parse_args()
    if not args.root.is_absolute() or not args.supervisor.is_absolute():
        raise ValueError('absolute paths required')
    print(render(args.root, args.supervisor))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('状态读取失败，请检查目录和权限。')
        sys.exit(1)
