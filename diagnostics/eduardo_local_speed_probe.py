#!/usr/bin/env python3
"""Station-local aggregate speed snapshot; no network or raw records leave stdout."""
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path('/workspace/results/own-test-499-v1')
WORK = Path('/workspace/eduardo27b-shared-wait-ecc7a92')
STATUS_SHA = '7f48cc6d6ce632f6f53e0bcee070dc2840df3515a99a95a8d0204f2f2c96b0cd'
SUPERVISOR_SHA = 'b642d842588a27644136d15783eff73d4747760cbd5e220a6e54a6f98d25ce33'
KEY = re.compile(r'^[0-9a-f]{24}$')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_status(work=WORK):
    path = work / 'status.py'
    if path.is_symlink() or digest_bytes(path.read_bytes()) != STATUS_SHA:
        raise ValueError('status hash mismatch')
    spec = importlib.util.spec_from_file_location('station_local_status', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def running_pids(script, proc=Path('/proc')):
    found = []
    for folder in proc.iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return None
        if os.fsencode(str(script)) in args and b'--run' in args:
            found.append(int(folder.name))
    return found


def active_receipt(root, pid, start_after=0):
    matches = []
    for path in (root / 'FIVEGPU_EXECUTIONS').glob('*.json'):
        match = re.fullmatch(r'(\d+)-(\d+)\.json', path.name)
        if not match or int(match.group(2)) != pid or path.is_symlink():
            continue
        stamp = int(match.group(1)) / 1_000_000_000
        if stamp < start_after:
            continue
        try:
            data = json.loads(path.read_bytes())
            if data.get('coordinator_sha256') == SUPERVISOR_SHA:
                matches.append((stamp, data))
        except (OSError, ValueError, AttributeError):
            continue
    return max(matches, key=lambda v: v[0]) if matches else None


def process_started_at(pid, proc=Path('/proc'), now=None):
    now = time.time() if now is None else now
    uptime = float((proc / 'uptime').read_text().split()[0])
    raw = (proc / str(pid) / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return now - uptime + int(fields[19]) / os.sysconf('SC_CLK_TCK')


def valid_marker(path, root):
    try:
        if path.is_symlink() or path.parent.is_symlink() or (path.parent / 'SOURCE_HELD.json').exists():
            return False
        data = json.loads(path.read_bytes())
        sid = data.get('source_id') if isinstance(data, dict) else None
        return (isinstance(sid, str) and bool(sid) and data.get('samples') == 96 and
                KEY.fullmatch(path.parent.name) is not None and
                digest_bytes(sid.encode())[:24] == path.parent.name)
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def marker_times(root, stages, after, now):
    out = []
    for stage in stages:
        for path in (root / stage / 'sources').glob('*/RAW_COMPLETE.json'):
            try:
                t = path.stat().st_mtime
                if after <= t <= now + 5 and valid_marker(path, root):
                    out.append(t)
            except OSError:
                continue
    return sorted(out)


def stage_rate(root, stages, after, now, counters, bulk):
    times = marker_times(root, stages, after, now)
    out = {'本次启动后生成完成数': len(times),
           '已有来源目录数': counters['sources_started'],
           '生成完成总数': counters['generation_complete'],
           '生成held数': counters['generation_held'],
           '生成partial数': counters['generation_partial'],
           '生成invalid数': counters['generation_invalid'],
           '判分待处理来源数': counters['grading_pending'],
           '判分未决作答数': counters['unknown']}
    if times:
        out['上次完成距今分钟'] = round(max(0, now - times[-1]) / 60, 1)
    recent = times[-10:]
    if bulk and len(recent) >= 3 and 0 <= now - recent[-1] <= 600 and recent[-1] > recent[0]:
        interval = (recent[-1] - recent[0]) / (len(recent) - 1)
        remaining = max(0, 499 - counters['sources_started'])
        out.update({'最近样本数': len(recent), '最近平均分钟每来源': round(interval / 60, 2),
                    '未开始来源生成粗估小时': round(remaining * interval / 3600, 1)})
    return out


def resolve_student(record):
    try:
        gpus, uuids, ports, routing = (record[k] for k in ('gpus', 'gpu_uuids', 'ports', 'routing'))
        if (not all(isinstance(x, list) for x in (gpus, uuids, ports)) or len(gpus) != 4 or
                len(uuids) != 4 or len(ports) != 4 or len(set(gpus)) != 4 or len(set(uuids)) != 4 or
                not isinstance(routing, dict) or set(routing) != {'EASY', 'MEDIUM', 'HARD'} or
                any(routing[x] != ports[3] for x in routing) or
                type(gpus[3]) is not int or not isinstance(uuids[3], str) or not uuids[3].startswith('GPU-')):
            return None
        if type(ports[3]) is not int or not 1024 <= ports[3] <= 65535:
            return None
        return gpus[3], uuids[3], ports[3]
    except (KeyError, TypeError):
        return None


def query_gpu():
    cmd = ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
           '--format=csv,noheader,nounits']
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=10)
    result = {}
    for line in raw.splitlines():
        values = [v.strip() for v in line.split(',')]
        if len(values) != 5:
            raise ValueError('invalid nvidia-smi output')
        index, uuid, util, used, total = values
        result[int(index)] = {'uuid': uuid, 'util': int(util), 'used': int(used), 'total': int(total)}
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def parse_metrics(text):
    """Select numeric values of exactly three known metric names, never labels."""
    allowed = {'vllm:generation_tokens_total', 'vllm:num_requests_running', 'vllm:num_requests_waiting'}
    values = {}
    pattern = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\r\n]*\})?\s+'
                         r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$')
    for line in text.splitlines():
        match = pattern.match(line)
        if not match or match.group(1) not in allowed:
            continue
        value = float(match.group(2))
        if math.isfinite(value) and value >= 0:
            values[match.group(1)] = values.get(match.group(1), 0) + value
    return values


def query_metrics(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(f'http://127.0.0.1:{port}/metrics', timeout=2) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('metrics too large')
    return parse_metrics(raw.decode('utf-8'))


def gpu_samples(student, query=query_gpu, sleep=time.sleep, metrics_query=query_metrics, clock=time.monotonic):
    index, expected, port = student
    samples, metrics = [], []
    for i in range(6):
        snapshot = query()
        row = snapshot.get(index)
        if not row or row['uuid'] != expected:
            raise ValueError('student GPU identity changed')
        samples.append(row)
        try:
            metrics.append((clock(), metrics_query(port)))
        except (OSError, ValueError, UnicodeError, urllib.error.URLError):
            pass
        if i != 5:
            sleep(2)
    utils = [s['util'] for s in samples]
    out = {'学生卡编号': index, '采样次数': len(samples), '采样间等待秒': 2,
            '整卡利用率均值百分比': round(sum(utils) / len(utils), 1),
            '整卡利用率最低百分比': min(utils), '整卡利用率最高百分比': max(utils),
            '显存已用MiB': samples[-1]['used'], '显存总MiB': samples[-1]['total']}
    gen_key, running_key, waiting_key = ('vllm:generation_tokens_total', 'vllm:num_requests_running',
                                         'vllm:num_requests_waiting')
    tokens = [(t, v[gen_key]) for t, v in metrics if gen_key in v]
    if len(tokens) >= 2 and tokens[-1][0] > tokens[0][0] and tokens[-1][1] >= tokens[0][1]:
        out['学生服务输出token每秒'] = round((tokens[-1][1] - tokens[0][1]) / (tokens[-1][0] - tokens[0][0]), 2)
    for key, label in ((running_key, '学生服务运行请求峰值'), (waiting_key, '学生服务排队请求峰值')):
        vals = [v[key] for _, v in metrics if key in v]
        if vals:
            out[label] = max(vals)
    if not metrics:
        out['服务指标'] = '本地指标不可用'
    return out


def inspect(root=ROOT, work=WORK, now=None, query=query_gpu, sleep=time.sleep, proc=Path('/proc')):
    now = time.time() if now is None else now
    status = load_status(work)
    state = status.process_status(work / 'supervisor.py', proc)
    phase = status.wait_phase(root)
    counts = status.snapshot(root)
    out = {'监督器状态': state['state'], '阶段': phase.get('state', 'unknown'),
           '注意': '时间仅按近期速度粗估未开始来源的生成，当前进行中或遗留partial、held、invalid和判分未决均需另计。GPU利用率为整卡短时采样，不能判定持续利用情况。'}
    pids = running_pids(work / 'supervisor.py', proc)
    if state['state'] != 'running' or pids is None or len(pids) != 1:
        out['速率'] = '监督器未运行或进程无法唯一识别'
        return out
    try:
        start = process_started_at(pids[0], proc, now)
        receipt = active_receipt(root, pids[0], start - 2)
    except (OSError, ValueError, IndexError):
        receipt = None
    if not receipt:
        out['速率'] = '本次启动尚无可核对运行回执'
        return out
    began, record = receipt
    out['本次启动后分钟'] = round(max(0, now - began) / 60, 1)
    groups = {'简单': ('PILOT', 'EASY'), '中等': ('MEDIUM',), '困难': ('HARD',)}
    fields = {'简单': 'easy', '中等': 'medium', '困难': 'hard'}
    out['各档'] = {name: stage_rate(root, stages, began, now, counts[fields[name]],
                                  phase.get('state') == 'BULK_RUNNING')
                    for name, stages in groups.items()}
    estimates = [v.get('未开始来源生成粗估小时') for v in out['各档'].values()]
    if all(type(v) in (int, float) for v in estimates):
        out['三档未开始来源生成粗估最慢小时'] = max(estimates)
    else:
        out['三档生成预计'] = '近期样本不足或没有持续进展'
    student = resolve_student(record)
    if student is None:
        out['学生显卡'] = '运行回执的服务路由无法核对'
    else:
        try:
            out['学生显卡'] = gpu_samples(student, query, sleep)
        except (OSError, ValueError, subprocess.SubprocessError):
            out['学生显卡'] = '采样失败或设备身份发生变化'
    return out


if __name__ == '__main__':
    try:
        print(json.dumps(inspect(), ensure_ascii=False, sort_keys=True))
    except Exception:
        print(json.dumps({'状态': '汇总失败'}, ensure_ascii=False))
        raise SystemExit(1)
