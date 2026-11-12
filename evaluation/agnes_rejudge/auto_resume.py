#!/usr/bin/env python3
"""Coarse retry supervisor for the pinned local Agnes rejudge process.

Reuses its existing AIMD window and immutable receipts; it does not change
judge code, print input data, or use GPUs. Per-cycle retries stay in the judge.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/workspace/results/own-test-499-v1')
OUTPUT = Path('/workspace/results/agnes-rejudge-v1')
OVERLAY = Path('/workspace/results/agnes-topbool-recovery-v1')
REJUDGE = Path('/workspace/edu-rejudge-auto-v1/rejudge_auto.py')
RECOVERY = Path('/workspace/edu-rejudge-ed7f64d/recovery.py')
JUDGE = Path('/workspace/agnes-eval-v3-4282afd/agnes_judge.py')
KEY = Path('/workspace/.secrets/agnes_api_key')
REJUDGE_SHA = '1f0fd73dda5c57a046ab64680cde5d0b925904038aae15fcc4e8f42a0bd46363'
RECOVERY_SHA = '5dd59dade528c45b395563d35bee3b63d3112703d5a5561eb446cecf696eaffc'
JUDGE_SHA = '2ffe746444939555b85e3557d2ddc3b61e3cd59d7997263909668876b924c871'
ROWS = ('API尝试作答', 'API请求次数', '本轮新判对', '本轮新判错', '本轮仍未决',
        '技术待处理', '格式熔断', '鉴权异常暂缓', '服务异常暂缓', '来源核验异常',
        '作答处理异常', '缺少目录', '目录无结果')
NETWORK_ERRORS = frozenset({'URLError', 'TimeoutError', 'ConnectionError', 'ConnectionResetError',
    'ConnectionAbortedError', 'ConnectionRefusedError', 'RemoteDisconnected', 'BrokenPipeError',
    'SSLError', 'OSError', 'Interrupted'})
FORMAT_ISSUES = frozenset({'response_not_json', 'response_truncated', 'response_structure_invalid',
                           'response_content_empty', 'response_duplicate_keys', 'response_schema_invalid'})


def checked(path, expected):
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('code mismatch')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(receipts):
    if receipts.is_symlink():
        raise ValueError('invalid receipts')
    if not receipts.exists():
        return set()
    if not receipts.is_dir():
        raise ValueError('invalid receipts')
    return set(receipts.rglob('RESULT.json'))


def fault_kind(response):
    """Fixed-category fail-closed classification, never returns raw response."""
    if not isinstance(response, dict):
        return 'unknown'
    code = response.get('http_status')
    if code in (401, 403):
        return 'auth'
    if code in (408, 429) or (type(code) is int and 500 <= code <= 599):
        return 'transient'
    if code is None and response.get('retryable') is True and response.get('error_type') in NETWORK_ERRORS:
        return 'transient'
    if code == 200:
        if response.get('retryable') is True and response.get('content_issue') in FORMAT_ISSUES:
            return 'transient_format'
        return 'format'
    return 'permanent'


def new_faults(receipts, before, classify):
    faults = []
    for path in snapshot(receipts) - before:
        if path.is_symlink():
            raise ValueError('invalid result')
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict) or data.get('status') != 'unknown':
            continue
        attempts = sorted(path.parent.glob('ATTEMPT-????-RESPONSE.json'))
        if not attempts or attempts[-1].is_symlink():
            faults.append('unknown')
            continue
        response = json.loads(attempts[-1].read_bytes())
        # Use the pinned grader's exact semantic-null logic, including fences.
        if classify(response) == 'semantic_null':
            continue
        faults.append(fault_kind(response))
    return faults


def parse_report(text):
    allowed_states = {'出现异常', '已到本轮上限', '仍有未决', '本轮扫描完成'}
    matches = re.findall(r'(?m)^本轮状态：([^\r\n]+)$', text)
    if len(matches) != 1 or matches[0] not in allowed_states:
        raise ValueError('missing state')
    table = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) != 7 or cells[0] or cells[-1] or cells[1] not in ROWS:
            continue
        if cells[1] in table or not all(re.fullmatch(r'\d+', s) for s in cells[2:6]):
            raise ValueError('invalid table')
        values = tuple(int(s) for s in cells[2:6])
        if sum(values[:3]) != values[3]:
            raise ValueError('inconsistent totals')
        table[cells[1]] = values
    if set(table) != set(ROWS):
        raise ValueError('incomplete table')
    return matches[0], table


def render_cycle(number, state, table):
    cells = [('项目', '简单', '中等', '困难', '合计')]
    cells += [(name, *(str(n) for n in table[name])) for name in ROWS]
    def width(value):
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1
                   for c in value)
    widths = [max(width(row[i]) for row in cells) for i in range(5)]
    border = '+' + '+'.join('-' * (n + 2) for n in widths) + '+'
    def line(row):
        padded = [(cell + ' ' * (widths[i] - width(cell))) if i == 0 else
                  (' ' * (widths[i] - width(cell)) + cell) for i, cell in enumerate(row)]
        return '| ' + ' | '.join(padded) + ' |'
    print(f'第 {number} 轮状态：{state}', border, line(cells[0]), border,
          *(line(row) for row in cells[1:]), border, sep='\n', flush=True)


def command(python, workers, reset_format_circuit=False, max_items=0):
    args = [str(python), str(REJUDGE), '--root', str(ROOT), '--output', str(OUTPUT),
            '--overlay', str(OVERLAY), '--judge', str(JUDGE), '--judge-sha256', JUDGE_SHA,
            '--recovery', str(RECOVERY), '--recovery-sha256', RECOVERY_SHA,
            '--key-file', str(KEY), '--max-items', str(max_items), '--workers', str(workers)]
    if reset_format_circuit:
        args.append('--reset-format-circuit')
    return args


def launch_child(cmd, stop):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    while True:
        try:
            stdout, _ = proc.communicate(timeout=.5)
            return SimpleNamespace(returncode=proc.returncode, stdout=stdout)
        except subprocess.TimeoutExpired:
            if stop.is_set():
                proc.terminate()
                try:
                    stdout, _ = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, _ = proc.communicate()
                return SimpleNamespace(returncode=proc.returncode, stdout=stdout)


def snapshot_decisions():
    root = OUTPUT / 'decisions'
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError('invalid decisions')
    return set(root.rglob('DECISION.json')) if root.exists() else set()


def pilot_verdict(before, report):
    """Require one newly persisted, API-based verdict agreeing with counters."""
    verdicts = []
    for path in snapshot_decisions() - before:
        if path.is_symlink():
            raise ValueError('invalid decision')
        doc = json.loads(path.read_bytes())
        if not isinstance(doc, dict):
            raise ValueError('invalid decision')
        if doc.get('method') == 'agnes_rejudge':
            verdicts.append(doc.get('status'))
    expected = (['correct'] * report['本轮新判对'][3] +
                ['incorrect'] * report['本轮新判错'][3] +
                ['unknown'] * report['本轮仍未决'][3])
    return (report['API尝试作答'][3] == 1 and report['API请求次数'][3] >= 1 and
            len(expected) >= 1 and sorted(verdicts) == sorted(expected))


def supervise(python, workers, classify, *, wait=None, launch=None, circuit_open=None, reset_format_circuit=False,
              pilot=True, verify_pilot=None):
    """Only transient service faults get unlimited capped cooldown cycles."""
    wait = wait or threading.Event()
    launch = launch or (lambda cmd: launch_child(cmd, wait))
    receipts = OUTPUT / 'receipts'
    verify_pilot = verify_pilot or pilot_verdict
    failures = 0
    cycle = 0
    while not wait.is_set():
        cycle += 1
        before = snapshot(receipts)
        before_decisions = snapshot_decisions() if pilot else None
        result = launch(command(python, 1 if pilot else workers, reset_format_circuit and cycle == 1,
                                max_items=1 if pilot else 0))
        if wait.is_set():
            return 'STOPPED'
        state, report = parse_report(result.stdout)
        render_cycle(cycle, state, report)
        if result.returncode == 0:
            if pilot:
                if state not in ('已到本轮上限', '仍有未决', '本轮扫描完成'):
                    return 'UNEXPECTED_STATUS'
                if not verify_pilot(before_decisions, report):
                    return 'PILOT_NOT_VERIFIED'
                print('单条真实补判试点通过，开始全量续跑。', flush=True)
                pilot = False
                failures = 0
                continue
            if state not in ('仍有未决', '本轮扫描完成'):
                return 'UNEXPECTED_STATUS'
            return 'DONE_WITH_UNRESOLVED' if state == '仍有未决' else 'DONE'
        if result.returncode != 2 or state != '出现异常':
            return 'EXECUTION_FAILED'
        fatal = ('鉴权异常暂缓', '来源核验异常', '作答处理异常', '缺少目录', '目录无结果')
        if any(report[name][3] for name in fatal):
            return 'NON_RETRYABLE'
        if report['格式熔断'][3]:
            if circuit_open is None or circuit_open(OUTPUT):
                return 'FORMAT_CIRCUIT_OPEN'
            print('并发成功判定已解除格式熔断，15秒后继续扫描。', flush=True)
            if wait.wait(15):
                return 'STOPPED'
            continue
        if report['服务异常暂缓'][3] < 1:
            return 'UNCLASSIFIED_FAILURE'
        faults = new_faults(receipts, before, classify)
        if not faults or any(kind not in ('transient', 'transient_format') for kind in faults):
            return 'NON_RETRYABLE_SERVICE'
        made_progress = sum(report[name][3] for name in ('本轮新判对','本轮新判错','本轮仍未决'))
        failures = 1 if made_progress else min(failures + 1, 6)
        systemic_format = sum(kind == 'transient_format' for kind in faults) >= 3
        delay = 300 if systemic_format else min(300, 15 * 2 ** (failures - 1))
        print(f'网络或服务响应异常；等待 {delay} 秒后自动续跑。每轮单请求仍最多重连3次。', flush=True)
        if wait.wait(delay):
            return 'STOPPED'
    return 'STOPPED'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5, choices=range(1, 6))
    parser.add_argument('--reset-format-circuit', action='store_true',
                        help='record explicit operator reset after investigation')
    args = parser.parse_args()
    if args.output != OUTPUT:
        raise ValueError('unexpected output')
    for p, h in ((REJUDGE, REJUDGE_SHA), (RECOVERY, RECOVERY_SHA), (JUDGE, JUDGE_SHA)):
        checked(p, h)
    if not KEY.is_file() or not KEY.stat().st_size or not OUTPUT.is_dir() or OUTPUT.is_symlink():
        raise ValueError('missing input')
    fd = os.open(OUTPUT / '.auto-rejudge.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        print('已启用自动续跑；每轮使用原拥塞窗口，并发最多5。', flush=True)
        r, judge = load_module(REJUDGE, 'pinned_rejudge'), load_module(JUDGE, 'pinned_judge')
        python = Path(sys.executable)
        if not python.is_absolute() or not python.is_file():
            raise ValueError('invalid Python executable')
        status = supervise(python, args.workers,
                           lambda response: r.classify_response(judge, response), wait=stop,
                           circuit_open=r.format_circuit_open,
                           reset_format_circuit=args.reset_format_circuit)
        labels = {'DONE':'扫描完成', 'DONE_WITH_UNRESOLVED':'扫描完成，仍有未决', 'STOPPED':'已停止',
                  'UNEXPECTED_STATUS':'状态异常，已暂停', 'EXECUTION_FAILED':'程序异常，已暂停',
                  'NON_RETRYABLE':'鉴权或数据异常，已暂停', 'UNCLASSIFIED_FAILURE':'故障未分类，已暂停',
                  'NON_RETRYABLE_SERVICE':'服务配置异常，已暂停',
                  'FORMAT_CIRCUIT_OPEN':'多个样本持续格式异常，已熔断暂停',
                  'PILOT_NOT_VERIFIED':'未取得可核验的单条试点结果，已暂停'}
        print('自动任务状态：' + labels.get(status, '未知，已暂停'), flush=True)
        return 0 if status in ('DONE', 'DONE_WITH_UNRESOLVED', 'STOPPED') else 2
    finally:
        os.close(fd)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('自动续跑未启动或异常暂停；已有记录保留。', flush=True)
        raise SystemExit(1)
