#!/usr/bin/env python3
"""Read-only aggregate progress of locally persisted rejudge decisions.

The output contains fixed labels and counters only: no source IDs, prompts,
student responses, local paths, credentials, or raw error messages. It does
not inspect or modify original result files or call any API.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path('/workspace/results/agnes-rejudge-v1')
SCRIPT = Path('/workspace/edu-rejudge-ed7f64d/rejudge.py')
ALTERNATE_SCRIPT = Path('/workspace/edu-rejudge-auto-v1/rejudge_auto.py')
SUPERVISOR = Path('/workspace/edu-rejudge-auto-v1/auto_resume.py')
STAGES = ('PILOT', 'EASY', 'MEDIUM', 'HARD')
GROUPS = (('简单', ('PILOT', 'EASY')), ('中等', ('MEDIUM',)), ('困难', ('HARD',)))
KEY24 = re.compile(r'^[0-9a-f]{24}$')
KEY64 = re.compile(r'^[0-9a-f]{64}$')
SCHEMA = 'standalone-rejudge-v1'
DEFERRED_SCHEMA = 'agnes-technical-deferred-v1'


def _digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _output_argument(args):
    values = []
    for i, part in enumerate(args):
        if part == '--output':
            if i + 1 >= len(args):
                return None
            values.append(args[i + 1])
        elif part.startswith('--output='):
            values.append(part.partition('=')[2])
    return values[0] if len(values) == 1 else None


def process_status(script, root, proc=Path('/proc')):
    """Report exact command matches, failing closed when inspection is incomplete."""
    if not script.is_absolute() or not root.is_absolute():
        raise ValueError('absolute path required')
    matches, unreadable = 0, 0
    try:
        paths = list(proc.iterdir())
    except OSError:
        return '无法确认', 0
    for path in paths:
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            raw = (path / 'cmdline').read_bytes().split(b'\0')
            args = [os.fsdecode(part) for part in raw if part]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            unreadable += 1
            continue
        if str(script) in args and _output_argument(args) == str(root):
            matches += 1
    if matches:
        return '运行中', matches
    return ('无法确认' if unreadable else '未发现进程'), 0


def validate_binding(doc, stage, source_key, identity_key, schema):
    """Check identity/hash structure; original receipts require a full audit."""
    if not isinstance(doc, dict) or doc.get('schema') != schema or doc.get('stage') != stage:
        raise ValueError('invalid record')
    if doc.get('source_key') != source_key:
        raise ValueError('source mismatch')
    ident = doc.get('identity')
    if not isinstance(ident, dict) or set(ident) != {'source_id', 'target', 'phase', 'index'}:
        raise ValueError('invalid identity')
    if (not isinstance(ident['source_id'], str) or not ident['source_id'] or
            not isinstance(ident['target'], str) or not ident['target'] or
            ident['phase'] not in ('pre', 'control', 'post') or
            type(ident['index']) is not int or not 0 <= ident['index'] < 16):
        raise ValueError('invalid identity fields')
    if hashlib.sha256(ident['source_id'].encode('utf-8')).hexdigest()[:24] != source_key:
        raise ValueError('source identity mismatch')
    if _digest({'identity': ident}) != identity_key:
        raise ValueError('identity mismatch')
    for field in ('request_sha256', 'source_result_sha256', 'input_sha256'):
        if not isinstance(doc.get(field), str) or not KEY64.fullmatch(doc[field]):
            raise ValueError('invalid binding')


def classify(doc, stage, source_key, identity_key):
    """Check local record structure and identity, without claiming full audit."""
    validate_binding(doc, stage, source_key, identity_key, SCHEMA)
    method, status, rounds = doc.get('method'), doc.get('status'), doc.get('rounds')
    if method == 'verified_topbool_overlay' and status in ('correct', 'incorrect') and type(rounds) is int and rounds == 0:
        return 'recovered'
    if method == 'agnes_rejudge' and status in ('correct', 'incorrect', 'unknown') and type(rounds) is int and 1 <= rounds <= 3:
        return {'correct': 'correct', 'incorrect': 'incorrect', 'unknown': 'unknown'}[status]
    raise ValueError('invalid method or status')


def validate_deferred(doc, stage, source_key, identity_key):
    validate_binding(doc, stage, source_key, identity_key, DEFERRED_SCHEMA)
    cycles = doc.get('format_cycles')
    if (doc.get('reason') != 'retryable_response_format' or type(cycles) is not int or
            not 2 <= cycles <= 1000):
        raise ValueError('invalid technical deferred')


def scan(root):
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('result root unavailable')
    base = root / 'decisions'
    if base.is_symlink():
        raise ValueError('invalid decisions root')
    counts = {s: Counter() for s in STAGES}
    if base.exists() and not base.is_dir():
        raise ValueError('invalid decisions root')
    seen_decisions = {}
    for stage in STAGES:
        stage_dir = base / stage
        if stage_dir.is_symlink():
            counts[stage]['error'] += 1
            continue
        try:
            sources = list(stage_dir.iterdir()) if stage_dir.exists() else []
        except OSError:
            counts[stage]['error'] += 1
            continue
        for source in sources:
            if source.is_symlink() or not KEY24.fullmatch(source.name) or not source.is_dir():
                counts[stage]['error'] += 1
                continue
            try:
                items = list(source.iterdir())
            except OSError:
                counts[stage]['error'] += 1
                continue
            for item in items:
                if item.is_symlink() or not KEY64.fullmatch(item.name) or not item.is_dir():
                    counts[stage]['error'] += 1
                    continue
                decision = item / 'DECISION.json'
                key = (stage, source.name, item.name)
                if decision.is_symlink():
                    seen_decisions[key] = False
                    counts[stage]['error'] += 1
                    continue
                try:
                    if not decision.exists():
                        # Directory can exist before the atomic decision appears.
                        continue
                    seen_decisions[key] = False
                    if not decision.is_file():
                        raise ValueError('invalid file')
                    with decision.open('r', encoding='utf-8') as f:
                        doc = json.load(f)
                    counts[stage][classify(doc, stage, source.name, item.name)] += 1
                    seen_decisions[key] = True
                except (OSError, ValueError, TypeError, KeyError, UnicodeError):
                    counts[stage]['error'] += 1

    pending = root / 'deferred'
    if pending.is_symlink() or (pending.exists() and not pending.is_dir()):
        raise ValueError('invalid deferred root')
    for stage in STAGES:
        directory = pending / stage
        if directory.is_symlink():
            counts[stage]['error'] += 1
            continue
        try:
            sources = list(directory.iterdir()) if directory.exists() else []
        except OSError:
            counts[stage]['error'] += 1
            continue
        for source in sources:
            if source.is_symlink() or not KEY24.fullmatch(source.name) or not source.is_dir():
                counts[stage]['error'] += 1
                continue
            try:
                items = list(source.iterdir())
            except OSError:
                counts[stage]['error'] += 1
                continue
            for item in items:
                if item.is_symlink() or not KEY64.fullmatch(item.name) or not item.is_dir():
                    counts[stage]['error'] += 1
                    continue
                entry = item / 'DEFERRED.json'
                key = (stage, source.name, item.name)
                try:
                    if entry.is_symlink():
                        raise ValueError('invalid deferred file')
                    if not entry.exists():
                        continue
                    if not entry.is_file():
                        raise ValueError('invalid deferred file')
                    with entry.open('r', encoding='utf-8') as f:
                        doc = json.load(f)
                    validate_deferred(doc, stage, source.name, item.name)
                    if key in seen_decisions:
                        # The resolved and deferred states must not both exist;
                        # the decision was counted once, never count it again.
                        if seen_decisions[key]:
                            counts[stage]['error'] += 1
                        continue
                    counts[stage]['technical_deferred'] += 1
                except (OSError, ValueError, TypeError, KeyError, UnicodeError):
                    counts[stage]['error'] += 1
    return counts


def width(value):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1 for c in str(value))


def table(headers, rows):
    data = [tuple(map(str, headers))] + [tuple(map(str, row)) for row in rows]
    widths = [max(width(row[i]) for row in data) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (n + 2) for n in widths) + '+'
    def line(row):
        cells = []
        for i, cell in enumerate(row):
            space = ' ' * (widths[i] - width(cell))
            cells.append(cell + space if i == 0 else space + cell)
        return '| ' + ' | '.join(cells) + ' |'
    return '\n'.join([border, line(data[0]), border, *(line(row) for row in data[1:]), border])


def describe_processes(worker, supervisor):
    """Conservatively describe process snapshots, not the reason for waiting."""
    worker_state, worker_count = worker
    supervisor_state, supervisor_count = supervisor
    if worker_state == '运行中':
        if supervisor_state == '运行中':
            return f'正在判分（守护进程在线；判分进程 {worker_count}，守护进程 {supervisor_count}）'
        if supervisor_state == '无法确认':
            return f'正在判分（判分进程 {worker_count}；守护状态无法确认）'
        return f'正在判分（独立判分进程 {worker_count}）'
    if supervisor_state == '运行中':
        if worker_state == '无法确认':
            return '守护进程运行中，判分进程状态无法确认'
        return '守护进程运行中，暂未发现判分进程（可能在退避或准备）'
    if worker_state == '无法确认' or supervisor_state == '无法确认':
        return '无法确认'
    return '未发现运行进程'


def combine_process_status(*states):
    """Count exact known child processes; preserve uncertainty when absent."""
    found = sum(count for state, count in states if state == '运行中')
    if found:
        return '运行中', found
    if any(state == '无法确认' for state, _ in states):
        return '无法确认', 0
    return '未发现进程', 0


def render(root, script, proc=Path('/proc'), supervisor=SUPERVISOR, alternate_script=ALTERNATE_SCRIPT):
    worker = combine_process_status(process_status(script, root, proc),
                                    process_status(alternate_script, root, proc))
    guardian = process_status(supervisor, root, proc)
    proc_state = describe_processes(worker, guardian)
    try:
        counts = scan(root)
    except (OSError, ValueError):
        return f'后台进程：{proc_state}\n结果目录尚未创建或无法读取，进度未知。'
    definitions = (('本任务已记录旧回执', 'recovered'), ('补判为对', 'correct'),
                   ('补判为错', 'incorrect'), ('补判仍未决', 'unknown'),
                   ('技术待处理', 'technical_deferred'), ('记录异常', 'error'))
    rows = []
    for label, key in definitions:
        vals = [sum(counts[s][key] for s in stages) for _, stages in GROUPS]
        rows.append((label, *vals, sum(vals)))
    vals = [sum(counts[s][key] for s in stages for key in ('correct', 'incorrect', 'unknown'))
            for _, stages in GROUPS]
    rows.insert(4, ('补判已落盘合计', *vals, sum(vals)))
    lines = [f'后台进程：{proc_state}',
             table(('已落盘项目', '简单', '中等', '困难', '合计'), rows),
             '说明：旧回执和新判结果保存在独立层，原判分文件不变。',
             '技术待处理是接口格式异常后的独立暂缓项，不计入语义未决或补判合计。',
             '本任务已记录旧回执仅统计已写入补判目录的候选，不是旧恢复层的总数。',
             '数字仅统计扫描时结构正常的已落盘记录；处理中、未创建及原报告未决不在表内。',
             '这是进度快照，不核验所有原回执或数学正确性；最终应使用完整汇总程序。']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--script', type=Path, default=SCRIPT)
    p.add_argument('--alternate-script', type=Path, default=ALTERNATE_SCRIPT)
    p.add_argument('--supervisor', type=Path, default=SUPERVISOR)
    a = p.parse_args()
    if not all(p.is_absolute() for p in (a.root, a.script, a.alternate_script, a.supervisor)):
        print('进度读取失败，请检查本机配置。')
        return 1
    try:
        print(render(a.root, a.script, supervisor=a.supervisor, alternate_script=a.alternate_script))
    except Exception:
        print('进度读取失败，请检查本机权限。')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
