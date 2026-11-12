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
STAGES = ('PILOT', 'EASY', 'MEDIUM', 'HARD')
GROUPS = (('简单', ('PILOT', 'EASY')), ('中等', ('MEDIUM',)), ('困难', ('HARD',)))
KEY24 = re.compile(r'^[0-9a-f]{24}$')
KEY64 = re.compile(r'^[0-9a-f]{64}$')
SCHEMA = 'standalone-rejudge-v1'


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


def classify(doc, stage, source_key, identity_key):
    """Check local record structure and identity, without claiming full audit."""
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA or doc.get('stage') != stage:
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
    method, status, rounds = doc.get('method'), doc.get('status'), doc.get('rounds')
    if method == 'verified_topbool_overlay' and status in ('correct', 'incorrect') and type(rounds) is int and rounds == 0:
        return 'recovered'
    if method == 'agnes_rejudge' and status in ('correct', 'incorrect', 'unknown') and type(rounds) is int and 1 <= rounds <= 3:
        return {'correct': 'correct', 'incorrect': 'incorrect', 'unknown': 'unknown'}[status]
    raise ValueError('invalid method or status')


def scan(root):
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('result root unavailable')
    base = root / 'decisions'
    if base.is_symlink():
        raise ValueError('invalid decisions root')
    counts = {s: Counter() for s in STAGES}
    if not base.exists():
        return counts
    if not base.is_dir():
        raise ValueError('invalid decisions root')
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
                if decision.is_symlink():
                    counts[stage]['error'] += 1
                    continue
                try:
                    if not decision.exists():
                        # Directory can exist before the atomic decision appears.
                        continue
                    if not decision.is_file():
                        raise ValueError('invalid file')
                    with decision.open('r', encoding='utf-8') as f:
                        doc = json.load(f)
                    counts[stage][classify(doc, stage, source.name, item.name)] += 1
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


def render(root, script, proc=Path('/proc')):
    proc_state, processes = process_status(script, root, proc)
    try:
        counts = scan(root)
    except (OSError, ValueError):
        return f'后台进程：{proc_state}\n结果目录尚未创建或无法读取，进度未知。'
    definitions = (('本任务已记录旧回执', 'recovered'), ('补判为对', 'correct'),
                   ('补判为错', 'incorrect'), ('补判仍未决', 'unknown'), ('记录异常', 'error'))
    rows = []
    for label, key in definitions:
        vals = [sum(counts[s][key] for s in stages) for _, stages in GROUPS]
        rows.append((label, *vals, sum(vals)))
    vals = [sum(counts[s][key] for s in stages for key in ('correct', 'incorrect', 'unknown'))
            for _, stages in GROUPS]
    rows.insert(4, ('补判已落盘合计', *vals, sum(vals)))
    lines = [f'后台进程：{proc_state}' + (f'（发现 {processes} 个匹配进程）' if processes else ''),
             table(('已落盘项目', '简单', '中等', '困难', '合计'), rows),
             '说明：旧回执和新判结果保存在独立层，原判分文件不变。',
             '本任务已记录旧回执仅统计已写入补判目录的候选，不是旧恢复层的总数。',
             '数字仅统计扫描时结构正常的已落盘记录；处理中、未创建及原报告未决不在表内。',
             '这是进度快照，不核验所有原回执或数学正确性；最终应使用完整汇总程序。']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--script', type=Path, default=SCRIPT)
    a = p.parse_args()
    if not a.root.is_absolute() or not a.script.is_absolute():
        print('进度读取失败，请检查本机配置。')
        return 1
    try:
        print(render(a.root, a.script))
    except Exception:
        print('进度读取失败，请检查本机权限。')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
