#!/usr/bin/env python3
"""Read-only aggregate counts for A-group judge logs; never print log content."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re

ROOT = Path('/workspace/A_group_repro_20261117')
WORK = 'pilots/smoke_shared3'
ACCESS = re.compile(r'"POST\s+/v1/chat/completions(?:\?[^\s"]*)?\s+HTTP/[^"\s]+"\s+(\d{3})\b')
STATUS = re.compile(r'\[judge\]\s+(?:transient failure|5xx)\s*\(([^)]{1,60})\)', re.I)
RETRY = re.compile(r'\[judge\].*(?:transient failure|5xx).*\bretry\b', re.I)
EXHAUSTED = re.compile(r'\[judge\].*gave up after.*using fallback', re.I)
NONRETRY = re.compile(r'\[judge\].*non-retriable.*using fallback', re.I)
PARSE_FAIL = re.compile(r'Failed to parse judge response, defaulting to REJECT', re.I)
PARSE_RECOVER = re.compile(r'Judge JSON malformed, recovered decision via regex', re.I)
DEDUP_WARNING = re.compile(r'\[repeated\s+\d+x\s+across\s+cluster\]', re.I)


def category(text):
    s = text.lower()
    if 'timeout' in s or 'timed out' in s:
        return '超时'
    if 'apiconnectionerror' in s or 'connection error' in s or 'connect error' in s or 'connection refused' in s:
        return '连接错误'
    if 'ratelimiterror' in s or re.search(r'(?<!\d)429(?!\d)', s):
        return '限流429'
    if 'internalservererror' in s or re.search(r'(?<!\d)5\d\d(?!\d)', s):
        return '服务端5xx'
    return '其他或无法识别'


def tally(path):
    c = Counter()
    with path.open('r', encoding='utf-8', errors='replace', newline='') as f:
        for line in f:
            if RETRY.search(line):
                c['裁判重试总行数'] += 1
                hit = STATUS.search(line)
                c['重试_' + category(hit.group(1) if hit else line)] += 1
            if EXHAUSTED.search(line):
                c['重试耗尽降级总行数'] += 1
                c['耗尽_' + category(line)] += 1
            if NONRETRY.search(line):
                c['不可重试降级行数'] += 1
            if PARSE_FAIL.search(line):
                c['解析失败判拒绝行数'] += 1
            if PARSE_RECOVER.search(line):
                c['JSON异常但正则恢复行数'] += 1
            if DEDUP_WARNING.search(line):
                c['Ray日志合并提示行数'] += 1
    return c


def unfinished_tail(path):
    try:
        if path.stat().st_size == 0:
            return False
        with path.open('rb') as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b'\n'
    except OSError:
        return None


def count_access(path):
    c = Counter()
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            for match in ACCESS.finditer(line):
                status = int(match.group(1))
                c['HTTP 200' if status == 200 else ('HTTP 其他2xx' if 200 <= status < 300 else 'HTTP 4xx' if 400 <= status < 500 else 'HTTP 5xx' if 500 <= status < 600 else 'HTTP 其他')] += 1
                c['HTTP 总计'] += 1
    return c


def ray_files(root, work):
    ray = Path((work / 'ray_tmp_path.txt').read_text(encoding='utf-8').strip()).resolve(strict=True)
    rel = ray.relative_to(root.resolve(strict=True))
    if len(rel.parts) != 1 or not re.fullmatch(r'r[0-9a-f]{6}', rel.name):
        raise RuntimeError('Ray目录与本试点不符')
    sessions = [p for p in ray.glob('session_*') if p.is_dir() and not p.is_symlink()]
    if not sessions:
        raise RuntimeError('未找到Ray会话')
    logroot = (max(sessions, key=lambda p: p.stat().st_mtime_ns) / 'logs').resolve(strict=True)
    files, seen = {'stdout': [], 'stderr': []}, set()
    stats = Counter()
    for p in sorted(logroot.iterdir(), key=lambda q: q.name):
        if not p.name.startswith('worker-'):
            continue
        match = re.search(r'\.(out|err)(?:\.\d+)?$', p.name)
        if not match:
            if re.search(r'\.(?:out|err)\.\d+\.gz$', p.name):
                stats['压缩轮转日志未扫描'] += 1
            continue
        stats['发现日志路径'] += 1
        try:
            if not p.is_file() or p.resolve(strict=True).parent != logroot:
                stats['异常或不可读路径'] += 1
                continue
            stat = p.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                stats['同inode重复路径'] += 1
                continue
            seen.add(inode)
            files['stdout' if match.group(1) == 'out' else 'stderr'].append(p)
        except OSError:
            stats['异常或不可读路径'] += 1
    return files, stats


def table(title, rows):
    print('\n' + title)
    print('| 项目 | 数量 |')
    print('|------|-----:|')
    for label, count in rows:
        print(f'| {label} | {count} |')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    work = root / WORK
    groups = {'主训练日志': [work / 'trainer.log'], 'Ray stdout': [], 'Ray stderr': []}
    try:
        files, stats = ray_files(root, work)
        groups['Ray stdout'], groups['Ray stderr'] = files['stdout'], files['stderr']
        table('Ray 文件范围', [(name, stats[name]) for name in ('发现日志路径', '同inode重复路径', '异常或不可读路径', '压缩轮转日志未扫描')] + [('唯一 stdout 文件', len(files['stdout'])), ('唯一 stderr 文件', len(files['stderr']))])
    except (OSError, ValueError, RuntimeError):
        print('Ray 文件范围：未能核验该试点的会话')
    labels = ['裁判重试总行数', '重试_超时', '重试_连接错误', '重试_限流429', '重试_服务端5xx', '重试_其他或无法识别', '重试耗尽降级总行数', '耗尽_超时', '耗尽_连接错误', '耗尽_限流429', '耗尽_服务端5xx', '耗尽_其他或无法识别', '不可重试降级行数', '解析失败判拒绝行数', 'JSON异常但正则恢复行数', 'Ray日志合并提示行数']
    print('\n错误日志分类（按日志行数；各组可能重复，不可相加）')
    print('| 项目 | 主训练日志 | Ray stdout | Ray stderr |')
    print('|------|-----------:|-----------:|-----------:|')
    result = {}
    quality = {}
    for label, paths in groups.items():
        counts, read, fail, incomplete = Counter(), 0, 0, 0
        for path in paths:
            try:
                counts.update(tally(path)); read += 1
                if unfinished_tail(path): incomplete += 1
            except OSError:
                fail += 1
        result[label] = counts
        quality[label] = (read, fail, incomplete)
    for label in labels:
        print('| ' + label + ' | ' + ' | '.join(str(result[g][label]) for g in groups) + ' |')
    for pos, label in enumerate(('成功读取文件数', '读取失败文件数', '末行无换行文件数')):
        print('| ' + label + ' | ' + ' | '.join(str(quality[g][pos]) for g in groups) + ' |')
    try:
        access = count_access(work / 'judge.log')
        table('裁判 HTTP 响应（包含虚构测试，可能包含重试）', [(label, access[label]) for label in ('HTTP 200', 'HTTP 其他2xx', 'HTTP 4xx', 'HTTP 5xx', 'HTTP 其他', 'HTTP 总计')])
    except OSError:
        print('\n裁判 HTTP 日志不可读取')
    print('\n说明：这是日志行而非去重后的票；trainer 与 Ray 可能重复。末行无换行只能反映文件尾部格式，不能证明日志完整或未轮转。')
    print('只读取文件；不输出日志正文、题目、答案、密钥或任意异常文本。')


if __name__ == '__main__':
    main()
