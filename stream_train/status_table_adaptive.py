"""Existing Chinese status table, additionally recognizing the new API producer."""
import argparse
import json
import os
from pathlib import Path

from stream_train import status_table as base


original_scan = base.scan_processes


def matches_adaptive(args, run):
    module = any(args[i:i + 2] == ['-m', 'stream_train.api_adaptive_external']
                 for i in range(len(args) - 1))
    return module and base.matches_path(base.option(args, '--db'), run / 'formal.sqlite')


def adaptive_scan(run):
    found, unreadable = original_scan(run)
    if unreadable is None:
        return found, None
    try:
        entries = list(Path('/proc').iterdir())
    except OSError:
        return found, None
    for p in entries:
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            args = [x.decode('utf-8', 'surrogateescape') for x in
                    (p / 'cmdline').read_bytes().split(b'\0') if x]
            if matches_adaptive(args, run):
                found['formal_api'] += 1
                found['adaptive_api'] += 1
        except FileNotFoundError:
            continue
        except OSError:
            unreadable += 1
    return found, unreadable


def window_status(run):
    """Return validated aggregate fields only, never echo an event or raw log."""
    log = run / 'api-adaptive.log'
    try:
        with log.open('rb') as f:
            f.seek(0, os.SEEK_END)
            length = f.tell()
            f.seek(max(0, length - 256 * 1024))
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return None, None, False
    current = maximum = None
    congestion = False
    for line in lines:
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get('event')
        if kind == 'start':
            # A fresh producer's events must not inherit the previous run.
            current = maximum = None
            congestion = False
        if kind in ('start', 'window', 'progress'):
            n = obj.get('maximum')
            if type(n) is int and 1 <= n <= 32:
                maximum = n
        if kind in ('start', 'window', 'progress'):
            n = obj.get('current')
            if type(n) is int and 1 <= n <= 32 and (maximum is None or n <= maximum):
                current = n
        if kind == 'window' and obj.get('reason') == 'congestion':
            congestion = True
    return current, maximum, congestion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    if not run.is_absolute():
        parser.error('运行目录必须使用绝对路径')
    run = run.resolve()
    base.scan_processes = adaptive_scan
    try:
        base.display(run)
        found, unreadable = adaptive_scan(run)
        print('其中自适应 API 生产者：%d' % found['adaptive_api'])
        if not found['adaptive_api']:
            if unreadable is None or unreadable:
                print('未发现自适应 API 进程；部分进程无法读取，不能确认是否运行。')
            else:
                print('自适应 API 未运行；下列窗口信息为历史日志记录。')
        current, maximum, congestion = window_status(run)
        print('日志最近记录的窗口：%s / %s' % (current if current is not None else '未知',
                                               maximum if maximum is not None else '未知'))
        print('近期日志是否出现拥塞：%s' % ('是：限流、服务或网络异常（未细分）' if congestion else '未发现；日志缺失时无法确认'))
    finally:
        base.scan_processes = original_scan


if __name__ == '__main__':
    main()
