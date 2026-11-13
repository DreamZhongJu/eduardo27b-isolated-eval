"""Show the existing aggregate table and retry supervisor status without raw logs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from stream_train import status_table as base
from stream_train import status_table_adaptive as adaptive


EVENTS = {
    'supervisor_started': '重试守护已启动',
    'probe_failed': '虚构题接口探测未通过，等待重试',
    'producer_already_active': '已有 API 生产者，守护等待检查',
    'producer_spawned': '已启动 API 生产者',
    'producer_exited': 'API 生产者已退出，等待重试',
    'temporary_failure': '暂时无法检查或启动，等待重试',
    'queue_drained': '队列已耗尽，可能仍有未决',
    'binding_changed': '队列或协议绑定变化，守护停止',
    'supervisor_stopped': '重试守护收到停止信号',
    'startup_failure': '重试守护启动失败',
}
STAGES = {
    'generate': '出题', 'review': '审核', 'equivalence': '答案核对',
    'grade': '判分', 'configuration': '配置', 'unknown': '未知',
    'probe_timeout_or_stop': '探测超时或被停止',
}
CATEGORIES = {'auth': '认证或权限异常', 'model': '模型身份异常',
              'protocol': '响应协议异常', 'congestion': '限流、服务或网络异常',
              'neutral': '其他异常', 'configuration': '配置异常', 'other': '其他异常'}
WAIT_EVENTS = frozenset(('probe_failed', 'producer_already_active',
                         'producer_exited', 'temporary_failure'))


def matches_retry(args, run):
    module = any(args[i:i + 2] == ['-m', 'stream_train.api_retry_supervisor']
                 for i in range(len(args) - 1))
    return module and base.matches_path(base.option(args, '--db'), run / 'formal.sqlite')


def scan_retry(run):
    count = unreadable = 0
    try:
        entries = list(Path('/proc').iterdir())
    except OSError:
        return 0, None
    for path in entries:
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.geteuid():
                continue
            args = [v.decode('utf-8', 'surrogateescape') for v in
                    (path / 'cmdline').read_bytes().split(b'\0') if v]
            if matches_retry(args, run):
                count += 1
        except FileNotFoundError:
            continue
        except OSError:
            unreadable += 1
    return count, unreadable


def retry_status(run, now=None):
    """Read a bounded tail and return only allowlisted aggregates."""
    try:
        with (run / 'api-retry.log').open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 256 * 1024))
            lines = stream.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return None, None, None, None
    latest = None
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (isinstance(event, dict) and isinstance(event.get('event'), str)
                and event['event'] in EVENTS):
            latest = event
    if latest is None:
        return None, None, None, None
    kind = latest['event']
    stage_value = latest.get('stage')
    stage = STAGES.get(stage_value) if kind == 'probe_failed' and isinstance(stage_value, str) else None
    category_value = latest.get('category')
    category = (CATEGORIES.get(category_value) if kind == 'probe_failed'
                and isinstance(category_value, str) else None)
    remaining = None
    wait = latest.get('next_wait_seconds')
    stamp = latest.get('time_utc')
    if kind in WAIT_EVENTS and type(wait) is int and 0 <= wait <= 3600 and isinstance(stamp, str):
        try:
            when = datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
            now = datetime.now(timezone.utc).timestamp() if now is None else now
            # Negative or far-future timestamps cannot become user-facing values.
            if math.isfinite(now) and when >= 0 and when - now <= 300:
                remaining = max(0, min(wait, math.ceil(when + wait - now)))
        except (ValueError, TypeError, OverflowError, OSError):
            pass
    return kind, stage, category, remaining


def display(run):
    old = base.scan_processes
    base.scan_processes = adaptive.adaptive_scan
    try:
        base.display(run)
    finally:
        base.scan_processes = old
    found, unreadable = adaptive.adaptive_scan(run)
    print('其中自适应 API 生产者：%d' % found['adaptive_api'])
    if not found['adaptive_api']:
        print('自适应 API 未发现；窗口为历史记录。' if unreadable == 0 else
              '自适应 API 进程无法确认；窗口可能为历史记录。')
    current, maximum, congestion = adaptive.window_status(run)
    print('日志最近记录的窗口：%s / %s' % (
        current if current is not None else '未知', maximum if maximum is not None else '未知'))
    print('近期日志是否出现拥塞：%s' % (
        '是：限流、服务或网络异常（未细分）' if congestion else '未发现；日志缺失时无法确认'))
    count, unreadable = scan_retry(run)
    print('API 自动重试守护进程：%d' % count)
    if not count:
        print('未发现守护；下方信息为历史日志。' if unreadable == 0 else
              '守护进程无法确认；下方信息可能为历史日志。')
    kind, stage, category, remaining = retry_status(run)
    print('最近重试状态：%s%s' % (
        EVENTS.get(kind, '暂无记录'), '（%s）' % stage if stage else ''))
    if category:
        print('最近探测失败类别：%s' % category)
    if remaining is not None:
        print('距下次检查约：%d 秒%s' % (remaining, '' if count else '（历史记录）'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    if not run.is_absolute():
        parser.error('运行目录必须使用绝对路径')
    display(run.resolve())


if __name__ == '__main__':
    main()
