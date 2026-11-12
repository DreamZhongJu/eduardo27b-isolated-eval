"""Stop a dedicated vLLM student only after the sealed run is fully drained.

Only local, read-only aggregate checks are performed. This tool never reads or
prints questions, answers, credentials or API responses. Call it only after the
initialization command has finished and no external source additions are planned.
Normal pipeline workers may continue creating child jobs as they process work.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import sqlite3
import sys
import time
from urllib.parse import quote, urlsplit
import urllib.request


STATES = frozenset(('pending', 'started', 'done', 'rejected', 'unknown'))
KINDS = frozenset(('generate', 'review', 'equivalence', 'student', 'grade'))


class GuardError(Exception):
    pass


def one_option(args, name):
    values = []
    for i, value in enumerate(args):
        if value == name:
            if i + 1 >= len(args):
                raise GuardError('进程参数不完整')
            values.append(args[i + 1])
        elif value.startswith(name + '='):
            values.append(value[len(name) + 1:])
    if len(values) != 1:
        raise GuardError('进程参数无法唯一确认')
    return values[0]


def proc_info(pid):
    if pid <= 1 or pid == os.getpid():
        raise GuardError('进程编号不安全')
    p = Path('/proc') / str(pid)
    try:
        if p.stat().st_uid != os.geteuid():
            raise GuardError('进程不属于当前用户')
        raw = (p / 'cmdline').read_bytes()
        args = [x.decode('utf-8', 'surrogateescape') for x in raw.split(b'\0') if x]
        stat = (p / 'stat').read_text()
        tail = stat.rsplit(')', 1)[1].split()
        started = int(tail[19])  # /proc stat field 22; tail begins at field 3.
        exe = os.path.realpath(os.readlink(str(p / 'exe')))
    except (OSError, IndexError, ValueError):
        raise GuardError('进程身份无法读取') from None
    try:
        same_python = os.path.samefile(exe, sys.executable)
    except OSError:
        same_python = False
    if not args or not same_python:
        raise GuardError('进程 Python 环境不匹配')
    return args, started


def has_module(args, module):
    return any(args[i:i + 2] == ['-m', module] for i in range(len(args) - 1))


def validate_service(pid, model_path, model_name, port, started=None):
    args, stamp = proc_info(pid)
    if started is not None and stamp != started:
        raise GuardError('模型服务进程身份已变化')
    if has_module(args, 'vllm.entrypoints.openai.api_server'):
        model = one_option(args, '--model')
    else:
        positions = [i for i in range(len(args) - 2)
                     if os.path.basename(args[i]) == 'vllm' and args[i + 1] == 'serve']
        if len(positions) != 1 or any(a == '--model' or a.startswith('--model=') for a in args):
            raise GuardError('不是可确认的 vLLM 服务进程')
        model = args[positions[0] + 2]
    if (not os.path.isabs(model) or os.path.realpath(model) != str(model_path)
            or one_option(args, '--served-model-name') != model_name
            or one_option(args, '--host') != '127.0.0.1'
            or one_option(args, '--port') != str(port)):
        raise GuardError('学生服务配置不匹配')
    if not owns_listening_socket(pid, port):
        raise GuardError('无法确认服务进程持有本机监听端口')
    return stamp


def owns_listening_socket(pid, port):
    """Ensure metrics port belongs to the selected service, not another server."""
    try:
        fds = Path('/proc') / str(pid) / 'fd'
        sockets = set()
        for fd in fds.iterdir():
            try:
                target = os.readlink(str(fd))
            except OSError:
                continue
            match = re.fullmatch(r'socket:\[(\d+)\]', target)
            if match:
                sockets.add(match.group(1))
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            addr, state, inode = fields[1], fields[3], fields[9]
            if addr == '0100007F:%04X' % port and state == '0A' and inode in sockets:
                return True
    except OSError:
        pass
    return False


def validate_consumer(pid, db_path, started=None):
    args, stamp = proc_info(pid)
    if started is not None and stamp != started:
        raise GuardError('学生消费者进程身份已变化')
    if not has_module(args, 'stream_train') or args.count('student') != 1:
        raise GuardError('不是训练学生消费者')
    value = one_option(args, '--db')
    if not os.path.isabs(value) or os.path.realpath(value) != str(db_path):
        raise GuardError('学生消费者数据库不匹配')
    return stamp


def consumer_pids(db_path):
    """Find same-user consumers of this database; never print command lines."""
    found = set()
    try:
        processes = list(Path('/proc').iterdir())
    except OSError:
        raise GuardError('无法核查消费者进程') from None
    for p in processes:
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            args = [x.decode('utf-8', 'surrogateescape') for x in
                    (p / 'cmdline').read_bytes().split(b'\0') if x]
            if not has_module(args, 'stream_train') or args.count('student') != 1:
                continue
            value = one_option(args, '--db')
            if os.path.isabs(value) and os.path.realpath(value) == str(db_path):
                found.add(int(p.name))
        except FileNotFoundError:
            continue  # Process exited during the scan.
        except (PermissionError, OSError, GuardError):
            raise GuardError('无法安全核查全部消费者') from None
    return found


def find_service(model_path, model_name, port):
    found = []
    try:
        processes = list(Path('/proc').iterdir())
    except OSError:
        raise GuardError('无法扫描服务进程') from None
    for p in processes:
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            args = (p / 'cmdline').read_bytes()
            if b'vllm' not in args:
                continue
            validate_service(int(p.name), model_path, model_name, port)
            found.append(int(p.name))
        except FileNotFoundError:
            continue
        except (OSError, GuardError):
            continue
    if len(found) != 1:
        raise GuardError('未能唯一识别学生服务，请检查或显式指定服务 PID')
    return found[0]


def snapshot(db_path, identity, namespace, model_name, port):
    """Read a consistent WAL-visible snapshot; never select task payloads."""
    try:
        st = db_path.stat()
        if (st.st_dev, st.st_ino) != identity:
            raise GuardError('队列文件身份已变化')
        db = sqlite3.connect('file:' + quote(str(db_path), safe='/') + '?mode=ro',
                             uri=True, timeout=5, isolation_level=None)
        try:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            row = db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
            if row is None:
                raise GuardError('队列缺少协议绑定')
            protocol = json.loads(row[0])
            base = urlsplit(protocol.get('student_base', ''))
            if (protocol.get('namespace') != namespace or
                    protocol.get('student_model') != model_name or
                    base.scheme != 'http' or base.hostname != '127.0.0.1' or base.port != port):
                raise GuardError('队列与学生服务绑定不一致')
            rows = db.execute('SELECT kind,state,COUNT(*),MAX(updated) '
                              'FROM jobs GROUP BY kind,state').fetchall()
            db.commit()
        finally:
            db.close()
        counts = Counter()
        for kind, state, count, _ in rows:
            if kind not in KINDS or state not in STATES or type(count) is not int or count < 0:
                raise GuardError('队列存在无法识别的状态')
            counts[state] += count
        if not sum(counts.values()):
            raise GuardError('队列为空，不能视为实验结束')
        fingerprint = tuple(sorted(rows))
        return counts, fingerprint
    except GuardError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, AttributeError):
        raise GuardError('无法安全读取队列状态') from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def metrics(port):
    """Both vLLM gauges must be explicitly present, finite, and zero."""
    names = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
    values = {name: [] for name in names}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        raw = None
        for suffix in ('/metrics', '/metrics/'):
            try:
                with opener.open('http://127.0.0.1:%d%s' % (port, suffix), timeout=5) as resp:
                    if resp.status == 200:
                        raw = resp.read(2 * 1024 * 1024 + 1)
                        break
            except OSError:
                continue
        if raw is None:
            raise GuardError('学生服务指标不可读')
        if len(raw) > 2 * 1024 * 1024:
            raise GuardError('学生服务指标过大')
        for line in raw.decode('utf-8').splitlines():
            for name in names:
                if re.match('^' + re.escape(name) + r'(?=[{\s]|$)', line):
                    pattern = r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+(\S+)(?:\s+\S+)?\s*$'
                    match = re.match(pattern, line)
                    if not match:
                        raise GuardError('学生服务指标格式异常')
                    val = float(match.group(1))
                    if not math.isfinite(val) or val < 0:
                        raise GuardError('学生服务指标数值异常')
                    values[name].append(val)
        if any(not values[name] for name in names):
            raise GuardError('学生服务缺少必要指标')
        return sum(values[names[0]]), sum(values[names[1]])
    except GuardError:
        raise
    except Exception:
        raise GuardError('学生服务指标不可读') from None


def open_pidfd(pid):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise GuardError('系统不支持安全进程句柄，未停止任何进程')
    try:
        return os.pidfd_open(pid, 0)
    except OSError:
        raise GuardError('无法登记进程句柄') from None


def stop_and_wait(fd, timeout=30):
    """Only the registered process receives SIGTERM; never signal a group."""
    try:
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        return bool(poller.poll(int(timeout * 1000)))
    except (OSError, AttributeError):
        return False


def monitor(args):
    if not args.confirm_initialized or not args.confirm_dedicated:
        raise GuardError('必须确认初始化完成、不再从外部追加来源且学生服务为本任务专用')
    db_path = Path(args.db)
    model_path = Path(args.model_path)
    if (not db_path.is_absolute() or not model_path.is_absolute() or
            db_path.is_symlink() or not db_path.is_file() or
            not 1 <= args.port <= 65535 or args.interval <= 0 or
            args.settle < 60 or args.interval > args.settle):
        raise GuardError('输入路径或等待参数无效')
    db_path, model_path = db_path.resolve(), model_path.resolve()
    st = db_path.stat()
    identity = (st.st_dev, st.st_ino)
    if args.service_pid is None:
        args.service_pid = find_service(model_path, args.student_model, args.port)
    if args.consumer_pid is None:
        candidates = consumer_pids(db_path)
        if len(candidates) != 1:
            raise GuardError('未能唯一识别学生消费者，请检查或显式指定消费者 PID')
        args.consumer_pid = next(iter(candidates))
    if args.service_pid == args.consumer_pid:
        raise GuardError('服务与消费者进程不能相同')
    service_fd = open_pidfd(args.service_pid)
    consumer_fd = None
    try:
        consumer_fd = open_pidfd(args.consumer_pid)
        service_start = validate_service(args.service_pid, model_path, args.student_model, args.port)
        consumer_start = validate_consumer(args.consumer_pid, db_path)
        if consumer_pids(db_path) != {args.consumer_pid}:
            raise GuardError('学生消费者数量不匹配，未停止任何进程')
        stable, fingerprint, last_status = None, None, None
        print('开始只读监测：队列耗尽并持续空闲后释放专用学生服务。', flush=True)
        while True:
            counts, mark = snapshot(db_path, identity, args.namespace, args.student_model, args.port)
            running, waiting = metrics(args.port)
            busy = counts['pending'] or counts['started'] or running or waiting
            status = (counts['pending'], counts['started'], counts['unknown'],
                      counts['rejected'], running, waiting)
            if status != last_status:
                print('等待=%d 进行=%d 未决=%d 拒绝=%d 模型运行=%g 模型排队=%g' % status,
                      flush=True)
                last_status = status
            if busy:
                stable, fingerprint = None, None
            elif stable is None or mark != fingerprint:
                stable, fingerprint = time.monotonic(), mark
            elif time.monotonic() - stable >= args.settle:
                break
            time.sleep(args.interval)

        # Revalidate all identities and idle conditions immediately before
        # stopping only the registered consumer.
        validate_service(args.service_pid, model_path, args.student_model, args.port, service_start)
        validate_consumer(args.consumer_pid, db_path, consumer_start)
        if consumer_pids(db_path) != {args.consumer_pid}:
            raise GuardError('消费者数量已变化，未停止服务')
        final_counts, final_mark = snapshot(db_path, identity, args.namespace, args.student_model, args.port)
        if final_mark != fingerprint or final_counts['pending'] or final_counts['started'] or any(metrics(args.port)):
            raise GuardError('队列或模型状态已变化，未停止服务')
        if not stop_and_wait(consumer_fd):
            raise GuardError('学生消费者未确认退出，未停止服务')

        # The producer may not enqueue after the user-declared seal. Verify
        # once more before touching the model if any outside actor violated it.
        validate_service(args.service_pid, model_path, args.student_model, args.port, service_start)
        if consumer_pids(db_path):
            raise GuardError('发现其他消费者，未停止服务')
        last_counts, last_mark = snapshot(db_path, identity, args.namespace, args.student_model, args.port)
        if last_mark != fingerprint or last_counts['pending'] or last_counts['started'] or any(metrics(args.port)):
            raise GuardError('队列或模型状态已变化，未停止服务')
        if not stop_and_wait(service_fd):
            raise GuardError('已请求学生服务退出，但未确认进程结束')
        print('学生消费者与指定模型服务已退出；未决=%d，拒绝=%d。' %
              (last_counts['unknown'], last_counts['rejected']), flush=True)
        if last_counts['unknown']:
            print('流程已耗尽但仍有未决，不能视为全部成功；记录保持不变。', flush=True)
        print('请用 nvidia-smi 核对实际显存释放情况。', flush=True)
    finally:
        if consumer_fd is not None:
            os.close(consumer_fd)
        os.close(service_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--student-model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--service-pid', type=int, help='可选；默认自动识别唯一学生服务')
    parser.add_argument('--consumer-pid', type=int, help='可选；默认自动识别唯一消费者')
    parser.add_argument('--confirm-initialized', action='store_true',
                        help='初始化已结束，后续不从外部追加来源；流水线可正常创建子任务')
    parser.add_argument('--confirm-dedicated', action='store_true',
                        help='该学生服务仅供此训练任务使用')
    parser.add_argument('--interval', type=float, default=15)
    parser.add_argument('--settle', type=float, default=120)
    args = parser.parse_args()
    try:
        monitor(args)
    except GuardError as exc:
        print('未自动释放：' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
