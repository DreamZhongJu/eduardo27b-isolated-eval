"""Gracefully stop only this formal run, without changing queue rows.

This program does not export, delete, requeue, or print data. It stops the
retry supervisor first, then the adaptive producer, waits for already claimed
work to be recorded, and stops the matching student consumer only while a
SQLite write reservation proves no student request is in progress.
"""
import argparse
import fcntl
import os
from pathlib import Path
import select
import signal
import sqlite3
import sys
import time
from urllib.parse import quote

from stream_train import __main__ as app
from stream_train import api_adaptive_external as producer
from stream_train import auto_stop_student as guard


class StopError(Exception):
    pass


def role(args):
    if guard.has_module(args, 'stream_train.launch_supervised'):
        return 'launcher'
    if guard.has_module(args, 'stream_train.api_retry_supervisor'):
        return 'retry'
    if guard.has_module(args, 'stream_train.api_adaptive_external'):
        return 'adaptive'
    if (guard.has_module(args, 'stream_train.api_handoff') or
            guard.has_module(args, 'stream_train.api_handoff_immediate')):
        return 'handoff'
    if guard.has_module(args, 'stream_train'):
        if args.count('api') == 1:
            return 'legacy_api'
        if args.count('student') == 1:
            return 'student'
    if any(os.path.basename(a) == 'auto_stop_student.py' for a in args):
        return 'watcher'
    return None


def matches_db(args, name, db):
    option = '--run-dir' if name == 'launcher' else '--db'
    expected = db.parent if name == 'launcher' else db
    value = guard.one_option(args, option)
    # A relative path might refer to this run via the process's cwd. Do not
    # silently classify it as unrelated or claim the run is quiescent.
    if not os.path.isabs(value):
        raise StopError('发现使用相对路径的相关进程，无法安全确认归属')
    return os.path.realpath(value) == str(expected)


def processes(db):
    found = {name: {} for name in ('retry', 'adaptive', 'legacy_api', 'student', 'watcher',
                                    'launcher', 'handoff')}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            args = [v.decode('utf-8', 'surrogateescape') for v in
                    (p / 'cmdline').read_bytes().split(b'\0') if v]
            name = role(args)
            if not name:
                continue
            if not matches_db(args, name, db):
                continue
            _, stamp = guard.proc_info(int(p.name))
            found[name][int(p.name)] = stamp
        except FileNotFoundError:
            continue
        except (OSError, guard.GuardError):
            raise StopError('无法安全识别进程') from None
    return found


def terminate(pid, stamp, expected, db, wait_seconds=0):
    try:
        fd = guard.open_pidfd(pid)
    except guard.GuardError:
        if not (Path('/proc') / str(pid)).exists():
            return True
        raise StopError('无法登记进程句柄') from None
    try:
        try:
            args, observed = guard.proc_info(pid)
        except guard.GuardError:
            if not (Path('/proc') / str(pid)).exists():
                return True
            raise
        if (observed != stamp or role(args) != expected or not matches_db(args, expected, db)):
            raise StopError('进程身份变化，未继续停止')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        if wait_seconds:
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            return bool(poll.poll(int(wait_seconds * 1000)))
        return True
    finally:
        os.close(fd)


def lock_api():
    import stat
    path = Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'
    if not path.exists():
        raise StopError('API 锁文件缺失，无法确认停止')
    fd = os.open(str(path), os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise StopError('API 锁身份异常')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def connect(db, timeout=1):
    return sqlite3.connect('file:' + quote(str(db), safe='/') + '?mode=rw',
                           uri=True, timeout=timeout, isolation_level=None)


def db_identity(db, identity):
    st = db.stat()
    if (st.st_dev, st.st_ino) != identity:
        raise StopError('队列文件身份变化')


def api_started(db, identity):
    db_identity(db, identity)
    connection = connect(db)
    try:
        row = connection.execute("SELECT COUNT(*) FROM jobs WHERE state='started' AND "
                                 "kind IN ('generate','review','equivalence','grade')").fetchone()
        return row[0]
    finally:
        connection.close()


def stop_student(db, identity, matches):
    """Stop the exact consumer under a write lock only if no sample is started."""
    connection = connect(db, timeout=.25)
    try:
        connection.execute('BEGIN IMMEDIATE')
        db_identity(db, identity)
        count = connection.execute("SELECT COUNT(*) FROM jobs WHERE kind='student' "
                                   "AND state='started'").fetchone()[0]
        if count:
            return False
        current = processes(db)['student']
        if current != matches or len(current) > 1:
            raise StopError('正式学生消费者数量或身份变化')
        for pid, stamp in current.items():
            if not terminate(pid, stamp, 'student', db, wait_seconds=5):
                raise StopError('学生消费者未确认退出')
        return True
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def run(db, config, timeout, release_vllm=False, model=None):
    if not db.is_absolute() or not config.is_absolute() or db.is_symlink() or not db.is_file():
        raise StopError('队列或配置路径无效')
    db, config = db.resolve(), config.resolve()
    cfg = app.config(config)
    if producer.read_protocol(db) != cfg:
        raise StopError('队列绑定与冻结配置不一致')
    st = db.stat()
    identity = (st.st_dev, st.st_ino)
    deadline = time.monotonic() + timeout
    signalled = set()
    last_message = 0
    print('正在停止当前队列的自动重试及 API，等待已开始请求写回。', flush=True)
    while True:
        current = processes(db)
        for name in ('retry', 'adaptive'):
            for pid, stamp in current[name].items():
                if (name, pid, stamp) not in signalled:
                    terminate(pid, stamp, name, db)
                    signalled.add((name, pid, stamp))
        if current['legacy_api']:
            raise StopError('发现旧版 API 生产者，未擅自中断')
        if not current['retry'] and not current['adaptive'] and not current['handoff']:
            try:
                api_fd = lock_api()
                break
            except BlockingIOError:
                pass
        if time.monotonic() >= deadline:
            raise StopError('API 尚未确认退出；请求可能仍在处理，数据保留')
        if time.monotonic() - last_message >= 15:
            print('等待 API 已开始请求结束。', flush=True)
            last_message = time.monotonic()
        time.sleep(1)
    try:
        # Holding api.lock prevents same-HOME API workers and probes from
        # starting during student drain and service release.
        db_identity(db, identity)
        current = processes(db)
        if current['retry'] or current['adaptive'] or current['legacy_api']:
            raise StopError('仍存在本队列 API 进程，未继续')
        orphaned = api_started(db, identity)
        print('API 已停止；仍标记进行中的 API 任务=%d（保留原状态）。' % orphaned, flush=True)
        for pid, stamp in current['watcher'].items():
            if not terminate(pid, stamp, 'watcher', db, wait_seconds=5):
                raise StopError('原自动停卡监控未确认退出')
        while True:
            current = processes(db)
            matches = current['student']
            if not matches:
                break
            try:
                if stop_student(db, identity, matches):
                    break
            except sqlite3.OperationalError as exc:
                if 'locked' not in str(exc).lower() and 'busy' not in str(exc).lower():
                    raise
            if time.monotonic() >= deadline:
                raise StopError('学生在途作答未确认结束；不强制中断')
            if time.monotonic() - last_message >= 15:
                print('等待学生在途作答落盘。', flush=True)
                last_message = time.monotonic()
            time.sleep(1)
        remaining = processes(db)
        if any(remaining[name] for name in ('retry', 'adaptive', 'legacy_api', 'student', 'watcher', 'handoff')):
            raise StopError('本队列仍有工作进程，未确认停止')
        # The original launcher waits on the watcher and should exit by
        # itself. Never kill it: it could be between supervision stages.
        launcher_deadline = min(deadline, time.monotonic() + 30)
        while True:
            remaining = processes(db)
            if any(remaining[name] for name in ('retry', 'adaptive', 'legacy_api', 'student', 'watcher', 'handoff')):
                raise StopError('本队列工作进程重新出现，未确认完全停止')
            if not remaining['launcher']:
                break
            if time.monotonic() >= launcher_deadline:
                raise StopError('原启动监督器仍在运行，未确认完全停止')
            time.sleep(1)
        if release_vllm:
            if not model or not model.is_absolute():
                raise StopError('释放模型需提供其绝对路径')
            from urllib.parse import urlsplit
            url = urlsplit(cfg['student_base'])
            if url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port:
                raise StopError('模型服务配置不安全')
            model = model.resolve()
            pid = guard.find_service(model, cfg['student_model'], url.port)
            stamp = guard.validate_service(pid, model, cfg['student_model'], url.port)
            if any(guard.metrics(url.port)):
                raise StopError('学生服务仍有请求，不关闭')
            fd = guard.open_pidfd(pid)
            try:
                guard.validate_service(pid, model, cfg['student_model'], url.port, stamp)
                if any(guard.metrics(url.port)):
                    raise StopError('学生服务状态变化，不关闭')
                if not guard.stop_and_wait(fd, timeout=30):
                    raise StopError('学生服务退出未确认')
            finally:
                os.close(fd)
            print('指定的学生服务已退出；请另用 nvidia-smi 检查显存。', flush=True)
        db_identity(db, identity)
        final = processes(db)
        if any(final[name] for name in final):
            raise StopError('本队列进程重新出现，未确认完全停止')
        print('本队列工作进程已停止；未删除、重置或重发任何记录。', flush=True)
    finally:
        os.close(api_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--release-vllm', action='store_true', help='仅用于专供本次任务的学生服务')
    parser.add_argument('--model-path')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 7200:
        print('停止未完成：等待时间不合法', file=sys.stderr)
        return 1
    try:
        run(Path(args.db), Path(args.config), args.timeout,
            args.release_vllm, Path(args.model_path) if args.model_path else None)
    except (StopError, guard.GuardError, OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        # Guard/StopError messages are fixed; never print arbitrary error text.
        message = str(exc) if isinstance(exc, (StopError, guard.GuardError)) else type(exc).__name__
        print('停止未完成：' + message, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
