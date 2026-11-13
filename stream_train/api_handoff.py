"""Safely hand an existing API producer to the adaptive producer.

The old producer is paused only briefly, with an independent pidfd watchdog.
No task, receipt, metadata, or result rows are updated. A non-blocking SQLite
write reservation proves there is no unfinished claim transaction; the old
producer is terminated only while the reservation is held and no API task is
in the started state. The student process and model service are not signalled.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
from urllib.parse import quote

from stream_train import __main__ as original
from stream_train import auto_stop_student as guard


API_KINDS = ('generate', 'review', 'equivalence', 'grade')


class HandoffError(Exception):
    pass


def acquire_handoff_lock(db_path):
    """Prevent concurrent handoffs of this run without touching the DB."""
    db = Path(db_path)
    if not db.is_absolute() or db.is_symlink() or not db.is_file():
        raise HandoffError('正式队列路径无效，未切换')
    lock_path = db.parent / ('.' + db.name + '.api-handoff.lock')
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(str(lock_path), flags, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise HandoffError('交接锁文件身份异常，未切换')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HandoffError('该队列已有交接任务，未重复切换') from None
        return fd
    except BaseException:
        os.close(fd)
        raise


def verify_process(pid, db, config, expected_start=None):
    args, start = guard.proc_info(pid)
    if (not guard.has_module(args, 'stream_train') or args.count('api') != 1
            or os.path.realpath(guard.one_option(args, '--db')) != str(db)
            or os.path.realpath(guard.one_option(args, '--config')) != str(config)
            or (expected_start is not None and start != expected_start)):
        raise HandoffError('旧 API 进程或队列身份不匹配')
    return start


def validate_binding(db, config):
    st = db.stat()
    if not db.is_file() or db.is_symlink():
        raise HandoffError('队列路径不可用')
    connection = sqlite3.connect('file:' + quote(str(db), safe='/') + '?mode=ro',
                                 uri=True, timeout=1)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
        if not row or json.loads(row[0]) != original.config(config):
            raise HandoffError('队列协议与冻结代码不匹配')
    finally:
        connection.close()
    return st.st_dev, st.st_ino


def assert_identity(db, identity):
    st = db.stat()
    if (st.st_dev, st.st_ino) != identity:
        raise HandoffError('队列文件身份发生变化')


def all_threads_stopped(pid):
    root = Path('/proc') / str(pid) / 'task'
    tids = list(root.iterdir())
    if not tids:
        return False
    for task in tids:
        try:
            tail = (task / 'stat').read_text().rsplit(')', 1)[1].split()
            if tail[0] not in ('T', 't', 'Z', 'X'):
                return False
        except (FileNotFoundError, IndexError):
            return False
    return True


WATCHDOG = r'''import os,select,signal,sys,time
pidfd=int(sys.argv[1]); pipe=int(sys.argv[2]); ready=int(sys.argv[3]); pid=int(sys.argv[4])
os.write(ready,b'R'); os.close(ready)
stopped_at=None
while True:
    r,_,_=select.select([pipe],[],[],.05)
    if r:
        if os.read(pipe,1)==b'D': break
        try: signal.pidfd_send_signal(pidfd,signal.SIGCONT)
        except OSError: pass
        break
    try:
        state=open('/proc/%d/stat'%pid).read().rsplit(')',1)[1].split()[0]
    except (OSError,IndexError): break
    if state in ('T','t'):
        if stopped_at is None: stopped_at=time.monotonic()
        if time.monotonic()-stopped_at>=1.0:
            try: signal.pidfd_send_signal(pidfd,signal.SIGCONT)
            except OSError: pass
            break
    else: stopped_at=None
'''


class PauseGuard:
    """Resume the same pidfd if this controller dies or the pause overruns."""
    def __init__(self, pidfd, pid):
        self.pidfd = pidfd
        self.pid = pid
        self.write_fd = None
        self.child = None

    def __enter__(self):
        r, w = os.pipe()
        ready_r, ready_w = os.pipe()
        try:
            env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL') if k in os.environ}
            self.child = subprocess.Popen(
                [sys.executable, '-c', WATCHDOG, str(self.pidfd), str(r), str(ready_w), str(self.pid)],
                pass_fds=(self.pidfd, r, ready_w), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, start_new_session=True)
            os.close(r); r = None
            os.close(ready_w); ready_w = None
            if not select.select([ready_r], [], [], 2)[0] or os.read(ready_r, 1) != b'R':
                raise HandoffError('无法启动暂停保护进程')
            self.write_fd = w; w = None
            signal.pidfd_send_signal(self.pidfd, signal.SIGSTOP)
            return self
        except BaseException:
            try: signal.pidfd_send_signal(self.pidfd, signal.SIGCONT)
            except OSError: pass
            raise
        finally:
            for fd in (r, w, ready_r, ready_w):
                if fd is not None:
                    os.close(fd)

    def __exit__(self, typ, value, tb):
        # Unconditionally CONT, including after SIGTERM was queued. The
        # inherited pidfd prevents signalling a process with a recycled PID.
        try:
            signal.pidfd_send_signal(self.pidfd, signal.SIGCONT)
        except OSError:
            pass
        if self.write_fd is not None:
            try: os.write(self.write_fd, b'D')
            except OSError: pass
            os.close(self.write_fd)
        if self.child is not None:
            try: self.child.wait(timeout=2)
            except subprocess.TimeoutExpired: pass


def attempt(pidfd, pid, start, db, config, identity, before_terminate=lambda: None):
    """Return (transferred, started_count); timeout on writer means retry."""
    with PauseGuard(pidfd, pid):
        end = time.monotonic() + 0.25
        while not all_threads_stopped(pid):
            if time.monotonic() >= end:
                return False, None
            time.sleep(.005)
        verify_process(pid, db, config, start)
        assert_identity(db, identity)
        conn = sqlite3.connect('file:' + quote(str(db), safe='/') + '?mode=rw',
                               uri=True, timeout=.05, isolation_level=None)
        try:
            try:
                conn.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() or 'busy' in str(e).lower():
                    return False, None
                raise
            assert_identity(db, identity)
            rows = conn.execute('SELECT kind,state,COUNT(*) FROM jobs GROUP BY kind,state').fetchall()
            if not rows or any(kind not in guard.KINDS or state not in guard.STATES
                               for kind, state, _ in rows):
                raise HandoffError('队列为空或包含未知状态，未切换')
            started = sum(count for kind, state, count in rows
                          if kind in API_KINDS and state == 'started')
            if started:
                return False, started
            # A pending TERM is made deliverable by CONT while the writer
            # reservation prevents new task claims. Never signal a group.
            verify_process(pid, db, config, start)
            before_terminate()
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            poller = select.poll(); poller.register(pidfd, select.POLLIN)
            if not poller.poll(1200):
                raise HandoffError('旧 API 未确认退出，未启动新生产者')
            return True, 0
        finally:
            if conn.in_transaction:
                conn.rollback()
            conn.close()


def handoff(pid, db, config, timeout=300, poll_interval=.5, journal=lambda state, **fields: None):
    db, config = Path(db), Path(config)
    if not db.is_absolute() or not config.is_absolute() or db.is_symlink():
        raise HandoffError('队列和配置必须使用绝对路径')
    db, config = db.resolve(), config.resolve()
    identity = validate_binding(db, config)
    start = verify_process(pid, db, config)
    pidfd = guard.open_pidfd(pid)
    try:
        verify_process(pid, db, config, start)
        deadline, next_report = time.monotonic() + timeout, 0
        while time.monotonic() < deadline:
            done, started = attempt(pidfd, pid, start, db, config, identity,
                                    lambda: journal('terminating_old', old_pid=pid))
            if done:
                journal('old_exited', old_pid=pid)
                print('旧 API 已在零在途任务时退出；原队列记录保持不变。', flush=True)
                return
            now = time.monotonic()
            if now >= next_report:
                print('等待 API 空档；在途任务=' + (str(started) if started is not None else '暂不可核验'), flush=True)
                next_report = now + 15
            time.sleep(poll_interval)
        raise HandoffError('等待空档超时；旧 API 已恢复，未切换')
    finally:
        os.close(pidfd)


def try_restore_original(db, config, identity, journal):
    """Only attempt rollback after replacement exited and the API lock is free."""
    db, config = Path(db).resolve(), Path(config).resolve()
    assert_identity(db, identity)
    if validate_binding(db, config) != identity:
        raise HandoffError('回退时队列身份发生变化')
    lock_path = Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            journal('fallback_blocked', reason='api_lock_busy')
            return None
        fcntl.flock(lock, fcntl.LOCK_UN)
    # The old program independently acquires this same non-blocking lock.
    # If another producer wins the race it exits instead of duplicating work.
    child = subprocess.Popen([sys.executable, '-m', 'stream_train', '--db', str(db),
                              '--config', str(config), 'api'],
                             stdin=subprocess.DEVNULL, start_new_session=True)
    journal('fallback_spawned', pid=child.pid)
    time.sleep(1)
    if child.poll() is not None:
        journal('fallback_exited', pid=child.pid, exit_code=child.returncode)
        return None
    return child.pid


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', required=True, type=int)
    p.add_argument('--db', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--max-concurrency', type=int, default=20)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--receipt', help='独立回执路径；默认在队列目录使用唯一名称')
    args = p.parse_args()
    if not 1 <= args.max_concurrency <= 32 or not 1 <= args.timeout <= 3600:
        print('参数不合法，未切换', file=sys.stderr, flush=True)
        return 1
    if not os.environ.get('AGNES_API_KEY', '').strip():
        print('缺少 Agnes 密钥，未切换', file=sys.stderr, flush=True)
        return 1
    receipt = Path(args.receipt) if args.receipt else Path(args.db).parent / ('api-handoff-' + uuid.uuid4().hex + '.jsonl')
    if not receipt.is_absolute():
        print('回执路径必须是绝对路径，未切换', file=sys.stderr, flush=True)
        return 1
    handoff_fd = None
    try:
        handoff_fd = acquire_handoff_lock(args.db)
        # Ensure the replacement module is importable before terminating old API.
        probe = subprocess.run([sys.executable, '-m', 'stream_train.api_adaptive_external', '--help'],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
        if probe.returncode:
            raise HandoffError('新生产者代码不可用，未切换')
        with receipt.open('x', encoding='utf-8') as receipt_file:
            os.chmod(receipt, 0o600)
            def journal(state, **fields):
                receipt_file.write(json.dumps({'state': state, 'time_utc': time.strftime(
                    '%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}, sort_keys=True) + '\n')
                receipt_file.flush()
                os.fsync(receipt_file.fileno())
            journal('waiting', old_pid=args.pid, max_concurrency=args.max_concurrency)
            print('独立交接回执：' + str(receipt), flush=True)
            try:
                identity = validate_binding(Path(args.db).resolve(), Path(args.config).resolve())
                handoff(args.pid, args.db, args.config, timeout=args.timeout, journal=journal)
                ready_r, ready_w = os.pipe()
                try:
                    env = os.environ.copy()
                    env['STREAM_PRODUCER_READY_FD'] = str(ready_w)
                    try:
                        child = subprocess.Popen([sys.executable, '-m', 'stream_train.api_adaptive_external',
                                                  '--db', args.db, '--config', args.config,
                                                  '--max-concurrency', str(args.max_concurrency)],
                                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                                 env=env, pass_fds=(ready_w,))
                    except OSError:
                        journal('replacement_spawn_failed')
                        fallback = try_restore_original(args.db, args.config, identity, journal)
                        if fallback is not None:
                            print('已尝试恢复原版 API，PID=%d；请检查生产者状态。' % fallback, flush=True)
                        raise HandoffError('旧 API 已退出，新生产者未能启动；检查回执') from None
                    os.close(ready_w); ready_w = None
                    journal('replacement_spawned', pid=child.pid)
                    if not select.select([ready_r], [], [], 30)[0] or os.read(ready_r, 1) != b'R':
                        journal('replacement_not_ready', pid=child.pid,
                                exited=child.poll() is not None)
                        if child.poll() is not None:
                            fallback = try_restore_original(args.db, args.config, identity, journal)
                            if fallback is not None:
                                print('已尝试恢复原版 API，PID=%d；请检查生产者状态。' % fallback, flush=True)
                        raise HandoffError('旧 API 已退出，但新生产者未确认就绪；检查回执，勿重复启动')
                    journal('replacement_ready', pid=child.pid)
                    print('新生产者已确认接管，PID=%d；原监督器和学生进程未更改。' % child.pid, flush=True)
                    os.close(handoff_fd)
                    handoff_fd = None
                finally:
                    os.close(ready_r)
                    if ready_w is not None:
                        os.close(ready_w)
                code = child.wait()
                journal('replacement_exited', pid=child.pid, exit_code=code)
                return code
            except Exception:
                journal('controller_error')
                raise
    except (HandoffError, guard.GuardError, OSError, sqlite3.Error, ValueError,
            subprocess.TimeoutExpired) as exc:
        print('未完成切换：' + str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        if handoff_fd is not None:
            os.close(handoff_fd)


if __name__ == '__main__':
    raise SystemExit(main())
