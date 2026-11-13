"""Interrupt a validated API producer and conservatively preserve its in-flight work.

This deliberately may interrupt up to five submitted API requests. Their
outcomes are unknown and they are never automatically retried. Task payloads,
existing results, code and protocol binding are not changed.
"""
import fcntl
import os
from pathlib import Path
import select
import signal
import sqlite3
import sys
import time
from urllib.parse import quote
import uuid

from stream_train import api_handoff as base
from stream_train import auto_stop_student as guard


def verify_api_lock(pid):
    """Require the validated API PID to have the expected contended HOME lock open."""
    lock_path = Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'
    st = lock_path.stat()
    found = False
    for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            actual = fd.stat()
            if (actual.st_dev, actual.st_ino) == (st.st_dev, st.st_ino):
                info = (Path('/proc') / str(pid) / 'fdinfo' / fd.name).read_text()
                for line in info.splitlines():
                    fields = line.split()
                    if (len(fields) >= 7 and fields[0] == 'lock:' and
                            fields[2:5] == ['FLOCK', 'ADVISORY', 'WRITE'] and
                            fields[5] == str(pid)):
                        found = True
                        break
                if found:
                    break
        except FileNotFoundError:
            continue
    if not found:
        raise base.HandoffError('旧 API 未持有预期的锁文件描述符')
    with lock_path.open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
    raise base.HandoffError('旧 API 独占锁未被占用')


def reject_other_producers(pid, db):
    """Do not classify started work while any other local producer exists."""
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in (pid, os.getpid()):
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
            args = [x.decode('utf-8', 'surrogateescape') for x in
                    (entry / 'cmdline').read_bytes().split(b'\0') if x]
            original = guard.has_module(args, 'stream_train') and args.count('api') == 1
            adaptive = guard.has_module(args, 'stream_train.api_adaptive_external')
            if (original or adaptive) and os.path.realpath(guard.one_option(args, '--db')) == str(db):
                raise base.HandoffError('该队列存在其他 API 生产者，未中断')
        except FileNotFoundError:
            continue
        except OSError:
            raise base.HandoffError('无法安全核查其他生产者') from None


def attempt_immediate(pidfd, pid, start, db, config, identity, journal):
    """Atomically audit and classify interrupted API tasks; never touch student jobs."""
    with base.PauseGuard(pidfd, pid):
        until = time.monotonic() + .25
        while not base.all_threads_stopped(pid):
            if time.monotonic() >= until:
                return False, None
            time.sleep(.005)
        base.verify_process(pid, db, config, start)
        verify_api_lock(pid)
        reject_other_producers(pid, db)
        base.assert_identity(db, identity)
        conn = sqlite3.connect('file:' + quote(str(db), safe='/') + '?mode=rw',
                               uri=True, timeout=.05, isolation_level=None)
        try:
            conn.execute('PRAGMA synchronous=FULL')
            try:
                conn.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as exc:
                if 'locked' in str(exc).lower() or 'busy' in str(exc).lower():
                    return False, None
                raise
            base.assert_identity(db, identity)
            groups = conn.execute('SELECT kind,state,COUNT(*) FROM jobs GROUP BY kind,state').fetchall()
            if not groups or any(k not in guard.KINDS or s not in guard.STATES for k, s, _ in groups):
                raise base.HandoffError('队列为空或状态无法识别')
            placeholders = ','.join('?' for _ in base.API_KINDS)
            params = ('started', *base.API_KINDS)
            count = conn.execute('SELECT COUNT(*) FROM jobs WHERE state=? AND kind IN (' +
                                 placeholders + ')', params).fetchone()[0]
            # The existing old producer has exactly five threads. A larger
            # count may include stale work or an unexpected second producer.
            if count > 5:
                raise base.HandoffError('在途 API 记录超过五条，未中断')
            nonempty = conn.execute('SELECT COUNT(*) FROM jobs WHERE state=? AND kind IN (' +
                                    placeholders + ') AND result IS NOT NULL', params).fetchone()[0]
            if nonempty:
                raise base.HandoffError('在途 API 记录已有结果，未中断')
            event_id = uuid.uuid4().hex
            conn.execute('CREATE TABLE IF NOT EXISTS api_handoff_interrupt_audit_v1 ('
                         'event_id TEXT NOT NULL, job_id TEXT NOT NULL, kind TEXT NOT NULL, '
                         'previous_result TEXT, previous_updated REAL, old_pid INTEGER NOT NULL, '
                         'recorded_at REAL NOT NULL, PRIMARY KEY(event_id,job_id))')
            conn.execute('INSERT INTO api_handoff_interrupt_audit_v1 '
                         '(event_id,job_id,kind,previous_result,previous_updated,old_pid,recorded_at) '
                         'SELECT ?,id,kind,result,updated,?,? FROM jobs WHERE state=? AND kind IN (' +
                         placeholders + ')', (event_id, pid, time.time(), *params))
            # Fail before touching the old process if the audit schema is not
            # writable, or if its row count cannot be reconciled.
            saved = conn.execute('SELECT COUNT(*) FROM api_handoff_interrupt_audit_v1 WHERE event_id=?',
                                 (event_id,)).fetchone()[0]
            if saved != count:
                raise base.HandoffError('中断审计核验失败')
            base.verify_process(pid, db, config, start)
            verify_api_lock(pid)
            reject_other_producers(pid, db)
            journal('terminating_old_immediately', old_pid=pid, api_started=count,
                    audit_event=event_id)
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            if not poller.poll(1200):
                raise base.HandoffError('旧 API 未确认退出，未启动新生产者')
            updated = conn.execute('UPDATE jobs SET state=? WHERE state=? AND kind IN (' +
                                   placeholders + ')', ('unknown', *params)).rowcount
            if updated != count:
                raise base.HandoffError('中断记录数量发生变化，未启动新生产者')
            conn.commit()
            print('旧 API 已退出，API 进行中转为未决=%d；已完成结果保持不变。' % count, flush=True)
            return True, count
        finally:
            if conn.in_transaction:
                conn.rollback()
            conn.close()


def immediate_handoff(pid, db, config, timeout=300, poll_interval=.5,
                      journal=lambda state, **fields: None):
    db, config = Path(db), Path(config)
    if not db.is_absolute() or not config.is_absolute() or db.is_symlink():
        raise base.HandoffError('队列和配置必须使用绝对路径')
    db, config = db.resolve(), config.resolve()
    identity = base.validate_binding(db, config)
    start = base.verify_process(pid, db, config)
    pidfd = guard.open_pidfd(pid)
    try:
        base.verify_process(pid, db, config, start)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            done, count = attempt_immediate(pidfd, pid, start, db, config, identity, journal)
            if done:
                journal('old_exited_and_interrupted_audited', old_pid=pid, interrupted=count)
                return
            time.sleep(poll_interval)
        raise base.HandoffError('等待数据库写锁超时，旧 API 已恢复，未切换')
    finally:
        os.close(pidfd)


def main():
    # Reuse the original handoff lock, READY handshake, rollback and monitoring
    # without modifying or replacing any frozen files or protocol metadata.
    base.handoff = immediate_handoff
    return base.main()


if __name__ == '__main__':
    sys.exit(main())
