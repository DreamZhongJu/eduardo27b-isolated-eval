"""Cancel only a provably waiting natural API handoff, without touching data.

The controller is frozen while its already-open receipt is inspected.  A
separate watchdog resumes it if this program exits or is killed; at timeout it
first kills this program and waits for it to exit, preventing a resume/check
race.  A receipt past the single 'waiting' event is never cancelled.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import time

from stream_train import api_handoff as h
from stream_train import api_handoff_immediate as immediate
from stream_train import auto_stop_student as g


WATCHDOG = r'''import os,select,signal,sys,time
target=int(sys.argv[1]); parent=int(sys.argv[2]); pipe=int(sys.argv[3]); ready=int(sys.argv[4])
os.write(ready,b'R'); os.close(ready)
end=time.monotonic()+5
while True:
    remaining=max(0,end-time.monotonic())
    readable,_,_=select.select([pipe,parent],[],[],remaining)
    if pipe in readable:
        if os.read(pipe,1)==b'D': break
        try: signal.pidfd_send_signal(target,signal.SIGCONT)
        except OSError: pass
        break
    if parent in readable:
        try: signal.pidfd_send_signal(target,signal.SIGCONT)
        except OSError: pass
        break
    if time.monotonic()>=end:
        try: signal.pidfd_send_signal(parent,signal.SIGKILL)
        except OSError: pass
        select.select([parent],[],[])
        try: signal.pidfd_send_signal(target,signal.SIGCONT)
        except OSError: pass
        break
'''


class FrozenController:
    def __init__(self, fd):
        self.fd = fd
        self.pipe = None
        self.watchdog = None

    def __enter__(self):
        parentfd = g.open_pidfd(os.getpid())
        r, w = os.pipe()
        ready_r, ready_w = os.pipe()
        try:
            env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL') if k in os.environ}
            self.watchdog = subprocess.Popen(
                [sys.executable, '-c', WATCHDOG, str(self.fd), str(parentfd), str(r), str(ready_w)],
                pass_fds=(self.fd, parentfd, r, ready_w), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            os.close(r); r = None
            os.close(ready_w); ready_w = None
            if not select.select([ready_r], [], [], 2)[0] or os.read(ready_r, 1) != b'R':
                raise h.HandoffError('无法启动等待器暂停保护')
            self.pipe = w; w = None
            signal.pidfd_send_signal(self.fd, signal.SIGSTOP)
            return self
        except BaseException:
            try:
                signal.pidfd_send_signal(self.fd, signal.SIGCONT)
            except OSError:
                pass
            raise
        finally:
            os.close(parentfd)
            for fd in (r, w, ready_r, ready_w):
                if fd is not None:
                    os.close(fd)

    def __exit__(self, typ, value, tb):
        try:
            signal.pidfd_send_signal(self.fd, signal.SIGCONT)
        except OSError:
            pass
        if self.pipe is not None:
            try:
                os.write(self.pipe, b'D')
            except OSError:
                pass
            os.close(self.pipe)
        if self.watchdog is not None:
            try:
                self.watchdog.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def verify_waiter(pid, db, config, old_pid, expected=None):
    args, stamp = g.proc_info(pid)
    if (not g.has_module(args, 'stream_train.api_handoff')
            or os.path.realpath(g.one_option(args, '--db')) != str(db)
            or os.path.realpath(g.one_option(args, '--config')) != str(config)
            or g.one_option(args, '--pid') != str(old_pid)
            or (expected is not None and stamp != expected)):
        raise h.HandoffError('自然交接控制器身份不匹配')
    return stamp


def find_waiters(db, config, old_pid):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            raw = (p / 'cmdline').read_bytes().split(b'\0')
            if b'stream_train.api_handoff' not in raw:
                continue
            args = [x.decode('utf-8', 'surrogateescape') for x in raw if x]
            if os.path.realpath(g.one_option(args, '--db')) != str(db):
                continue
            verify_waiter(int(p.name), db, config, old_pid)
            found.append(int(p.name))
        except FileNotFoundError:
            continue
        except (OSError, g.GuardError):
            raise h.HandoffError('无法安全核查自然交接控制器') from None
    return found


def verify_waiting_receipt(pid, run_dir, old_pid):
    """Read only the controller's open receipt, never outputting its content."""
    matches = []
    for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            target = Path(os.readlink(str(fd)))
            if (target.parent != run_dir or
                    not re.fullmatch(r'api-handoff-[0-9a-f]{32}\.jsonl', target.name)):
                continue
            if not target.is_file() or target.is_symlink():
                raise h.HandoffError('交接回执身份不可确认')
            st_a, st_b = fd.stat(), target.stat()
            if (st_a.st_dev, st_a.st_ino) != (st_b.st_dev, st_b.st_ino):
                raise h.HandoffError('交接回执身份发生变化')
            if st_b.st_size > 65536:
                raise h.HandoffError('交接回执过大')
            data = fd.read_bytes()
            if len(data) > 65536 or not data.endswith(b'\n'):
                raise h.HandoffError('交接回执不完整')
            lines = data.splitlines()
            if len(lines) != 1:
                raise h.HandoffError('自然交接已进入切换阶段，不能取消')
            value = json.loads(lines[0])
            if value.get('state') != 'waiting' or value.get('old_pid') != old_pid:
                raise h.HandoffError('自然交接已进入切换阶段，不能取消')
            matches.append(target)
        except FileNotFoundError:
            continue
    if len(matches) != 1:
        raise h.HandoffError('未能唯一核验自然交接回执')


def acquire_cancel_lock(db):
    path = db.parent / ('.' + db.name + '.api-cancel-waiter.lock')
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(str(path), flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise h.HandoffError('取消等待锁身份异常')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise h.HandoffError('已有等待器取消操作进行中') from None
        return fd
    except BaseException:
        os.close(fd)
        raise


def cancel(db, config, old_pid):
    db, config = Path(db), Path(config)
    if not db.is_absolute() or not config.is_absolute() or db.is_symlink():
        raise h.HandoffError('队列和配置必须是绝对路径')
    db, config = db.resolve(), config.resolve()
    if not db.is_file():
        raise h.HandoffError('正式队列不存在')
    lock_fd = acquire_cancel_lock(db)
    try:
        return cancel_locked(db, config, old_pid)
    finally:
        os.close(lock_fd)


def cancel_locked(db, config, old_pid):
    identity = h.validate_binding(db, config)
    api_start = h.verify_process(old_pid, db, config)
    oldfd = g.open_pidfd(old_pid)
    try:
        h.verify_process(old_pid, db, config, api_start)
        immediate.verify_api_lock(old_pid)
        immediate.reject_other_producers(old_pid, db)
        waiters = find_waiters(db, config, old_pid)
        if not waiters:
            state = (Path('/proc') / str(old_pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('T', 't'):
                raise h.HandoffError('旧 API 已暂停，无法确认原因，未切换')
            print('未发现自然交接等待器；旧 API 进程存在。', flush=True)
            return
        if len(waiters) != 1:
            raise h.HandoffError('自然交接控制器不唯一，未取消')
        pid = waiters[0]
        start = verify_waiter(pid, db, config, old_pid)
        state = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        if state in ('T', 't'):
            raise h.HandoffError('自然交接控制器已暂停，未接管')
        fd = g.open_pidfd(pid)
        try:
            with FrozenController(fd):
                deadline = time.monotonic() + .3
                while not h.all_threads_stopped(pid):
                    if time.monotonic() >= deadline:
                        raise h.HandoffError('无法暂停自然交接控制器')
                    time.sleep(.005)
                verify_waiter(pid, db, config, old_pid, start)
                h.assert_identity(db, identity)
                verify_waiting_receipt(pid, db.parent, old_pid)
                h.verify_process(old_pid, db, config, api_start)
                immediate.verify_api_lock(old_pid)
                immediate.reject_other_producers(old_pid, db)
                if len(find_waiters(db, config, old_pid)) != 1:
                    raise h.HandoffError('自然交接控制器数量变化')
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                signal.pidfd_send_signal(fd, signal.SIGCONT)
                poll = select.poll(); poll.register(fd, select.POLLIN)
                if not poll.poll(2000):
                    raise h.HandoffError('自然交接控制器未确认退出')
            h.verify_process(old_pid, db, config, api_start)
            signal.pidfd_send_signal(oldfd, signal.SIGCONT)
            immediate.verify_api_lock(old_pid)
            fd_lock = h.acquire_handoff_lock(db)
            os.close(fd_lock)
            print('已取消仍在等待的自然交接；旧 API 已恢复。', flush=True)
        finally:
            os.close(fd)
    finally:
        os.close(oldfd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', required=True, type=int)
    p.add_argument('--db', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    try:
        cancel(a.db, a.config, a.pid)
    except (h.HandoffError, g.GuardError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as exc:
        print('未取消自然交接：' + str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
