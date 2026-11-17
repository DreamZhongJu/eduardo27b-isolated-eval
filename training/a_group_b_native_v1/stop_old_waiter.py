#!/usr/bin/env python3
"""Stop only a verified, frozen A-group GPU-waiting controller."""
import ctypes
import fcntl
import json
import os
import platform
import select
import signal
import time
from pathlib import Path

ROOT = Path('/workspace/A_group_repro_20261117')
LOCK = ROOT / '.pilot.lock'
CANDIDATES = (('pilot-v2', 'smoke1'), ('pilot-v3', 'smoke_reserve1'))


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('平台不支持安全切换')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('平台不支持安全切换')
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.syscall(424, fd, int(sig), 0, 0)
    if rc < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def unlocked():
    with LOCK.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def read_args(pid):
    return [x for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]


def child_pids(pid):
    # /proc task children are thread-specific; inspect every thread.
    children = set()
    tasks = Path(f'/proc/{pid}/task')
    for task in tasks.iterdir():
        if task.name.isdigit():
            try:
                children.update(int(x) for x in (task/'children').read_text().split())
            except FileNotFoundError:
                # If a task disappeared we cannot prove a complete scan.
                raise RuntimeError('进程线程变化，未切换')
    return children


def is_own_holder(child_pid, parent_pid):
    p = Path(f'/proc/{child_pid}')
    try:
        if p.stat().st_uid != os.geteuid():
            return False
        args = read_args(child_pid)
        expected = str(ROOT / 'pilot-v3/reserve_worker.py').encode()
        if expected not in args or not args or args[-1] != str(parent_pid).encode():
            return False
        status = (p/'status').read_text()
        return any(line == f'PPid:\t{parent_pid}' for line in status.splitlines())
    except (FileNotFoundError, ProcessLookupError):
        # Transient children should be rescanned before permitting a stop.
        return False


def find_live_candidates():
    found = []
    for version, name in CANDIDATES:
        state_path = ROOT/'pilots'/name/'status.json'
        try:
            state = json.loads(state_path.read_text())
            pid = state.get('pid')
            if type(pid) is not int or pid <= 1:
                continue
            args = read_args(pid)
            expected = str(ROOT/version/'pilot.py').encode()
            if (Path(f'/proc/{pid}').stat().st_uid == os.geteuid() and
                expected in args and b'--wait-gpus' in args and
                b'--name' in args and name.encode() in args and
                b'--steps' in args and b'1' in args):
                found.append((version, name, state_path, pid))
        except (OSError, ValueError):
            continue
    return found


def run():
    if unlocked():
        print('无运行中的旧控制器，可启动新任务。')
        return
    matches = find_live_candidates()
    if len(matches) != 1:
        raise RuntimeError('无法唯一确认旧控制器身份，未切换')
    version, name, state_path, pid = matches[0]
    fd = pidfd_open(pid)
    frozen = False
    try:
        # Revalidate identity after fixing PID, then freeze to close the phase race.
        expected = str(ROOT/version/'pilot.py').encode()
        args = read_args(pid)
        if expected not in args or name.encode() not in args:
            raise RuntimeError('旧控制器身份变化，未切换')
        pidfd_signal(fd, signal.SIGSTOP)
        frozen = True
        for _ in range(50):
            status = Path(f'/proc/{pid}/status').read_text()
            if any(line.startswith('State:') and '\tT' in line for line in status.splitlines()):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('无法暂停旧控制器，未切换')
        current = json.loads(state_path.read_text())
        if current.get('pid') != pid or current.get('phase') != 'waiting_gpu':
            raise RuntimeError('旧任务已开始或状态不明，已保留')
        # nvidia-smi can be a transient child. Frozen controller cannot create
        # any more, so give pre-existing transient children time to exit.
        for _ in range(30):
            ids = child_pids(pid)
            if not ids or (version == 'pilot-v3' and all(is_own_holder(c, pid) for c in ids)):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('发现非本任务预留进程，已保留旧任务')
        # No new children can be started while parent is frozen. SIGTERM is
        # delivered after SIGCONT and the controller cleans up only its holders.
        pidfd_signal(fd, signal.SIGTERM)
        pidfd_signal(fd, signal.SIGCONT)
        frozen = False
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        if not poll.poll(30000):
            raise RuntimeError('旧控制器未及时退出，不启动新任务')
    finally:
        if frozen:
            try:
                pidfd_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
        os.close(fd)
    for _ in range(50):
        if unlocked():
            print('旧等待任务已安全退出，可启动新任务。')
            return
        time.sleep(.1)
    raise RuntimeError('任务锁仍被占用，不启动新任务')


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print('未切换：' + str(e))
        raise SystemExit(1)
