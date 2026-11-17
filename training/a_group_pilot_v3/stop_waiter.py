#!/usr/bin/env python3
"""Safely stop only the old smoke1 controller if it is still waiting for GPUs."""
import fcntl
import ctypes
import json
import os
import platform
from pathlib import Path
import select
import signal
import time

ROOT = Path('/workspace/A_group_repro_20261117')
OLD = ROOT / 'pilot-v2/pilot.py'
STATE = ROOT / 'pilots/smoke1/status.json'
LOCK = ROOT / '.pilot.lock'

def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('平台不支持安全切换')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(434, pid, 0)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result

def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('平台不支持安全切换')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(424, fd, int(sig), 0, 0)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))

def unlocked():
    with LOCK.open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True

def main():
    if unlocked():
        print('没有运行中的旧控制器，可启动新任务。')
        return
    state = json.loads(STATE.read_text())
    pid = state.get('pid')
    if type(pid) is not int or pid <= 1:
        raise RuntimeError('无法确认旧任务身份')
    fd = pidfd_open(pid)
    stopped = False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if (str(OLD).encode() not in args or b'--wait-gpus' not in args or
            b'--name' not in args or b'smoke1' not in args):
            raise RuntimeError('旧进程身份不符，未停止任务')
        if Path(f'/proc/{pid}').stat().st_uid != os.geteuid():
            raise RuntimeError('旧进程用户不符，未停止任务')
        pidfd_signal(fd, signal.SIGSTOP)
        stopped = True
        for _ in range(50):
            text = Path(f'/proc/{pid}/status').read_text()
            if any(line.startswith('State:') and '\tT' in line for line in text.splitlines()):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('无法冻结旧进程，已取消切换')
        current = json.loads(STATE.read_text())
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        if current.get('pid') != pid or current.get('phase') != 'waiting_gpu' or children:
            raise RuntimeError('旧任务已经进入其他阶段或有子进程，已保留')
        pidfd_signal(fd, signal.SIGTERM)
        pidfd_signal(fd, signal.SIGCONT)
        stopped = False
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        if not poll.poll(30000):
            raise RuntimeError('旧任务未及时退出，请检查；未启动新任务')
    finally:
        if stopped:
            try: pidfd_signal(fd, signal.SIGCONT)
            except ProcessLookupError: pass
        os.close(fd)
    if not unlocked():
        raise RuntimeError('旧任务锁仍占用；未启动新任务')
    print('已安全停止旧等待任务，锁已释放。')

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print('未切换：' + str(e))
        raise SystemExit(1)
