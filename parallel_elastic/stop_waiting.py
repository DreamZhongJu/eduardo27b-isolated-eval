#!/usr/bin/env python3
"""Stop exactly the old, currently waiting supervisor; never start a job.

Output is fixed aggregate state only.  SIGSTOP closes the race between the
status check and requesting termination; a finally block always resumes it.
"""
import fcntl
import importlib.util
import json
import os
import select
import signal
import stat
import sys
import time
from pathlib import Path

OLD_DIR = Path('/workspace/eduardo27b-shared-wait-ecc7a92')
ROOT = Path('/workspace/results/own-test-499-v1')
LOCK = Path('/workspace/.own-test-499.lock')
INTERRUPTS = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)


class HandoffInterrupted(Exception):
    pass


def install_signal_handlers():
    triggered = [False]
    def interrupt(_signum, _frame):
        if triggered[0]:
            return
        triggered[0] = True
        raise HandoffInterrupted('interrupted')
    for signum in INTERRUPTS:
        signal.signal(signum, interrupt)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_args(pid, proc):
    return (proc / str(pid) / 'cmdline').read_bytes().split(b'\0')


def matches(args, script):
    return os.fsencode(str(script)) in args and b'--run' in args and b'--wait-gpus' in args


def start_token(pid, proc):
    raw = (proc / str(pid) / 'stat').read_text()
    end = raw.rfind(')')
    if end < 0:
        raise ValueError('invalid stat')
    fields = raw[end + 2:].split()
    return fields[19]


def stopped(pid, proc):
    for line in (proc / str(pid) / 'status').read_text().splitlines():
        if line.startswith('State:'):
            return line.split()[1] in ('T', 't')
    raise ValueError('missing state')


def children(pid, proc):
    found = []
    for task in (proc / str(pid) / 'task').iterdir():
        if task.name.isdigit():
            found.extend((task / 'children').read_text().split())
    return found


def exited(fd, timeout_ms=0):
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(timeout_ms))


def ready(root, lock, guard):
    fd = os.open(lock, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            return not guard.prior_conflict(root)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def stop_waiting(script, root, lock, guard, status, proc=Path('/proc'), timeout=30):
    candidates = []
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if matches(process_args(int(path.name), proc), script):
                candidates.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return 'PROCESS_INSPECTION_FAILED'
    if not candidates:
        return 'ALREADY_STOPPED_READY' if ready(root, lock, guard) else 'CONFLICT_OR_LOCKED'
    if len(candidates) != 1:
        return 'MULTIPLE_MATCHES'
    pid = candidates[0]
    before = start_token(pid, proc)
    fd = guard.open_pidfd(pid)
    paused = False
    try:
        if exited(fd) or before != start_token(pid, proc) or not matches(process_args(pid, proc), script):
            return 'PROCESS_CHANGED'
        paused = True
        guard.signal_pidfd(fd, signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while not stopped(pid, proc):
            if exited(fd) or time.monotonic() >= deadline:
                return 'STOP_FAILED'
            time.sleep(.05)
        # No generation may start while the supervisor is stopped.  Any child
        # (including a transient nvidia-smi probe) makes this attempt defer.
        if (before != start_token(pid, proc) or not matches(process_args(pid, proc), script)
                or status.wait_phase(root).get('state') != 'WAITING_FOR_FREE_GPUS'):
            return 'PHASE_CHANGED'
        if children(pid, proc):
            return 'CHILD_ACTIVE_TRY_AGAIN'
        guard.signal_pidfd(fd, signal.SIGTERM)
        guard.signal_pidfd(fd, signal.SIGCONT)
        paused = False
        if not exited(fd, int(timeout * 1000)):
            return 'STOP_PENDING'
    finally:
        # A second HUP/TERM/INT must not interrupt the cleanup before SIGCONT.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, INTERRUPTS)
        try:
            if paused:
                try:
                    guard.signal_pidfd(fd, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            os.close(fd)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return 'STOPPED_READY' if ready(root, lock, guard) else 'CONFLICT_OR_LOCKED'


if __name__ == '__main__':
    try:
        install_signal_handlers()
        guard = load_module('handoff_process_guard', OLD_DIR / 'process_guard.py')
        status = load_module('handoff_wait_status', OLD_DIR / 'status.py')
        result = stop_waiting(OLD_DIR / 'supervisor.py', ROOT, LOCK, guard, status)
    except Exception:
        result = 'INSPECTION_FAILED'
    print(json.dumps({'state': result}))
    sys.exit(0 if result in ('STOPPED_READY', 'ALREADY_STOPPED_READY') else 2)
