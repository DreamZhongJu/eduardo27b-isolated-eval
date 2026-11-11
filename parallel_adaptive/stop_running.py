#!/usr/bin/env python3
"""Explicitly stop ONLY the pinned old supervisor and await its own cleanup.

This action may interrupt an active source or API call. It never deletes or
rewrites records, does not signal child processes, and does not start another
controller. Run only with explicit authorization for this interruption.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import select
import signal
import stat
import time
from pathlib import Path

SCRIPT = Path('/workspace/eduardo27b-shared-wait-ecc7a92/supervisor.py')
SCRIPT_SHA = 'b642d842588a27644136d15783eff73d4747760cbd5e220a6e54a6f98d25ce33'
ROOT = Path('/workspace/results/own-test-499-v1')
ROOT_LOCK = Path('/workspace/.own-test-499.lock')
PROC = Path('/proc')


def exact_supervisor(args):
    wanted = os.fsencode(str(SCRIPT))
    if args.count(wanted) != 1 or args.count(b'--run') != 1 or args.count(b'--wait-gpus') != 1:
        return False
    # Repeated or custom output root is a different operation.
    roots = [i for i, item in enumerate(args) if item == b'--root' or item.startswith(b'--root=')]
    if len(roots) > 1:
        return False
    if roots:
        at = roots[0]
        item = args[at]
        root = item.partition(b'=')[2] if item.startswith(b'--root=') else (args[at + 1] if at + 1 < len(args) else b'')
        if root != os.fsencode(str(ROOT)):
            return False
    return True


def cmdline(path):
    return [x for x in (path / 'cmdline').read_bytes().split(b'\0') if x]


def list_matches(proc=PROC):
    matches = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if exact_supervisor(cmdline(p)):
                matches.append(int(p.name))
        except (ProcessLookupError, FileNotFoundError):
            continue
        except OSError as exc:
            raise ValueError('process inspection unavailable') from exc
    return matches


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.machine() != 'x86_64':
        raise OSError('pidfd unsupported')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    return fd


def send_pidfd(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
        return
    if platform.machine() != 'x86_64':
        raise OSError('pidfd unsupported')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, sig, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')


def exited(fd):
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def verify_script():
    if SCRIPT.is_symlink() or not SCRIPT.is_file():
        raise ValueError('missing pinned supervisor')
    if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != SCRIPT_SHA:
        raise ValueError('supervisor hash mismatch')


def lock_free(lock=ROOT_LOCK):
    if lock.is_symlink():
        raise ValueError('lock symlink')
    fd = os.open(lock, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('invalid lock')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def conflict(proc=PROC):
    root = os.fsencode(str(ROOT))
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = cmdline(p)
            if exact_supervisor(args) or (b'--run' in args and any(a.endswith(b'/supervisor.py') for a in args)):
                return True
            guard = any(a.endswith(b'/evaluator_guard.py') for a in args)
            if guard and any(a in {root + b'/EASY', root + b'/MEDIUM', root + b'/HARD'} for a in args):
                return True
            potentially_tagged = (b'vllm.entrypoints.openai.api_server' in args or
                                  any(a.endswith(b'/watch_sources.py') or b'EngineCore' in a or b'VLLM::' in a
                                      for a in args))
            if potentially_tagged:
                env = (p / 'environ').read_bytes().split(b'\0')
                if b'FIVEGPU_RUN_ROOT=' + root in env:
                    return True
        except (ProcessLookupError, FileNotFoundError):
            continue
        except OSError as exc:
            raise ValueError('process inspection unavailable') from exc
    return False


def stop(timeout=180, proc=PROC):
    verify_script()
    matches = list_matches(proc)
    if len(matches) > 1:
        return 'MULTIPLE_MATCHES'
    if not matches:
        return 'ALREADY_STOPPED_READY' if not conflict(proc) and lock_free() else 'CONFLICT_OR_LOCKED'
    pid = matches[0]
    fd = open_pidfd(pid)
    try:
        if exited(fd) or not exact_supervisor(cmdline(proc / str(pid))):
            return 'PROCESS_CHANGED'
        # Only this supervisor is signalled. It owns cleanup of its own children.
        send_pidfd(fd, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if exited(fd) and not conflict(proc) and lock_free():
                return 'STOPPED_READY'
            time.sleep(.5)
        return 'STOP_PENDING_OR_CONFLICT'
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description='Explicitly stop the pinned old supervisor; active work can be interrupted.')
    p.add_argument('--stop', action='store_true', required=True)
    p.add_argument('--accept-interrupted-source', action='store_true', required=True)
    args = p.parse_args()
    try:
        state = stop()
    except Exception:
        state = 'INSPECTION_FAILED'
    print(json.dumps({'state': state, 'active_work_may_be_interrupted': True}), flush=True)
    return 0 if state in ('STOPPED_READY', 'ALREADY_STOPPED_READY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
