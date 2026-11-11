"""Identify and signal only processes carrying this supervisor's unique tag."""
import os
import signal
import time
import ctypes
import platform
from pathlib import Path


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.machine() != 'x86_64':
        raise OSError('pidfd fallback supports x86_64 only')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, 'pidfd_open failed')
    return fd


def signal_pidfd(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
        return
    if platform.machine() != 'x86_64':
        raise OSError('pidfd fallback supports x86_64 only')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(424, fd, sig, 0, 0)
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, 'pidfd_send_signal failed')


def stat_session(path):
    # /proc/<pid>/stat comm may contain spaces and ')' characters.
    raw = path.read_text()
    end = raw.rfind(')')
    if end < 0:
        raise ValueError('invalid process stat')
    fields = raw[end + 2:].split()
    return int(fields[3])


def arguments(argv, option):
    try:
        i = argv.index(option)
        return argv[i + 1]
    except (ValueError, IndexError):
        return None


def prior_conflict(root, proc_root=Path('/proc')):
    root_bytes = str(root).encode()
    for p in proc_root.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = (p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            raise ValueError('cannot inspect process')
        is_guard = any(a.endswith(b'/evaluator_guard.py') for a in argv)
        is_supervisor = any(a.endswith(b'/supervisor.py') for a in argv) and b'--run' in argv
        if is_guard and arguments(argv, b'--results') in {root_bytes + b'/' + s for s in (b'EASY', b'MEDIUM', b'HARD')}:
            return True
        if is_supervisor:
            # Production supervisor is restricted to this one root.
            return True
        is_service = (b'vllm.entrypoints.openai.api_server' in argv or
                      any(a.endswith(b'/watch_sources.py') or b'EngineCore' in a or b'VLLM::' in a for a in argv))
        if is_service:
            try:
                env = (p/'environ').read_bytes().split(b'\0')
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                raise ValueError('cannot inspect service environment')
            if b'FIVEGPU_RUN_ROOT=' + root_bytes in env:
                return True
    return False


def signal_session_tagged(session, run_tag, sig, proc_root=Path('/proc')):
    """Use pidfds; no signalling based only on a potentially reused PGID."""
    sent = 0
    expected = b'FIVEGPU_RUN_ID=' + run_tag.encode()
    for p in proc_root.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if stat_session(p/'stat') != session:
                continue
            pid = int(p.name)
            fd = open_pidfd(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            # pidfd pins the identity during this recheck and signal.
            if stat_session(p/'stat') != session:
                continue
            env = (p/'environ').read_bytes().split(b'\0')
            if expected not in env:
                continue
            signal_pidfd(fd, sig)
            sent += 1
        except (FileNotFoundError, ProcessLookupError):
            continue
        finally:
            os.close(fd)
    return sent


def terminate_tagged(proc, run_tag, grace=10):
    # Even if the leader has exited, tagged children in the original session
    # are found; unrelated processes cannot be selected by PGID alone.
    signal_session_tagged(proc.pid, run_tag, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(.1)
    signal_session_tagged(proc.pid, run_tag, signal.SIGKILL)
    if proc.poll() is None:
        proc.wait(timeout=10)
