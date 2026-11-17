"""Guard one task-owned process tree after a controller dies.

Linux x86-64 only.  The guard is a subreaper; it polls its own controller's
parent relationship.  On controller death or SIGTERM it signals only current
and adopted descendants via pidfd (no process-name matching, no NVML PID kill).
This file is a draft and is not wired to a runnable training entrypoint.
"""
from __future__ import annotations
import ctypes
import errno
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER=36
SYS_PIDFD_SEND_SIGNAL=424
SYS_PIDFD_OPEN=434
libc=ctypes.CDLL(None,use_errno=True)


def syscall(number,*args):
    result=libc.syscall(number,*args)
    if result<0:
        err=ctypes.get_errno()
        raise OSError(err,os.strerror(err))
    return result


def subreaper():
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','amd64'):
        raise RuntimeError('仅支持 Linux x86_64')
    if libc.prctl(PR_SET_CHILD_SUBREAPER,1,0,0,0)!=0:
        err=ctypes.get_errno(); raise OSError(err,os.strerror(err))
    # Check pidfd syscalls before starting any task process.
    fd=syscall(SYS_PIDFD_OPEN,os.getpid(),0)
    try: syscall(SYS_PIDFD_SEND_SIGNAL,fd,0,0,0)
    finally: os.close(fd)


def stat_of(pid):
    p=Path('/proc')/str(pid)/'stat'
    line=p.read_text()
    fields=line[line.rfind(')')+2:].split()
    return {'state':fields[0], 'ppid':int(fields[1]), 'start':int(fields[19])}


def child_ids(pid):
    result=set()
    task=Path('/proc')/str(pid)/'task'
    try:
        entries=list(task.iterdir())
    except FileNotFoundError:
        return result
    for thread in entries:
        try:
            text=(thread/'children').read_text()
            result.update(int(x) for x in text.split())
        except FileNotFoundError:
            continue
    return result


def descendants(guard_pid):
    """Collect only the guard's descendant tree, using each thread's children."""
    result={}; pending=[guard_pid]; seen={guard_pid}
    while pending:
        parent=pending.pop()
        for pid in child_ids(parent):
            if pid in seen: continue
            seen.add(pid)
            try: row=stat_of(pid)
            except (FileNotFoundError,ProcessLookupError): continue
            # A child may exit/reparent during scanning; fail closed on mismatch.
            if row['ppid']!=parent: continue
            result[pid]=row; pending.append(pid)
    return result


def send_owned(pid, expected_start, sig):
    """A pidfd pins the observed process so PID reuse cannot target another."""
    try:
        fd=syscall(SYS_PIDFD_OPEN,pid,0)
    except OSError as e:
        if e.errno==errno.ESRCH: return
        raise
    try:
        try: row=stat_of(pid)
        except (FileNotFoundError,ProcessLookupError): return
        if row['start']!=expected_start: return
        try: syscall(SYS_PIDFD_SEND_SIGNAL,fd,int(sig),0,0)
        except OSError as e:
            if e.errno!=errno.ESRCH: raise
    finally: os.close(fd)


def reap_children():
    while True:
        try:
            pid,_=os.waitpid(-1,os.WNOHANG)
            if not pid: return
        except ChildProcessError: return


def cleanup(guard_pid, grace=5, kill_wait=10):
    """Conservatively terminate only our descendants, including adopted ones."""
    for sig,duration in ((signal.SIGTERM,grace),(signal.SIGKILL,kill_wait)):
        deadline=time.monotonic()+duration
        while True:
            reap_children()
            rows=descendants(guard_pid)
            if not rows: return
            # Child-first where possible, to reduce reparenting churn.
            for pid,row in reversed(list(rows.items())):
                if row['state']!='Z': send_owned(pid,row['start'],sig)
            if time.monotonic()>=deadline: break
            time.sleep(.05)
    reap_children()
    if descendants(guard_pid):
        raise RuntimeError('无法确认本任务全部后代退出')


def guard(parent_pid, command):
    if not command: raise ValueError('missing owned child command')
    subreaper()
    if os.getppid()!=parent_pid:
        return 3
    stopped=[False]
    def request_stop(_sig,_frame): stopped[0]=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    # The controller might die at any point; check once more immediately before spawn.
    if os.getppid()!=parent_pid: return 3
    child=subprocess.Popen(command,env=os.environ.copy(),start_new_session=True)
    try:
        while True:
            if stopped[0] or os.getppid()!=parent_pid:
                return 3
            rc=child.poll()
            if rc is not None:
                return rc if rc>=0 else 128-rc
            time.sleep(.1)
    finally:
        cleanup(os.getpid())


def main(argv):
    if len(argv)<4 or argv[1]!='--parent' or argv[3]!='--':
        raise ValueError('usage: parent_guard.py --parent PID -- command ...')
    return guard(int(argv[2]),argv[4:])

if __name__=='__main__':
    try: sys.exit(main(sys.argv))
    except Exception as e:
        print('任务进程清理异常：'+type(e).__name__,file=sys.stderr)
        sys.exit(1)
