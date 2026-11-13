import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from test_api_handoff import RealSubprocessTests
from stream_train import api_cancel_waiter as c
from stream_train import api_handoff as h


PREFIX = ("import ctypes,signal\nL=ctypes.CDLL(None,use_errno=True)\n"
          "def _send(fd,sig):\n r=L.syscall(424,fd,sig,0,0)\n"
          " if r<0: raise OSError(ctypes.get_errno(),'pidfd')\n"
          "signal.pidfd_send_signal=_send\n")


class CancelRealTests(RealSubprocessTests):
    def start_waiter(self, extra=False):
        fake = self.root / 'stream_train' / 'api_handoff.py'
        fake.write_text('''import fcntl,json,sys,time\nfrom pathlib import Path\na=sys.argv; db=Path(a[a.index('--db')+1]); old=int(a[a.index('--pid')+1])\nlock=(db.parent/('.'+db.name+'.api-handoff.lock')).open('a')\nfcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\nr=(db.parent/('api-handoff-'+'a'*32+'.jsonl')).open('w')\nr.write(json.dumps({'state':'waiting','old_pid':old})+'\\n');r.flush()\n'''+("r.write(json.dumps({'state':'terminating_old'})+'\\n');r.flush()\n" if extra else '')+
                        'while True:time.sleep(1)\n')
        env = os.environ.copy(); env['PYTHONPATH'] = str(self.root); env['HOME'] = str(self.root)
        proc = subprocess.Popen([sys.executable, '-m', 'stream_train.api_handoff',
                                 '--pid', str(self.proc.pid), '--db', str(self.db),
                                 '--config', str(self.config)], cwd=str(self.root), env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self.addCleanup(self.kill_waiter, proc)
        deadline = time.monotonic() + 4
        receipt = self.root / ('api-handoff-' + 'a'*32 + '.jsonl')
        while time.monotonic() < deadline:
            if receipt.exists() and receipt.stat().st_size and proc.poll() is None:
                return proc
            time.sleep(.01)
        self.fail('synthetic handoff did not become ready')

    def kill_waiter(self, proc):
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGCONT)
            proc.terminate()
            proc.wait(timeout=3)

    def test_waiting_controller_is_stopped_but_old_api_and_jobs_remain(self):
        waiter = self.start_waiter()
        with patch.dict(os.environ, {'HOME': str(self.root)}), \
             patch.object(c, 'WATCHDOG', PREFIX + c.WATCHDOG):
            c.cancel(self.db, self.config, self.proc.pid)
        waiter.wait(timeout=2)
        self.assertIsNone(self.proc.poll())
        self.assertEqual(self.store.counts()[0]['state'], 'pending')

    def test_advanced_receipt_must_not_kill_controller_or_api(self):
        waiter = self.start_waiter(extra=True)
        with patch.dict(os.environ, {'HOME': str(self.root)}), \
             patch.object(c, 'WATCHDOG', PREFIX + c.WATCHDOG):
            with self.assertRaises(h.HandoffError):
                c.cancel(self.db, self.config, self.proc.pid)
        self.assertIsNone(waiter.poll())
        self.assertIsNone(self.proc.poll())

    def test_cancellation_preserves_api_started_state(self):
        waiter = self.start_waiter()
        self.store.claim(['generate'])
        with patch.dict(os.environ, {'HOME': str(self.root)}), \
             patch.object(c, 'WATCHDOG', PREFIX + c.WATCHDOG):
            c.cancel(self.db, self.config, self.proc.pid)
        waiter.wait(timeout=2)
        self.assertEqual(self.store.counts()[0]['state'], 'started')
        self.assertIsNone(self.proc.poll())

    def test_no_waiter_with_api_already_stopped_refuses(self):
        os.kill(self.proc.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = (Path('/proc') / str(self.proc.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('T', 't'):
                break
            time.sleep(.02)
        with patch.dict(os.environ, {'HOME': str(self.root)}):
            with self.assertRaises(h.HandoffError):
                c.cancel(self.db, self.config, self.proc.pid)

    def test_watchdog_recovers_controller_if_canceller_crashes(self):
        waiter = self.start_waiter()
        helper = r'''import ctypes,os,signal,sys
from stream_train import api_cancel_waiter as c
L=ctypes.CDLL(None,use_errno=True)
def send(fd,sig):
 if L.syscall(424,fd,sig,0,0)<0: raise OSError(ctypes.get_errno(),'pidfd')
def op(pid,flags=0):
 fd=L.syscall(434,pid,flags)
 if fd<0: raise OSError(ctypes.get_errno(),'pidfd')
 return fd
signal.pidfd_send_signal=send; os.pidfd_open=op
prefix="import ctypes,signal\nL=ctypes.CDLL(None,use_errno=True)\ndef s(fd,sig):\n if L.syscall(424,fd,sig,0,0)<0: raise OSError(ctypes.get_errno(),'pidfd')\nsignal.pidfd_send_signal=s\n"
c.WATCHDOG=prefix+c.WATCHDOG
fd=op(int(sys.argv[1]))
with c.FrozenController(fd): os._exit(23)
'''
        env = os.environ.copy(); env['PYTHONPATH'] = str(Path(c.__file__).resolve().parent.parent)
        result = subprocess.run([sys.executable, '-c', helper, str(waiter.pid)], env=env,
                                cwd=env['PYTHONPATH'], timeout=5)
        self.assertEqual(result.returncode, 23)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = (Path('/proc') / str(waiter.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state not in ('T', 't'):
                break
            time.sleep(.02)
        self.assertNotIn(state, ('T', 't'))
        self.assertIsNone(waiter.poll())

    def test_timeout_kills_canceller_before_resuming_controller(self):
        waiter = self.start_waiter()
        helper = r'''import ctypes,os,signal,sys,time
from stream_train import api_cancel_waiter as c
L=ctypes.CDLL(None,use_errno=True)
def send(fd,sig):
 if L.syscall(424,fd,sig,0,0)<0: raise OSError(ctypes.get_errno(),'pidfd')
def op(pid,flags=0):
 fd=L.syscall(434,pid,flags)
 if fd<0: raise OSError(ctypes.get_errno(),'pidfd')
 return fd
signal.pidfd_send_signal=send; os.pidfd_open=op
prefix="import ctypes,signal\nL=ctypes.CDLL(None,use_errno=True)\ndef s(fd,sig):\n if L.syscall(424,fd,sig,0,0)<0: raise OSError(ctypes.get_errno(),'pidfd')\nsignal.pidfd_send_signal=s\n"
c.WATCHDOG=prefix+c.WATCHDOG.replace('end=time.monotonic()+5','end=time.monotonic()+0.5')
fd=op(int(sys.argv[1]))
with c.FrozenController(fd): time.sleep(5)
'''
        env = os.environ.copy(); env['PYTHONPATH'] = str(Path(c.__file__).resolve().parent.parent)
        result = subprocess.run([sys.executable, '-c', helper, str(waiter.pid)], env=env,
                                cwd=env['PYTHONPATH'], timeout=5)
        self.assertEqual(result.returncode, -signal.SIGKILL)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = (Path('/proc') / str(waiter.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state not in ('T', 't'):
                break
            time.sleep(.02)
        self.assertNotIn(state, ('T', 't'))
        self.assertIsNone(waiter.poll())

    def test_cancel_lock_rejects_duplicate(self):
        fd = c.acquire_cancel_lock(self.db)
        try:
            with self.assertRaises(h.HandoffError):
                c.acquire_cancel_lock(self.db)
        finally:
            os.close(fd)


if __name__ == '__main__':
    import unittest
    unittest.main()
