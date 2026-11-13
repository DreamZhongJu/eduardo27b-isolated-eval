import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import select
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from stream_train import __main__ as original
from stream_train import api_handoff as h
from stream_train.store import Store


LIBC = ctypes.CDLL(None, use_errno=True)


def pidfd_open(pid, flags=0):
    fd = LIBC.syscall(434, pid, flags)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    return fd


def pidfd_send_signal(fd, sig):
    if LIBC.syscall(424, fd, sig, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_send_signal')


class RealSubprocessTests(unittest.TestCase):
    def setUp(self):
        if sys.platform != 'linux' or os.uname().machine != 'x86_64':
            self.skipTest('real pidfd integration requires x86_64 Linux')
        self.tmp = tempfile.TemporaryDirectory(dir='/dev/shm')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'queue.sqlite'
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'namespace': 'synthetic',
            'api_base': 'https://api.agnes-ai.cn/v1', 'api_model': 'agnes-3.0-flash',
            'student_base': 'http://127.0.0.1:8000/v1', 'student_model': 'student-8b'}))
        self.store = Store(self.db)
        self.store.bind('protocol', original.config(self.config))
        self.store.enqueue('sample', 'generate', {})
        fake = self.root / 'stream_train'
        fake.mkdir()
        (fake / '__init__.py').write_text('')
        (fake / '__main__.py').write_text(
            'import fcntl,os,time\nfrom pathlib import Path\n'
            'p=Path.home()/".cache"/"math-transfer-stream"/"api.lock"\n'
            'p.parent.mkdir(parents=True,exist_ok=True)\n'
            'f=p.open("a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
            'while True: time.sleep(1)\n')
        env = os.environ.copy(); env['PYTHONPATH'] = str(self.root); env['HOME'] = str(self.root)
        self.proc = subprocess.Popen([sys.executable, '-m', 'stream_train', '--db', str(self.db),
            '--config', str(self.config), 'api'], cwd=str(self.root), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.cleanup_child)
        p1 = patch.object(os, 'pidfd_open', pidfd_open, create=True); p1.start(); self.addCleanup(p1.stop)
        p2 = patch.object(signal, 'pidfd_send_signal', pidfd_send_signal, create=True); p2.start(); self.addCleanup(p2.stop)
        self.fd = pidfd_open(self.proc.pid)
        self.addCleanup(os.close, self.fd)
        # This Python 3.8 build lacks signal.pidfd_send_signal, so give the
        # independent watchdog the same Linux syscall adapter used above.
        prefix = ("import ctypes,signal\nL=ctypes.CDLL(None,use_errno=True)\n"
                  "def _send(fd,sig):\n r=L.syscall(424,fd,sig,0,0)\n"
                  " if r<0: raise OSError(ctypes.get_errno(),'pidfd')\n"
                  "signal.pidfd_send_signal=_send\n")
        p3 = patch.object(h, 'WATCHDOG', prefix + h.WATCHDOG); p3.start(); self.addCleanup(p3.stop)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                self.start = h.verify_process(self.proc.pid, self.db, self.config)
                break
            except h.HandoffError:
                time.sleep(.01)
        else:
            self.fail('synthetic API process did not start')
        lockpath = self.root / '.cache/math-transfer-stream/api.lock'
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if lockpath.exists():
                with lockpath.open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    except BlockingIOError:
                        break
            time.sleep(.01)
        else:
            self.fail('synthetic API did not obtain the lock')
        st = self.db.stat(); self.identity = (st.st_dev, st.st_ino)

    def cleanup_child(self):
        if self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGCONT)
            self.proc.terminate()
            self.proc.wait(timeout=3)

    def attempt(self):
        return h.attempt(self.fd, self.proc.pid, self.start,
                         self.db, self.config, self.identity)

    def test_started_request_resumes_process_without_mutating_queue(self):
        self.store.claim(['generate'])
        done, started = self.attempt()
        self.assertFalse(done)
        self.assertEqual(started, 1)
        self.assertIsNone(self.proc.poll())
        self.assertEqual(self.store.counts()[0]['state'], 'started')

    def test_busy_writer_resumes_process_without_waiting_for_writer(self):
        writer = sqlite3.connect(str(self.db), isolation_level=None)
        try:
            writer.execute('BEGIN IMMEDIATE')
            before = time.monotonic()
            done, started = self.attempt()
            self.assertFalse(done)
            self.assertIsNone(started)
            self.assertLess(time.monotonic() - before, 2)
            self.assertIsNone(self.proc.poll())
        finally:
            writer.rollback(); writer.close()

    def test_zero_started_terminates_only_old_process_and_keeps_pending(self):
        self.assertEqual(self.attempt(), (True, 0))
        self.proc.wait(timeout=2)
        self.assertEqual(self.store.counts()[0]['state'], 'pending')

    def test_identity_and_protocol_mismatch_refused(self):
        with self.assertRaises(h.HandoffError):
            h.verify_process(self.proc.pid, self.db, self.root / 'wrong.json')
        self.config.write_text('{}')
        with self.assertRaises(ValueError):
            h.validate_binding(self.db, self.config)

    def test_watchdog_resumes_process_if_controller_is_killed(self):
        # The subprocess exits abruptly inside the paused context. Its pipe
        # closes, so the independent watchdog must send SIGCONT to the pidfd.
        helper = r'''
import ctypes, os, signal, sys
from stream_train import api_handoff as h
L=ctypes.CDLL(None,use_errno=True)
def send(fd,sig):
    if L.syscall(424,fd,sig,0,0)<0: raise OSError(ctypes.get_errno(),'pidfd')
signal.pidfd_send_signal=send
prefix="import ctypes,signal\nL=ctypes.CDLL(None,use_errno=True)\ndef s(fd,sig):\n if L.syscall(424,fd,sig,0,0)<0: raise OSError(ctypes.get_errno(),'pidfd')\nsignal.pidfd_send_signal=s\n"
h.WATCHDOG=prefix+h.WATCHDOG
fd=L.syscall(434,int(sys.argv[1]),0)
with h.PauseGuard(fd,int(sys.argv[1])):
    os._exit(23)
'''
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(h.__file__).resolve().parent.parent)
        result = subprocess.run([sys.executable, '-c', helper, str(self.proc.pid)],
                                cwd=env['PYTHONPATH'], env=env, timeout=5)
        self.assertEqual(result.returncode, 23)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = (Path('/proc') / str(self.proc.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state not in ('T', 't'):
                break
            time.sleep(.02)
        self.assertNotIn(state, ('T', 't'))
        self.assertIsNone(self.proc.poll())

    def test_watchdog_resumes_if_pause_exceeds_one_second(self):
        with h.PauseGuard(self.fd, self.proc.pid):
            time.sleep(1.4)
            state = (Path('/proc') / str(self.proc.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            self.assertNotIn(state, ('T', 't'))

    def test_real_new_producer_readiness_and_policy_without_network(self):
        # No pending jobs => no HTTP calls; use the real replacement module.
        self.store.claim(['generate'])
        self.store.finish('sample', {'error_type': 'synthetic'}, state='unknown')
        # Exercise actual stop-under-writer-lock before the real producer
        # acquires the same HOME lock; the unknown task must remain intact.
        h.handoff(self.proc.pid, self.db, self.config, timeout=3, poll_interval=.01)
        r, w = os.pipe()
        env = os.environ.copy()
        env['HOME'] = str(self.root)
        env['PYTHONPATH'] = str(Path(h.__file__).resolve().parent.parent)
        env['AGNES_API_KEY'] = 'synthetic-never-used'
        env['STREAM_PRODUCER_READY_FD'] = str(w)
        policy = self.root / 'policy.json'
        producer = subprocess.Popen([sys.executable, '-m', 'stream_train.api_adaptive_external',
            '--db', str(self.db), '--config', str(self.config), '--max-concurrency', '2',
            '--policy-output', str(policy)], cwd=env['PYTHONPATH'], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            pass_fds=(w,))
        os.close(w)
        try:
            self.assertTrue(select.select([r], [], [], 5)[0])
            self.assertEqual(os.read(r, 1), b'R')
            self.assertEqual(json.loads(policy.read_text())['maximum'], 2)
            self.assertEqual(self.store.counts()[0]['state'], 'unknown')
        finally:
            os.close(r)
            if producer.poll() is None:
                producer.terminate()
            producer.wait(timeout=5)


class ReadyHandshakeTests(unittest.TestCase):
    def run_main(self, replacement_ready):
        with tempfile.TemporaryDirectory() as d:
            receipt = str(Path(d) / 'receipt.jsonl')
            db = Path(d) / 'queue.sqlite'
            db.touch()
            argv = ['api_handoff', '--pid', '1234', '--db', str(db),
                    '--config', '/tmp/config.json', '--receipt', receipt]
            checks = []
            def check_locked(*args, **kwargs):
                with self.assertRaises(h.HandoffError):
                    h.acquire_handoff_lock(db)
                checks.append('waiting')
            class Child:
                pid = 9123
                def poll(self): return None if replacement_ready else 1
                def wait(self): return 0
            def spawn(*args, **kwargs):
                with self.assertRaises(h.HandoffError):
                    h.acquire_handoff_lock(db)
                checks.append('ready_wait')
                if replacement_ready:
                    os.write(int(kwargs['env']['STREAM_PRODUCER_READY_FD']), b'R')
                return Child()
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'AGNES_API_KEY': 'test'}), \
                 patch.object(h.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                 patch.object(h.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(h, 'validate_binding', return_value=(1, 2)), \
                 patch.object(h, 'handoff', side_effect=check_locked), \
                 patch.object(h, 'try_restore_original', return_value=9124) as fallback:
                code = h.main()
                events = [json.loads(row)['state'] for row in Path(receipt).read_text().splitlines()]
                self.assertEqual(checks, ['waiting', 'ready_wait'])
                unlocked = h.acquire_handoff_lock(db)
                os.close(unlocked)
                return code, events, fallback.call_count

    def test_ready_event_is_recorded_only_after_pipe_signal(self):
        code, events, fallback = self.run_main(True)
        self.assertEqual(code, 0)
        self.assertIn('replacement_ready', events)
        self.assertNotIn('replacement_not_ready', events)
        self.assertEqual(fallback, 0)

    def test_dead_replacement_attempts_fallback_without_claiming_success(self):
        code, events, fallback = self.run_main(False)
        self.assertEqual(code, 1)
        self.assertIn('replacement_not_ready', events)
        self.assertNotIn('replacement_ready', events)
        self.assertEqual(fallback, 1)

    def test_second_handoff_for_same_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / 'queue.sqlite'
            db.touch()
            fd = h.acquire_handoff_lock(db)
            try:
                with self.assertRaises(h.HandoffError):
                    h.acquire_handoff_lock(db)
            finally:
                os.close(fd)
            fd = h.acquire_handoff_lock(db)
            os.close(fd)


if __name__ == '__main__':
    unittest.main()
