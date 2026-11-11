import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from stop_waiting import matches, stop_waiting

GUARD = Path(__file__).resolve().parents[1] / 'parallel_shared_wait' / 'process_guard.py'
spec = importlib.util.spec_from_file_location('real_process_guard', GUARD)
guard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_module)
status_spec = importlib.util.spec_from_file_location('real_wait_status', GUARD.with_name('status.py'))
status_module = importlib.util.module_from_spec(status_spec)
status_spec.loader.exec_module(status_module)


class Handoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script = self.root / 'supervisor.py'
        self.lock = self.root / 'old.lock'
        self.lock.touch()
        self.ready = self.root / 'ready'
        self.script.write_text(
            'import fcntl, pathlib, signal, sys, time\n'
            'lock = open(sys.argv[3], "r+")\n'
            'fcntl.flock(lock, fcntl.LOCK_EX)\n'
            'signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n'
            'pathlib.Path(sys.argv[4]).write_text("ready")\n'
            'while True: time.sleep(0.1)\n')
        self.guard = types.SimpleNamespace(open_pidfd=guard_module.open_pidfd,
            signal_pidfd=guard_module.signal_pidfd, prior_conflict=lambda _: False)
        self.proc = None

    def tearDown(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.tmp.cleanup()

    def start(self):
        self.proc = subprocess.Popen([sys.executable, str(self.script), '--run', '--wait-gpus',
                                      str(self.lock), str(self.ready)])
        end = time.monotonic() + 5
        while not self.ready.exists():
            if self.proc.poll() is not None or time.monotonic() >= end:
                self.fail('synthetic supervisor failed')
            time.sleep(.02)

    def status(self, phase):
        return types.SimpleNamespace(wait_phase=lambda _: {'state': phase})

    def test_waiting_stops_and_releases_lock(self):
        self.start()
        outcome = stop_waiting(self.script, self.root, self.lock, self.guard,
                               self.status('WAITING_FOR_FREE_GPUS'), timeout=5)
        self.assertEqual(outcome, 'STOPPED_READY')
        self.assertEqual(self.proc.wait(timeout=5), 0)

    def test_changed_phase_is_resumed_and_not_stopped(self):
        self.start()
        outcome = stop_waiting(self.script, self.root, self.lock, self.guard,
                               self.status('PILOT_RUNNING'), timeout=5)
        self.assertEqual(outcome, 'PHASE_CHANGED')
        self.assertIsNone(self.proc.poll())
        state = (Path('/proc') / str(self.proc.pid) / 'status').read_text()
        self.assertNotIn('State:\tT', state)

    def test_active_child_defers_and_resumes(self):
        self.start()
        with mock.patch('stop_waiting.children', return_value=['123']):
            outcome = stop_waiting(self.script, self.root, self.lock, self.guard,
                                   self.status('WAITING_FOR_FREE_GPUS'), timeout=5)
        self.assertEqual(outcome, 'CHILD_ACTIVE_TRY_AGAIN')
        self.assertIsNone(self.proc.poll())
        state = (Path('/proc') / str(self.proc.pid) / 'status').read_text()
        self.assertNotIn('State:\tT', state)

    def test_inspection_error_resumes_process(self):
        self.start()
        def fail(_):
            raise OSError('synthetic')
        with self.assertRaises(OSError):
            stop_waiting(self.script, self.root, self.lock, self.guard,
                         types.SimpleNamespace(wait_phase=fail), timeout=5)
        self.assertIsNone(self.proc.poll())
        state = (Path('/proc') / str(self.proc.pid) / 'status').read_text()
        self.assertNotIn('State:\tT', state)

    def test_conflict_after_exit_prevents_ready(self):
        self.start()
        self.guard.prior_conflict = lambda _: True
        outcome = stop_waiting(self.script, self.root, self.lock, self.guard,
                               self.status('WAITING_FOR_FREE_GPUS'), timeout=5)
        self.assertEqual(outcome, 'CONFLICT_OR_LOCKED')
        self.assertEqual(self.proc.wait(timeout=5), 0)

    def test_real_phase_reader_stops_only_for_latest_wait_event(self):
        self.start()
        log = self.root / 'shared-supervisor-wait.log'
        log.write_text('{"state":"PILOT_RUNNING"}\n{"state":"WAITING_FOR_FREE_GPUS"}\n')
        outcome = stop_waiting(self.script, self.root, self.lock, self.guard, status_module, timeout=5)
        self.assertEqual(outcome, 'STOPPED_READY')
        self.assertEqual(self.proc.wait(timeout=5), 0)

    def test_real_phase_reader_refuses_if_pilot_started(self):
        self.start()
        log = self.root / 'shared-supervisor-wait.log'
        log.write_text('{"state":"WAITING_FOR_FREE_GPUS"}\n{"state":"PILOT_RUNNING"}\n')
        outcome = stop_waiting(self.script, self.root, self.lock, self.guard, status_module, timeout=5)
        self.assertEqual(outcome, 'PHASE_CHANGED')
        self.assertIsNone(self.proc.poll())

    def test_helper_hup_or_term_resumes_frozen_supervisor(self):
        self.start()
        handoff = Path(__file__).resolve().parent
        for sig in (signal.SIGHUP, signal.SIGTERM):
            marker = self.root / ('inside-' + str(sig))
            code = '''
import importlib.util, pathlib, sys, time, types
sys.path.insert(0, sys.argv[1])
import stop_waiting as h
spec = importlib.util.spec_from_file_location("g", sys.argv[2])
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
guard = types.SimpleNamespace(open_pidfd=g.open_pidfd, signal_pidfd=g.signal_pidfd,
                              prior_conflict=lambda _: False)
def phase(_):
    pathlib.Path(sys.argv[6]).write_text("frozen")
    time.sleep(30)
    return {"state":"WAITING_FOR_FREE_GPUS"}
h.install_signal_handlers()
try:
    h.stop_waiting(pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]),
                   pathlib.Path(sys.argv[5]), guard, types.SimpleNamespace(wait_phase=phase))
except h.HandoffInterrupted:
    pass
'''
            helper = subprocess.Popen([sys.executable, '-c', code, str(handoff), str(GUARD),
                                       str(self.script), str(self.root), str(self.lock), str(marker)])
            try:
                end = time.monotonic() + 5
                while not marker.exists():
                    if helper.poll() is not None or time.monotonic() >= end:
                        self.fail('helper failed to reach frozen phase')
                    time.sleep(.02)
                state = (Path('/proc') / str(self.proc.pid) / 'status').read_text()
                self.assertIn('State:\tT', state)
                helper.send_signal(sig)
                self.assertEqual(helper.wait(timeout=5), 0)
                end = time.monotonic() + 2
                while time.monotonic() < end:
                    state = (Path('/proc') / str(self.proc.pid) / 'status').read_text()
                    if 'State:\tT' not in state:
                        break
                    time.sleep(.02)
                self.assertNotIn('State:\tT', state)
                self.assertIsNone(self.proc.poll())
            finally:
                if helper.poll() is None:
                    helper.kill()
                    helper.wait(timeout=5)

    def test_exact_path_and_flags(self):
        path = self.script
        self.assertTrue(matches([b'python', os.fsencode(str(path)), b'--run', b'--wait-gpus'], path))
        self.assertFalse(matches([b'python', os.fsencode(str(path)) + b'.other', b'--run', b'--wait-gpus'], path))
        self.assertFalse(matches([b'python', os.fsencode(str(path)), b'--run'], path))


if __name__ == '__main__':
    unittest.main()
