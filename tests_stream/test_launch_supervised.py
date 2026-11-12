import json
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from stream_train import launch_supervised as m


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'pilot.sqlite'
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE jobs(kind text,state text,payload text,result text)')

    def insert(self, kind, state, payload, result):
        with sqlite3.connect(self.db) as c:
            c.execute('INSERT INTO jobs VALUES(?,?,?,?)', (kind, state, json.dumps(payload), json.dumps(result)))

    def fill(self, n=16, mismatch=False, api=True):
        for i in range(n):
            payload = {'candidate': {'candidate_id': 'candidate', 'origin': 'self_generated'},
                       'sample_id': str(i), 'sample_index': i}
            self.insert('student', 'done', payload, {})
            p = dict(payload)
            if mismatch and i == 0: p['sample_id'] = 'other'
            result = {'decision': 'correct'}
            if api and i == 0:
                result['meta'] = {'model': 'test'}
                result['verdict'] = {'decision': 'correct'}
            self.insert('grade', 'done', p, result)

    def test_complete_chain_requires_16_matching_and_real_api(self):
        self.fill()
        self.assertEqual(m.pilot_state(self.db), (0, 0, True))

    def test_partial_or_missing_api_does_not_pass(self):
        self.fill(n=15)
        self.assertFalse(m.pilot_state(self.db)[2])

    def test_wrong_pair_does_not_pass(self):
        self.fill(mismatch=True)
        self.assertFalse(m.pilot_state(self.db)[2])

    def test_local_only_grades_do_not_pass(self):
        self.fill(api=False)
        self.assertFalse(m.pilot_state(self.db)[2])

    def test_unresolved_and_pending_counted(self):
        self.fill()
        self.insert('generate', 'started', {}, {})
        self.insert('review', 'unknown', {}, {})
        self.assertEqual(m.pilot_state(self.db), (1, 1, True))


class AdmissionTests(unittest.TestCase):
    def test_absent_authorization_has_no_side_effect(self):
        class Args:
            authorized_external = False
            confirm_dedicated = True
        with patch.object(m.guard, 'find_service') as service:
            with self.assertRaises(m.LaunchError): m.launch(Args())
            service.assert_not_called()

    def test_pidfd_preflight_failure_stops_launch(self):
        with patch.object(m.guard, 'open_pidfd', side_effect=m.guard.GuardError('unsupported')):
            with self.assertRaises(m.guard.GuardError):
                m.preflight_pidfd(123, Path('/model'), 'student', 8000, 1)

    def test_pidfd_signal_permission_failure_stops_launch(self):
        import os
        fd = os.open('/dev/null', os.O_RDONLY)
        with patch.object(m.guard, 'open_pidfd', return_value=fd), \
                patch.object(m.guard, 'validate_service'), \
                patch.object(m.signal, 'pidfd_send_signal', side_effect=PermissionError(), create=True):
            with self.assertRaises(m.LaunchError):
                m.preflight_pidfd(123, Path('/model'), 'student', 8000, 1)

    def test_api_lock_conflict_stops_launch(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / '.cache/math-transfer-stream/api.lock'
            lock.parent.mkdir(parents=True)
            with lock.open('a') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(m.Path, 'home', return_value=Path(d)):
                    with self.assertRaises(m.LaunchError): m.preflight_processes()


class OrchestrationTests(unittest.TestCase):
    def test_pilot_then_formal_once_with_separate_databases(self):
        from argparse import Namespace
        import os
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'sources'; source.write_text('{}\n')
            cfgfile = root / 'config'; cfgfile.write_text('{}')
            model = root / 'model'; model.mkdir()
            args = Namespace(sources=str(source), config=str(cfgfile), model_path=str(model),
                run_dir=str(root / 'run'), workers=32, expected_sources=1, pilot_timeout=60,
                sources_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                authorized_external=True, confirm_dedicated=True)
            cfg = {'student_base':'http://127.0.0.1:8000/v1', 'student_model':'student-8b', 'namespace':'run'}
            launched=[]
            class Child:
                def __init__(self, cmd):
                    self.cmd=cmd; self.pid=100 + len(launched); self.returncode=None
                    self.watcher='--confirm-initialized' in cmd
                    launched.append(self)
                def poll(self):
                    if self.watcher: self.returncode=0
                    return self.returncode
                def terminate(self): self.returncode=0
                def wait(self, timeout=None): self.returncode=0; return 0
            def init(cmd, **kwargs):
                Path(cmd[cmd.index('--db') + 1]).touch()
                return SimpleNamespace(returncode=0)
            with patch.dict(os.environ, {'AGNES_API_KEY':'fake'}), \
                    patch.object(m.app, 'config', return_value=cfg), \
                    patch.object(m.guard, 'find_service', return_value=42), \
                    patch.object(m.guard, 'validate_service', return_value=1), \
                    patch.object(m, 'preflight_pidfd'), patch.object(m, 'preflight_processes'), \
                    patch.object(m.guard, 'metrics', return_value=(0, 0)), \
                    patch.object(m, 'pilot_state', return_value=(0, 0, True)), \
                    patch.object(m.subprocess, 'Popen', side_effect=lambda cmd, **kw: Child(cmd)), \
                    patch.object(m.subprocess, 'run', side_effect=init), \
                    patch.object(m.time, 'sleep'):
                m.launch(args)
            self.assertEqual(len(launched), 5)
            self.assertIn('pilot.sqlite', ' '.join(launched[0].cmd))
            self.assertIn('formal.sqlite', ' '.join(launched[2].cmd))
            self.assertTrue(launched[4].watcher)
            self.assertEqual((root / 'run' / 'empty-hold.txt').read_text(), '')
            with self.assertRaises(m.LaunchError):
                with patch.dict(os.environ, {'AGNES_API_KEY':'fake'}), \
                        patch.object(m.app, 'config', return_value=cfg), \
                        patch.object(m.guard, 'find_service', return_value=42), \
                        patch.object(m.guard, 'validate_service', return_value=1), \
                        patch.object(m, 'preflight_pidfd'), patch.object(m, 'preflight_processes'), \
                        patch.object(m.guard, 'metrics', return_value=(0, 0)):
                    m.launch(args)


if __name__ == '__main__': unittest.main()
