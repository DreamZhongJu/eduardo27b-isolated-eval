import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stream_train import __main__ as app
from stream_train import api_retry_supervisor as s
from stream_train import pipeline
from stream_train.store import Store


class ProbeTests(unittest.TestCase):
    def test_probe_child_accepts_success_and_fixed_stage(self):
        import io
        class Child:
            def __init__(self, code, output):
                self.returncode = code
                self.stdout = io.BytesIO(output)
            def poll(self): return self.returncode
        class Stop:
            def wait(self, seconds): return False
        with patch.object(s.subprocess, 'Popen', return_value=Child(0, b'{"stage":"grade","status":"ok"}\n')):
            self.assertEqual(s.probe('/tmp/config.json', 'dummy', Stop(), 7), (True, 'ok', 'none'))
        with patch.object(s.subprocess, 'Popen', return_value=Child(1, b'{"stage":"review","status":"failed"}\n')):
            self.assertEqual(s.probe('/tmp/config.json', 'dummy', Stop(), 7), (False, 'review', 'other'))

    def test_probe_returns_allowlisted_auth_category(self):
        import io
        class Child:
            returncode = 1
            stdout = io.BytesIO(b'{"stage":"generate","status":"failed","category":"auth"}\n')
            def poll(self): return 1
        class Stop:
            def wait(self, seconds): return False
        with patch.object(s.subprocess, 'Popen', return_value=Child()):
            self.assertEqual(s.probe('/tmp/config.json', 'dummy', Stop(), 7),
                             (False, 'generate', 'auth'))

    def test_probe_never_reports_untrusted_category(self):
        import io
        class Child:
            returncode = 1
            stdout = io.BytesIO(b'{"stage":"grade","status":"failed","category":"secret"}\n')
            def poll(self): return 1
        class Stop:
            def wait(self, seconds): return False
        with patch.object(s.subprocess, 'Popen', return_value=Child()):
            self.assertEqual(s.probe('/tmp/config.json', 'dummy', Stop(), 7),
                             (False, 'grade', 'other'))

    def test_probe_never_reports_untrusted_json(self):
        import io
        class Child:
            returncode = 1
            stdout = io.BytesIO(b'{"stage":"secret value","status":"failed"}\n')
            def poll(self): return 1
        class Stop:
            def wait(self, seconds): return False
        with patch.object(s.subprocess, 'Popen', return_value=Child()):
            self.assertEqual(s.probe('/tmp/config.json', 'dummy', Stop(), 7), (False, 'unknown', 'other'))


class QueueTests(unittest.TestCase):
    def test_started_unknown_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.sqlite'
            store = Store(path)
            store.enqueue('a', 'generate', {'x': 1})
            store.enqueue('b', 'review', {'x': 2})
            store.claim(['generate'])
            store.claim(['review'])
            store.finish('b', {'error_type': 'UnknownOutcome'}, state='unknown')
            counts = s.read_counts(path)
            self.assertEqual(counts['started'], 1)
            self.assertEqual(counts['unknown'], 1)
            self.assertFalse(s.is_drained(counts))
            with sqlite3.connect(str(path)) as conn:
                self.assertEqual(conn.execute("SELECT state FROM jobs WHERE id='a'").fetchone()[0], 'started')
                self.assertEqual(conn.execute("SELECT state FROM jobs WHERE id='b'").fetchone()[0], 'unknown')

    def test_missing_db_never_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.sqlite'
            with self.assertRaises(ValueError):
                s.read_counts(path)
            self.assertFalse(path.exists())

    def test_unknown_does_not_block_drained(self):
        from collections import Counter
        self.assertTrue(s.is_drained(Counter({'unknown': 3, 'done': 1})))
        self.assertFalse(s.is_drained(Counter({'started': 1})))


class LockTests(unittest.TestCase):
    def test_probe_child_holds_lock_if_supervisor_dies(self):
        import os, subprocess, sys
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / 'api.lock'
            fd = s.acquire_lock(lock)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     pass_fds=(fd,))
            os.close(fd)
            try:
                with self.assertRaises(BlockingIOError):
                    s.acquire_lock(lock)
            finally:
                child.terminate()
                child.wait(timeout=5)
            fd = s.acquire_lock(lock)
            os.close(fd)

    def test_probe_does_not_overlap_live_producer(self):
        import os
        class Stop:
            def wait(self, seconds): return False
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / 'api.lock'
            fd = s.acquire_lock(lock)
            try:
                with patch.object(s, 'api_lock_path', return_value=lock), \
                     patch.object(s.subprocess, 'Popen') as spawn:
                    self.assertEqual(s.try_probe('/tmp/config.json', 'dummy', Stop()),
                                     (False, 'producer_active', 'none'))
                    spawn.assert_not_called()
            finally:
                os.close(fd)

    def test_supervisor_lock_prevents_duplicates(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / 'guard.lock'
            fd = s.acquire_lock(lock)
            try:
                with self.assertRaises(BlockingIOError):
                    s.acquire_lock(lock)
            finally:
                os.close(fd)


class RetryTests(unittest.TestCase):
    def test_probe_duration_counts_towards_hour_cap(self):
        self.assertEqual(s.retry_wait(3600, 3600, 240), 3360)
        self.assertEqual(s.retry_wait(60, 3600, 80), 0)

    def test_producer_failure_preserves_pending_and_unknown(self):
        class Stop:
            stopped = False
            def is_set(self): return self.stopped
            def wait(self, seconds): self.stopped = True; return True
        class Child:
            def poll(self): return 1
            def wait(self): return 1
        stopper = Stop()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, cfg, log = root/'formal.sqlite', root/'config.json', root/'adaptive.log'
            cfg.write_text(json.dumps({'namespace':'s', 'api_base':'https://api.agnes-ai.cn/v1',
                'api_model':'agnes-3.0-flash', 'student_base':'http://127.0.0.1:18761/v1',
                'student_model':'student-8b'}))
            store = Store(db)
            store.bind('protocol', app.config(cfg))
            store.enqueue('pending', 'generate', {'x': 1})
            store.enqueue('uncertain', 'review', {'x': 2})
            store.claim(['review'])
            store.finish('uncertain', {'error_type':'UnknownOutcome'}, state='unknown')
            with patch.dict('os.environ', {'AGNES_API_KEY':'dummy'}), \
                 patch.object(s, 'try_probe', return_value=(True, 'ok', 'none')), \
                 patch.object(s.subprocess, 'Popen', return_value=Child()):
                rc = s.supervise(db, cfg, 8, log, stopper)
            self.assertEqual(rc, 0)
            self.assertEqual(s.read_counts(db)['pending'], 1)
            self.assertEqual(s.read_counts(db)['unknown'], 1)

    def test_failure_backoff_doubles_and_caps(self):
        class Stop:
            def __init__(self): self.waits = []
            def is_set(self): return len(self.waits) >= 4
            def wait(self, seconds): self.waits.append(seconds); return self.is_set()
        stopper = Stop()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, cfg, log = root/'formal.sqlite', root/'config.json', root/'adaptive.log'
            cfg.write_text(json.dumps({'namespace':'s', 'api_base':'https://api.agnes-ai.cn/v1',
                'api_model':'agnes-3.0-flash', 'student_base':'http://127.0.0.1:18761/v1',
                'student_model':'student-8b'}))
            store = Store(db)
            store.bind('protocol', app.config(cfg))
            store.enqueue('pending', 'generate', {'x': 1})
            with patch.dict('os.environ', {'AGNES_API_KEY':'dummy'}), \
                 patch.object(s, 'try_probe', return_value=(False, 'generate', 'protocol')):
                rc = s.supervise(db, cfg, 8, log, stopper, minimum_wait=60, max_wait=180)
            self.assertEqual(rc, 0)
            for actual, expected in zip(stopper.waits, [60, 120, 180, 180]):
                self.assertAlmostEqual(actual, expected, delta=0.05)
            self.assertEqual(s.read_counts(db)['pending'], 1)
            self.assertFalse(log.exists())


if __name__ == '__main__': unittest.main()
