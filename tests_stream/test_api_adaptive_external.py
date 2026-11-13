import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stream_train import __main__ as app
from stream_train import api_adaptive_external as m
from stream_train.store import Store


class WindowTest(unittest.TestCase):
    def test_increases_to_new_cap_and_congestion_halves(self):
        with patch.object(m, 'event'):
            w = m.Window(20)
            for _ in range(30):
                w.report(w.epoch, 'success')
            self.assertEqual(w.limit, 20)
            old = w.epoch
            w.report(old, 'congestion')
            self.assertEqual(w.limit, 10)
            w.report(old, 'success')
            self.assertEqual(w.limit, 10)

    def test_auth_and_protocol_latch_and_close(self):
        with patch.object(m, 'event'):
            w = m.Window(7)
            for _ in range(3):
                w.report(w.epoch, 'protocol')
            self.assertEqual(w.stop_reason, 'protocol')
            self.assertIsNone(w.acquire())
            w = m.Window(7)
            w.report(w.epoch, 'auth')
            self.assertEqual(w.stop_reason, 'auth')
            self.assertIsNone(w.acquire())

    def test_invalid_maximum(self):
        for n in (0, 33):
            with self.assertRaises(ValueError):
                m.Window(n)


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'queue.sqlite'
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'namespace': 'test',
            'api_base': 'https://api.agnes-ai.cn/v1', 'api_model': 'agnes-3.0-flash',
            'student_base': 'http://127.0.0.1:18761/v1', 'student_model': 'student-8b'}))
        self.store = Store(self.db)
        self.cfg = app.config(self.config)
        self.store.bind('protocol', self.cfg)
        self.store.enqueue('id', 'generate', {})

    def test_protocol_mismatch_does_not_create_policy(self):
        changed = json.loads(self.config.read_text())
        changed['namespace'] = 'different'
        self.config.write_text(json.dumps(changed))
        policy = self.root / 'policy.json'
        with patch.dict(os.environ, {'AGNES_API_KEY': 'fake'}):
            with self.assertRaises(ValueError):
                m.run(self.db, self.config, 10, policy)
        self.assertFalse(policy.exists())
        self.assertEqual(self.store.counts()[0]['state'], 'pending')

    def test_lock_conflict_does_not_create_policy_or_claim(self):
        lock = self.root / '.cache/math-transfer-stream/api.lock'
        lock.parent.mkdir(parents=True)
        policy = self.root / 'policy.json'
        with patch.dict(os.environ, {'HOME': str(self.root), 'AGNES_API_KEY': 'fake'}):
            with lock.open('a') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    m.run(self.db, self.config, 10, policy)
        self.assertFalse(policy.exists())
        self.assertEqual(self.store.counts()[0]['state'], 'pending')

    def test_unknown_preserved_and_safe_drain(self):
        self.store.claim(['generate'])
        self.store.finish('id', {'reason': 'http_429'}, state='unknown')
        policy = self.root / 'policy.json'
        clock = iter(range(0, 100000, 100))
        with patch.dict(os.environ, {'HOME': str(self.root), 'AGNES_API_KEY': 'fake'}), \
                patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = m.run(self.db, self.config, 1, policy, idle_seconds=60, poll_seconds=0.01)
        self.assertEqual(result, 0)
        self.assertEqual(self.store.counts()[0]['state'], 'unknown')
        self.assertIn('queue_drained', output.getvalue())
        self.assertEqual(json.loads(policy.read_text())['maximum'], 1)
        self.assertEqual(m.read_protocol(self.db), self.cfg)

    def test_ready_sent_only_after_lock_and_policy_receipt(self):
        self.store.claim(['generate'])
        self.store.finish('id', {'reason': 'http_429'}, state='unknown')
        policy = self.root / 'policy.json'
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        clock = iter(range(0, 100000, 100))
        with patch.dict(os.environ, {'HOME': str(self.root), 'AGNES_API_KEY': 'fake',
                                     'STREAM_PRODUCER_READY_FD': str(write_fd)}), \
                patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.run(self.db, self.config, 1, policy,
                                   idle_seconds=60, poll_seconds=0.01), 0)
        self.assertEqual(os.read(read_fd, 2), b'R')
        self.assertTrue(policy.is_file())

    def test_protocol_mismatch_sends_no_ready(self):
        changed = json.loads(self.config.read_text())
        changed['namespace'] = 'different'
        self.config.write_text(json.dumps(changed))
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        with patch.dict(os.environ, {'HOME': str(self.root), 'AGNES_API_KEY': 'fake',
                                     'STREAM_PRODUCER_READY_FD': str(write_fd)}), \
                patch.object(m.os, 'write') as writer:
            with self.assertRaises(ValueError):
                m.run(self.db, self.config, 1, self.root / 'policy.json')
        writer.assert_not_called()

    def test_real_pipeline_congestion_is_unknown_and_not_retried(self):
        policy = self.root / 'policy.json'
        clock = iter(range(0, 100000, 100))
        with patch.dict(os.environ, {'HOME': str(self.root), 'AGNES_API_KEY': 'secret-key'}), \
                patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)), \
                patch.object(app.p, 'generate', side_effect=app.p.UnknownOutcome('http_429')) as request, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = m.run(self.db, self.config, 1, policy, idle_seconds=60, poll_seconds=0.01)
        self.assertEqual(result, 0)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(self.store.counts()[0]['state'], 'unknown')
        self.assertIn('congestion', output.getvalue())
        self.assertNotIn('secret-key', output.getvalue())

    def test_missing_db_never_created(self):
        missing = self.root / 'missing.sqlite'
        with self.assertRaises(ValueError):
            m.read_protocol(missing)
        self.assertFalse(missing.exists())


if __name__ == '__main__':
    unittest.main()
