"""AIMD tests use local mock tasks; HOME locks are not cross-host coordination."""
import contextlib
import fcntl
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from stream_train import __main__ as cli
from stream_train import pipeline as p
from stream_train.store import Store


class APIWindowTest(unittest.TestCase):
    def setUp(self):
        ram = Path('/dev/shm')
        temp_dir = str(ram) if ram.is_dir() and os.access(ram, os.W_OK) else None
        self.tmp = tempfile.TemporaryDirectory(dir=temp_dir)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'queue.sqlite')

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_grows_one_at_a_time_to_five(self):
        window = cli.AdaptiveAPIWindow()
        limits = []
        for _ in range(6):
            lease = window.acquire()
            window.report(lease, 'success')
            window.release()
            limits.append(window.limit)
        self.assertEqual(limits, [2, 3, 4, 5, 5, 5])
        with self.assertRaises(ValueError):
            cli.AdaptiveAPIWindow(maximum=6)

    def test_429_halves_and_stale_success_does_not_cancel_backoff(self):
        window = cli.AdaptiveAPIWindow(cooldown_base=.1, cooldown_cap=.1)
        lease = window.acquire()
        window.report(lease, 'success')
        window.release()
        first, second = window.acquire(), window.acquire()
        self.assertEqual(window.limit, 2)
        window.report(first, cli.api_outcome(p.UnknownOutcome('http_429')))
        window.release()
        window.report(second, 'success')
        window.release()
        self.assertEqual(window.limit, 1)
        self.assertGreater(window.blocked_until, time.monotonic())

    def test_status_and_transport_classification(self):
        for reason in ('http_429', 'http_500', 'http_503', 'http_408',
                       'http_retries_exhausted_500', 'URLError', 'TimeoutError'):
            self.assertEqual(cli.api_outcome(p.UnknownOutcome(reason)), 'congestion')
        for reason in ('http_401', 'http_403'):
            self.assertEqual(cli.api_outcome(p.UnknownOutcome(reason)), 'auth')
        for reason in ('invalid_json_response', 'response_schema', 'unparseable_content',
                       'nonobject_response', 'generation_schema', 'review_schema',
                       'review_enum', 'equivalence_schema', 'grade_schema',
                       'missing_content', 'incomplete_response', 'http_400'):
            self.assertEqual(cli.api_outcome(p.UnknownOutcome(reason)), 'protocol')
        self.assertEqual(cli.api_outcome(AttributeError('bad shape')), 'protocol')
        self.assertEqual(cli.api_outcome(p.UnknownOutcome('unexpected_provider_model')), 'model')

    def test_model_mismatch_stops_immediately(self):
        for i in range(10):self.store.enqueue(f'g-{i}', 'generate', {})
        window = cli.AdaptiveAPIWindow()
        with patch.object(p, 'generate', side_effect=p.UnknownOutcome('unexpected_provider_model')):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                ok = cli.loop(self.store, {}, 'fake', ['generate'], 5, False,
                              api_window=window)
        self.assertFalse(ok)
        self.assertEqual(window.stop_reason, 'model')
        self.assertIn('模型', error.getvalue())
        self.assertEqual(self.store.counts(), [
            {'kind': 'generate', 'state': 'pending', 'count': 9},
            {'kind': 'generate', 'state': 'unknown', 'count': 1}])

    def test_three_http200_invalid_structures_stop_instead_of_sweeping(self):
        source = {'source_id': 'synthetic', 'source_row_index': 1,
                  'problem': 'Synthetic', 'answer': '1'}
        for i in range(10):
            self.store.enqueue(f'g-{i}', 'generate',
                               {'source': source, 'proposal': 1, 'feedback': []})
        cfg = {'api_base': 'https://example.invalid/v1', 'api_model': 'test'}
        window = cli.AdaptiveAPIWindow()
        # Valid JSON array, but not a valid response object: raw.get would
        # raise AttributeError instead of a provider-specific UnknownOutcome.
        with patch.object(p, 'request_json', return_value=([], [200])) as request:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                ok = cli.loop(self.store, cfg, 'fake', ['generate'], 5, False,
                              api_window=window)
        self.assertFalse(ok)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(window.stop_reason, 'protocol')
        self.assertIn('连续三次', error.getvalue())
        self.assertEqual(self.store.counts(), [
            {'kind': 'generate', 'state': 'pending', 'count': 7},
            {'kind': 'generate', 'state': 'unknown', 'count': 3}])

    def test_valid_new_response_resets_protocol_streak(self):
        window = cli.AdaptiveAPIWindow()
        lease = window.acquire(); window.report(lease, 'protocol'); window.release()
        self.assertEqual(window.protocol_failures, 1)
        lease = window.acquire(); window.report(lease, 'success'); window.release()
        self.assertEqual(window.protocol_failures, 0)
        lease = window.acquire(); window.report(lease, 'protocol'); window.release()
        self.assertEqual(window.protocol_failures, 1)
        self.assertIsNone(window.stop_reason)

    def test_inflight_success_resets_streak_but_cannot_unlock_latched_circuit(self):
        window = cli.AdaptiveAPIWindow()
        lease = window.acquire(); window.report(lease, 'success'); window.release()
        first, old_success = window.acquire(), window.acquire()
        window.report(first, 'protocol'); window.release()
        window.report(old_success, 'success'); window.release()
        # A valid in-flight result interrupts the streak before it trips.
        self.assertEqual(window.protocol_failures, 0)
        for _ in range(3):
            lease = window.acquire(); window.report(lease, 'protocol'); window.release()
        self.assertEqual(window.stop_reason, 'protocol')
        # Once latched, even a late in-flight success cannot reopen it.
        window.report(old_success, 'success')
        self.assertEqual(window.stop_reason, 'protocol')
        self.assertIsNone(window.acquire())

    def test_normal_semantic_unknown_is_valid_api_response(self):
        self.store.enqueue('one', 'review', {})
        window = cli.AdaptiveAPIWindow()
        lease = window.acquire(); window.report(lease, 'protocol'); window.release()
        def review_unknown(store, job, payload, cfg, key):
            store.finish(job, {'verdict': 'uncertain'}, state='unknown')
        with patch.object(p, 'review', side_effect=review_unknown):
            self.assertTrue(cli.loop(self.store, {}, 'fake', ['review'], 1, True,
                                     api_window=window))
        self.assertEqual(window.protocol_failures, 0)
        self.assertIsNone(window.stop_reason)

    def test_cooldown_leaves_unrequested_job_pending(self):
        self.store.enqueue('one', 'generate', {})
        window = cli.AdaptiveAPIWindow(cooldown_base=.3, cooldown_cap=.3)
        lease = window.acquire()
        window.report(lease, 'congestion')
        window.release()
        started = threading.Event()

        def fake_generate(store, job, payload, cfg, key):
            started.set()
            store.finish(job, {'ok': True})

        with patch.object(p, 'generate', side_effect=fake_generate):
            thread = threading.Thread(target=cli.loop,
                args=(self.store, {}, 'fake', ['generate'], 1, True),
                kwargs={'api_window': window})
            thread.start()
            self.assertFalse(started.wait(.05))
            self.assertEqual(self.store.counts(),
                             [{'kind': 'generate', 'state': 'pending', 'count': 1}])
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(started.is_set())

    def test_unavailable_permit_precedes_claim(self):
        self.store.enqueue('one', 'generate', {})
        window = cli.AdaptiveAPIWindow()
        held = window.acquire()
        started = threading.Event()

        def fake_generate(store, job, payload, cfg, key):
            started.set()
            store.finish(job, {'ok': True})

        with patch.object(p, 'generate', side_effect=fake_generate):
            thread = threading.Thread(target=cli.loop,
                args=(self.store, {}, 'fake', ['generate'], 1, True),
                kwargs={'api_window': window})
            thread.start()
            self.assertFalse(started.wait(.05))
            self.assertEqual(self.store.counts(),
                             [{'kind': 'generate', 'state': 'pending', 'count': 1}])
            window.release()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(started.is_set())
        self.assertEqual(held, 0)

    def test_401_and_403_stop_before_scanning_pending_queue(self):
        for code in (401, 403):
            with self.subTest(code=code):
                store = Store(self.root / f'queue-{code}.sqlite')
                for i in range(10):store.enqueue(f'g-{i}', 'generate', {})
                window = cli.AdaptiveAPIWindow()
                with patch.object(p, 'generate', side_effect=p.UnknownOutcome(f'http_{code}')):
                    with contextlib.redirect_stderr(io.StringIO()) as error:
                        ok = cli.loop(store, {}, 'fake', ['generate'], 5, False,
                                      api_window=window)
                self.assertFalse(ok)
                self.assertTrue(window.auth_failed)
                self.assertIn('401/403', error.getvalue())
                self.assertEqual(store.counts(), [
                    {'kind': 'generate', 'state': 'pending', 'count': 9},
                    {'kind': 'generate', 'state': 'unknown', 'count': 1}])

    def test_ambiguous_transport_remains_unknown_and_not_reclaimed(self):
        self.store.enqueue('first', 'generate', {})
        self.store.enqueue('second', 'generate', {})
        observed = []
        with patch.object(p, 'generate', side_effect=p.UnknownOutcome('URLError')):
            self.assertTrue(cli.run_one(self.store, {}, 'fake', ['generate'], observed.append))
        self.assertEqual(observed, ['congestion'])
        self.assertEqual(self.store.claim(['generate'])[0], 'second')
        self.assertIsNone(self.store.claim(['generate']))

    def test_locally_truncated_grade_does_not_increase_window(self):
        self.store.enqueue('truncated', 'grade',
                           {'candidate': {}, 'student': {'finish_reason': 'length'}})
        window = cli.AdaptiveAPIWindow()
        self.assertTrue(cli.loop(self.store, {}, 'fake', ['grade'], 1, True,
                                 api_window=window))
        self.assertEqual(window.limit, 1)

    def test_api_concurrency_never_exceeds_five(self):
        for i in range(5):self.store.enqueue(f'g-{i}', 'generate', {})
        window = cli.AdaptiveAPIWindow()
        guard = threading.Lock()
        counters = {'active': 0, 'peak': 0}

        def fake_generate(store, job, payload, cfg, key):
            with guard:
                counters['active'] += 1
                counters['peak'] = max(counters['peak'], counters['active'])
            time.sleep(.03)
            with guard:counters['active'] -= 1
            store.finish(job, {'ok': True})

        with patch.object(p, 'generate', side_effect=fake_generate):
            self.assertTrue(cli.loop(self.store, {}, 'fake', ['generate'], 5, True,
                                     api_window=window))
        self.assertGreaterEqual(counters['peak'], 2)
        self.assertLessEqual(counters['peak'], 5)

    def test_home_lock_only_coordinates_same_underlying_home(self):
        # Different HOME directories (or container filesystems) use different
        # inodes: this cannot enforce an account-wide or cross-host limit.
        home_a, home_b = self.root / 'a', self.root / 'b'
        for home in (home_a, home_b):
            (home / '.cache' / 'math-transfer-stream').mkdir(parents=True)
        path = home_a / '.cache' / 'math-transfer-stream' / 'api.lock'
        child = ("import fcntl,os,pathlib; "
                 "p=pathlib.Path.home()/'.cache'/'math-transfer-stream'/'api.lock'; "
                 "f=p.open('a'); "
                 "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('acquired')")
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            a = subprocess.run([sys.executable, '-c', child],
                               env={**os.environ, 'HOME': str(home_a)}, capture_output=True)
            b = subprocess.run([sys.executable, '-c', child],
                               env={**os.environ, 'HOME': str(home_b)}, capture_output=True)
        self.assertNotEqual(a.returncode, 0)
        self.assertEqual(b.returncode, 0)


if __name__ == '__main__':
    unittest.main()
