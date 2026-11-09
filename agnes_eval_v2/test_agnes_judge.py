"""No network calls: concurrency, retry, response binding and crash-recovery tests."""
import concurrent.futures
import json
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import agnes_judge as j
import watch_sources as watch


def sample(i):
    return {'source_id': 'synthetic', 'target': 'paper', 'phase': 'post', 'index': i,
            'problem': 'What is 2+2?', 'reference_answer': '4',
            'student_response': 'The answer is 4.'}


def response(value=True):
    return {'model': j.MODEL, 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps({'correct': value})}}]}


def multiworker(root, worker, active, high, mutex, start):
    base = Path(root)
    def fake(_body, _key):
        with mutex:
            active.value += 1
            high.value = max(high.value, active.value)
        time.sleep(0.12)
        with mutex:
            active.value -= 1
        return 200, response()
    start.wait(10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda i: j.grade_one(sample(worker * 100 + i), 'test-only',
            base / f'receipts-{worker}', base / 'slots', caller=fake, sleep=lambda _: None), range(5)))
    assert all(r['status'] == 'correct' for r in results)


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'slots').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_three_variable_fields_leave(self):
        row = dict(sample(0), private_extra='DO_NOT_SEND', finish_reason='length')
        request = j.payload(row)
        self.assertEqual(request['model'], j.MODEL)
        self.assertEqual(set(json.loads(request['messages'][1]['content'])),
                         {'problem', 'reference_answer', 'student_response'})
        self.assertNotIn('DO_NOT_SEND', json.dumps(request))
        self.assertNotIn('source_id', json.dumps(request))

    def test_fenced_json_verdict(self):
        r = response(True)
        r['choices'][0]['message']['content'] = '```json\n{"correct": true}\n```'
        self.assertIs(j.verdict(r), True)
        self.assertIsNone(j.verdict(None))

    def test_fixed_endpoint_and_no_redirect_or_proxy(self):
        calls = []
        class Reply:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _n): return json.dumps(response()).encode()
        class Opener:
            def open(self, request, timeout):
                calls.append((request.full_url, timeout))
                return Reply()
        def build(*handlers):
            self.assertTrue(any(isinstance(x, j.NoRedirect) for x in handlers))
            self.assertTrue(any(isinstance(x, j.urllib.request.ProxyHandler) for x in handlers))
            return Opener()
        with patch.object(j.urllib.request, 'build_opener', side_effect=build):
            code, raw = j.api_call(j.payload(sample(1)), 'test-only')
        self.assertEqual((code, j.verdict(raw)), (200, True))
        self.assertEqual(calls, [(j.ENDPOINT, 120)])

    def test_initial_plus_three_retry_and_result_reuse(self):
        codes = [500, 429, 503, 200]
        calls = []
        def fake(_body, _key):
            calls.append(1)
            c = codes.pop(0)
            return c, response() if c == 200 else None
        x = j.grade_one(sample(2), 'test-only', self.root/'r', self.root/'slots', fake, lambda _: None)
        self.assertEqual((x['status'], len(calls)), ('correct', 4))
        y = j.grade_one(sample(2), 'test-only', self.root/'r', self.root/'slots',
                        lambda *_: self.fail('must not repeat'), lambda _: None)
        self.assertEqual(x, y)

    def test_ambiguous_transport_retries_then_reuses_unknown(self):
        calls = []
        def fake(*_):
            calls.append(1)
            raise TimeoutError()
        x = j.grade_one(sample(3), 'test-only', self.root/'r', self.root/'slots', fake, lambda _: None)
        self.assertEqual((x['status'], x['reason'], len(calls)), ('unknown', 'retry_limit', 4))
        y = j.grade_one(sample(3), 'test-only', self.root/'r', self.root/'slots', fake, lambda _: None)
        self.assertEqual((x, len(calls)), (y, 4))

    def test_interrupted_intent_adds_new_attempt(self):
        row = sample(32); body = j.payload(row); ident = j.identity(row); rhash = j.digest(body)
        folder = self.root/'r'/j.digest({'identity': ident})
        folder.mkdir(parents=True)
        j.put_once(folder/'BINDING.json', {'identity': ident, 'request_sha256': rhash,
                    'model': j.MODEL, 'grader': 'agnes_full_response_v2'})
        j.put_once(folder/'ATTEMPT-0001-INTENT.json', {'attempt': 1, 'request_sha256': rhash})
        calls = []
        def fake(*_):
            calls.append(1)
            return 200, response()
        result = j.grade_one(row, 'test-only', self.root/'r', self.root/'slots', fake, lambda _: None)
        self.assertEqual((result['status'], len(calls)), ('correct', 1))
        self.assertTrue((folder/'ATTEMPT-0002-INTENT.json').is_file())
        self.assertEqual(j.read_json(folder/'ATTEMPT-0001-RESPONSE.json')['status'], 'unknown')

    def test_atomic_receipt_not_published_if_link_fails(self):
        target = self.root/'atomic'/'RESULT.json'
        with patch.object(j.os, 'link', side_effect=OSError('test')):
            with self.assertRaises(OSError):
                j.put_once(target, {'status': 'correct'})
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()), [])

    def test_adaptive_window_growth_halving_and_rate_limit(self):
        w = j.AdaptiveWindow(5)
        for _ in range(4):
            w.acquire(); w.release(True)
        self.assertEqual(w.window, 5)
        with patch.object(j.time, 'monotonic', return_value=100):
            w.acquire(); w.release(False, rate_limited=True)
            self.assertEqual(w.window, 2)
            self.assertEqual(w.blocked_until, 160)

    def test_saved_200_recovers_after_crash_without_network(self):
        row = sample(4); body = j.payload(row); ident = j.identity(row); rhash = j.digest(body)
        folder = self.root/'r'/j.digest({'identity': ident})
        folder.mkdir(parents=True)
        j.put_once(folder/'BINDING.json', {'identity': ident, 'request_sha256': rhash,
                    'model': j.MODEL, 'grader': 'agnes_full_response_v2'})
        j.put_once(folder/'ATTEMPT-0001-INTENT.json', {'attempt': 1, 'request_sha256': rhash})
        j.put_once(folder/'ATTEMPT-0001-RESPONSE.json', {'attempt': 1, 'http_status': 200,
                   'retryable': False, 'raw_response': response(False)})
        x = j.grade_one(row, 'test-only', self.root/'r', self.root/'slots',
                        lambda *_: self.fail('must recover locally'), lambda _: None)
        self.assertEqual(x['status'], 'incorrect')

    def test_existing_identity_different_input_fails_closed(self):
        row = sample(5)
        j.grade_one(row, 'test-only', self.root/'r', self.root/'slots', lambda *_: (200, response()), lambda _: None)
        with self.assertRaises(ValueError):
            j.grade_one(dict(row, reference_answer='changed'), 'test-only', self.root/'r',
                        self.root/'slots', lambda *_: self.fail('must not call'), lambda _: None)

    def test_five_slot_limit_across_processes(self):
        ctx = multiprocessing.get_context('fork')
        active = ctx.Value('i', 0, lock=False); high = ctx.Value('i', 0, lock=False)
        mutex = ctx.Lock(); start = ctx.Event()
        jobs = [ctx.Process(target=multiworker, args=(str(self.root), n, active, high, mutex, start)) for n in range(3)]
        for job in jobs: job.start()
        start.set()
        for job in jobs:
            job.join(15)
            self.assertEqual(job.exitcode, 0)
        self.assertGreater(high.value, 1)
        self.assertLessEqual(high.value, 5)

    def test_source_watcher_waits_for_complete_and_reuses(self):
        source_id = 'synthetic'
        source_key = __import__('hashlib').sha256(source_id.encode()).hexdigest()[:24]
        folder = self.root/'sources'/source_key
        folder.mkdir(parents=True)
        rows = [sample(10), sample(11)]
        (folder/'SCORE_INPUT.json').write_text(json.dumps(rows))
        calls = []
        def fake(*_):
            calls.append(1)
            return 200, response()
        self.assertFalse(watch.process_source(folder, self.root/'out', self.root/'receipts',
                                              self.root/'slots', 'test-only', fake))
        self.assertEqual(len(calls), 0)
        input_sha = __import__('hashlib').sha256((folder/'SCORE_INPUT.json').read_bytes()).hexdigest()
        (folder/'RAW_COMPLETE.json').write_text(json.dumps({'source_id': source_id, 'samples': 2,
                                                           'score_input_sha256': input_sha}))
        self.assertTrue(watch.process_source(folder, self.root/'out', self.root/'receipts',
                                             self.root/'slots', 'test-only', fake))
        self.assertEqual(len(calls), 2)
        self.assertTrue(watch.process_source(folder, self.root/'out', self.root/'receipts',
                                             self.root/'slots', 'test-only', fake))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(json.loads((self.root/'out'/'sources'/source_key/'RESULTS.json').read_text())['results']), 2)

    def test_source_watcher_rejects_mismatched_source(self):
        source_id = 'synthetic'
        source_key = __import__('hashlib').sha256(source_id.encode()).hexdigest()[:24]
        folder = self.root/'sources'/source_key
        folder.mkdir(parents=True)
        (folder/'SCORE_INPUT.json').write_text(json.dumps([dict(sample(10), source_id='different')]))
        input_sha = __import__('hashlib').sha256((folder/'SCORE_INPUT.json').read_bytes()).hexdigest()
        (folder/'RAW_COMPLETE.json').write_text(json.dumps({'source_id': source_id, 'samples': 1,
                                                           'score_input_sha256': input_sha}))
        with self.assertRaises(ValueError):
            watch.process_source(folder, self.root/'out', self.root/'receipts', self.root/'slots',
                                 'test-only', lambda *_: self.fail('must not send'))

    def test_source_watcher_rejects_tampered_score_input(self):
        source_id = 'synthetic'
        source_key = __import__('hashlib').sha256(source_id.encode()).hexdigest()[:24]
        folder = self.root/'sources'/source_key
        folder.mkdir(parents=True)
        (folder/'SCORE_INPUT.json').write_text(json.dumps([sample(10)]))
        (folder/'RAW_COMPLETE.json').write_text(json.dumps({'source_id': source_id, 'samples': 1,
                                                           'score_input_sha256': '0' * 64}))
        with self.assertRaises(ValueError):
            watch.process_source(folder, self.root/'out', self.root/'receipts', self.root/'slots',
                                 'test-only', lambda *_: self.fail('must not send'))

    def test_public_own_marker_reconstruction_and_reuse(self):
        source_id = 'synthetic'
        key = __import__('hashlib').sha256(source_id.encode()).hexdigest()[:24]
        folder = self.root/'sources'/key
        folder.mkdir(parents=True)
        source = {'source_id': source_id, 'problem': 'source question', 'answer': '4',
                  'transfer_problem': 'transfer question', 'transfer_answer': '5',
                  'transfer_target': 'ours_easy', 'transfer_candidate_id': 'c1', 'transfer_position': 3,
                  'student_correct': 12, 'student_n_scored': 16, 'score_tie': False}
        rows = []
        for target in ('source', 'ours_easy'):
            for phase in ('pre', 'control', 'post'):
                for index in range(16):
                    row = {'source_id': source_id, 'target': target, 'phase': phase, 'index': index,
                           'problem': source['problem'] if target == 'source' else source['transfer_problem'],
                           'reference_answer': source['answer'] if target == 'source' else source['transfer_answer'],
                           'student_response': 'A final answer.', 'finish_reason': 'length'}
                    if target != 'source':
                        row.update(candidate_id='c1', difficulty='easy', position_by_student_correct=3,
                                   historical_student_correct=12, historical_student_n_scored=16,
                                   historical_group_score_tie=False)
                    rows.append(row)
        marker = {'source_id': source_id, 'samples': 96, 'grading_status': 'ungraded',
                  'source_sha256': j.digest(source)}
        (folder/'SOURCE.json').write_text(json.dumps({'source_id': source_id, 'sha256': j.digest(source)}))
        (folder/'RAW_COMPLETE.json').write_text(json.dumps(marker))
        (folder/'SCORE_INPUT.json').write_text(json.dumps(rows))
        calls = []
        def fake(body, _key):
            calls.append(body)
            return 200, response()
        self.assertTrue(watch.process_source(folder, self.root/'out', self.root/'receipts',
                                             self.root/'slots', 'test-only', fake))
        self.assertEqual(len(calls), 96)
        self.assertTrue(all('finish_reason' not in json.dumps(body) for body in calls))
        self.assertTrue(watch.process_source(folder, self.root/'out', self.root/'receipts',
                                             self.root/'slots', 'test-only', fake))
        self.assertEqual(len(calls), 96)
        changed = list(rows)
        changed[0] = dict(changed[0], problem='tampered')
        (folder/'SCORE_INPUT.json').write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            watch.process_source(folder, self.root/'out', self.root/'receipts', self.root/'slots',
                                 'test-only', fake)


if __name__ == '__main__':
    unittest.main()
