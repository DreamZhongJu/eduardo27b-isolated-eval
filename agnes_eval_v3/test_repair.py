import hashlib
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import agnes_judge as new
import watch_sources as watcher

test_parent = Path(__file__).resolve().parent.parent
legacy_candidates = [test_parent / 'agnes_eval_v2' / 'agnes_judge.py',
                     test_parent / 'eduardo27b-github-update-20261109' / 'agnes_eval_v2' / 'agnes_judge.py']
legacy_path = next(p for p in legacy_candidates if p.is_file())
legacy_spec = importlib.util.spec_from_file_location('legacy_judge', legacy_path)
legacy = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy)


def answer(text, finish='stop', model=new.MODEL):
    return {'model': model, 'choices': [{'finish_reason': finish, 'message': {'content': text}}]}


def row(index=0):
    return {'source_id': 'local-synthetic', 'target': 'source', 'phase': 'pre',
            'index': index, 'problem': 'Q', 'reference_answer': 'A', 'student_response': 'R'}


class TestRepair(unittest.TestCase):
    def test_http_envelope_rejects_duplicate_keys(self):
        class Response:
            status = 200
            def __init__(self, raw):
                self.raw = raw
            def read(self, _limit):
                return self.raw
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
        class Opener:
            def __init__(self, raw):
                self.raw = raw
            def open(self, *_args, **_kwargs):
                return Response(self.raw)
        valid = b'{"model":"agnes-3.0-flash","choices":[]}'
        duplicate = b'{"model":"agnes-3.0-flash","model":"other","choices":[]}'
        nested = b'{"model":"agnes-3.0-flash","choices":[{"x":1,"x":2}]}'
        with patch('urllib.request.build_opener', return_value=Opener(valid)):
            self.assertEqual(new.api_call({}, 'fake')[1]['model'], new.MODEL)
        for raw in (duplicate, nested):
            with patch('urllib.request.build_opener', return_value=Opener(raw)):
                code, body = new.api_call({}, 'fake')
                self.assertEqual(code, 200)
                self.assertIsNone(body)
                self.assertEqual(new.retryable_content_issue(body), 'response_not_json')

    def test_formats(self):
        cases = [('true', True), ('false', False), ('{"correct":true}', True),
                 ('{"correct":false}', False), ('null', None), ('{"correct":null}', None),
                 ('"true"', None), ('{"correct":"false"}', None), ('YES', None)]
        for content, expected in cases:
            self.assertIs(new.verdict(answer(content)), expected)
        self.assertIsNone(new.verdict(answer('true', finish='length')))
        self.assertIsNone(new.verdict(answer('true', model='wrong')))
        self.assertIsNone(new.retryable_content_issue(answer('null')))
        self.assertIsNone(new.retryable_content_issue(answer('{"correct":null}')))
        self.assertIsNone(new.retryable_content_issue(answer('true', model='wrong')))
        self.assertEqual(new.retryable_content_issue(answer('')), 'response_content_empty')
        self.assertEqual(new.retryable_content_issue(answer('garbage')), 'response_not_json')
        self.assertEqual(new.retryable_content_issue(answer('true', finish='length')), 'response_truncated')
        self.assertEqual(new.retryable_content_issue(answer('{"correct":"true"}')), 'response_schema_invalid')
        for content in ('{"correct":true,"correct":false}',
                        '{"correct":true,"extra":{"x":1,"x":2}}',
                        '[{"x":1,"x":2}]'):
            self.assertIsNone(new.verdict(answer(content)))
            self.assertEqual(new.retryable_content_issue(answer(content)), 'response_duplicate_keys')

    def test_retry_format_but_not_null_or_regrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen = []
            sequence = iter([answer(''), answer('not-json'), answer('false')])
            def caller(body, key):
                seen.append(1)
                return 200, next(sequence)
            w = new.AdaptiveWindow(5)
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                result = new.grade_one(row(), 'fake', root/'receipts', root/'locks', caller=caller,
                                       sleep=lambda _: None, window=w)
                self.assertEqual(result['status'], 'incorrect')
                self.assertEqual(result['parser_version'], new.PARSER_VERSION)
                self.assertEqual(w.window, 2)  # Invalid formats are neutral; final bool grows the window.
                self.assertEqual(w.failures, 0)
                self.assertEqual(w.blocked_until, 0)
                self.assertEqual(len(seen), 3)
                before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (root/'receipts').rglob('*') if p.is_file()}
                again = new.grade_one(row(), 'fake', root/'receipts', root/'locks',
                                      caller=lambda *_: self.fail('regrade'), sleep=lambda _: None)
                after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root/'receipts').rglob('*') if p.is_file()}
                self.assertEqual(before, after)
                self.assertEqual(again, result)

            null_calls = []
            def null_caller(*_):
                null_calls.append(1)
                return 200, answer('{"correct":null}')
            null_result = new.grade_one(row(1), 'fake', root/'receipts', root/'locks',
                                        caller=null_caller, sleep=lambda _: None)
            self.assertEqual(null_result['status'], 'unknown')
            self.assertEqual(len(null_calls), 1)

    def test_four_attempts_and_transport_backoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, sleeps = [], []
            def caller(*_):
                calls.append(1)
                return 200, answer('bad')
            window = new.AdaptiveWindow(5)
            window.window = 3
            result = new.grade_one(row(), 'fake', root/'receipts', root/'locks',
                                   caller=caller, sleep=sleeps.append, window=window)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['reason'], 'retry_limit')
            self.assertEqual(len(calls), 4)
            self.assertEqual(window.window, 3)
            self.assertEqual(window.blocked_until, 0)
            self.assertEqual(sleeps, [1, 2, 4])
            responses = list((root/'receipts').rglob('ATTEMPT-*-RESPONSE.json'))
            self.assertEqual(len(responses), 4)
            self.assertTrue(all(json.loads(p.read_text())['content_issue'] == 'response_not_json' for p in responses))
            w = new.AdaptiveWindow(5)
            w.window = 5
            w.active = 1
            w.release(None)
            self.assertEqual(w.window, 5)
            self.assertEqual(w.blocked_until, 0)
            w.active = 1
            w.release(False, rate_limited=True)
            self.assertEqual(w.window, 2)
            self.assertGreater(w.blocked_until, time.monotonic())

    def test_inflight_success_does_not_undo_429_cooldown(self):
        w = new.AdaptiveWindow(5)
        w.window = 5
        w.active = 5
        with patch.object(new.time, 'monotonic', return_value=100):
            w.release(False, rate_limited=True)
        self.assertEqual((w.window, w.failures, w.blocked_until), (2, 1, 160))
        with patch.object(new.time, 'monotonic', return_value=110):
            for _ in range(4):
                w.release(True)
        self.assertEqual((w.window, w.failures), (2, 1))
        w.active = 1
        with patch.object(new.time, 'monotonic', return_value=161):
            w.release(True)
        self.assertEqual((w.window, w.failures), (3, 0))

    def test_v2_result_immutable_and_mixed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_id = 'local-synthetic'
            source_key = hashlib.sha256(source_id.encode()).hexdigest()[:24]
            src = root/'stages'/'EASY'/'sources'/source_key
            src.mkdir(parents=True)
            source_rows = [row(0), row(1)]
            score = json.dumps(source_rows).encode()
            (src/'SCORE_INPUT.json').write_bytes(score)
            (src/'RAW_COMPLETE.json').write_text(json.dumps({'source_id': source_id, 'samples': 2,
                                    'score_input_sha256': hashlib.sha256(score).hexdigest()}))
            receipts = root/'agnes'/'receipts'/'EASY'
            lockdir = root/'locks'
            legacy_result = legacy.grade_one(row(0), 'fake', receipts/source_key, lockdir,
                                               caller=lambda *_: (200, answer('true')), sleep=lambda _: None)
            self.assertEqual(legacy_result['status'], 'unknown')
            self.assertNotIn('parser_version', legacy_result)
            total_calls = []
            def caller(*_):
                total_calls.append(1)
                return 200, answer('true')
            self.assertTrue(watcher.process_source(src, root/'agnes'/'EASY', receipts, lockdir,
                                                   'fake', caller=caller, workers=1))
            doc = json.loads((root/'agnes'/'EASY'/'sources'/source_key/'RESULTS.json').read_text())
            self.assertEqual(doc['results'][0], legacy_result)
            self.assertEqual(doc['results'][1]['status'], 'correct')
            self.assertEqual(doc['results'][1]['parser_version'], new.PARSER_VERSION)
            self.assertEqual(doc['completion_parser_version'], new.PARSER_VERSION)
            self.assertEqual(len(total_calls), 1)
            self.assertTrue(watcher.process_source(src, root/'agnes'/'EASY', receipts, lockdir,
                                                   'fake', caller=lambda *_: self.fail('source regraded'), workers=1))

    def test_resume_saved_response_and_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipts, locks = root/'receipts', root/'locks'
            old = legacy.grade_one(row(), 'fake', receipts, locks,
                                   caller=lambda *_: (200, answer('true')), sleep=lambda _: None)
            folder = receipts / new.digest({'identity': new.identity(row())})
            (folder/'RESULT.json').unlink()  # Simulate old process dying after the response write.
            recovered = new.grade_one(row(), 'fake', receipts, locks,
                                      caller=lambda *_: self.fail('saved response must be reused'), sleep=lambda _: None)
            self.assertEqual(old['status'], 'unknown')
            self.assertEqual(recovered['status'], 'correct')
            self.assertEqual(recovered['parser_version'], new.PARSER_VERSION)
            ident = new.identity(row(1))
            request_hash = new.digest(new.payload(row(1)))
            folder2 = receipts / new.digest({'identity': ident})
            folder2.mkdir(parents=True)
            new.put_once(folder2/'BINDING.json', {'identity': ident, 'request_sha256': request_hash,
                                                  'model': new.MODEL, 'grader': 'agnes_full_response_v2'})
            new.put_once(folder2/'ATTEMPT-0001-INTENT.json', {'attempt': 1, 'request_sha256': request_hash})
            result = new.grade_one(row(1), 'fake', receipts, locks,
                                   caller=lambda *_: (200, answer('false')), sleep=lambda _: None)
            self.assertEqual(result['status'], 'incorrect')
            self.assertEqual(json.loads((folder2/'ATTEMPT-0001-RESPONSE.json').read_text())['error_type'], 'Interrupted')
            self.assertTrue((folder2/'ATTEMPT-0002-RESPONSE.json').exists())

    def test_shared_slots_never_exceed_five(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mutex = threading.Lock()
            active = maximum = 0
            def caller(*_):
                nonlocal active, maximum
                with mutex:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.03)
                with mutex:
                    active -= 1
                return 200, answer('true')
            def one(i):
                return new.grade_one(row(i), 'fake', root/'receipts', root/'locks',
                                     caller=caller, sleep=lambda _: None)['status']
            with ThreadPoolExecutor(max_workers=12) as pool:
                self.assertEqual(list(pool.map(one, range(12))), ['correct'] * 12)
            self.assertLessEqual(maximum, 5)
            self.assertGreater(maximum, 1)


if __name__ == '__main__':
    unittest.main()
