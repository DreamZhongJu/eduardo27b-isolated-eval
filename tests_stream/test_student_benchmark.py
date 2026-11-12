import json
import io
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest
from unittest.mock import patch

from stream_train import student_benchmark as b
from stream_train.pipeline import PARAMS, STUDENT_PROMPT


class Handler(BaseHTTPRequestHandler):
    lock = threading.Lock()
    active = 0
    peak = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        request = json.loads(self.rfile.read(length))
        if request['model'] == 'http-failure':
            self.send_response(429)
            self.end_headers()
            return
        with self.lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
        time.sleep(.02)
        response = {'model': request['model'],
                    'choices': [{'finish_reason': 'stop',
                                 'message': {'content': 'generated-content-must-not-be-reported'}}]}
        if request['model'] != 'missing-usage':
            response['usage'] = {'completion_tokens': 11}
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        with self.lock:
            type(self).active -= 1


class StudentBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:%d/v1' % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_only_local_http_without_credentials_or_query(self):
        self.assertEqual(b.endpoint(self.base), self.base + '/chat/completions')
        for bad in ('https://example.com/v1', 'http://example.com/v1',
                    'http://user:password@127.0.0.1/v1', 'http://127.0.0.1/v1?key=x',
                    'http://127.0.0.1/v1#fragment'):
            with self.subTest(base=bad), self.assertRaises(ValueError):
                b.endpoint(bad)

    def test_preserves_sampling_and_16_sample_groups(self):
        rows = [b.payload('student', i, 512) for i in range(32)]
        self.assertEqual(len({r['seed'] for r in rows}), 32)
        self.assertEqual(len({r['messages'][0]['content'] for r in rows[:16]}), 1)
        self.assertNotEqual(rows[0]['messages'], rows[16]['messages'])
        self.assertEqual(rows[0]['max_tokens'], 512)
        self.assertEqual(rows[0]['n'], 1)
        self.assertIn('\\boxed{}', rows[0]['messages'][0]['content'])
        for key, value in PARAMS.items():
            if key != 'max_tokens':
                self.assertEqual(rows[0][key], value)
        self.assertTrue(STUDENT_PROMPT.endswith('Problem: {}'))

    def test_real_local_concurrent_http_returns_only_aggregate(self):
        Handler.peak = 0
        result = b.run_level(b.endpoint(self.base), 'student', 4, 8, 128, 5, 'not-a-real-key')
        self.assertEqual((result['succeeded'], result['failed'], result['truncated']), (8, 0, 0))
        self.assertEqual((result['submitted'], result['unsubmitted']), (8, 0))
        self.assertEqual(result['mean_output_tokens'], 11)
        self.assertGreaterEqual(Handler.peak, 2)
        self.assertLessEqual(Handler.peak, 4)
        self.assertGreater(result['tokens_s'], 0)
        self.assertGreaterEqual(result['p95_s'], result['p50_s'])
        self.assertNotIn('generated-content-must-not-be-reported', repr(result))
        self.assertNotIn('not-a-real-key', b.table([result]))

    def test_failures_are_counted_without_retries_or_contents(self):
        result = b.run_level(b.endpoint(self.base), 'http-failure', 2, 2, 128, 5, 'secret')
        self.assertEqual((result['succeeded'], result['failed']), (0, 2))
        self.assertEqual(result['failure_reasons'], {'http_429': 2})
        self.assertIsNone(result['tokens_s'])
        self.assertIsNone(result['mean_output_tokens'])
        self.assertIsNone(result['p50_s'])
        self.assertNotIn('secret', repr(result))

    def test_missing_usage_is_not_fabricated(self):
        result = b.run_level(b.endpoint(self.base), 'missing-usage', 2, 2, 128, 5, 'local')
        self.assertEqual(result['succeeded'], 2)
        self.assertIsNone(result['tokens_s'])
        self.assertIsNone(result['mean_output_tokens'])
        self.assertIn('不可用', b.table([result]))

    def test_rejects_underfilled_or_oversized_concurrency(self):
        with self.assertRaises(ValueError):
            b.run_level(b.endpoint(self.base), 'student', 4, 3, 128, 5, 'local')
        with self.assertRaises(ValueError):
            b.run_level(b.endpoint(self.base), 'student', 65, 65, 128, 5, 'local')

    def test_first_failure_stops_submitting_current_level(self):
        def fake(url, model, index, max_tokens, timeout, key):
            if index == 0:
                return {'ok': False, 'reason': 'http_429'}
            time.sleep(.03)
            return {'ok': True, 'seconds': .03, 'tokens': 11, 'truncated': False}
        with patch.object(b, 'one_request', side_effect=fake):
            result = b.run_level('local', 'student', 4, 256, 128, 5, 'local')
        self.assertEqual(result['submitted'], 4)
        self.assertEqual(result['unsubmitted'], 252)
        self.assertEqual((result['succeeded'], result['failed']), (3, 1))

    def test_failed_level_stops_sweep_and_warns_to_drain(self):
        failed = {'workers': 16, 'requested': 256, 'submitted': 16,
                  'unsubmitted': 240, 'succeeded': 15, 'failed': 1, 'truncated': 0,
                  'failure_reasons': {'connection_or_parse_error': 1}, 'seconds': 1,
                  'samples_s': 15, 'tokens_s': 165, 'mean_output_tokens': 11,
                  'p50_s': .1, 'p95_s': .2}
        output = io.StringIO()
        with patch.object(b, 'one_request', return_value={'ok': True}), \
             patch.object(b, 'run_level', return_value=failed) as level, \
             patch('sys.argv', ['student_benchmark', '--student-model', 'student']), \
             contextlib.redirect_stdout(output):
            self.assertEqual(b.main(), 1)
        self.assertEqual(level.call_count, 1)
        self.assertEqual(level.call_args.args[3], 256)
        self.assertIn('后续档位', output.getvalue())
        self.assertIn('排空', output.getvalue())
        self.assertIn('平均输出token', output.getvalue())

    def test_failed_warmup_stops_without_more_requests(self):
        output = io.StringIO()
        with patch.object(b, 'one_request', return_value={'ok': False, 'reason': 'http_503'}) as request, \
             patch.object(b, 'run_level') as level, \
             patch('sys.argv', ['student_benchmark', '--student-model', 'student']), \
             contextlib.redirect_stdout(output):
            self.assertEqual(b.main(), 1)
        self.assertEqual(request.call_count, 1)
        level.assert_not_called()
        self.assertIn('排空', output.getvalue())


if __name__ == '__main__':
    unittest.main()
