from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stream_train import status_table_retry as m


class RetryStatusTests(unittest.TestCase):
    def test_process_matches_only_exact_module_and_formal_database(self):
        run = Path('/workspace/results/run')
        correct = ['python', '-m', 'stream_train.api_retry_supervisor',
                   '--db', '/workspace/results/run/formal.sqlite']
        self.assertTrue(m.matches_retry(correct, run))
        self.assertFalse(m.matches_retry(correct, Path('/workspace/results/other')))
        self.assertFalse(m.matches_retry(['python', '-m', 'stream_train', '--db',
                                         '/workspace/results/run/formal.sqlite', 'api'], run))

    def test_log_tail_allowlists_fields_and_computes_time(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            events = [
                {'event': 'producer_spawned', 'time_utc': '2026-01-01T00:00:00Z'},
                {'event': 'probe_failed', 'stage': 'grade', 'category': 'protocol',
                 'time_utc': '2026-01-01T00:00:00Z',
                 'next_wait_seconds': 3600, 'secret': 'PRIVATE ANSWER'},
                {'event': ['PRIVATE'], 'stage': 'PRIVATE'},
                {'event': 'not_a_fixed_event', 'time_utc': 'PRIVATE'},
            ]
            (run / 'api-retry.log').write_text('\n'.join(json.dumps(e) for e in events))
            kind, stage, category, remaining = m.retry_status(run, now=1767228900)
            self.assertEqual((kind, stage, category, remaining), ('probe_failed', '判分', '响应协议异常', 300))

    def test_untrusted_stage_and_wait_never_echoed(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'api-retry.log').write_text(json.dumps({
                'event': 'probe_failed', 'stage': ['PRIVATE'], 'category': 'PRIVATE',
                'time_utc': '2026-01-01T00:00:00Z', 'next_wait_seconds': 'PRIVATE'}))
            self.assertEqual(m.retry_status(run, now=1767225600), ('probe_failed', None, None, None))

    def test_probe_failure_displays_only_fixed_category(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'api-retry.log').write_text(json.dumps({
                'event': 'probe_failed', 'stage': 'generate', 'category': 'auth',
                'reason': 'PRIVATE_PROVIDER_RESPONSE',
                'time_utc': '2026-01-01T00:00:00Z', 'next_wait_seconds': 60}))
            with patch.object(m.adaptive, 'adaptive_scan', return_value=(Counter(), 0)), \
                    patch.object(m, 'scan_retry', return_value=(1, 0)), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.display(run)
            self.assertIn('最近探测失败类别：认证或权限异常', out.getvalue())
            self.assertNotIn('PRIVATE_PROVIDER_RESPONSE', out.getvalue())

    def test_missing_and_invalid_log_timestamp(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self.assertEqual(m.retry_status(run), (None, None, None, None))
            (run / 'api-retry.log').write_text(json.dumps({
                'event': 'temporary_failure', 'time_utc': 'PRIVATE', 'next_wait_seconds': 100}))
            self.assertEqual(m.retry_status(run), ('temporary_failure', None, None, None))

    def test_display_marks_history_and_never_echoes_secret(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'api-retry.log').write_text(json.dumps({
                'event': 'producer_exited', 'time_utc': '2026-01-01T00:00:00Z',
                'next_wait_seconds': 60, 'raw_response': 'PRIVATE_ANSWER'}))
            with patch.object(m.adaptive, 'adaptive_scan', return_value=(Counter(), 0)), \
                    patch.object(m, 'scan_retry', return_value=(0, 0)), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.display(run)
            self.assertIn('API 自动重试守护进程：0', out.getvalue())
            self.assertIn('信息为历史日志', out.getvalue())
            self.assertNotIn('PRIVATE', out.getvalue())

    def test_inaccessible_processes_not_definitely_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(m.adaptive, 'adaptive_scan', return_value=(Counter(), None)), \
                    patch.object(m, 'scan_retry', return_value=(0, None)), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.display(Path(d))
            self.assertIn('守护进程无法确认', out.getvalue())
            self.assertNotIn('未发现守护', out.getvalue())


if __name__ == '__main__':
    unittest.main()
