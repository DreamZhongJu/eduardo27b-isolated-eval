from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stream_train import status_table_adaptive as m


class StatusTest(unittest.TestCase):
    def test_exact_module_and_database_match(self):
        run = Path('/workspace/results/formal')
        args = ['python', '-m', 'stream_train.api_adaptive_external',
                '--db', '/workspace/results/formal/formal.sqlite']
        self.assertTrue(m.matches_adaptive(args, run))
        self.assertFalse(m.matches_adaptive(args, Path('/workspace/results/other')))
        self.assertFalse(m.matches_adaptive(['python', '-m', 'stream_train', 'api'], run))

    def test_output_uses_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(m, 'original_scan', return_value=(Counter(), 0)), \
                    patch.object(m, 'adaptive_scan', return_value=(Counter({'adaptive_api': 1}), 0)), \
                    patch('sys.argv', ['status', '--run-dir', d]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.main()
            self.assertIn('其中自适应 API 生产者：1', out.getvalue())
            self.assertIn('正式队列暂无可显示的统计', out.getvalue())

    def test_window_extracts_only_validated_aggregate_fields(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self.assertEqual(m.window_status(run), (None, None, False))
            events = [
                {'event': 'start', 'maximum': 20, 'current': 1},
                {'event': 'window', 'maximum': 20, 'current': 10,
                 'reason': 'congestion', 'student_response': 'SECRET_ANSWER'},
                {'event': 'progress', 'maximum': 20, 'current': 12},
                {'event': 'window', 'maximum': True, 'current': 'SECRET_KEY',
                 'reason': 'SECRET_REASON'},
            ]
            (run / 'api-adaptive.log').write_text('\n'.join(json.dumps(x) for x in events))
            self.assertEqual(m.window_status(run), (12, 20, True))
            with patch.object(m, 'original_scan', return_value=(Counter(), 0)), \
                    patch.object(m, 'adaptive_scan', return_value=(Counter({'adaptive_api': 1}), 0)), \
                    patch('sys.argv', ['status', '--run-dir', d]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.main()
            self.assertIn('日志最近记录的窗口：12 / 20', out.getvalue())
            self.assertIn('近期日志是否出现拥塞：是', out.getvalue())
            self.assertNotIn('SECRET', out.getvalue())

    def test_new_start_resets_previous_congestion(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            events = [
                {'event': 'window', 'maximum': 20, 'current': 10, 'reason': 'congestion'},
                {'event': 'start', 'maximum': 8, 'current': 1},
            ]
            (run / 'api-adaptive.log').write_text('\n'.join(json.dumps(x) for x in events))
            self.assertEqual(m.window_status(run), (1, 8, False))

    def test_historical_window_is_labeled_when_process_absent(self):
        with tempfile.TemporaryDirectory() as d:
            events = [{'event': 'start', 'maximum': 20, 'current': 1}]
            (Path(d) / 'api-adaptive.log').write_text(json.dumps(events[0]))
            with patch.object(m, 'original_scan', return_value=(Counter(), 0)), \
                    patch.object(m, 'adaptive_scan', return_value=(Counter(), 0)), \
                    patch('sys.argv', ['status', '--run-dir', d]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.main()
            self.assertIn('自适应 API 未运行', out.getvalue())
            self.assertIn('日志最近记录的窗口：1 / 20', out.getvalue())

    def test_unreadable_process_scan_does_not_claim_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(m, 'original_scan', return_value=(Counter(), None)), \
                    patch.object(m, 'adaptive_scan', return_value=(Counter(), None)), \
                    patch('sys.argv', ['status', '--run-dir', d]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                m.main()
            self.assertIn('不能确认是否运行', out.getvalue())
            self.assertNotIn('自适应 API 未运行', out.getvalue())


if __name__ == '__main__':
    unittest.main()
