import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from stream_train import auto_stop_student as m
from stream_train.store import Store


class QueueTests(unittest.TestCase):
    def test_uncheckpointed_wal_visible(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'queue.sqlite'
            s = Store(path)
            s.bind('protocol', {'namespace': 'sample', 'student_base': 'http://127.0.0.1:8000/v1',
                                'student_model': 'student-8b'})
            writer = sqlite3.connect(str(path))
            try:
                writer.execute("INSERT INTO jobs VALUES ('x','student','{}','started',NULL,1,1)")
                writer.commit()
                self.assertTrue(Path(str(path) + '-wal').is_file())
                stat = path.stat()
                counts, _ = m.snapshot(path, (stat.st_dev, stat.st_ino), 'sample', 'student-8b', 8000)
                self.assertEqual(counts['started'], 1)
            finally:
                writer.close()

    def test_wal_snapshot_and_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'queue.sqlite'
            s = Store(path)
            s.bind('protocol', {'namespace': 'sample', 'student_base': 'http://127.0.0.1:8000/v1',
                                'student_model': 'student-8b'})
            s.enqueue('a', 'student', {'sensitive': 'never print'})
            s.claim(['student'])
            s.finish('a', {}, state='unknown')
            stat = path.stat()
            c, mark = m.snapshot(path, (stat.st_dev, stat.st_ino), 'sample', 'student-8b', 8000)
            self.assertEqual(c['unknown'], 1)
            self.assertEqual(c['pending'], 0)
            self.assertEqual(mark[0][:3], ('student', 'unknown', 1))
            with self.assertRaises(m.GuardError):
                m.snapshot(path, (stat.st_dev, stat.st_ino), 'wrong', 'student-8b', 8000)

    def test_empty_queue_is_not_finished(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'queue.sqlite'
            s = Store(path)
            s.bind('protocol', {'namespace': 'sample', 'student_base': 'http://127.0.0.1:8000/v1',
                                'student_model': 'student-8b'})
            stat = path.stat()
            with self.assertRaises(m.GuardError):
                m.snapshot(path, (stat.st_dev, stat.st_ino), 'sample', 'student-8b', 8000)


class Response:
    status = 200
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, n): return self.value


class MetricsTests(unittest.TestCase):
    def read(self, text):
        with patch.object(m.urllib.request, 'build_opener') as build:
            build.return_value.open.return_value = Response(text.encode())
            return m.metrics(8000)

    def test_all_labels_and_other_names(self):
        result = self.read('vllm:num_requests_running{model_name="a"} 0\n'
                           'vllm:num_requests_running_total 7\n'
                           'vllm:num_requests_waiting{model_name="a"} 2\n')
        self.assertEqual(result, (0, 2))

    def test_missing_nan_and_bad_values_fail_closed(self):
        for text in ('vllm:num_requests_running 0\n',
                     'vllm:num_requests_running NaN\nvllm:num_requests_waiting 0\n',
                     'vllm:num_requests_running -1\nvllm:num_requests_waiting 0\n',
                     'vllm:num_requests_running bad\nvllm:num_requests_waiting 0\n'):
            with self.subTest(text=text), self.assertRaises(m.GuardError): self.read(text)


class ProcessTests(unittest.TestCase):
    def test_cli_and_module_model_matching(self):
        common = ['--served-model-name', 'student-8b', '--host', '127.0.0.1', '--port', '8000']
        for a in (['/venv/bin/vllm', 'serve', '/models/student'] + common,
                  ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', '/models/student'] + common):
            with patch.object(m, 'proc_info', return_value=(a, 123)), patch.object(m, 'owns_listening_socket', return_value=True):
                self.assertEqual(m.validate_service(100, Path('/models/student'), 'student-8b', 8000), 123)
                with self.assertRaises(m.GuardError):
                    m.validate_service(100, Path('/models/other'), 'student-8b', 8000)
                with self.assertRaises(m.GuardError):
                    m.validate_service(100, Path('/models/student'), 'student-8b', 8000, 124)

    def test_pidfd_unavailable_fails_closed(self):
        with patch.object(m.os, 'pidfd_open', side_effect=PermissionError() , create=True):
            with patch.object(m.signal, 'pidfd_send_signal', create=True):
                with self.assertRaises(m.GuardError): m.open_pidfd(123)


class MonitorTests(unittest.TestCase):
    def run_monitor(self, snapshots=None, metrics=None, consumers=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / 'queue.sqlite'; db.touch()
        args = argparse.Namespace(db=str(db), model_path='/models/student', port=8000,
            interval=1, settle=60, namespace='sample', student_model='student-8b',
            service_pid=100, consumer_pid=101, confirm_initialized=True, confirm_dedicated=True)
        good = (Counter({'done': 2, 'unknown': 1}), ('stable',))
        snapshots = snapshots or [good] * 4
        metrics = metrics or [(0, 0)] * 4
        consumers = consumers or [{101}, {101}, set()]
        log = io.StringIO()
        patches = [patch.object(m, 'open_pidfd', side_effect=[8, 9]),
            patch.object(m, 'validate_service', return_value=1),
            patch.object(m, 'validate_consumer', return_value=2),
            patch.object(m, 'consumer_pids', side_effect=consumers),
            patch.object(m, 'snapshot', side_effect=snapshots),
            patch.object(m, 'metrics', side_effect=metrics),
            patch.object(m, 'stop_and_wait', return_value=True),
            patch.object(m.os, 'close'), patch.object(m.time, 'monotonic', side_effect=[1, 62]),
            patch.object(m.time, 'sleep')]
        mocks=[]
        for p in patches:
            mocks.append(p.start()); self.addCleanup(p.stop)
        with redirect_stdout(log):
            m.monitor(args)
        return mocks, log.getvalue()

    def test_drained_unknown_stops_consumer_first_and_reports(self):
        mocks, output = self.run_monitor()
        self.assertEqual([c.args[0] for c in mocks[6].call_args_list], [9, 8])
        self.assertIn('仍有未决', output)

    def test_queue_changed_after_consumer_no_service_stop(self):
        g = (Counter({'done': 2}), ('stable',))
        changed = (Counter({'pending': 1}), ('changed',))
        with self.assertRaises(m.GuardError):
            self.run_monitor(snapshots=[g, g, g, changed])

    def test_metrics_failure_never_stops(self):
        with self.assertRaises(m.GuardError):
            self.run_monitor(metrics=[m.GuardError('no metrics')])

    def test_pending_and_started_never_stop(self):
        for state in ('pending', 'started'):
            with self.subTest(state=state):
                busy = (Counter({state: 1}), ('busy',))
                with self.assertRaises(StopIteration):
                    self.run_monitor(snapshots=[busy, busy, busy])
                self.doCleanups()


if __name__ == '__main__': unittest.main()
