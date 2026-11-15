import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from stream_train import quiesce_formal as q


class StudentDrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = (Path(self.tmp.name) / 'formal.sqlite').resolve()
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute('CREATE TABLE jobs(kind TEXT, state TEXT, payload TEXT, result TEXT)')
            conn.executemany('INSERT INTO jobs VALUES (?,?,?,?)', [
                ('student', 'pending', 'keep-pending', None),
                ('student', 'done', 'keep-done', 'keep-result'),
                ('grade', 'unknown', 'keep-unknown', None),
            ])
        st = self.db.stat()
        self.identity = (st.st_dev, st.st_ino)

    def rows(self):
        with sqlite3.connect(str(self.db)) as conn:
            return conn.execute('SELECT * FROM jobs ORDER BY payload').fetchall()

    def test_idle_student_stops_while_write_lock_held_without_row_changes(self):
        before = self.rows()
        def fake_terminate(pid, stamp, role, db, wait_seconds):
            self.assertEqual((pid, stamp, role), (123, 7, 'student'))
            other = sqlite3.connect(str(self.db), timeout=0)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    other.execute("INSERT INTO jobs VALUES ('student','started','racing',NULL)")
            finally:
                other.close()
            return True
        with patch.object(q, 'processes', return_value={'student': {123: 7}}), \
             patch.object(q, 'terminate', side_effect=fake_terminate):
            self.assertTrue(q.stop_student(self.db, self.identity, {123: 7}))
        self.assertEqual(before, self.rows())

    def test_started_student_is_never_signalled(self):
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute("INSERT INTO jobs VALUES ('student','started','in-flight',NULL)")
        before = self.rows()
        with patch.object(q, 'terminate') as terminate:
            self.assertFalse(q.stop_student(self.db, self.identity, {123: 7}))
            terminate.assert_not_called()
        self.assertEqual(before, self.rows())

    def test_multiple_or_changed_students_refuse_stop(self):
        with patch.object(q, 'processes', return_value={'student': {123: 7, 124: 8}}), \
             patch.object(q, 'terminate') as terminate:
            with self.assertRaises(q.StopError):
                q.stop_student(self.db, self.identity, {123: 7})
            terminate.assert_not_called()

    def test_database_identity_change_refuses_stop(self):
        with patch.object(q, 'terminate') as terminate:
            with self.assertRaises(q.StopError):
                q.stop_student(self.db, (-1, -1), {123: 7})
            terminate.assert_not_called()


class ProcessIdentityTests(unittest.TestCase):
    def test_exact_run_binding_and_roles(self):
        db = Path('/tmp/run/formal.sqlite')
        retry = ['python', '-m', 'stream_train.api_retry_supervisor', '--db', str(db)]
        adaptive = ['python', '-m', 'stream_train.api_adaptive_external', '--db', str(db)]
        launch = ['python', '-m', 'stream_train.launch_supervised', '--run-dir', '/tmp/run']
        student = ['python', '-m', 'stream_train', '--db', str(db), 'student']
        other = ['python', '-m', 'stream_train', '--db', '/tmp/other/formal.sqlite', 'student']
        for args, expected in ((retry, 'retry'), (adaptive, 'adaptive'), (launch, 'launcher'),
                               (student, 'student')):
            self.assertEqual(q.role(args), expected)
            self.assertTrue(q.matches_db(args, expected, db))
        self.assertFalse(q.matches_db(other, 'student', db))

    def test_signal_checks_pid_start_and_db_before_signal(self):
        db = Path('/tmp/run/formal.sqlite')
        args = ['python', '-m', 'stream_train', '--db', str(db), 'student']
        with patch.object(q.guard, 'open_pidfd', return_value=12), \
             patch.object(q.guard, 'proc_info', return_value=(args, 10)), \
             patch.object(q.signal, 'pidfd_send_signal', create=True) as send, \
             patch.object(q.os, 'close'):
            with self.assertRaises(q.StopError):
                q.terminate(123, 11, 'student', db)
            send.assert_not_called()

    def test_relative_db_and_run_directory_fail_closed(self):
        db = Path('/tmp/run/formal.sqlite')
        student = ['python', '-m', 'stream_train', '--db', 'formal.sqlite', 'student']
        launcher = ['python', '-m', 'stream_train.launch_supervised', '--run-dir', '.']
        with self.assertRaises(q.StopError):
            q.matches_db(student, 'student', db)
        with self.assertRaises(q.StopError):
            q.matches_db(launcher, 'launcher', db)


class OrchestrationTests(unittest.TestCase):
    def test_drains_only_formal_processes_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = (Path(tmp) / 'formal.sqlite').resolve()
            cfgpath = (Path(tmp) / 'config.json').resolve()
            cfgpath.write_text('{}')
            with sqlite3.connect(str(db)) as conn:
                conn.execute('CREATE TABLE jobs(kind TEXT, state TEXT, payload TEXT, result TEXT)')
                conn.executemany('INSERT INTO jobs VALUES (?,?,?,?)', [
                    ('review', 'pending', 'protected', None),
                    ('grade', 'unknown', 'unknown', None),
                    ('student', 'pending', 'next-student', None),
                ])
            states = {'retry': {10: 1}, 'adaptive': {11: 2}, 'legacy_api': {},
                      'student': {13: 4}, 'watcher': {12: 3},
                      'launcher': {14: 5}, 'handoff': {}}
            stopped = []
            def processes(_):
                return {key: value.copy() for key, value in states.items()}
            def terminate(pid, stamp, name, db_path, wait_seconds=0):
                stopped.append(name)
                states[name].pop(pid)
                if name == 'watcher':
                    states['launcher'].clear()
                return True
            def student(*args):
                states['student'].clear()
                return True
            def api_lock():
                return os.open(str(Path(tmp) / 'lock'), os.O_CREAT | os.O_RDWR, 0o600)
            with patch.object(q.app, 'config', return_value={'name': 'fixed'}), \
                 patch.object(q.producer, 'read_protocol', return_value={'name': 'fixed'}), \
                 patch.object(q, 'processes', side_effect=processes), \
                 patch.object(q, 'terminate', side_effect=terminate), \
                 patch.object(q, 'stop_student', side_effect=student), \
                 patch.object(q, 'lock_api', side_effect=api_lock), \
                 patch.object(q.time, 'sleep'):
                q.run(db, cfgpath, 10)
            self.assertEqual(stopped, ['retry', 'adaptive', 'watcher'])
            with sqlite3.connect(str(db)) as conn:
                actual = conn.execute('SELECT kind,state,payload,result FROM jobs ORDER BY payload').fetchall()
            self.assertEqual(actual, [('student', 'pending', 'next-student', None),
                                      ('review', 'pending', 'protected', None),
                                      ('grade', 'unknown', 'unknown', None)])


if __name__ == '__main__':
    unittest.main()
