import contextlib
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest

from stream_train import status_table as status


class ReadOnlyStatusTests(unittest.TestCase):
    def test_missing_database_stays_missing(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / 'formal.sqlite'
            self.assertEqual(status.read_queue(db), ('尚未创建', None))
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_locked_database_returns_safe_status(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'formal.sqlite'
            db = sqlite3.connect(str(path), isolation_level=None)
            try:
                db.execute('CREATE TABLE jobs(kind TEXT,state TEXT)')
                db.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
                db.execute('BEGIN EXCLUSIVE')
                self.assertEqual(status.read_queue(path), ('无法读取或数据库暂时锁定', None))
            finally:
                db.close()

    def test_aggregates_without_printing_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            db = sqlite3.connect(str(run / 'formal.sqlite'))
            db.execute('CREATE TABLE jobs(kind TEXT,state TEXT,payload TEXT)')
            db.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
            db.execute("INSERT INTO metadata VALUES ('protocol', 'private-value')")
            db.executemany('INSERT INTO jobs VALUES (?,?,?)', [
                ('student', 'done', 'secret-question'),
                ('student', 'unknown', 'secret-answer'),
                ('grade', 'pending', 'secret-output')])
            db.commit()
            db.close()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status.display(run)
            text = output.getvalue()
            self.assertIn('学生作答', text)
            self.assertIn('判分', text)
            self.assertNotIn('secret', text)
            self.assertNotIn('private-value', text)
            _, counts = status.read_queue(run / 'formal.sqlite')
            self.assertEqual(counts[('student', 'done')], 1)
            self.assertEqual(counts[('student', 'unknown')], 1)
            self.assertEqual(counts[('grade', 'pending')], 1)

    def test_unknown_states_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'formal.sqlite'
            db = sqlite3.connect(str(path))
            db.executescript("CREATE TABLE jobs(kind TEXT,state TEXT);"
                             "CREATE TABLE metadata(key TEXT,value TEXT);"
                             "INSERT INTO metadata VALUES('protocol', '{}');"
                             "INSERT INTO jobs VALUES('student', 'something-new');")
            db.commit()
            db.close()
            self.assertEqual(status.read_queue(path), ('结构或状态异常', None))


if __name__ == '__main__':
    unittest.main()
