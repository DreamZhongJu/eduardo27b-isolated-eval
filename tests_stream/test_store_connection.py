"""Connections must close without weakening SQLite transaction semantics."""
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from stream_train.store import Store


class StoreConnectionTest(unittest.TestCase):
    def setUp(self):
        ram = Path('/dev/shm')
        temp_dir = str(ram) if ram.is_dir() and os.access(ram, os.W_OK) else None
        self.tmp = tempfile.TemporaryDirectory(dir=temp_dir)
        self.store = Store(Path(self.tmp.name) / 'queue.sqlite')

    def tearDown(self):
        self.tmp.cleanup()

    def test_successful_context_commits_and_closes(self):
        with self.store.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('INSERT INTO metadata VALUES (?,?)', ('success', 'value'))
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')
        with self.store.connect() as fresh:
            self.assertEqual(fresh.execute('SELECT value FROM metadata WHERE key=?',
                                           ('success',)).fetchone(), ('value',))

    def test_exception_rolls_back_and_closes(self):
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with self.store.connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                connection.execute('INSERT INTO metadata VALUES (?,?)', ('rolled_back', 'value'))
                raise RuntimeError('failure')
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')
        with self.store.connect() as fresh:
            self.assertIsNone(fresh.execute('SELECT value FROM metadata WHERE key=?',
                                            ('rolled_back',)).fetchone())

    def test_finish_collision_rolls_back_all_children_and_parent(self):
        self.store.enqueue('parent', 'generate', {'i': 1})
        self.store.enqueue('existing', 'grade', {'i': 1})
        self.assertEqual(self.store.claim(['generate'])[0], 'parent')
        with self.assertRaisesRegex(ValueError, 'identity collision'):
            self.store.finish('parent', {'result': 'done'}, [
                ('new-child', 'grade', {'i': 2}),
                ('existing', 'grade', {'i': 99}),
            ])
        with self.store.connect() as connection:
            self.assertEqual(connection.execute('SELECT state FROM jobs WHERE id=?',
                                                ('parent',)).fetchone(), ('started',))
            self.assertIsNone(connection.execute('SELECT state FROM jobs WHERE id=?',
                                                 ('new-child',)).fetchone())

    def test_pragma_failure_still_closes(self):
        fake = Mock()
        fake.execute.side_effect = sqlite3.OperationalError('pragma failed')
        with patch('stream_train.store.sqlite3.connect', return_value=fake):
            with self.assertRaises(sqlite3.OperationalError):
                with self.store.connect():
                    self.fail('context body must not run')
        fake.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
