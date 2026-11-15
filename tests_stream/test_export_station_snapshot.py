import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from stream_train import export_station_snapshot as s
from stream_train.store import Store


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.code = self.root / 'code'
        self.code.mkdir()
        self.source = self.root / 'sources.jsonl'
        self.source.write_text('{"problem":"PRIVATE_SOURCE"}\n')
        self.output = self.root / 'out.zip'
        config = {'namespace': 'test', 'api_base': 'https://api.agnes-ai.cn/v1',
                  'api_model': 'test', 'student_base': 'http://127.0.0.1:18761/v1',
                  'student_model': 'student-8b'}
        for name in ('__main__.py', 'pipeline.py', 'store.py'):
            (self.code / name).write_text('frozen-code-' + name)
        protocol = dict(config, code_sha256={name: s.digest(self.code / name)
                         for name in ('__main__.py', 'pipeline.py', 'store.py')})
        (self.run / 'config.json').write_text(json.dumps(config))
        self.db = self.run / 'formal.sqlite'
        store = Store(self.db)
        store.bind('protocol', protocol)
        store.bind('source_sha256', hashlib.sha256(self.source.read_bytes()).hexdigest())
        store.bind('held_source_ids_sha256', 'held')
        store.enqueue('pending', 'review', {'problem': 'PRIVATE_PROMPT'})
        store.enqueue('started', 'student', {'answer': 'PRIVATE_ANSWER'})
        self.started = store.claim(['student'])
        self.patch_home = patch.dict(os.environ, {'HOME': str(self.root)})
        self.patch_home.start()
        self.addCleanup(self.patch_home.stop)
        self.patch_live = patch.object(s, 'live_run_processes', return_value=[])
        self.patch_live.start()
        self.addCleanup(self.patch_live.stop)
        # The fixtures are tiny; /dev/shm in CI may be smaller than the
        # production safety reserve despite having room for these tests.
        self.patch_space = patch.object(s.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12))
        self.patch_space.start()
        self.addCleanup(self.patch_space.stop)

    def snapshot(self):
        s.create_snapshot(self.run, self.source, self.code, self.output, 60)

    def test_wal_and_incomplete_states_preserved_without_input_duplicate(self):
        writer = sqlite3.connect(str(self.db))
        try:
            writer.execute("INSERT INTO jobs VALUES ('unknown','grade','{}','unknown',NULL,1,1)")
            writer.commit()
            self.assertTrue(Path(str(self.db) + '-wal').exists())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.snapshot()
            self.assertNotIn('PRIVATE_', output.getvalue())
            with zipfile.ZipFile(self.output) as z:
                self.assertEqual(set(z.namelist()), {'formal.sqlite', 'config.json',
                    'runtime_receipts.json', 'manifest.json'})
                manifest = json.loads(z.read('manifest.json'))
                self.assertTrue(all(x['matches'] for x in manifest['code_sha256'].values()))
                env = manifest['environment']
                self.assertEqual(set(env['package_versions']), {'vllm', 'torch', 'transformers'})
                self.assertEqual(env['model']['weights_checksum'], 'not_checked')
                extracted = self.root / 'extracted.sqlite'
                extracted.write_bytes(z.read('formal.sqlite'))
            with sqlite3.connect(str(extracted)) as con:
                rows = con.execute('SELECT id,state FROM jobs ORDER BY id').fetchall()
            self.assertEqual(rows, [('pending', 'pending'), ('started', 'started'), ('unknown', 'unknown')])
        finally:
            writer.close()

    def test_reject_running_tasks_and_preserve_existing_zip(self):
        with patch.object(s, 'live_run_processes', return_value=['worker']):
            with self.assertRaises(s.SnapshotError):
                self.snapshot()
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b'existing')
        with self.assertRaises(s.SnapshotError):
            self.snapshot()
        self.assertEqual(self.output.read_bytes(), b'existing')

    def test_reject_mismatched_source(self):
        self.source.write_text('changed')
        with self.assertRaises(s.SnapshotError):
            self.snapshot()
        self.assertFalse(self.output.exists())

    def test_reject_suspected_credential_without_echo(self):
        with sqlite3.connect(str(self.db)) as con:
            con.execute('UPDATE jobs SET payload=? WHERE id=?', ('sk-' + 'a' * 30, 'pending'))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(s.SnapshotError):
            self.snapshot()
        self.assertFalse(self.output.exists())
        self.assertNotIn('sk-', output.getvalue())

    def test_space_check_needs_capacity_before_opening_snapshot(self):
        with patch.object(s.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaises(s.SnapshotError):
                s.check_disk(self.run, self.output)
            with self.assertRaises(s.SnapshotError):
                self.snapshot()
        self.assertFalse(self.output.exists())

    def test_reserved_api_lock_prevents_snapshot(self):
        lock_path = self.root / '.cache' / 'math-transfer-stream' / 'api.lock'
        lock_path.parent.mkdir(parents=True)
        with lock_path.open('w+') as f:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(s.SnapshotError):
                self.snapshot()
        self.assertFalse(self.output.exists())

    def test_detect_relative_worker_and_direct_watcher(self):
        self.patch_live.stop()
        proc = self.root / 'proc'
        proc.mkdir()
        cwd = self.run / 'nested'
        cwd.mkdir()
        for pid, args in (
            ('7777771', ['python', '-m', 'stream_train', '--db', '../formal.sqlite', 'student']),
            ('7777772', ['python', '/x/auto_stop_student.py', '--db', '../formal.sqlite']),
        ):
            entry = proc / pid
            entry.mkdir()
            (entry / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in args) + b'\0')
            (entry / 'cwd').symlink_to(cwd)
        found = s.live_run_processes(self.db.resolve(), self.run.resolve(), proc)
        self.assertEqual(found, ['worker_or_retry', 'worker_or_retry'])

    def test_related_process_without_db_fails_closed(self):
        self.patch_live.stop()
        proc = self.root / 'proc'
        proc.mkdir()
        entry = proc / '7777773'
        entry.mkdir()
        (entry / 'cmdline').write_bytes(b'python\0-m\0stream_train.api_retry_supervisor\0')
        with self.assertRaises(s.SnapshotError):
            s.live_run_processes(self.db.resolve(), self.run.resolve(), proc)

    def test_only_model_config_hashes_are_recorded(self):
        model = self.root / 'student-model'
        model.mkdir()
        (model / 'config.json').write_text('{"model":"sample"}')
        (model / 'tokenizer_config.json').write_text('{}')
        # A malformed or huge weight file must not be read or included.
        (model / 'model.safetensors').write_bytes(b'NEVER_READ_WEIGHT')
        with contextlib.redirect_stdout(io.StringIO()):
            s.create_snapshot(self.run, self.source, self.code, self.output, 60, model)
        with zipfile.ZipFile(self.output) as z:
            manifest = json.loads(z.read('manifest.json'))
            self.assertNotIn('model.safetensors', z.namelist())
        m = manifest['environment']['model']
        self.assertEqual(m['config_sha256'], s.digest(model / 'config.json'))
        self.assertEqual(m['tokenizer_config_sha256'], s.digest(model / 'tokenizer_config.json'))
        self.assertEqual(m['weights_checksum'], 'not_checked')


if __name__ == '__main__':
    unittest.main()
