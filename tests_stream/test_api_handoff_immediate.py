import os
from pathlib import Path
import json
import select
import sqlite3
import subprocess
import sys
from unittest.mock import patch

from test_api_handoff import RealSubprocessTests
from stream_train import api_handoff as base
from stream_train import api_handoff_immediate as immediate


class ImmediateRealTests(RealSubprocessTests):
    def run_immediate(self, journal=lambda state, **fields: None):
        with patch.dict(os.environ, {'HOME': str(self.root)}):
            return immediate.attempt_immediate(self.fd, self.proc.pid, self.start,
                                               self.db, self.config, self.identity, journal)

    def test_only_started_api_rows_become_unknown_and_full_audit_is_atomic(self):
        self.store.enqueue('sample2', 'review', {'not': 'displayed'})
        self.store.enqueue('student', 'student', {'not': 'displayed'})
        self.store.enqueue('finished', 'grade', {'not': 'displayed'})
        self.store.enqueue('pending', 'grade', {'not': 'displayed'})
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute("UPDATE jobs SET state='started' WHERE id IN ('sample','sample2')")
            conn.execute("UPDATE jobs SET state='started',result='old-result' WHERE id='student'")
            conn.execute("UPDATE jobs SET state='done',result='finished-result' WHERE id='finished'")
            before = {row[0]: row[1:] for row in conn.execute('SELECT id,kind,payload,state,result,updated FROM jobs')}
        events = []
        result = self.run_immediate(lambda state, **fields: events.append((state, fields)))
        self.assertEqual(result, (True, 2))
        self.proc.wait(timeout=3)
        with sqlite3.connect(str(self.db)) as conn:
            after = {row[0]: row[1:] for row in conn.execute('SELECT id,kind,payload,state,result,updated FROM jobs')}
            audit = conn.execute('SELECT job_id,kind,previous_result,previous_updated,old_pid '
                                 'FROM api_handoff_interrupt_audit_v1 ORDER BY job_id').fetchall()
        for jid in before:
            want = list(before[jid])
            if jid in ('sample', 'sample2'):
                want[2] = 'unknown'
            self.assertEqual(after[jid], tuple(want))
        self.assertEqual([row[0] for row in audit], ['sample', 'sample2'])
        self.assertTrue(all(row[2] is None and row[4] == self.proc.pid for row in audit))
        self.assertEqual(events[0][1]['api_started'], 2)

    def test_more_than_five_api_started_refuses_without_stopping_or_mutating(self):
        for i in range(6):
            self.store.enqueue('over-%d' % i, 'grade', {})
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute("UPDATE jobs SET state='started' WHERE id LIKE 'over-%'")
        with self.assertRaises(base.HandoffError):
            self.run_immediate()
        self.assertIsNone(self.proc.poll())
        with sqlite3.connect(str(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM jobs WHERE state='started'").fetchone()[0], 6)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='api_handoff_interrupt_audit_v1'").fetchone()[0], 0)

    def test_started_with_nonempty_result_is_never_changed(self):
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute("UPDATE jobs SET state='started',result='existing-result' WHERE id='sample'")
        with self.assertRaises(base.HandoffError):
            self.run_immediate()
        self.assertIsNone(self.proc.poll())
        with sqlite3.connect(str(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT state,result FROM jobs WHERE id='sample'").fetchone(),
                             ('started', 'existing-result'))

    def test_failure_before_signal_leaves_process_and_database_untouched(self):
        self.store.claim(['generate'])
        def fail(*args, **kwargs):
            raise OSError('synthetic journal failure')
        with self.assertRaises(OSError):
            self.run_immediate(fail)
        self.assertIsNone(self.proc.poll())
        self.assertEqual(self.store.counts()[0]['state'], 'started')
        with sqlite3.connect(str(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='api_handoff_interrupt_audit_v1'").fetchone()[0], 0)

    def test_busy_db_does_not_interrupt_process(self):
        writer = sqlite3.connect(str(self.db), isolation_level=None)
        try:
            writer.execute('BEGIN IMMEDIATE')
            self.assertEqual(self.run_immediate(), (False, None))
            self.assertIsNone(self.proc.poll())
        finally:
            writer.rollback()
            writer.close()

    def test_interrupted_job_is_preserved_when_actual_new_producer_is_ready(self):
        self.store.claim(['generate'])
        with patch.dict(os.environ, {'HOME': str(self.root)}):
            immediate.immediate_handoff(self.proc.pid, self.db, self.config,
                                        timeout=3, poll_interval=.01)
        self.proc.wait(timeout=3)
        r, w = os.pipe()
        env = os.environ.copy()
        env.update({'HOME': str(self.root),
                    'PYTHONPATH': str(Path(base.__file__).resolve().parent.parent),
                    'AGNES_API_KEY': 'synthetic-never-used',
                    'STREAM_PRODUCER_READY_FD': str(w)})
        policy = self.root / 'policy-immediate.json'
        producer = subprocess.Popen([sys.executable, '-m', 'stream_train.api_adaptive_external',
             '--db', str(self.db), '--config', str(self.config), '--max-concurrency', '2',
             '--policy-output', str(policy)], cwd=env['PYTHONPATH'], env=env,
             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
             pass_fds=(w,))
        os.close(w)
        try:
            self.assertTrue(select.select([r], [], [], 5)[0])
            self.assertEqual(os.read(r, 1), b'R')
            self.assertEqual(json.loads(policy.read_text())['maximum'], 2)
            with sqlite3.connect(str(self.db)) as conn:
                self.assertEqual(conn.execute("SELECT state,result FROM jobs WHERE id='sample'").fetchone(),
                                 ('unknown', None))
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM api_handoff_interrupt_audit_v1').fetchone()[0], 1)
        finally:
            os.close(r)
            if producer.poll() is None:
                producer.terminate()
            producer.wait(timeout=5)
