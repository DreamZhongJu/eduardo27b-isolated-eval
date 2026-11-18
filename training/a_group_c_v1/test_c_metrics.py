import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch
import logging
import os
from pathlib import Path
import tempfile
import unittest

import c_metrics as c


class CReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_http_config_and_boundary(self):
        try:
            from uvicorn.config import LOGGING_CONFIG
            from uvicorn.logging import AccessFormatter
        except ImportError:
            self.skipTest('uvicorn missing')
        original_format = LOGGING_CONFIG['formatters']['access']['fmt']
        config = json.loads(c.write_access_config(self.root).read_text())
        self.assertEqual(LOGGING_CONFIG['formatters']['access']['fmt'], original_format)
        fmt = AccessFormatter(config['formatters']['access']['fmt'], use_colors=False)
        def log(ts, method, path, code):
            rec = logging.LogRecord('uvicorn.access', 20, 'file', 1, '%s - "%s %s HTTP/%s" %d',
                                    ('local', method, path, '1.1', code), None)
            rec.created = ts
            return fmt.format(rec)
        path = self.root / 'server.log'
        path.write_text('\n'.join([log(1000, 'POST', '/v1/chat/completions', 200),
                                   log(1003, 'POST', '/v1/chat/completions', 500),
                                   log(1008, 'GET', '/health', 200),
                                   log(1009, 'POST', '/v1/chat/completions', 200)]))
        counts = c.tally_access(path, 1004.0)
        self.assertEqual(dict(counts), {'all': 3, 'train': 1, 'val': 1, 'unassigned': 1, 'non_2xx': 1})
        self.assertEqual(c.tally_access(path, None)['unassigned'], 3)

    def test_timing_filter(self):
        log = self.root / 'trainer.log'
        log.write_text('timing_s/step: 42\ntraining/global_step:10 timing_s/step:90\n'
                       'training/global_step:1 - timing_s/step: 20 - timing_s/gen: 12 - '
                       'timing_s/testing: 30 - timing_s/update_actor: 1.5\n'
                       'training/global_step:1 timing_s/gen:15\n')
        self.assertEqual(c.read_timings(log), {'step': 20, 'testing': 30, 'update_actor': 1.5})

    def test_persisted_marker_crosschecked_against_record_sets(self):
        marker = self.root / 'latest_checkpointed_iteration.txt'
        marker.write_text('1')
        os.utime(marker, (100, 100))
        stamps = {'train': [99.0] * 256, 'val': [101.0] * 100}
        self.assertEqual(c.checkpoint_boundary(marker, stamps), 100)
        stamps['val'][0] = 98.0
        self.assertIsNone(c.checkpoint_boundary(marker, stamps))
        stamps['val'].pop(0)
        self.assertIsNone(c.checkpoint_boundary(marker, stamps))

    def test_identity_no_order_assumption_and_unusable_records(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                return list(text)
        def row(uid, src, content):
            return dict(uuid=uid, problem=src, messages=[dict(role='assistant', content=content)],
                        gated=1, note='ok', ts=50.)
        teacher = row('train-1', 't', 'a'*662)
        teacher['messages'].append({'role': 'user', 'content': 'some text'})
        observations = [row('val-1', 'z', ''), teacher,
                        row('train-1', 't', 'other'), row('val-2', 'out', ''), {}]
        p = self.root / 'records.jsonl'
        p.write_text('\n'.join(json.dumps(r) for r in observations))
        splits = {c.source_hash('z'): 'val', c.source_hash('t'): 'train'}
        data, _, err = c.tally_trajectories([p], splits, str.strip, Tokenizer())
        self.assertEqual(data['train']['rows'], 1)
        self.assertEqual(data['train']['at_cap'], 1)
        self.assertEqual(data['train']['possible_omitted_teacher'], 1)
        self.assertEqual(data['val']['possible_omitted_teacher'], 0)
        self.assertEqual(data['val']['invisible'], 1)
        self.assertEqual(err['重复UUID'], 1)
        self.assertEqual(err['题单未绑定'], 1)
        self.assertEqual(err['记录格式'], 1)
        self.assertEqual(err['来源重复次数不符'], 1)

    def test_estimate_separates_final_validation_and_save(self):
        self.assertEqual(c.estimate_seconds(dict(step=100, save_checkpoint=20, testing=60)),
                         {10: 880, 100: 8320})
        self.assertFalse(c.estimate_seconds(dict(step=2, save_checkpoint=3, testing=6)))

    def test_cue_queue_and_unstamped_record(self):
        p = self.root / 'service.log'
        p.write_text('Engine 000: Running: 2 reqs, Waiting: 3 reqs.\n'
                     'Engine 000: Running: 0 reqs, Waiting: 0 reqs.\n'
                     'INFO: "POST /v1/chat/completions HTTP/1.1" 200 OK\n')
        self.assertEqual(c.judge_queue(p)['peak_waiting'], 3)
        self.assertEqual(c.judge_queue(p)['active'], 1)
        self.assertEqual(c.tally_access(p, 1234)['unassigned'], 1)

    def test_parquet_membership_and_cross_split_conflict(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow missing')
        def write(split, text):
            row = {'extra_info': {'problem': text,'split': split}}
            pq.write_table(pa.Table.from_pylist([row]), self.root / (split + '.parquet'))
        write('train', 'train-only')
        write('val', 'val-only')
        self.assertEqual(c.load_splits(self.root)[c.source_hash('val-only')], 'val')
        write('val', 'train-only')
        with self.assertRaises(ValueError):
            c.load_splits(self.root)

    def test_full_synthetic_summary_and_fallback_verdict(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                return list(text)
        w = self.root / 'run'
        source_dir = w / 'eduardo_conversations'
        source_dir.mkdir(parents=True)
        marker = w / 'verl_ckpts/c-paper-one-step/latest_checkpointed_iteration.txt'
        marker.parent.mkdir(parents=True)
        marker.write_text('1')
        os.utime(marker, (1000, 1000))
        split_map = {}
        with (source_dir / 'conversations_1.jsonl').open('w') as sink:
            for split, n, mult, ts in [('val', 100, 1, 1020), ('train', 16, 16, 980)]:
                for i in range(n):
                    src = f'PRIVATE-{split}-{i}'
                    split_map[c.source_hash(src)] = split
                    for j in range(mult):
                        row = dict(problem=src, uuid=f'{src}-{j}', messages=[
                            {'role': 'assistant', 'content': 'PRIVATE-response'}],
                            gated=False, note='ok', ts=ts)
                        print(json.dumps(row), file=sink)
        for role in ['student', 'judge']:
            with (w / (role + '.log')).open('w') as sink:
                for n, ts in [(256*6, 978), (100*6, 1019)]:
                    for _ in range(n):
                        print(f'[C_HTTP {ts}.250000] "POST /v1/chat/completions HTTP/1.1" 200 OK', file=sink)
        (w / 'trainer.log').write_text('training/global_step:1 timing_s/step:90 timing_s/testing:20')
        raytmp = self.root / 'r123abc'
        logs = raytmp / 'session_1/logs'
        logs.mkdir(parents=True)
        (w / 'ray_tmp_path.txt').write_text(str(raytmp))
        for name in ['worker-fake-100.out','worker-fake-100.err']:
            (logs / name).write_text('line\n')
        with patch.object(c.audit, 'load_stripper', return_value=str.strip), \
             patch.object(c.audit, 'load_tokenizer', return_value=Tokenizer()), \
             patch.object(c, 'load_splits', return_value=split_map):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(c.report(self.root, w), 0)
            self.assertIn('| 裁判 | 2136 | 1536 | 600 | 0 | 0 |', out.getvalue())
            self.assertNotIn('PRIVATE', out.getvalue())
            (logs / 'worker-fake-100.err').write_text('[judge] gave up after 3 using fallback\n')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(c.report(self.root, w), 2)


if __name__ == '__main__':
    unittest.main()
