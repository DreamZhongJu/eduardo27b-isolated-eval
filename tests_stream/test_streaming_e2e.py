import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from stream_train import __main__ as cli
from stream_train import pipeline as p
from stream_train.store import Store


class StreamingEndToEndTest(unittest.TestCase):
    def test_fair_workers_complete_one_candidate_without_real_services(self):
        ram = Path('/dev/shm')
        temp_dir = str(ram) if ram.is_dir() and os.access(ram, os.W_OK) else None
        with tempfile.TemporaryDirectory(dir=temp_dir) as d:
            store = Store(Path(d) / 'test.sqlite')
            source = {'source_id': 'synthetic:1', 'source_row_index': 1,
                      'source_split': 'train', 'problem': 'Synthetic source',
                      'answer': '5', 'direction_band': 'mixed'}
            store.enqueue(p.job_id('generate', [source['source_id'], 1]), 'generate',
                          {'source': source, 'proposal': 1, 'feedback': []})
            generated = {'construction_status': 'proposed', 'question': 'Synthetic question',
                         'answer': '5', 'solution': 'Synthetic solution', 'notes': ''}
            reviewed = {'solvability': 'solvable', 'independent_answer': '5',
                        'independent_solution': 'Synthetic solution', 'core_relation': 'related',
                        'direct_repeat': 'no', 'explanation': ''}
            graded = {'final_answer_complete': 'yes', 'decision': 'correct',
                      'student_final_answer': '5', 'reason': ''}
            cfg = {'namespace': 'test', 'api_base': 'https://example.invalid/v1',
                   'api_model': 'model', 'student_base': 'http://127.0.0.1:8000/v1',
                   'student_model': 'student'}

            def fake_api(base, key, model, prompt, **kwargs):
                if prompt.startswith('Create one'): return generated, {}
                if prompt.startswith('Independently'): return reviewed, {}
                return graded, {}

            def fake_student(url, body, key, **kwargs):
                return {'model': 'student', 'choices': [
                    {'message': {'content': '\\boxed{5}'}, 'finish_reason': 'stop'}]}, [200]

            window = cli.AdaptiveAPIWindow()
            with patch.object(p, 'api_message', side_effect=fake_api), \
                 patch.object(p, 'request_json', side_effect=fake_student):
                for _ in range(20):
                    self.assertTrue(cli.loop(store, cfg, 'test-key',
                        ['generate', 'review', 'equivalence', 'grade'], 5, True,
                        api_window=window))
                    self.assertTrue(cli.loop(store, cfg, None, ['student'], 32, True))
                    counts = {(x['kind'], x['state']): x['count'] for x in store.counts()}
                    if counts.get(('grade', 'done'), 0) == 16:
                        break
            self.assertEqual(counts.get(('student', 'done')), 16)
            self.assertEqual(counts.get(('grade', 'done')), 16)
            self.assertFalse(any(state in ('started', 'unknown') for _, state in counts))
            with store.connect() as db:
                ids = db.execute("SELECT id FROM jobs WHERE kind='student'").fetchall()
            self.assertEqual(len({row[0] for row in ids}), 16)


if __name__ == '__main__':
    unittest.main()
