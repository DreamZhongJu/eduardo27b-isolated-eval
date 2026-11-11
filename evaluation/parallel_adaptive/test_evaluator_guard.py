import asyncio
import json
import tempfile
import unittest
import sys
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluator_guard as guard
from student_router import StudentRouter


class FakeClient:
    calls = []
    fail = False

    def __init__(self, **settings):
        self.url = settings['base_url']
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **params):
        self.calls.append((self.url, dict(params)))
        await asyncio.sleep(0)
        if self.fail:
            raise TimeoutError('synthetic')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='synthetic'), finish_reason='stop')], model='student')

    async def close(self):
        pass


def router_factory(endpoints, **kwargs):
    return StudentRouter(endpoints, client_factory=FakeClient, **kwargs)


class GuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        FakeClient.calls = []
        FakeClient.fail = False
        self.done = set()
        self.attempts = []
        self.seen_dialogue = []
        self.evaluator = SimpleNamespace(local_url=lambda v: v, run_one=self.run_one,
                                         source_dir=self.source_dir, put_once=self.put_once)

    def tearDown(self):
        self.temp.cleanup()

    def source_dir(self, row):
        folder = self.root / row['source_id']
        folder.mkdir(exist_ok=True)
        return folder

    def put_once(self, path, data):
        path.write_text(json.dumps(data))

    def inspect(self, stage, row):
        if row['source_id'] == 'old':
            return 'complete'
        if row['source_id'] == 'partial':
            return 'partial'
        return 'complete' if row['source_id'] in self.done else 'new'

    async def run_one(self, row, teacher, teacher_url, student_url, client):
        self.attempts.append(row['source_id'])
        self.seen_dialogue.append(student_url)
        # Model the frozen evaluator's Semaphore(4) and unchanged sample seeds.
        sem = asyncio.Semaphore(4)
        async def sample(i):
            async with sem:
                return await client.chat.completions.create(model='student', seed=i,
                    temperature=.6, max_tokens=1024, messages=[{'role': 'user', 'content': 'synthetic'}])
        await asyncio.gather(*(sample(i) for i in range(16)))
        self.done.add(row['source_id'])

    async def run_guard(self, rows, **kwargs):
        with patch.object(guard, 'inspect_source', side_effect=self.inspect):
            return await guard.run(rows, self.root, self.evaluator, Path('/teacher'),
                                   'http://127.0.0.1:18001/v1', 'http://127.0.0.1:18004/v1',
                                   'http://127.0.0.1:18005/v1', router_factory=router_factory, **kwargs)

    async def test_dual_sampling_preserves_identity_and_skips_existing(self):
        counts = await self.run_guard([{'source_id': x} for x in ('old', 'partial', 'fresh')])
        self.assertEqual((counts['complete'], counts['partial'], counts['new_complete']), (1, 1, 1))
        self.assertEqual(self.attempts, ['fresh'])
        self.assertEqual(self.seen_dialogue, ['http://127.0.0.1:18004/v1'])
        self.assertEqual({url for url, _ in FakeClient.calls},
                         {'http://127.0.0.1:18004/v1', 'http://127.0.0.1:18005/v1'})
        self.assertEqual(sorted(args['seed'] for _, args in FakeClient.calls), list(range(16)))
        self.assertTrue(all(a['temperature'] == .6 and a['max_tokens'] == 1024 for _, a in FakeClient.calls))

    async def test_uncertain_service_failure_holds_one_source_and_stops(self):
        FakeClient.fail = True
        counts = await self.run_guard([{'source_id': 'first'}, {'source_id': 'second'}])
        self.assertEqual(counts['failed'], 1)
        self.assertEqual(self.attempts, ['first'])
        self.assertTrue((self.root/'first'/'SOURCE_HELD.json').exists())

    async def test_drain_before_next_source(self):
        counts = await self.run_guard([{'source_id': 'first'}], should_drain=lambda: True)
        self.assertEqual(counts['new_complete'], 0)
        self.assertEqual(self.attempts, [])

    async def test_pilot_attempts_only_one_source(self):
        counts = await self.run_guard([{'source_id': 'a'}, {'source_id': 'b'}], max_new=1)
        self.assertEqual(counts['new_complete'], 1)
        self.assertEqual(self.attempts, ['a'])

    async def test_frozen_evaluator_completes_synthetic_source_on_two_students(self):
        try:
            import openai  # noqa: F401
        except ImportError:
            self.skipTest('SDK absent')
        app = str(Path(__file__).resolve().parents[1] / 'app')
        sys.path.insert(0, app)
        try:
            import offline_eval
        finally:
            sys.path.remove(app)
        row = {'source_id': 'synthetic-only', 'problem': 'What is 2+2?', 'answer': '4',
               'transfer_problem': 'What is 3+3?', 'transfer_answer': '6', 'transfer_target': 'paper'}
        async def fake_dialogue(source, selected, args):
            self.assertEqual(args.student_api_base, 'http://127.0.0.1:18004/v1')
            return {'source_id': source['source_id'], 'stage': 'completed',
                    'reward_messages': [{'role': 'user', 'content': 'I tried.'},
                                        {'role': 'assistant', 'content': 'SYNTHETIC TEACHER HINT'}]}
        with patch.dict(os.environ, {'RESULTS_DIR': str(self.root)}), \
                patch.object(offline_eval, 'make_dialogue', fake_dialogue):
            counts = await guard.run([row], self.root, offline_eval, Path('/teacher'),
                                     'http://127.0.0.1:18001/v1', 'http://127.0.0.1:18004/v1',
                                     'http://127.0.0.1:18005/v1', router_factory=router_factory)
            self.assertEqual(counts['new_complete'], 1)
            folder = offline_eval.source_dir(row)
            entries = json.loads((folder / 'SCORE_INPUT.json').read_text())
            self.assertEqual(len(entries), 96)
            self.assertEqual(len(FakeClient.calls), 96)
            self.assertEqual({url for url, _ in FakeClient.calls},
                             {'http://127.0.0.1:18004/v1', 'http://127.0.0.1:18005/v1'})
            self.assertEqual(len(list((folder/'samples').glob('*/*/*/RESPONSE.json'))), 96)
            for _, params in FakeClient.calls:
                self.assertNotIn('SYNTHETIC TEACHER HINT', json.dumps(params['messages']))


if __name__ == '__main__':
    unittest.main()
