"""CPU-only mock checks; no real model, API, network or user data."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import offline_eval
import dialogue
import protocol
import score_local


class FakeCompletion:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        choice = SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=r'\boxed{4}'))
        return SimpleNamespace(choices=[choice], model='fake-student')


class TestOffline(unittest.TestCase):
    def test_refuses_non_loopback(self):
        self.assertEqual(offline_eval.local_url('http://127.0.0.1:8001/v1'), 'http://127.0.0.1:8001/v1')
        for endpoint in ('https://example.org/v1', 'http://10.0.0.1:8001/v1', 'http://user@localhost/v1'):
            with self.assertRaises(ValueError):
                offline_eval.local_url(endpoint)

    def test_ungradeable_is_unknown(self):
        for response in ('The answer is 4.', r'\boxed{4} but I change my mind'):
            self.assertEqual(score_local.score({'student_response': response, 'reference_answer': '4'})[0], 'unknown')
        self.assertEqual(score_local.score({'student_response': r'\boxed{4}', 'reference_answer': '4, 5'})[0], 'unknown')

    def test_local_simple_numeric_grader(self):
        self.assertEqual(score_local.score({'student_response': r'Therefore \boxed{4}.', 'reference_answer': '4'})[0], 'correct')
        self.assertEqual(score_local.score({'student_response': r'Therefore \boxed{5}.', 'reference_answer': '4'})[0], 'incorrect')
        self.assertEqual(score_local.score({'student_response': r'\boxed{\frac{1}{2}}', 'reference_answer': '0.5'})[0], 'correct')

    def test_sample_journal_and_teacher_mask(self):
        async def fake_dialogue(source, row, args):
            return {'source_id': source['source_id'], 'stage': 'completed',
                    'reward_messages': [{'role': 'user', 'content': 'I tried.'},
                                        {'role': 'assistant', 'content': 'SECRET TEACHER HINT'}]}

        sample = {'source_id': 'synthetic-only', 'problem': 'What is 2+2?', 'answer': '4',
                  'transfer_problem': 'What is 3+3?', 'transfer_answer': '6'}
        completion = FakeCompletion()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RESULTS_DIR': temp}), \
                patch.object(offline_eval, 'make_dialogue', fake_dialogue):
            asyncio.run(offline_eval.run_one(sample, '/fake', 'http://127.0.0.1:8001/v1',
                                             'http://127.0.0.1:8002/v1', client))
            self.assertEqual(len(completion.calls), 96)
            folder = offline_eval.source_dir(sample)
            self.assertEqual(len(json.loads((folder / 'SCORE_INPUT.json').read_text())), 96)
            self.assertEqual(json.loads((folder / 'RAW_COMPLETE.json').read_text())['grading_status'], 'ungraded')
            for call in completion.calls:
                self.assertNotIn('SECRET TEACHER HINT', json.dumps(call['messages']))
            # Completed sample receipts are not generated again.
            asyncio.run(offline_eval.run_one(sample, '/fake', 'http://127.0.0.1:8001/v1',
                                             'http://127.0.0.1:8002/v1', client))
            self.assertEqual(len(completion.calls), 96)

    def test_actual_dialogue_with_in_memory_clients(self):
        class FakeTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))
            def apply_chat_template(self, *args, **kwargs):
                return '<think>'
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.calls = 0
            async def create(self, **kwargs):
                self.calls += 1
                if kwargs['model'] == 'Eduardo-27b':
                    text = 'private</think>Try the relation.'
                    if self.calls >= 4:
                        text += '<end_of_conversation>'
                else:
                    text = 'I tried.'
                choice = SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=text))
                return SimpleNamespace(choices=[choice], model=kwargs['model'], id='fake')
        t, s = Client(), Client()
        source = {'source_id': 'synthetic-dialogue', 'problem': '2+2?', 'answer': '4'}
        row = protocol._row('2+2?', '4', 0.0, '3+3?', '6', 0.0, 'test')
        args = SimpleNamespace(model='Eduardo-27b', teacher_tokenizer_path='/fake',
            teacher_api_key_env='LOCAL_ONLY_KEY', teacher_api_base='http://127.0.0.1:8001/v1',
            student_api_key_env='LOCAL_ONLY_KEY', student_api_base='http://127.0.0.1:8002/v1',
            student_model='Llama-3.1-8B-Instruct', student_concurrency=1, teacher_temperature=0.6,
            teacher_top_p=0.95, student_temperature=0.6, max_tokens_per_turn=4192, student_max_tokens=1024,
            max_conversation_tokens=16384, max_teacher_turns=11, min_teacher_turns_before_eoc=3,
            enable_thinking=True, num_samples_per_problem=1, assume_thinking='auto', seed=42, num_retries=0)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RESULTS_DIR': temp, 'LOCAL_ONLY_KEY': 'dummy'}), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=FakeTokenizer()), \
             patch.object(dialogue.openai, 'AsyncClient', return_value=t), \
             patch.object(protocol, 'get_async_openai_client', return_value=s):
            result = asyncio.run(dialogue.make_dialogue(source, row, args))
            self.assertEqual(result['stage'], 'completed')
            self.assertEqual(result['num_teacher_turns'], 4)
            self.assertNotIn('private', json.dumps(result['judge_messages']))


if __name__ == '__main__':
    unittest.main()
