import contextlib
import io
import json
import unittest
from unittest.mock import patch

from stream_train import agnes_protocol_probe as probe
from stream_train import pipeline as p


CFG = {'namespace': 'synthetic', 'api_base': 'https://api.agnes-ai.cn/v1',
       'api_model': 'agnes-3.0-flash', 'student_base': 'http://127.0.0.1:1/v1',
       'student_model': 'student-8b'}


def success(body):
    prompt = body['messages'][0]['content']
    if prompt.startswith('Create one'):
        obj = {'construction_status': 'proposed', 'question': 'What is 2+4?',
               'answer': '6', 'solution': '2+4=6', 'notes': 'ok'}
    elif prompt.startswith('Independently solve'):
        obj = {'solvability': 'uncertain', 'independent_answer': '6',
               'independent_solution': '2+4=6', 'core_relation': 'related',
               'direct_repeat': 'no', 'explanation': 'uncertain'}
    elif prompt.startswith('Compare these'):
        obj = {'equivalence': 'uncertain', 'reason': 'uncertain'}
    elif prompt.startswith('Judge only'):
        obj = {'decision': 'unknown', 'final_answer_complete': 'uncertain',
               'student_final_answer': '6', 'reason': 'uncertain'}
    else:
        raise AssertionError('wrong request')
    return ({'model': CFG['api_model'], 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(obj)}}]}, [200])


class ProbeTest(unittest.TestCase):
    def test_all_four_actual_pipelines_and_semantic_unknown(self):
        calls = []
        def request(url, body, key, **kwargs):
            calls.append((url, body, kwargs))
            return success(body)
        out = io.StringIO()
        with patch.object(p, 'request_json', side_effect=request), contextlib.redirect_stdout(out):
            self.assertTrue(probe.probe(CFG, 'synthetic-secret', probe.KINDS, timeout=12))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c[2]['timeout'] == 12 for c in calls))
        self.assertEqual([json.loads(line)['status'] for line in out.getvalue().splitlines()],
                         ['ok'] * 4)
        self.assertNotIn('synthetic-secret', out.getvalue())

    def test_single_kind_and_non_json_circuit(self):
        def request(url, body, key, **kwargs):
            return ({'model': CFG['api_model'], 'choices': [{'finish_reason': 'stop',
                'message': {'content': 'not JSON; sensitive'}}]}, [200])
        out = io.StringIO()
        with patch.object(p, 'request_json', side_effect=request), contextlib.redirect_stdout(out):
            self.assertFalse(probe.probe(CFG, 'secret', ('review',)))
        result = json.loads(out.getvalue())
        self.assertEqual(result['reason'], 'unparseable_content')
        self.assertEqual(result['category'], 'protocol')
        self.assertNotIn('sensitive', out.getvalue())

    def test_schema_failure_of_actual_stage(self):
        def request(url, body, key, **kwargs):
            response, statuses = success(body)
            response['choices'][0]['message']['content'] = '{}'
            return response, statuses
        out = io.StringIO()
        with patch.object(p, 'request_json', side_effect=request), contextlib.redirect_stdout(out):
            self.assertFalse(probe.probe(CFG, 'secret', ('grade',)))
        self.assertEqual(json.loads(out.getvalue())['reason'], 'grade_schema')

    def test_http_429_is_sanitized(self):
        out = io.StringIO()
        with patch.object(p, 'request_json', side_effect=p.UnknownOutcome('http_429',
                 {'raw_response': 'sensitive'})), contextlib.redirect_stdout(out):
            self.assertFalse(probe.probe(CFG, 'secret', ('generate',)))
        self.assertEqual(json.loads(out.getvalue())['category'], 'congestion')
        self.assertNotIn('sensitive', out.getvalue())

    def test_unrecognized_exception_is_not_exposed(self):
        out = io.StringIO()
        with patch.object(p, 'request_json', side_effect=ValueError('secret raw')), \
                contextlib.redirect_stdout(out):
            self.assertFalse(probe.probe(CFG, 'secret', ('generate',)))
        self.assertNotIn('secret', out.getvalue())
        self.assertEqual(json.loads(out.getvalue())['category'], 'protocol')

    def test_bounded_restores_original_on_failure(self):
        original = p.request_json
        out = io.StringIO()
        with patch.object(p, 'api_message', side_effect=RuntimeError('bad')), \
                contextlib.redirect_stdout(out):
            self.assertFalse(probe.probe(CFG, 'secret', ('review',)))
        self.assertIs(p.request_json, original)


if __name__ == '__main__': unittest.main()
