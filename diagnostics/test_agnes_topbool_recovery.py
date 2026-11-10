import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base / 'agnes_eval_v2'))
import agnes_judge as judge
spec = importlib.util.spec_from_file_location('recovery', Path(__file__).with_name('agnes_topbool_recovery.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def snapshot(folder):
    return {str(x.relative_to(folder)): hashlib.sha256(x.read_bytes()).hexdigest()
            for x in folder.rglob('*') if x.is_file()}


def setup_source(root, responses):
    stage = 'EASY'
    source_id = 'SYNTHETIC_ID'
    key = hashlib.sha256(source_id.encode()).hexdigest()[:24]
    src = root / stage / 'sources' / key
    src.mkdir(parents=True)
    locks = root / 'locks'
    locks.mkdir(exist_ok=True)
    rows = [{'source_id': source_id, 'target': 'source', 'phase': 'pre', 'index': i,
             'problem': 'SYNTHETIC_QUESTION', 'reference_answer': 'SYNTHETIC_REFERENCE',
             'student_response': str(i)} for i in range(len(responses))]
    score = json.dumps(rows).encode()
    (src / 'SCORE_INPUT.json').write_bytes(score)
    outcomes = []
    for i, row in enumerate(rows):
        def caller(*_, i=i):
            return 200, {'model': judge.MODEL, 'choices': [{'finish_reason': 'stop',
                    'message': {'content': responses[i]}}]}
        outcomes.append(judge.grade_one(row, 'fake', root/'AGNES'/'receipts'/stage/key, locks,
                                        caller=caller, sleep=lambda _:None))
    out = root/'AGNES'/stage/'sources'/key
    out.mkdir(parents=True)
    report = {'source_id': source_id, 'input_sha256': hashlib.sha256(score).hexdigest(),
              'grader': 'agnes_full_response_v2', 'model': judge.MODEL, 'results': outcomes}
    (out/'RESULTS.json').write_text(json.dumps(report))
    return key, report


class TestRecovery(unittest.TestCase):
    def test_offline_safe_idempotent_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            root, dest = folder/'raw', folder/'overlay'
            root.mkdir()
            key, report = setup_source(root, ['true', 'false', 'null', '{"correct":true}', '"true"'])
            self.assertEqual([x['status'] for x in report['results']], ['unknown','unknown','unknown','correct','unknown'])
            before = snapshot(root)
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                first = r.run(root, dest)
                second = r.run(root, dest)
            self.assertEqual(snapshot(root), before)
            self.assertEqual(first['EASY']['候选条数'], 2)
            self.assertEqual(second['EASY']['已有相同来源'], 1)
            overlay = dest/'EASY'/key/'OVERLAY.json'
            data = json.loads(overlay.read_text())
            self.assertEqual([v['decision'] for v in data['recoveries']], [True, False])
            self.assertNotIn('SYNTHETIC_QUESTION', overlay.read_text())
            old_overlay = overlay.read_bytes()
            overlay.write_text('{}')
            third = r.run(root, dest)
            self.assertEqual(third['EASY']['来源异常'], 1)
            self.assertEqual(overlay.read_text(), '{}')
            self.assertNotEqual(overlay.read_bytes(), old_overlay)
            self.assertEqual(snapshot(root), before)

    def test_input_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            root, dest = folder/'raw', folder/'overlay'
            root.mkdir()
            key, _ = setup_source(root, ['true'])
            score = root/'EASY'/'sources'/key/'SCORE_INPUT.json'
            score.write_bytes(score.read_bytes() + b' ')
            result = r.run(root, dest)
            self.assertEqual(result['EASY']['来源异常'], 1)
            self.assertFalse((dest/'EASY'/key/'OVERLAY.json').exists())

    def test_changed_intent_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            root, dest = folder/'raw', folder/'overlay'
            root.mkdir()
            key, _ = setup_source(root, ['true'])
            intents = list((root/'AGNES'/'receipts'/'EASY'/key).glob('*/ATTEMPT-0001-INTENT.json'))
            intent = json.loads(intents[0].read_text())
            intent['request_sha256'] = '0' * 64
            intents[0].write_text(json.dumps(intent))
            self.assertEqual(r.run(root, dest)['EASY']['来源异常'], 1)
            self.assertFalse((dest/'EASY'/key/'OVERLAY.json').exists())

    def test_attempt_history(self):
        cases = ('valid', 'mismatched_numbers', 'gap', 'old_intent_hash',
                 'old_response_number', 'old_not_retryable', 'old_not_transient',
                 'final_retryable', 'final_wrong_reason')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                root, dest = folder/'raw', folder/'overlay'
                root.mkdir()
                key, report = setup_source(root, ['true'])
                receipts = root/'AGNES'/'receipts'/'EASY'/key
                request = next(p for p in receipts.iterdir() if p.is_dir())
                first_intent_path = request/'ATTEMPT-0001-INTENT.json'
                first_response_path = request/'ATTEMPT-0001-RESPONSE.json'
                intent = json.loads(first_intent_path.read_text())
                last_response = json.loads(first_response_path.read_text())
                if case == 'mismatched_numbers':
                    first_response_path.rename(request/'ATTEMPT-0002-RESPONSE.json')
                elif case == 'gap':
                    first_intent_path.rename(request/'ATTEMPT-0002-INTENT.json')
                    first_response_path.rename(request/'ATTEMPT-0002-RESPONSE.json')
                else:
                    retry = {'attempt': 1, 'http_status': 429, 'retryable': True,
                             'raw_response': None}
                    if case == 'old_intent_hash':
                        intent['request_sha256'] = '0' * 64
                        first_intent_path.write_text(json.dumps(intent))
                    if case == 'old_response_number':
                        retry['attempt'] = 2
                    if case == 'old_not_retryable':
                        retry['retryable'] = False
                    if case == 'old_not_transient':
                        retry['http_status'] = 200
                    first_response_path.write_text(json.dumps(retry))
                    final_intent = {'attempt': 2, 'request_sha256': report['results'][0]['request_sha256']}
                    (request/'ATTEMPT-0002-INTENT.json').write_text(json.dumps(final_intent))
                    last_response['attempt'] = 2
                    if case == 'final_retryable':
                        last_response['retryable'] = True
                    (request/'ATTEMPT-0002-RESPONSE.json').write_text(json.dumps(last_response))
                    if case == 'final_wrong_reason':
                        report['results'][0]['reason'] = 'retry_limit'
                        (request/'RESULT.json').write_text(json.dumps(report['results'][0]))
                        (root/'AGNES'/'EASY'/'sources'/key/'RESULTS.json').write_text(json.dumps(report))
                outcome = r.run(root, dest)['EASY']
                if case == 'valid':
                    self.assertEqual(outcome['候选条数'], 1)
                else:
                    self.assertEqual(outcome['来源异常'], 1)
                    self.assertFalse((dest/'EASY'/key/'OVERLAY.json').exists())

    def test_verdict_cases(self):
        def response(content, finish='stop', model=r.MODEL):
            return {'http_status': 200, 'raw_response': {'model': model,
                    'choices': [{'finish_reason': finish, 'message': {'content': content}}]}}
        self.assertIs(r.top_boolean(response('true')), True)
        self.assertIs(r.top_boolean(response('```json\nfalse\n```')), False)
        for value in ('null', '"true"', '{"correct":true}', 'yes', 'True'):
            self.assertIsNone(r.top_boolean(response(value)))
        self.assertIsNone(r.top_boolean(response('true', finish='length')))
        self.assertIsNone(r.top_boolean(response('true', model='other')))


if __name__ == '__main__':
    unittest.main()
