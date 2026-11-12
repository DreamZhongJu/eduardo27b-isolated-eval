import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rejudge', HERE/'rejudge.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
spec = importlib.util.spec_from_file_location('judge', HERE.parent/'agnes_eval_v3'/'agnes_judge.py')
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(j.canonical(obj) + b'\n')


def response(text):
    return {'model': j.MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}


class TestRejudge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root, self.out, self.overlay = base/'root', base/'out', base/'oldoverlay'
        (self.root/'AGNES'/'locks').mkdir(parents=True)
        self.overlay.mkdir()
        self.reference = None
        self.recovery = SimpleNamespace(source_overlay=self.reference_overlay)
        self.id = 'synthetic:1'
        self.key = hashlib.sha256(self.id.encode()).hexdigest()[:24]
        self.rows = []
        meta = {'candidate_id':'synthetic-candidate', 'difficulty':'easy', 'position_by_student_correct':2,
                'historical_student_correct':10, 'historical_student_n_scored':16,
                'historical_group_score_tie':False}
        for target in ('source', 'ours_easy'):
            for phase in ('pre', 'control', 'post'):
                for i in range(16):
                    row = {'source_id':self.id, 'target':target, 'phase':phase, 'index':i,
                           'problem':'2+3?' if target == 'source' else '3+3?',
                           'reference_answer':'5' if target == 'source' else '6',
                           'student_response':'5', 'finish_reason':'stop'}
                    if target != 'source': row.update(meta)
                    self.rows.append(row)
        self.items = [{'identity': j.identity(row), 'request_sha256': j.digest(j.payload(row)),
                       'grader': 'agnes_full_response_v2', 'model': j.MODEL,
                       'status': 'unknown' if 1 <= i <= 3 else 'correct'} for i, row in enumerate(self.rows)]
        self.score = self.root/'EASY'/'sources'/self.key/'SCORE_INPUT.json'
        self.result = self.root/'AGNES'/'EASY'/'sources'/self.key/'RESULTS.json'
        save(self.score, self.rows)
        save(self.result, {'source_id': self.id, 'input_sha256': r.sha(self.score.read_bytes()),
                           'grader': 'agnes_full_response_v2', 'model': j.MODEL, 'results': self.items})
        self.source = self.score.parent
        selected = {'source_id':self.id, 'problem':'2+3?', 'answer':'5', 'transfer_problem':'3+3?',
                    'transfer_answer':'6', 'transfer_target':'ours_easy',
                    'transfer_candidate_id':meta['candidate_id'], 'transfer_position':meta['position_by_student_correct'],
                    'student_correct':meta['historical_student_correct'],
                    'student_n_scored':meta['historical_student_n_scored'],
                    'score_tie':meta['historical_group_score_tie']}
        source_sha = j.digest(selected)
        save(self.source/'SOURCE.json', {'source_id':self.id, 'sha256':source_sha})
        save(self.source/'RAW_COMPLETE.json', {'source_id':self.id, 'samples':96,
                                               'grading_status':'ungraded', 'source_sha256':source_sha})
        for row in self.rows:
            folder = self.source/'samples'/row['target']/row['phase']/f"{row['index']:02d}"
            identity = {**j.identity(row), 'request_sha256':j.digest(j.identity(row))}
            save(folder/'INTENT.json', identity)
            save(folder/'RESPONSE.json', {'identity':identity, 'text':row['student_response'], 'finish_reason':'stop'})
        self.calls = 0

    def reference_overlay(self, root, stage, key):
        if self.reference is not None:
            return json.loads(json.dumps(self.reference))
        return {'schema': 'agnes-topbool-recovery-v1', 'stage': stage,
                'input_sha256': r.sha(self.score.read_bytes()),
                'source_result_sha256': r.sha(self.result.read_bytes()), 'recoveries': []}

    def tearDown(self):
        self.tmp.cleanup()

    def overlay_one(self):
        item = self.items[1]
        k = j.digest({'identity': item['identity']})
        rec = self.root/'AGNES'/'receipts'/'EASY'/self.key/k/'ATTEMPT-0001-RESPONSE.json'
        save(rec, {'http_status': 200, 'raw_response': response('true')})
        save(self.overlay/'EASY'/self.key/'OVERLAY.json',
             {'schema':'agnes-topbool-recovery-v1', 'stage':'EASY',
              'input_sha256':r.sha(self.score.read_bytes()), 'source_result_sha256':r.sha(self.result.read_bytes()),
              'recoveries':[{'identity': item['identity'], 'request_sha256':item['request_sha256'],
                             'response_sha256':r.sha(rec.read_bytes()), 'original_status':'unknown', 'decision': True}]})
        self.reference = json.loads((self.overlay/'EASY'/self.key/'OVERLAY.json').read_text())

    def test_overlay_bounded_null_resume_and_original_unchanged(self):
        self.overlay_one()
        before = self.result.read_bytes()
        def caller(body, key):
            self.calls += 1
            # null is explicit and ends a round; all rounds persist unknown
            return 200, response('null')
        first = r.run(j, self.root, self.out, self.overlay, 'synthetic', 2, caller, max_rounds=3, recovery=self.recovery)
        self.assertEqual(first['preserved'], 93)
        self.assertEqual(first['overlay_correct'], 1)
        self.assertEqual(first['rejudged_unknown'], 2)
        self.assertEqual(self.calls, 6)
        second = r.run(j, self.root, self.out, self.overlay, 'synthetic', 2, caller, max_rounds=3, recovery=self.recovery)
        self.assertEqual(second['already_correct'], 1)
        self.assertEqual(second['already_unknown'], 2)
        self.assertEqual(self.calls, 6)
        self.assertEqual(self.result.read_bytes(), before)

    def test_null_then_boolean_and_mismatch(self):
        def caller(body, key):
            self.calls += 1
            return 200, response('null' if self.calls == 1 else 'false')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller, max_rounds=2, recovery=self.recovery)
        self.assertEqual(counts['rejudged_incorrect'], 3)
        self.assertEqual(self.calls, 4)
        self.items[1]['request_sha256'] = 'bad'
        save(self.result, {'source_id': self.id, 'input_sha256': r.sha(self.score.read_bytes()),
                           'grader': 'agnes_full_response_v2', 'model': j.MODEL, 'results': self.items})
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller, max_rounds=2, recovery=self.recovery)
        self.assertEqual(counts['source_error'], 1)
        self.assertEqual(self.calls, 4)

    def test_auth_failure_pauses_and_later_resumes(self):
        def denied(body, key):
            self.calls += 1
            return 401, None
        counts = r.run(j, self.root, self.out, self.overlay, 'bad', 1, denied, recovery=self.recovery)
        self.assertEqual(counts['authentication_failed'], 1)
        self.assertEqual(self.calls, 1)
        self.assertFalse(list((self.out/'decisions').rglob('DECISION.json')))
        def allowed(body, key):
            self.calls += 1
            return 200, response('true')
        counts = r.run(j, self.root, self.out, self.overlay, 'valid', 1, allowed, recovery=self.recovery)
        self.assertEqual(counts['rejudged_correct'], 3)
        self.assertEqual(self.calls, 4)

    def test_modified_overlay_refused_without_api_calls(self):
        self.overlay_one()
        path = self.overlay/'EASY'/self.key/'OVERLAY.json'
        data = json.loads(path.read_text())
        data['recoveries'][0]['decision'] = False
        save(path, data)
        def caller(body, key):
            self.calls += 1
            return 200, response('true')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller, recovery=self.recovery)
        self.assertEqual(counts['source_error'], 1)
        self.assertEqual(self.calls, 0)

    def test_missing_recoverable_overlay_never_calls_api(self):
        self.overlay_one()
        (self.overlay/'EASY'/self.key/'OVERLAY.json').unlink()
        def caller(body, key):
            self.calls += 1
            return 200, response('true')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                       recovery=self.recovery)
        self.assertEqual(counts['source_error'], 1)
        self.assertEqual(self.calls, 0)

    def test_trial_limits_api_items_but_not_overlay_and_can_continue(self):
        self.overlay_one()
        def caller(body, key):
            self.calls += 1
            return 200, response('true')
        stages = {}
        first = r.run(j, self.root, self.out, self.overlay, 'synthetic', 2, caller,
                      max_items=1, recovery=self.recovery, per_stage=stages)
        self.assertEqual(first['overlay_correct'], 1)
        self.assertEqual(first['rejudged_correct'], 1)
        self.assertEqual(first['limit_deferred'], 1)
        self.assertEqual(first['limit_reached'], 1)
        self.assertNotIn('MEDIUM', stages)
        self.assertNotIn('HARD', stages)
        self.assertEqual(first['api_samples'], 1)
        self.assertEqual(first['api_requests'], 1)
        self.assertEqual(self.calls, 1)
        second = r.run(j, self.root, self.out, self.overlay, 'synthetic', 2, caller,
                       max_items=1, recovery=self.recovery)
        self.assertEqual(second['rejudged_correct'], 1)
        self.assertEqual(self.calls, 2)

    def test_semantic_null_gets_one_default_round(self):
        def caller(body, key):
            self.calls += 1
            return 200, response('null')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                       recovery=self.recovery)
        self.assertEqual(counts['rejudged_unknown'], 3)
        self.assertEqual(self.calls, 3)

    def test_retry_limit_stops_batch_and_next_run_resumes(self):
        def outage(body, key):
            self.calls += 1
            return 429, None
        original = j.grade_one
        def no_sleep(*args, **kwargs):
            return original(*args, sleep=lambda _: None, **kwargs)
        class NoWaitWindow:
            def acquire(self): pass
            def release(self, *args, **kwargs): pass
        with patch.object(j, 'grade_one', no_sleep), patch.object(j, 'AdaptiveWindow', return_value=NoWaitWindow()):
            counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, outage,
                           max_rounds=3, recovery=self.recovery)
        self.assertEqual(counts['service_failed'], 1)
        self.assertEqual(self.calls, 4)
        self.assertEqual(counts['api_samples'], 1)
        self.assertEqual(counts['api_requests'], 4)
        self.assertFalse(list((self.out/'decisions').rglob('DECISION.json')))
        def available(body, key):
            self.calls += 1
            return 200, response('true')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, available,
                       recovery=self.recovery)
        self.assertEqual(counts['rejudged_correct'], 3)
        self.assertEqual(self.calls, 7)

    def test_wrong_model_or_nonstop_is_technical_not_semantic(self):
        wrong_model = response('true')
        wrong_model['model'] = 'unexpected'
        wrong_finish = response('null')
        wrong_finish['choices'][0]['finish_reason'] = 'other'
        for raw in (wrong_model, wrong_finish):
            self.assertEqual(r.classify_response(j, {'http_status': 200, 'raw_response': raw}), 'technical')
        self.assertEqual(r.classify_response(j, {'http_status': 200, 'raw_response': response('null')}),
                         'semantic_null')
        def caller(body, key):
            self.calls += 1
            return 200, wrong_model
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                       recovery=self.recovery)
        self.assertEqual(counts['service_failed'], 1)
        self.assertEqual(self.calls, 1)
        self.assertFalse(list((self.out/'decisions').rglob('DECISION.json')))

    def test_missing_stage_and_chinese_aggregate_table(self):
        def caller(body, key):
            return 200, response('true')
        stages = {}
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                       recovery=self.recovery, per_stage=stages)
        self.assertEqual(counts['missing_stage'], 3)
        table = r.format_report(stages, '出现异常')
        self.assertIn('中等', table)
        self.assertIn('档位已检查', table)
        self.assertIn('缺少目录', table)
        self.assertNotIn(self.id, table)
        self.assertNotIn('2+3?', table)

    def test_partially_scanned_easy_and_no_pilot_column(self):
        text = r.format_report({'PILOT': {'visited': 1, 'sources_scanned': 1}}, '已到本轮上限')
        self.assertIn('部分', text)
        self.assertIn('1/4', text)
        self.assertIn('简单', text)
        self.assertNotIn('| 试点 ', text)

    def test_completed_receipt_recovery_does_not_spend_trial_quota(self):
        row = self.rows[1]
        request_key = j.digest({'identity': j.identity(row)})
        epoch = self.out/'receipts'/'EASY'/self.key/'round-1'/'auth-0000'/request_key
        request_hash = j.digest(j.payload(row))
        save(epoch/'BINDING.json', {'identity': j.identity(row), 'request_sha256': request_hash,
                                   'model': j.MODEL, 'grader': 'agnes_full_response_v2'})
        save(epoch/'ATTEMPT-0001-INTENT.json', {'attempt': 1, 'request_sha256': request_hash})
        save(epoch/'ATTEMPT-0001-RESPONSE.json', {'attempt': 1, 'http_status': 200,
                                                  'retryable': False, 'raw_response': response('true')})
        def caller(body, key):
            self.calls += 1
            return 200, response('true')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                       max_items=1, recovery=self.recovery)
        self.assertEqual(counts['rejudged_correct'], 2)
        self.assertEqual(counts['limit_deferred'], 1)
        self.assertEqual(counts['api_samples'], 1)
        self.assertEqual(counts['api_requests'], 1)
        self.assertEqual(self.calls, 1)

    def test_five_sample_trial_never_sends_sixth(self):
        for item in self.items[:7]:
            item['status'] = 'unknown'
        save(self.result, {'source_id':self.id, 'input_sha256':r.sha(self.score.read_bytes()),
                           'grader':'agnes_full_response_v2', 'model':j.MODEL, 'results':self.items})
        def caller(body, key):
            self.calls += 1
            return 200, response('true')
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 5, caller,
                       max_items=5, recovery=self.recovery)
        self.assertEqual(counts['rejudged_correct'], 5)
        self.assertEqual(counts['limit_deferred'], 2)
        self.assertEqual(counts['api_samples'], 5)
        self.assertEqual(counts['api_requests'], 5)
        self.assertEqual(self.calls, 5)

    def test_fifth_sample_service_failure_precedes_batch_limit(self):
        for item in self.items[:5]:
            item['status'] = 'unknown'
        save(self.result, {'source_id':self.id, 'input_sha256':r.sha(self.score.read_bytes()),
                           'grader':'agnes_full_response_v2', 'model':j.MODEL, 'results':self.items})
        def caller(body, key):
            self.calls += 1
            raw = response('true')
            if self.calls == 5:
                raw['model'] = 'unexpected'
            return 200, raw
        stages = {}
        counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                       max_items=5, recovery=self.recovery, per_stage=stages)
        self.assertEqual(counts['service_failed'], 1)
        self.assertEqual(counts['api_samples'], 5)
        self.assertEqual(counts['api_requests'], 5)
        self.assertEqual(counts['rejudged_correct'], 4)
        self.assertNotIn('limit_reached', counts)

    def test_invalid_source_marker_or_missing_sample_never_calls_api(self):
        marker = self.source/'RAW_COMPLETE.json'
        response_file = self.source/'samples'/'source'/'pre'/'00'/'RESPONSE.json'
        intent_file = response_file.parent/'INTENT.json'
        originals = {p:p.read_bytes() for p in (marker, response_file, intent_file)}
        def caller(body, key):
            self.calls += 1
            return 200, response('true')
        for case in ('corrupt_marker', 'missing_sample', 'wrong_sample', 'missing_intent', 'mismatched_intent', 'invalid_hash'):
            with self.subTest(case=case):
                for path, blob in originals.items(): path.write_bytes(blob)
                if case == 'corrupt_marker':
                    value = json.loads(marker.read_bytes())
                    value['samples'] = 95
                    save(marker, value)
                elif case == 'missing_sample':
                    response_file.unlink()
                elif case == 'wrong_sample':
                    value = json.loads(response_file.read_bytes())
                    value['text'] = 'changed'
                    save(response_file, value)
                elif case == 'missing_intent':
                    intent_file.unlink()
                elif case == 'mismatched_intent':
                    value = json.loads(intent_file.read_bytes())
                    value['request_sha256'] = 'a'*64
                    save(intent_file, value)
                else:
                    value = json.loads(intent_file.read_bytes())
                    value['request_sha256'] = 'not-a-sha256'
                    save(intent_file, value)
                    response_value = json.loads(response_file.read_bytes())
                    response_value['identity'] = value
                    save(response_file, response_value)
                counts = r.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                               recovery=self.recovery)
                self.assertEqual(counts['source_error'], 1)
                self.assertEqual(self.calls, 0)

    def test_cli_exit_code_distinguishes_error_from_limited_pilot(self):
        secret = self.root/'key'
        secret.write_text('sensitive-synthetic-key')
        command = r'''import importlib.util, sys
spec = importlib.util.spec_from_file_location('local_rejudge', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.load_judge = lambda *a: object()
m.load_recovery = lambda *a: object()
mode = sys.argv[2]
def fake_run(*args, **kwargs):
    if kwargs['max_items'] != 5:
        raise RuntimeError('unsafe default sample limit')
    kwargs['per_stage']['EASY'] = {'visited': 1}
    return {'source_error': 1} if mode == 'error' else {'limit_deferred': 1}
m.run = fake_run
sys.argv = ['rejudge', '--root', sys.argv[3], '--output', sys.argv[4], '--overlay', sys.argv[5],
            '--judge', '/synthetic/judge.py', '--judge-sha256', m.APPROVED_JUDGE,
            '--recovery', '/synthetic/recovery.py', '--recovery-sha256', m.APPROVED_RECOVERY,
            '--key-file', sys.argv[6]]
sys.exit(m.main())
'''
        for mode, expected in (('error', 2), ('limited', 0)):
            child = subprocess.run([sys.executable, '-c', command, str(HERE/'rejudge.py'), mode,
                                    str(self.root), str(self.out), str(self.overlay), str(secret)],
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(child.returncode, expected, child.stderr)
            self.assertIn('本轮状态', child.stdout)
            self.assertIn('不会在同一输出目录自动重判', child.stdout)
            self.assertNotIn('sensitive-synthetic-key', child.stdout + child.stderr)


if __name__ == '__main__':
    unittest.main()
