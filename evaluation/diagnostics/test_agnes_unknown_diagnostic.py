"""Offline integration tests against the real local Agnes grading code."""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / 'agnes_eval_v2'
sys.path.insert(0, str(APP))
import agnes_judge as grader

spec = importlib.util.spec_from_file_location('diagnostic_v2', Path(__file__).with_name('agnes_unknown_diagnostic.py'))
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def response(content, finish='stop', model=None):
    return {'model': model or grader.MODEL,
            'choices': [{'finish_reason': finish, 'message': {'content': content}}]}


class Integration(unittest.TestCase):
    def test_real_grader_receipts_and_four_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locks = root / 'locks'
            locks.mkdir()
            calls = []
            staged = {
                'PILOT': [('null', 200, response('{"correct":null}'))],
                'EASY': [('length', 200, response('PRIVATE_RESPONSE', 'length')),
                         ('bad_json', 200, response('PRIVATE_RESPONSE'))],
                'MEDIUM': [('wrong_model', 200, response('{}', model='OTHER')),
                           ('rate_limit', 429, None)],
                'HARD': [('empty', 200, response('')),
                         ('timeout', None, None),
                         ('complete', 200, response('{"correct":true}'))],
            }
            for stage, cases in staged.items():
                source_id = 'PRIVATE_SOURCE_' + stage
                source_key = hashlib.sha256(source_id.encode()).hexdigest()[:24]
                receipt_root = root / 'AGNES' / 'receipts' / stage / source_key
                result_dir = root / 'AGNES' / stage / 'sources' / source_key
                result_dir.mkdir(parents=True)
                results = []
                for i, (name, http, raw) in enumerate(cases):
                    row = {'source_id': source_id, 'target': 'source', 'phase': 'pre', 'index': i,
                           'problem': 'PRIVATE_QUESTION', 'reference_answer': 'PRIVATE_ANSWER',
                           'student_response': 'PRIVATE_STUDENT_RESPONSE'}
                    def fake(_body, _key, h=http, r=raw):
                        calls.append(name)
                        if name == 'timeout':
                            raise TimeoutError('PRIVATE_TIMEOUT_DETAIL')
                        return h, r
                    result = grader.grade_one(row, 'NOT_A_REAL_KEY', receipt_root, locks,
                                              caller=fake, sleep=lambda _: None)
                    results.append(result)
                (result_dir / 'RESULTS.json').write_text(json.dumps({'results': results}))

            # Missing receipt and old/wrong identity hash must remain explicitly missing.
            source_id = 'PRIVATE_MISSING'
            source_key = hashlib.sha256(source_id.encode()).hexdigest()[:24]
            identity = {'source_id': source_id, 'target': 'source', 'phase': 'pre', 'index': 0}
            result_dir = root / 'AGNES' / 'HARD' / 'sources' / source_key
            result_dir.mkdir(parents=True)
            (result_dir / 'RESULTS.json').write_text(json.dumps({'results': [
                {'identity': identity, 'status': 'unknown', 'reason': 'retry_limit'}]}))
            wrong_key = hashlib.sha256(grader.canonical(identity)).hexdigest()
            wrong = root / 'AGNES' / 'receipts' / 'HARD' / source_key / wrong_key
            wrong.mkdir(parents=True)
            (wrong / 'ATTEMPT-0001-RESPONSE.json').write_text('{"http_status":429}')

            def snapshot():
                return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob('*') if p.is_file()}

            before = snapshot()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = diagnostic.diagnose(root)
                print(json.dumps(result, ensure_ascii=False))
            self.assertEqual(before, snapshot(), 'diagnostic must not modify files')
            text = out.getvalue()
            for secret in ('PRIVATE_SOURCE', 'PRIVATE_QUESTION', 'PRIVATE_ANSWER',
                           'PRIVATE_STUDENT_RESPONSE', 'PRIVATE_RESPONSE', 'NOT_A_REAL_KEY'):
                self.assertNotIn(secret, text)

            easy, medium, hard = result['简单'], result['中等'], result['困难']
            self.assertEqual(easy['未决作答数'], 3)
            self.assertEqual(easy['最后一次请求']['HTTP_200'], 3)
            self.assertEqual(easy['HTTP200响应类别']['判分模型返回null'], 1)
            self.assertEqual(easy['HTTP200响应类别']['判分输出截断'], 1)
            self.assertEqual(easy['HTTP200响应类别']['判分正文非JSON'], 1)
            self.assertEqual(medium['未决作答数'], 2)
            self.assertEqual(medium['HTTP200响应类别']['模型标识不匹配'], 1)
            self.assertEqual(medium['最后一次请求']['HTTP_429'], 1)
            self.assertEqual(medium['未决样本全部尝试次数']['HTTP_429'], 4)
            self.assertEqual(hard['未决作答数'], 3)
            self.assertEqual(hard['HTTP200响应类别']['判分正文为空'], 1)
            self.assertEqual(hard['最后一次请求']['TimeoutError'], 1)
            self.assertEqual(hard['未决样本全部尝试次数']['TimeoutError'], 4)
            self.assertEqual(hard['缺少尝试回执的未决样本'], 1)
            self.assertEqual(len(calls), 14)


if __name__ == '__main__':
    unittest.main()
