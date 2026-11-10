"""Offline integration test using actual grading receipts and synthetic inputs."""
import contextlib
import hashlib
import importlib.util
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / 'agnes_eval_v2'
sys.path.insert(0, str(APP))
import agnes_judge as grader

spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('agnes_unknown_shape_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def response(content, finish='stop', model=None, reasoning=None):
    message = {'content': content}
    if reasoning is not None:
        message['reasoning_content'] = reasoning
    return {'model': model or grader.MODEL,
            'choices': [{'finish_reason': finish, 'message': message}]}


class Integration(unittest.TestCase):
    def test_real_grader_receipts_partition_and_privacy(self):
        cases = [
            ('PILOT', 200, response('{"correct":true,"explanation":"SECRET_EXPLANATION"}'), 'correct布尔与说明字段候选'),
            ('EASY', 200, response('{"correct":false,"is_correct":true}'), 'correct布尔与其他判定字段'),
            ('EASY', 200, response('{"correct":true,"PRIVATE_KEY":"SECRET_VALUE"}'), 'correct布尔与其他字段'),
            ('EASY', 200, response('{"correct":true,"correct":false,"reason":"PRIVATE"}'), 'JSON含重复键'),
            ('EASY', 200, response('{"correct":true,"explanation":{"x":1,"x":2}}'), 'JSON含重复键'),
            ('EASY', 200, response('{"correct":true,"explanation":[{"x":1,"x":2}]}'), 'JSON含重复键'),
            ('EASY', 200, response('{"correct":"true"}'), 'correct字符串true_false候选'),
            ('EASY', 200, response('{"is_correct":true}'), 'is_correct布尔候选'),
            ('EASY', 200, response('true'), 'JSON顶层布尔候选'),
            ('EASY', 200, response('yes'), '非JSON精确yes_no_correct_incorrect候选'),
            ('EASY', 200, response('PRIVATE_ANSWER free text'), '非JSON其他'),
            ('MEDIUM', 200, response('{"correct":null}'), 'correct为null'),
            ('MEDIUM', 200, response('', reasoning='PRIVATE_REASONING'), '判分正文为空'),
            ('MEDIUM', 200, response(''), '判分正文为空'),
            ('MEDIUM', 200, response('{"correct":true}', finish='length'), '判分输出截断'),
            ('MEDIUM', 200, response('{}', model='PRIVATE_OTHER_MODEL'), '模型标识不匹配'),
            ('HARD', 429, None, '非HTTP200或网络异常'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locks = root / 'locks'
            locks.mkdir()
            by_stage = {}
            for i, (stage, http, raw, label) in enumerate(cases):
                source_id = 'PRIVATE_SOURCE_' + stage
                source_key = hashlib.sha256(source_id.encode()).hexdigest()[:24]
                receipt_root = root / 'AGNES' / 'receipts' / stage / source_key
                row = {'source_id': source_id, 'target': 'source', 'phase': 'pre', 'index': i,
                       'problem': 'PRIVATE_QUESTION', 'reference_answer': 'PRIVATE_REFERENCE',
                       'student_response': 'PRIVATE_STUDENT_RESPONSE'}
                def fake(_body, _key, h=http, r=raw):
                    return h, r
                with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
                    outcome = grader.grade_one(row, 'NOT_A_REAL_KEY', receipt_root, locks,
                                               caller=fake, sleep=lambda _: None)
                self.assertEqual(outcome['status'], 'unknown', label)
                by_stage.setdefault(stage, []).append(outcome)
            for stage, results in by_stage.items():
                source_key = hashlib.sha256(('PRIVATE_SOURCE_' + stage).encode()).hexdigest()[:24]
                folder = root / 'AGNES' / stage / 'sources' / source_key
                folder.mkdir(parents=True)
                (folder / 'RESULTS.json').write_text(json.dumps({'results': results}))

            def snapshot():
                return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob('*') if p.is_file()}
            before = snapshot()
            with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    data = probe.diagnose(root)
                    print(json.dumps(data, ensure_ascii=False))
            self.assertEqual(before, snapshot())
            for word in ('PRIVATE_', 'SECRET_', 'NOT_A_REAL_KEY'):
                self.assertNotIn(word, out.getvalue())
            counts = {}
            for group in data.values():
                self.assertEqual(group['未决作答数'], sum(group['结构分类'].values()))
                for name, count in group['结构分类'].items():
                    counts[name] = counts.get(name, 0) + count
            for _, _, _, label in cases:
                self.assertGreaterEqual(counts.get(label, 0), 1, label)
            self.assertEqual(counts.get('JSON含重复键'), 3)
            self.assertEqual(data['中等']['空正文中有推理字段'], 1)
            self.assertEqual(data['中等']['空正文中无推理字段'], 1)


if __name__ == '__main__':
    unittest.main()
