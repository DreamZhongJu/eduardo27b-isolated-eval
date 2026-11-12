import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import progress as p


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.decisions = self.root / 'decisions'
        self.decisions.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def record(self, stage, source_id, index, method, status, rounds):
        source = hashlib.sha256(source_id.encode()).hexdigest()[:24]
        ident = {'source_id': source_id, 'target':'source', 'phase':'pre', 'index':index}
        token = p._digest({'identity':ident})
        doc = {'schema':p.SCHEMA, 'stage':stage, 'source_key':source, 'identity':ident,
               'request_sha256':'a'*64, 'source_result_sha256':'b'*64, 'input_sha256':'c'*64,
               'method':method, 'status':status, 'rounds':rounds}
        path = self.decisions/stage/source/token/'DECISION.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding='utf-8')
        return path

    def deferred(self, stage, source_id, index, cycles=2):
        source = hashlib.sha256(source_id.encode()).hexdigest()[:24]
        ident = {'source_id': source_id, 'target':'source', 'phase':'pre', 'index':index}
        token = p._digest({'identity':ident})
        doc = {'schema':p.DEFERRED_SCHEMA, 'stage':stage, 'source_key':source, 'identity':ident,
               'request_sha256':'a'*64, 'source_result_sha256':'b'*64, 'input_sha256':'c'*64,
               'reason':'retryable_response_format', 'format_cycles':cycles}
        path = self.root/'deferred'/stage/source/token/'DEFERRED.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding='utf-8')
        return path

    def test_grouped_stable_records_and_privacy(self):
        self.record('PILOT','sensitive-source-1',0,'verified_topbool_overlay','correct',0)
        self.record('EASY','sensitive-source-2',1,'agnes_rejudge','incorrect',1)
        self.record('MEDIUM','sensitive-source-3',2,'agnes_rejudge','correct',1)
        self.record('HARD','sensitive-source-4',3,'agnes_rejudge','unknown',1)
        proc = self.root/'proc'
        proc.mkdir()
        before = {x: x.read_bytes() for x in self.decisions.rglob('DECISION.json')}
        with patch.object(p, 'process_status', return_value=('运行中',1)):
            with patch('socket.socket', side_effect=AssertionError('network prohibited')):
                out = p.render(self.root, Path('/test/rejudge.py'), proc)
        self.assertIn('后台进程：正在判分', out)
        self.assertIn('本任务已记录旧回执', out)
        self.assertIn('补判仍未决', out)
        self.assertIn('补判已落盘合计', out)
        self.assertNotIn('sensitive-source', out)
        self.assertNotIn(str(self.root), out)
        self.assertEqual(before, {x: x.read_bytes() for x in self.decisions.rglob('DECISION.json')})
        counts = p.scan(self.root)
        self.assertEqual(counts['PILOT']['recovered'], 1)
        self.assertEqual(counts['EASY']['incorrect'], 1)
        self.assertEqual(counts['MEDIUM']['correct'], 1)
        self.assertEqual(counts['HARD']['unknown'], 1)

    def test_incomplete_ignored_but_corrupt_record_counted(self):
        good = self.record('EASY','example',0,'agnes_rejudge','correct',1)
        good.write_text('{broken', encoding='utf-8')
        other = self.record('EASY','example',1,'agnes_rejudge','correct',1)
        other.unlink()
        c = p.scan(self.root)['EASY']
        self.assertEqual(c['error'], 1)
        self.assertEqual(c['correct'], 0)

    def test_identity_and_methods_are_checked(self):
        file = self.record('HARD','example',0,'agnes_rejudge','correct',1)
        doc = json.loads(file.read_text())
        doc['identity']['index'] = 1
        file.write_text(json.dumps(doc))
        self.assertEqual(p.scan(self.root)['HARD']['error'], 1)

    def test_technical_deferred_is_separate_from_semantic_unknown(self):
        self.record('EASY','example',0,'agnes_rejudge','unknown',1)
        self.deferred('EASY','example',1,cycles=3)
        counts = p.scan(self.root)['EASY']
        self.assertEqual((counts['unknown'], counts['technical_deferred'], counts['error']), (1,1,0))
        with patch.object(p, 'process_status', return_value=('未发现进程', 0)):
            out = p.render(self.root, Path('/test/rejudge.py'), self.root/'unused')
        self.assertIn('技术待处理', out)
        self.assertIn('补判仍未决', out)
        self.assertNotIn('example', out)

    def test_deferred_collision_is_error_and_not_double_counted(self):
        self.record('HARD','example',0,'agnes_rejudge','correct',1)
        self.deferred('HARD','example',0)
        counts = p.scan(self.root)['HARD']
        self.assertEqual((counts['correct'], counts['technical_deferred'], counts['error']), (1,0,1))

    def test_invalid_deferred_bindings_and_cycles_are_rejected(self):
        file = self.deferred('HARD','example',0)
        for field, value in (('request_sha256', 'broken'), ('format_cycles', 1),
                             ('reason', 'other'), ('schema', p.SCHEMA)):
            original = json.loads(file.read_text())
            changed = dict(original)
            changed[field] = value
            file.write_text(json.dumps(changed))
            self.assertEqual((p.scan(self.root)['HARD']['technical_deferred'],
                              p.scan(self.root)['HARD']['error']), (0,1))
            file.write_text(json.dumps(original))

    def test_deferred_count_works_without_decisions_directory(self):
        self.decisions.rmdir()
        self.deferred('MEDIUM', 'example', 0)
        self.assertEqual(p.scan(self.root)['MEDIUM']['technical_deferred'], 1)

    def test_pid_requires_exact_script_and_output(self):
        proc = self.root/'proc'
        proc.mkdir()
        script = Path('/workspace/edu-rejudge-ed7f64d/rejudge.py')
        process = proc/'12345'
        process.mkdir()
        process.joinpath('cmdline').write_bytes(os.fsencode(f'python\0{script}\0--output\0{self.root}\0'))
        self.assertEqual(p.process_status(script, self.root, proc), ('运行中', 1))
        process.joinpath('cmdline').write_bytes(os.fsencode(f'python\0{script}\0--output\0/somewhere\0'))
        self.assertEqual(p.process_status(script, self.root, proc), ('未发现进程', 0))
        process.joinpath('cmdline').write_bytes(os.fsencode(f'python\0{script}\0--output={self.root}\0'))
        self.assertEqual(p.process_status(script, self.root, proc), ('运行中', 1))

    def test_unknown_on_unreadable_process(self):
        proc = self.root/'proc'
        proc.mkdir()
        process = proc/'12345'
        process.mkdir()
        process.joinpath('cmdline').mkdir()
        self.assertEqual(p.process_status(Path('/test/rejudge.py'), self.root, proc), ('无法确认', 0))

    def test_guardian_and_worker_use_exact_script_and_output(self):
        proc = self.root/'proc'
        proc.mkdir()
        guardian = proc/'11111'
        guardian.mkdir()
        guardian.joinpath('cmdline').write_bytes(os.fsencode(
            f'python\0{p.SUPERVISOR}\0--output\0{self.root}\0'))
        worker_state = p.process_status(p.SCRIPT, self.root, proc)
        guard_state = p.process_status(p.SUPERVISOR, self.root, proc)
        self.assertEqual(worker_state, ('未发现进程', 0))
        self.assertEqual(guard_state, ('运行中', 1))
        self.assertIn('可能在退避或准备', p.describe_processes(worker_state, guard_state))

        worker = proc/'22222'
        worker.mkdir()
        worker.joinpath('cmdline').write_bytes(os.fsencode(
            f'python\0{p.SCRIPT}\0--output={self.root}\0'))
        worker_state = p.process_status(p.SCRIPT, self.root, proc)
        self.assertIn('正在判分', p.describe_processes(worker_state, guard_state))
        self.assertIn('守护进程在线', p.describe_processes(worker_state, guard_state))
        # A similarly named path and a different output must not count.
        guardian.joinpath('cmdline').write_bytes(os.fsencode(
            f'python\0/tmp/auto_resume.py\0--output\0{self.root}\0'))
        self.assertEqual(p.process_status(p.SUPERVISOR, self.root, proc), ('未发现进程', 0))

    def test_unreadable_never_claims_waiting_or_stopped(self):
        self.assertEqual(p.describe_processes(('无法确认', 0), ('运行中', 1)),
                         '守护进程运行中，判分进程状态无法确认')
        self.assertEqual(p.describe_processes(('未发现进程', 0), ('无法确认', 0)), '无法确认')
        self.assertEqual(p.describe_processes(('未发现进程', 0), ('未发现进程', 0)), '未发现运行进程')

    def test_old_and_new_workers_both_supported(self):
        proc = self.root/'proc'
        proc.mkdir()
        for index, script in enumerate((p.SCRIPT, p.ALTERNATE_SCRIPT), 1):
            folder = proc/str(12340 + index)
            folder.mkdir()
            folder.joinpath('cmdline').write_bytes(os.fsencode(
                f'python\0{script}\0--output\0{self.root}\0'))
        combined = p.combine_process_status(p.process_status(p.SCRIPT, self.root, proc),
                                            p.process_status(p.ALTERNATE_SCRIPT, self.root, proc))
        self.assertEqual(combined, ('运行中', 2))

    def test_cjk_columns_align(self):
        result = p.table(('项目','简单','困难'), [('补判为对',1,3)])
        widths = [p.width(line) for line in result.splitlines()]
        self.assertEqual(len(set(widths)), 1)


if __name__ == '__main__':
    unittest.main()
