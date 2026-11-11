"""CPU-only tests; no model or network requests."""
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

import lane_launch as lane


class TestLane(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.teacher = self.root / 'teacher'
        self.student = self.root / 'student'
        self.repo = self.root / 'repo'
        for path in (self.teacher, self.student, self.repo / 'app'):
            path.mkdir(parents=True)
        (self.teacher / 'config.json').write_text('{"teacher":1}')
        (self.student / 'config.json').write_text('{"student":1}')
        self.source = self.root / 'input.jsonl'
        self.source.write_text('{"source_id":"test"}\n')
        self.args = lane.parse_args(['--difficulty', 'medium', '--teacher-gpu', '2',
            '--student-gpu', '3', '--teacher-port', '18111', '--student-port', '18112',
            '--source', str(self.source), '--results', str(self.root / 'out'),
            '--repo', str(self.repo), '--teacher-model', str(self.teacher),
            '--student-model', str(self.student)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_frozen_config_uses_original_fields_and_values(self):
        versions = {'vllm': '0.18.0', 'torch': '2.10.0', 'transformers': '5.0',
                    'openai': '1', 'math-verify': '1'}
        cfg = lane.config_for(self.args, versions)
        expected = {'protocol': 'own-transfer-fresh-offline-v1',
            'source_sha256': lane.file_sha(self.source),
            'teacher_config_sha256': lane.file_sha(self.teacher / 'config.json'),
            'student_config_sha256': lane.file_sha(self.student / 'config.json'),
            'teacher_dtype': 'bfloat16', 'teacher_tp': 1, 'student_dtype': 'float16',
            'engine': 'vllm-0.18.0', 'teacher_max_model_len': 20000,
            'teacher_max_tokens': 4192, 'teacher_timeout': 1800,
            'student_max_tokens': 1024, 'student_samples_per_condition': 16,
            'student_temperature': 0.6, 'grading': 'ungraded-local-only',
            'transfer_difficulty': 'medium', 'installed_versions': versions}
        self.assertEqual(cfg, expected)

    def test_independent_service_ports_and_frozen_parameters(self):
        teacher, student, evaluation = lane.service_commands(self.args)
        for command, port in ((teacher, '18111'), (student, '18112')):
            self.assertEqual(command[command.index('--port') + 1], port)
            self.assertEqual(command[command.index('--max-model-len') + 1], '20000')
            self.assertEqual(command[command.index('--seed') + 1], '42')
        self.assertEqual(teacher[teacher.index('--tensor-parallel-size') + 1], '1')
        self.assertEqual(teacher[teacher.index('--gpu-memory-utilization') + 1], '0.85')
        self.assertEqual(student[student.index('--gpu-memory-utilization') + 1], '0.90')
        self.assertEqual(evaluation[evaluation.index('--teacher-url') + 1], 'http://127.0.0.1:18111/v1')
        self.assertEqual(evaluation[evaluation.index('--student-url') + 1], 'http://127.0.0.1:18112/v1')

    def test_resource_identity_validation(self):
        fake = SimpleNamespace(stdout='0, GPU-aaa\n1, GPU-bbb\n')
        with patch.object(lane.subprocess, 'run', return_value=fake):
            self.assertEqual(lane.gpu_uuids(0, 1), ('GPU-aaa', 'GPU-bbb'))
            with self.assertRaises(ValueError):
                lane.gpu_uuids(0, 2)
        with self.assertRaises(SystemExit):
            lane.parse_args(['--difficulty','easy','--teacher-gpu','0','--student-gpu','0',
                '--teacher-port','1','--student-port','2','--source',str(self.source),
                '--results',str(self.root / 'x')])

    def test_frozen_config_does_not_overwrite(self):
        path = self.root / 'config.json'
        def put_once(dest, value):
            with dest.open('x') as f:
                json.dump(value, f)
        lane.bind_config(path, {'x': 1}, put_once)
        lane.bind_config(path, {'x': 1}, put_once)
        with self.assertRaises(ValueError):
            lane.bind_config(path, {'x': 2}, put_once)
        self.assertEqual(json.loads(path.read_text()), {'x': 1})

    def test_exclusive_lock_matches_frozen_filename(self):
        output = self.root / 'output'
        output.mkdir()
        fd = lane.lock_results(output)
        try:
            with self.assertRaises(BlockingIOError):
                lane.lock_results(output)
        finally:
            os.close(fd)

    def test_ports_are_rejected_if_already_bound(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            with self.assertRaises(OSError):
                lane.check_ports([s.getsockname()[1]])

    def test_ready_checks_health_expected_model_and_live_process(self):
        class Response:
            status = 200
            def __init__(self, data=b''):
                self.data = data
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, size):
                return self.data[:size]
        opener = SimpleNamespace(open=unittest.mock.Mock(side_effect=[
            Response(), Response(b'{"data":[{"id":"Eduardo-27b"}]}')]))
        proc = SimpleNamespace(poll=unittest.mock.Mock(return_value=None))
        with patch.object(lane.urllib.request, 'build_opener', return_value=opener) as build:
            lane.ready(proc, 18111, 'Eduardo-27b')
        self.assertEqual(proc.poll.call_count, 2)
        self.assertTrue(opener.open.call_args_list[0].args[0].endswith('/health'))
        self.assertTrue(opener.open.call_args_list[1].args[0].endswith('/v1/models'))
        self.assertEqual(build.call_args.args[0].proxies, {})

    def test_ready_rejects_wrong_model_and_process_exit(self):
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, size):
                return b'{"data":[{"id":"wrong-model"}]}'
        opener = SimpleNamespace(open=unittest.mock.Mock(side_effect=lambda *a, **k: Response()))
        proc = SimpleNamespace(poll=unittest.mock.Mock(side_effect=[None, 1]))
        with patch.object(lane.urllib.request, 'build_opener', return_value=opener), \
             patch.object(lane.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                lane.ready(proc, 18111, 'Eduardo-27b')

    def test_keys_and_proxy_are_not_inherited(self):
        with patch.dict(os.environ, {'AGNES_API_KEY': 'test-only', 'OPENAI_API_KEY': 'test-only',
                                     'HTTPS_PROXY': 'http://bad.invalid'}):
            env = lane.clean_env(self.repo, self.root / 'out')
        self.assertNotIn('AGNES_API_KEY', env)
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('HTTPS_PROXY', env)
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_launcher_wiring_and_append_only_lane_metadata(self):
        (self.repo / 'app' / 'offline_eval.py').touch()
        offline = ModuleType('offline_eval')
        offline.parse_sources = lambda source, difficulty: [1]
        scoring = ModuleType('scoring')
        def put_once(dest, data):
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('x') as f:
                json.dump(data, f)
        scoring.put_once = put_once
        calls = []
        class FakeProc:
            def __init__(self, cmd, **kwargs):
                self.pid = 9000 + len(calls)
                calls.append((cmd, kwargs))
            def wait(self, timeout=None):
                return 0
        argv = ['--difficulty', 'medium', '--teacher-gpu', '2', '--student-gpu', '3',
            '--teacher-port', '18111', '--student-port', '18112', '--source', str(self.source),
            '--results', str(self.root / 'out'), '--repo', str(self.repo),
            '--teacher-model', str(self.teacher), '--student-model', str(self.student)]
        with patch.dict('sys.modules', {'offline_eval': offline, 'scoring': scoring}), \
             patch.object(lane.importlib.metadata, 'version', side_effect=lambda name: name + '-version'), \
             patch.object(lane, 'gpu_uuids', return_value=('GPU-2', 'GPU-3')), \
             patch.object(lane, 'check_ports'), patch.object(lane, 'ready'), \
             patch.object(lane, 'terminate_group'), patch.object(lane.subprocess, 'Popen', side_effect=FakeProc):
            lane.main(argv)
            lane.main(argv)
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[0][1]['env']['CUDA_VISIBLE_DEVICES'], 'GPU-2')
        self.assertEqual(calls[1][1]['env']['CUDA_VISIBLE_DEVICES'], 'GPU-3')
        self.assertIn('http://127.0.0.1:18111/v1', calls[2][0])
        self.assertIn('http://127.0.0.1:18112/v1', calls[2][0])
        output = self.root / 'out'
        metadata = list((output / 'LANE_EXECUTIONS').glob('*.json'))
        self.assertEqual(len(metadata), 2)
        self.assertEqual(json.loads(metadata[0].read_text())['teacher_gpu_uuid'], 'GPU-2')
        self.assertEqual(json.loads((output / 'RUN_CONFIGURATION.json').read_text())['teacher_tp'], 1)


if __name__ == '__main__':
    unittest.main()
