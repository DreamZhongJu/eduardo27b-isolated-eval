"""CPU-only tests: no GPU, network, real inputs or external API calls."""
import argparse
import ast
import contextlib
import hashlib
import io
import itertools
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import coordinator as c


class TestCoordination(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'results'
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_distinct_gpu_and_port_pairs_required(self):
        self.assertEqual(c.parse_pairs(['0,1', '2,3', '4,5'], 'gpus')['HARD'], (4, 5))
        with self.assertRaises(ValueError):
            c.parse_pairs(['0,1', '1,2', '3,4'], 'gpus')
        with self.assertRaises(ValueError):
            c.parse_pairs(['8001,8002', '8002,8003', '8004,8005'], 'ports')

    def test_resource_checks_fail_closed_for_busy_gpu_and_port(self):
        gpus = c.parse_pairs(['0,1', '2,3', '4,5'], 'gpus')
        ports = c.parse_pairs(['21001,21002', '21003,21004', '21005,21006'], 'ports')
        snapshot = {i: {'uuid': 'GPU-' + str(i), 'name': 'NVIDIA A100 80GB',
                        'total': 81000, 'free': 80000, 'util': 0} for i in range(6)}
        snapshot[3]['free'] = 100
        with self.assertRaisesRegex(ValueError, 'gpu_busy'):
            c.check_resources(gpus, ports, snapshot)
        snapshot[3]['free'] = 80000
        with socket.socket() as s:
            s.bind(('127.0.0.1', 21003))
            with self.assertRaises(OSError):
                c.check_resources(gpus, ports, snapshot)

    def test_automatic_pairs_exclude_busy_reserved_and_duplicate_uuid(self):
        snapshot = {i: {'uuid': 'GPU-' + str(i), 'name': 'NVIDIA A100 80GB',
                        'total': 81000, 'free': 80000, 'util': 0} for i in range(8)}
        for i in (4, 5, 6):
            snapshot[i]['free'] = 100
        first = c.available_pair(snapshot, set())
        self.assertEqual(first, (0, 1))
        second = c.available_pair(snapshot, set(first))
        self.assertEqual(second, (2, 3))
        self.assertIsNone(c.available_pair(snapshot, set(first + second)))
        snapshot[7]['uuid'] = snapshot[0]['uuid']
        self.assertIsNone(c.available_pair(snapshot, {0, 1, 2, 3}))

    def test_one_free_pair_is_reused_only_after_lane_exit(self):
        for stage in c.STAGES:
            (self.root / stage).mkdir()
        (self.root / 'CODE_BINDING.json').write_text('{"old":"binding"}')
        key = self.base / 'key'; key.write_text('test-only')
        launcher = self.base / 'lane_launch.py'; launcher.write_text('# fake')
        data = self.base / 'data'; data.write_text('{}\n')
        args = argparse.Namespace(root=self.root, repo=self.base / 'repo', python=Path('/usr/bin/python3'),
            data=data, teacher=self.base / 'teacher', student=self.base / 'student', key=key,
            launcher=launcher, gpu_pairs=None, port_pairs=None, max_lanes=2, poll_seconds=2)
        events = []
        live = set()
        class Fake:
            def __init__(self, name, watcher=False): self.name, self.watcher, self.stopped = name, watcher, False
            def poll(self):
                if self.watcher: return 0 if self.stopped else None
                if self.name in live:
                    live.remove(self.name)
                    events.append(('finished', self.name))
                return 0
            def terminate(self): self.stopped = True
            def wait(self): return 0
            def send_signal(self, _): self.stopped = True
        def popen(cmd, **_):
            if 'watch_sources.py' in cmd[1]: return Fake('watcher', watcher=True)
            name = cmd[cmd.index('--difficulty') + 1]
            pair = (int(cmd[cmd.index('--teacher-gpu') + 1]), int(cmd[cmd.index('--student-gpu') + 1]))
            self.assertEqual(pair, (0, 1))
            self.assertFalse(live)
            if name == 'hard': self.assertTrue(any(e[0] == 'finished' for e in events))
            live.add(name); events.append(('started', name))
            return Fake(name)
        snapshot = {i: {'uuid': 'GPU-' + str(i), 'name': 'A100 80GB', 'total': 81000,
                        'free': 80000 if i in (0, 1) else 100, 'util': 0} for i in range(8)}
        with patch.object(c, 'BASE', self.base), patch.object(c, 'old_process_running', return_value=False), \
             patch.object(c, 'verify_old_binding'), patch.object(c, 'verify_inputs'), patch.object(c, 'verify_pilot'), \
             patch.object(c, 'verify_versions'), patch.object(c, 'verify_models'), patch.object(c, 'check_stage_locks'), \
             patch.object(c, 'gpu_snapshot', return_value=snapshot), patch.object(c, 'ports_available', return_value=True), \
             patch.object(c, 'grading_pending', return_value=0), \
             patch.object(c, 'verify_terminal', return_value={'complete': 1, 'held': 0}), \
             patch.object(c.subprocess, 'Popen', side_effect=popen), patch.object(c.time, 'sleep'):
            c.run(args)
        self.assertEqual([name for event, name in events if event == 'started'], ['easy', 'medium', 'hard'])

    def test_two_lanes_run_together_and_hard_waits_for_real_exit(self):
        for stage in c.STAGES:
            (self.root / stage).mkdir()
        (self.root / 'CODE_BINDING.json').write_text('{"old":"binding"}')
        key = self.base / 'key'; key.write_text('test-only')
        launcher = self.base / 'lane_launch.py'; launcher.write_text('# fake')
        data = self.base / 'data'; data.write_text('{}\n')
        args = argparse.Namespace(root=self.root, repo=self.base / 'repo', python=Path('/usr/bin/python3'),
            data=data, teacher=self.base / 'teacher', student=self.base / 'student', key=key,
            launcher=launcher, gpu_pairs=None, port_pairs=None, max_lanes=2, poll_seconds=2)
        events, live, peak = [], {}, [0]
        class Fake:
            def __init__(self, name, watcher=False):
                self.name, self.watcher, self.stopped, self.polls = name, watcher, False, 0
            def poll(self):
                if self.watcher: return 0 if self.stopped else None
                self.polls += 1
                if self.polls < 2: return None
                if self.name in live:
                    del live[self.name]
                    events.append(('finished', self.name))
                return 0
            def terminate(self): self.stopped = True
            def wait(self): return 0
            def send_signal(self, _): self.stopped = True
        def popen(cmd, **_):
            if 'watch_sources.py' in cmd[1]: return Fake('watcher', watcher=True)
            name = cmd[cmd.index('--difficulty') + 1]
            pair = (int(cmd[cmd.index('--teacher-gpu') + 1]), int(cmd[cmd.index('--student-gpu') + 1]))
            self.assertFalse(set(pair) & {4, 5, 6})
            self.assertFalse(set(pair) & {x for p in live.values() for x in p})
            if name == 'hard': self.assertTrue(any(e[0] == 'finished' for e in events))
            live[name] = pair; peak[0] = max(peak[0], len(live)); events.append(('started', name))
            return Fake(name)
        snapshot = {i: {'uuid': 'GPU-' + str(i), 'name': 'A100 80GB', 'total': 81000,
                        'free': 100 if i in (4, 5, 6) else 80000, 'util': 0} for i in range(8)}
        with patch.object(c, 'BASE', self.base), patch.object(c, 'old_process_running', return_value=False), \
             patch.object(c, 'verify_old_binding'), patch.object(c, 'verify_inputs'), patch.object(c, 'verify_pilot'), \
             patch.object(c, 'verify_versions'), patch.object(c, 'verify_models'), patch.object(c, 'check_stage_locks'), \
             patch.object(c, 'gpu_snapshot', return_value=snapshot), patch.object(c, 'ports_available', return_value=True), \
             patch.object(c, 'grading_pending', return_value=0), \
             patch.object(c, 'verify_terminal', return_value={'complete': 1, 'held': 0}), \
             patch.object(c.subprocess, 'Popen', side_effect=popen), patch.object(c.time, 'sleep'):
            c.run(args)
        self.assertEqual(peak[0], 2)
        self.assertEqual([name for event, name in events if event == 'started'], ['easy', 'medium', 'hard'])

    def test_original_serial_lock_blocks_second_owner(self):
        path = self.base / 'lock'
        fd = c.take_lock(path)
        try:
            with self.assertRaises(BlockingIOError):
                c.take_lock(path)
        finally:
            os.close(fd)

    def test_precheck_visible_and_locked_out_controller_cannot_overwrite_status(self):
        args = argparse.Namespace(root=self.root, repo=self.base / 'repo', gpu_pairs=None,
                                  port_pairs=None, max_lanes=2, poll_seconds=2)
        marker = self.root / 'PARALLEL_V4_STATUS.json'
        marker.write_text('{"state":"EXISTING"}')
        fd = c.take_lock(self.base / '.own-test-499.lock')
        try:
            with patch.object(c, 'BASE', self.base), self.assertRaises(BlockingIOError):
                c.run(args)
            self.assertEqual(c.load(marker)['state'], 'EXISTING')
        finally:
            os.close(fd)
        def check(*_):
            self.assertEqual(c.load(marker)['state'], 'PRECHECK')
            raise ValueError('old_process_still_running')
        with patch.object(c, 'BASE', self.base), patch.object(c, 'old_process_running', side_effect=check):
            with self.assertRaisesRegex(ValueError, 'old_process_still_running'):
                c.run(args)
        self.assertEqual(c.load(marker)['state'], 'FAILED')
        self.assertEqual(c.load(marker)['code'], 'old_process_still_running')

    def test_other_vllm_process_does_not_require_environ_access(self):
        proc_root = self.base / 'fake_proc'
        other = proc_root / '999999'
        other.mkdir(parents=True)
        (other / 'cmdline').write_bytes(b'python\0-m\0vllm.entrypoints.openai.api_server\0'
            b'--port\0' + b'8001\0--served-model-name\0another-model\0--model\0/other/model\0')
        # Reading this directory as bytes would raise an OSError.
        (other / 'environ').mkdir()
        real_path = Path
        with patch.object(c, 'Path', side_effect=lambda x: proc_root if x == '/proc' else real_path(x)):
            self.assertFalse(c.old_process_running(self.root, self.base / 'repo'))
        old = proc_root / '999998'
        old.mkdir()
        (old / 'cmdline').write_bytes(b'python\0-m\0vllm.entrypoints.openai.api_server\0'
            b'--port\0' + b'8001\0--served-model-name\0Eduardo-27b\0'
            b'--model\0/workspace/models/Eduardo-27b-3bbf17e\0')
        (old / 'environ').write_bytes(b'RESULTS_DIR=' + str(self.root / 'EASY').encode() + b'\0')
        with patch.object(c, 'Path', side_effect=lambda x: proc_root if x == '/proc' else real_path(x)):
            self.assertTrue(c.old_process_running(self.root, self.base / 'repo'))

    def test_prior_v3_and_v4_orphan_lanes_are_detected_without_process_environment(self):
        proc_root = self.base / 'fake_proc'
        lane = proc_root / '999997'
        lane.mkdir(parents=True)
        # An orphan may be between startup and creating its vLLM process.
        (lane / 'environ').mkdir()
        real_path = Path
        for version in (b'v3', b'v4'):
            with self.subTest(version=version):
                lane_cmd = (b'python\0/workspace/clone/parallel_eval_' + version + b'/lane_launch.py\0'
                            b'--results\0' + str(self.root / 'MEDIUM').encode() + b'\0')
                (lane / 'cmdline').write_bytes(lane_cmd)
                with patch.object(c, 'Path', side_effect=lambda x: proc_root if x == '/proc' else real_path(x)):
                    self.assertTrue(c.old_process_running(self.root, self.base / 'repo'))
                (lane / 'cmdline').write_bytes(lane_cmd.replace(str(self.root / 'MEDIUM').encode(), b'/other/result'))
                with patch.object(c, 'Path', side_effect=lambda x: proc_root if x == '/proc' else real_path(x)):
                    self.assertFalse(c.old_process_running(self.root, self.base / 'repo'))

    def test_prior_v3_and_v4_service_on_recorded_custom_port_is_detected(self):
        proc_root = self.base / 'fake_proc'
        service = proc_root / '999996'
        service.mkdir(parents=True)
        (service / 'cmdline').write_bytes(b'python\0-m\0vllm.entrypoints.openai.api_server\0'
            b'--port\0' + b'9281\0--served-model-name\0Eduardo-27b\0'
            b'--model\0/workspace/models/Eduardo-27b-3bbf17e\0')
        (service / 'environ').write_bytes(b'RESULTS_DIR=' + str(self.root / 'MEDIUM').encode() + b'\0')
        real_path = Path
        for version in ('V3', 'V4'):
            with self.subTest(version=version):
                receipts = self.root / f'PARALLEL_{version}_EXECUTIONS'
                receipts.mkdir()
                receipt = receipts / 'run.json'
                receipt.write_text(json.dumps({'port_pair': [9281, 9282]}))
                with patch.object(c, 'Path', side_effect=lambda x: proc_root if x == '/proc' else real_path(x)):
                    self.assertTrue(c.old_process_running(self.root, self.base / 'repo'))
                receipt.unlink()

    def test_existing_data_and_reuse_bound_to_exact_lines(self):
        data = self.base / 'test.jsonl'
        lines = [json.dumps({'source_id': str(i)}).encode() + b'\n' for i in range(499)]
        data.write_bytes(b''.join(lines))
        (self.root / 'pilot.jsonl').write_bytes(lines[0])
        (self.root / 'easy-rest.jsonl').write_bytes(b''.join(lines[1:]))
        with patch.object(c, 'DATA_SHA', c.sha(data)):
            c.verify_inputs(self.root, data)
            (self.root / 'pilot.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen_input_mismatch'):
                c.verify_inputs(self.root, data)

    def test_source_terminal_requires_exact_sources_and_receipts(self):
        data = self.base / 'one.jsonl'
        data.write_text(json.dumps({'source_id': 'synthetic'}) + '\n')
        key = hashlib.sha256(b'synthetic').hexdigest()[:24]
        source = self.root / 'EASY' / 'sources' / key
        source.mkdir(parents=True)
        (self.root / 'EASY' / 'RUN_CONFIGURATION.json').write_text(json.dumps({
            'protocol': 'own-transfer-fresh-offline-v1', 'transfer_difficulty': 'easy',
            'source_sha256': c.sha(data)}))
        (source / 'RAW_COMPLETE.json').write_text('{"samples":96}')
        with self.assertRaisesRegex(ValueError, 'invalid_source_receipt'):
            c.verify_terminal(self.root, 'EASY', data)
        for i in range(96):
            p = source / 'samples' / 'source' / 'pre' / str(i)
            p.mkdir(parents=True)
            (p / 'RESPONSE.json').write_text('{}')
        self.assertEqual(c.verify_terminal(self.root, 'EASY', data), {'complete': 1, 'held': 0})
        (source / 'RAW_COMPLETE.json').unlink()
        (source / 'SOURCE_HELD.json').write_text('{}')
        self.assertEqual(c.verify_terminal(self.root, 'EASY', data), {'complete': 0, 'held': 1})

    def test_status_only_reports_aggregate_and_merges_easy_pilot(self):
        for stage in ('PILOT', 'EASY'):
            source = self.root / stage / 'sources' / 'source-hash'
            source.mkdir(parents=True)
            (source / 'RAW_COMPLETE.json').write_text('{}')
            grade = self.root / 'AGNES' / stage / 'sources' / 'source-hash'
            grade.mkdir(parents=True)
            (grade / 'RESULTS.json').write_text(json.dumps({'results': [
                {'status': 'correct', 'student_response': 'SHOULD_NOT_LEAK'}, {'status': 'unknown'}]}))
        result = c.status_summary(self.root)
        self.assertEqual(result['easy']['generation_complete'], 2)
        self.assertEqual(result['easy']['correct'], 2)
        self.assertEqual(result['easy']['unknown'], 2)
        self.assertNotIn('SHOULD_NOT_LEAK', json.dumps(result))

    def test_worker_versions_must_match_pilot_and_existing_stage(self):
        versions = {'vllm': '0.18.0', 'torch': '2.10.0', 'transformers': '5.0',
                    'openai': '2.0', 'math-verify': '0.8'}
        pilot = self.root / 'PILOT' / 'RUN_CONFIGURATION.json'
        pilot.parent.mkdir(parents=True)
        pilot.write_text(json.dumps({'installed_versions': versions}))
        easy = self.root / 'EASY' / 'RUN_CONFIGURATION.json'
        easy.parent.mkdir(parents=True)
        easy.write_text(json.dumps({'installed_versions': versions}))
        with patch.object(c.subprocess, 'check_output', return_value=json.dumps(versions)) as query:
            c.verify_versions(Path('/chosen/worker/python'), self.root)
            self.assertEqual(query.call_args.args[0][0], '/chosen/worker/python')
            ast.parse(query.call_args.args[0][2])
            easy.write_text(json.dumps({'installed_versions': {**versions, 'torch': 'different'}}))
            with self.assertRaisesRegex(ValueError, 'environment_version_mismatch'):
                c.verify_versions(Path('/chosen/worker/python'), self.root)
            easy.unlink()
            pilot.unlink()
            with self.assertRaisesRegex(ValueError, 'pilot_config_missing'):
                c.verify_versions(Path('/chosen/worker/python'), self.root)
        pilot.write_text(json.dumps({'installed_versions': versions}))
        with patch.object(c.subprocess, 'check_output', return_value=json.dumps({**versions, 'vllm': '0.17.0'})):
            with self.assertRaisesRegex(ValueError, 'environment_version_mismatch'):
                c.verify_versions(Path('/chosen/worker/python'), self.root)

    def test_three_lanes_resume_and_record_failure_without_overwriting(self):
        for stage in c.STAGES:
            (self.root / stage).mkdir()
        (self.root / 'CODE_BINDING.json').write_text('{"old":"binding"}')
        old_status = self.root / 'PARALLEL_V3_STATUS.json'
        old_binding = self.root / 'PARALLEL_V3_BINDING.json'
        old_status.write_text('{"state":"FAILED_V3"}')
        old_binding.write_text('{"code":"old-v3"}')
        old_receipt = self.root / 'PARALLEL_V3_EXECUTIONS' / 'old.json'
        old_receipt.parent.mkdir()
        old_receipt.write_text('{"port_pair":[8011,8012]}')
        existing_score = self.root / 'AGNES' / 'EASY' / 'sources' / 'existing' / 'RESULTS.json'
        existing_score.parent.mkdir(parents=True)
        existing_score.write_text('{"results":[{"status":"correct"}]}')
        before = {p: p.read_bytes() for p in (old_status, old_binding, old_receipt, existing_score)}
        key = self.base / 'key'
        key.write_text('test-only')
        launcher = self.base / 'lane_launch.py'
        launcher.write_text('# fake')
        data = self.base / 'data'
        data.write_text('{}\n')
        args = argparse.Namespace(root=self.root, repo=self.base / 'repo', python=Path('/usr/bin/python3'),
            data=data, teacher=self.base / 'teacher', student=self.base / 'student', key=key,
            launcher=launcher, gpu_pairs=['0,1', '2,3', '4,5'],
            port_pairs=['21001,21002', '21003,21004', '21005,21006'], max_lanes=2, poll_seconds=2)
        class Fake:
            def __init__(self, watcher=False, code=0): self.watcher, self.code, self.stopped = watcher, code, False
            def poll(self): return (0 if self.stopped else None) if self.watcher else self.code
            def terminate(self): self.stopped = True
            def wait(self): return 0
            def send_signal(self, _): self.stopped = True
        calls = []
        def popen(cmd, **kwargs):
            if 'watch_sources.py' in cmd[1]:
                return Fake(watcher=True)
            difficulty = cmd[cmd.index('--difficulty') + 1]
            calls.append(difficulty)
            return Fake(code=1 if difficulty == 'medium' else 0)
        snapshot = {i: {'uuid': 'GPU-' + str(i), 'name': 'A100 80GB', 'total': 81000,
                        'free': 80000, 'util': 0} for i in range(6)}
        with patch.object(c, 'BASE', self.base), patch.object(c, 'old_process_running', return_value=False), \
             patch.object(c, 'verify_old_binding'), patch.object(c, 'verify_inputs'), patch.object(c, 'verify_pilot'), \
             patch.object(c, 'verify_versions'), patch.object(c, 'verify_models'), patch.object(c, 'check_stage_locks'), \
             patch.object(c, 'gpu_snapshot', return_value=snapshot), patch.object(c, 'ports_available', return_value=True), \
             patch.object(c, 'grading_pending', return_value=0), \
             patch.object(c, 'verify_terminal', return_value={'complete': 1, 'held': 0}), \
             patch.object(c.subprocess, 'Popen', side_effect=popen), patch.object(c.time, 'sleep'), \
             patch.object(c.time, 'monotonic', side_effect=itertools.count(100, 10)):
            c.run(args)
            state = c.load(self.root / 'PARALLEL_V4_STATUS.json')
            self.assertEqual(state['state'], 'FINISHED_WITH_FAILURES_OR_UNRESOLVED')
            self.assertEqual(state['stages']['MEDIUM'], 'failed')
            self.assertEqual(calls.count('easy'), 1)
            self.assertEqual(calls.count('medium'), 4)
            self.assertEqual(calls.count('hard'), 1)
            self.assertTrue((self.root / 'EASY' / 'STAGE_FINISHED').exists())
            self.assertFalse((self.root / 'MEDIUM' / 'STAGE_FINISHED').exists())
            self.assertEqual((self.root / 'CODE_BINDING.json').read_text(), '{"old":"binding"}')
            self.assertTrue((self.root / 'PARALLEL_V4_BINDING.json').exists())
            self.assertTrue((self.root / 'PARALLEL_V4_EXECUTIONS').is_dir())
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            calls.clear()
            c.run(args)
            self.assertEqual(calls, ['medium'] * 4)
            self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == '__main__':
    unittest.main()
