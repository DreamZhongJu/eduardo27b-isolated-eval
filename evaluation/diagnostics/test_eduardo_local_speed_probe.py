import contextlib
import hashlib
import io
import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import eduardo_local_speed_probe as p


def marker(root, stage, sid, timestamp, held=False):
    key = hashlib.sha256(sid.encode()).hexdigest()[:24]
    folder = root/stage/'sources'/key
    folder.mkdir(parents=True)
    path = folder/'RAW_COMPLETE.json'
    path.write_text(json.dumps({'source_id': sid, 'samples': 96}))
    os.utime(path, (timestamp, timestamp))
    if held:
        (folder/'SOURCE_HELD.json').write_text('{}')


class Tests(unittest.TestCase):
    def test_sources_with_recent_valid_times_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            now = 100000
            for i, age in enumerate((5000, 500, 400, 300, 200)):
                marker(root, 'EASY', f'PRIVATE_SOURCE_{i}', now-age)
            marker(root, 'EASY', 'PRIVATE_HELD', now-100, held=True)
            counters = {'sources_started': 13, 'generation_complete': 10, 'generation_held': 1, 'generation_partial': 2, 'generation_invalid': 0,
                        'grading_pending': 3, 'unknown': 50}
            out = p.stage_rate(root, ('EASY',), now-600, now, counters, True)
            self.assertEqual(out['本次启动后生成完成数'], 4)
            self.assertEqual(out['最近样本数'], 4)
            self.assertEqual(out['最近平均分钟每来源'], 1.67)
            self.assertEqual(out['未开始来源生成粗估小时'], 13.5)
            self.assertEqual(out['已有来源目录数'], 13)
            self.assertEqual(out['生成partial数'], 2)
            self.assertNotIn('PRIVATE_', json.dumps(out))
            self.assertNotIn('未开始来源生成粗估小时',
                             p.stage_rate(root, ('EASY',), now-600, now+1000, counters, True))
            self.assertNotIn('未开始来源生成粗估小时',
                             p.stage_rate(root, ('EASY',), now-600, now, counters, False))

    def test_active_receipts_match_pid_and_frozen_supervisor(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            folder = root/'FIVEGPU_EXECUTIONS'
            folder.mkdir()
            (folder/'100000000000-42.json').write_text(json.dumps({'coordinator_sha256': p.SUPERVISOR_SHA}))
            (folder/'110000000000-43.json').write_text(json.dumps({'coordinator_sha256': p.SUPERVISOR_SHA}))
            (folder/'120000000000-42.json').write_text(json.dumps({'coordinator_sha256': 'different'}))
            self.assertEqual(p.active_receipt(root, 42, 90)[0], 100)
            self.assertIsNone(p.active_receipt(root, 42, 101))

    def test_student_route_and_uuid(self):
        rec = {'gpus': [0, 1, 2, 3], 'gpu_uuids': ['GPU-0', 'GPU-1', 'GPU-2', 'GPU-3'],
               'ports': [18001, 18002, 18003, 18004],
               'routing': {'EASY': 18004, 'MEDIUM': 18004, 'HARD': 18004}}
        self.assertEqual(p.resolve_student(rec), (3, 'GPU-3', 18004))
        rec['routing']['HARD'] = 18003
        self.assertIsNone(p.resolve_student(rec))
        rows = {3: {'uuid': 'GPU-3', 'util': 40, 'used': 70000, 'total': 81920}}
        samples = p.gpu_samples((3, 'GPU-3', 18004), lambda: rows, lambda _: None,
                                lambda _: {}, lambda: 1)
        self.assertEqual(samples['整卡利用率均值百分比'], 40)
        self.assertEqual(samples['采样间等待秒'], 2)
        rows[3]['uuid'] = 'GPU-OTHER'
        with self.assertRaises(ValueError):
            p.gpu_samples((3, 'GPU-3', 18004), lambda: rows, lambda _: None,
                          lambda _: {}, lambda: 1)

    def test_metric_allowlist_never_emits_labels_or_raw_values(self):
        text = ('vllm:generation_tokens_total{model="PRIVATE_MODEL"} 123\n'
                'vllm:num_requests_running{SECRET="PRIVATE"} 2\n'
                'vllm:num_requests_waiting{model="PRIVATE"} 3\n'
                'vllm:PRIVATE_METRIC{PRIVATE="x"} 99\n')
        result = p.parse_metrics(text)
        self.assertEqual(result['vllm:generation_tokens_total'], 123)
        self.assertEqual(result['vllm:num_requests_running'], 2)
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertNotIn('SECRET', json.dumps(result))

    def test_no_remote_or_raw_output_on_failure(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(socket.socket, 'connect', side_effect=AssertionError('network must not be used')):
                with self.assertRaises((OSError, ValueError)):
                    p.load_status(Path(t))


if __name__ == '__main__':
    unittest.main()
