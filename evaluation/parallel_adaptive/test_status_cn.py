import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import status_cn as status


def receipt():
    ports = [18001, 18002, 18003, 18004, 18005]
    return {'coordinator_sha256': status.SUPERVISOR_SHA, 'gpus': [0, 2, 4, 5, 7],
            'gpu_uuids': ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d', 'GPU-e'], 'ports': ports,
            'routing': {'teacher': dict(zip(('EASY', 'MEDIUM', 'HARD'), ports[:3])),
                        'student': {k: ports[3:] for k in ('EASY', 'MEDIUM', 'HARD')}}}


class StatusCnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.rec = receipt()
        self.gpus = dict(zip(self.rec['gpus'], self.rec['gpu_uuids']))

    def test_validate_roles_and_routing(self):
        roles = status.gpu_roles(self.rec, self.gpus)
        self.assertEqual([r[0] for r in roles], [0, 2, 4, 5, 7])
        self.assertEqual([r[1] for r in roles], ['教师：简单', '教师：中等', '教师：困难', '学生：主服务', '学生：第二服务'])
        self.rec['routing']['student']['HARD'] = [18005]
        self.assertIsNone(status.gpu_roles(self.rec, self.gpus))

    def test_uuid_or_duplicate_mismatch_never_claims_roles(self):
        self.gpus[2] = 'GPU-other'
        self.assertIsNone(status.gpu_roles(self.rec, self.gpus))
        self.rec['gpus'][1] = 0
        self.assertIsNone(status.gpu_roles(self.rec, self.gpus))

    def test_receipt_requires_pid_start_and_code_hash(self):
        d = self.root/'FIVEGPU_EXECUTIONS'
        d.mkdir()
        (d/'12000000000-12.json').write_text(json.dumps(self.rec))
        self.assertEqual(status.active_receipt(self.root, 12, 10), self.rec)
        self.assertIsNone(status.active_receipt(self.root, 13, 10))
        self.assertIsNone(status.active_receipt(self.root, 12, 13))
        self.rec['coordinator_sha256'] = 'wrong'
        (d/'12000000000-12.json').write_text(json.dumps(self.rec))
        self.assertIsNone(status.active_receipt(self.root, 12, 10))

    def test_service_processes_require_parent_port_device_tag(self):
        proc = self.root/'proc'
        proc.mkdir()
        roles = status.gpu_roles(self.rec, self.gpus)
        for i, (gpu, _, port) in enumerate(roles):
            p = proc / str(100 + i)
            p.mkdir()
            (p/'stat').write_text(f'{100+i} (svc) S 12 0 0')
            (p/'cmdline').write_bytes(b'python\0-m\0vllm.entrypoints.openai.api_server\0--port\0' + str(port).encode()+b'\0')
            env = [f'CUDA_VISIBLE_DEVICES={gpu}', f'FIVEGPU_RUN_ROOT={self.root}',
                   'FIVEGPU_RUN_ID='+'a'*32]
            (p/'environ').write_bytes(('\0'.join(env)+'\0').encode())
        self.assertEqual(status.verified_services(12, roles, self.root, proc), {r[2] for r in roles})
        env = proc/'100'/'environ'
        env.write_text('CUDA_VISIBLE_DEVICES=99\0')
        self.assertIsNone(status.verified_services(12, roles, self.root, proc))

    def test_active_process_hash_receipt_and_race(self):
        sup = self.root/'supervisor.py'
        sup.write_text('synthetic supervisor')
        digest = hashlib.sha256(sup.read_bytes()).hexdigest()
        self.rec['coordinator_sha256'] = digest
        d = self.root/'FIVEGPU_EXECUTIONS'
        d.mkdir()
        (d/'12000000000-12.json').write_text(json.dumps(self.rec))
        with patch.object(status, 'SUPERVISOR_SHA', digest), \
             patch.object(status, 'process_start', return_value=10), \
             patch.object(status, 'running_pids', side_effect=[[12], [12]]), \
             patch.object(status, 'verified_services', return_value={18001, 18004}):
            result = status.get_gpu_roles(self.root, sup, self.root, lambda:self.gpus)
        self.assertEqual(len(result[0]), 5)
        self.assertEqual(result[1], {18001, 18004})
        with patch.object(status, 'SUPERVISOR_SHA', digest), \
             patch.object(status, 'process_start', return_value=10), \
             patch.object(status, 'running_pids', side_effect=[[12], []]):
            self.assertIsNone(status.get_gpu_roles(self.root, sup, self.root, lambda:self.gpus))
        with patch.object(status, 'SUPERVISOR_SHA', 'a'*64):
            self.assertIsNone(status.get_gpu_roles(self.root, sup, self.root, lambda:self.gpus))

    def test_device_query_error_has_no_assumed_roles(self):
        sup = self.root/'supervisor.py'
        sup.write_text('synthetic supervisor')
        digest = hashlib.sha256(sup.read_bytes()).hexdigest()
        self.rec['coordinator_sha256'] = digest
        d = self.root/'FIVEGPU_EXECUTIONS'
        d.mkdir()
        (d/'12000000000-12.json').write_text(json.dumps(self.rec))
        def fail():
            raise OSError('private error text')
        with patch.object(status, 'SUPERVISOR_SHA', digest), \
             patch.object(status, 'process_start', return_value=10), \
             patch.object(status, 'running_pids', return_value=[12]):
            self.assertIsNone(status.get_gpu_roles(self.root, sup, self.root, fail))

    def test_table_is_chinese_and_never_shows_raw_ids(self):
        c = {k: 0 for k in status.COUNTERS}
        c.update(generation_complete=382, grading_complete=380, unknown=347, generation_partial=1)
        roles = status.gpu_roles(self.rec, self.gpus)
        with patch.object(status, 'running_pids', return_value=[12]), \
             patch.object(status, 'wait_phase', return_value={'state':'BULK_RUNNING'}), \
             patch.object(status, 'get_gpu_roles', return_value=(roles, {18001,18004})), \
             patch.object(status, 'snapshot', return_value={k:c for k in ('easy','medium','hard')}):
            output = status.render(self.root, Path('/run/supervisor.py'))
        self.assertIn('控制器：运行中', output)
        self.assertIn('批量评测中', output)
        self.assertIn('已核验的配置显卡数：5', output)
        self.assertIn('| GPU 5 | 学生：主服务 | 运行中 |', output)
        self.assertIn('| GPU 7 | 学生：第二服务 | 未确认 |', output)
        self.assertIn('382 / 499', output)
        self.assertNotIn('GPU-a', output)
        self.assertNotIn('18001', output)

    def test_unknown_and_stopped_never_shows_assigned_roles(self):
        c = {k: 0 for k in status.COUNTERS}
        with patch.object(status, 'running_pids', return_value=[]), \
             patch.object(status, 'wait_phase', return_value={'state':'BULK_RUNNING'}), \
             patch.object(status, 'snapshot', return_value={k:c for k in ('easy','medium','hard')}):
            output = status.render(self.root, Path('/run/supervisor.py'))
        self.assertIn('控制器：未运行', output)
        self.assertIn('已核验的配置显卡数：未知', output)
        self.assertIn('可能滞后', output)
        self.assertNotIn('GPU 0', output)


if __name__ == '__main__': unittest.main()
