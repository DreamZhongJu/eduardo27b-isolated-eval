"""Unit tests for memory_pro5000 and stage1_collect (CPU-only)."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memory_pro5000
import stage1_collect


class Stage1CollectTest(unittest.TestCase):
    def test_memory_pro5000_offload_decision(self):
        res = memory_pro5000.summarize_pro5000(73415.0)
        self.assertFalse(res['offload_false']['meets_8gib_reserve'])
        self.assertTrue(res['offload_false']['within_95_pct'])
        self.assertTrue(res['offload_true']['meets_8gib_reserve'])
        self.assertTrue(res['offload_true']['within_95_pct'])
        self.assertAlmostEqual(res['offload_true']['peak_gib'], res['offload_true']['update_gib'])

    def test_detect_and_select_idle_prefers_numa(self):
        gpu_csv = '\n'.join([
            f'{i}, GPU-{i}, NVIDIA RTX PRO 5000, {"68512" if i == 1 else "5"}, '
            f'{"4903" if i == 1 else "73410"}, 73415, {"96" if i == 1 else "0"}'
            for i in range(8)
        ])
        app_csv = 'GPU-1, 2677716, 68512\n'
        snap = stage1_collect.parse_gpu_snapshot(gpu_csv, app_csv)
        idle, busy = stage1_collect.detect_idle_gpus(snap, snap)
        self.assertEqual([g['index'] for g in busy], [1])
        self.assertEqual([g['index'] for g in idle], [0, 2, 3, 4, 5, 6, 7])
        chosen4, same_numa = stage1_collect.select_idle_group(idle, 4)
        self.assertEqual([g['index'] for g in chosen4], [4, 5, 6, 7])
        self.assertTrue(same_numa)

    def test_parse_vllm_kv_logs(self):
        sample = (
            'INFO ... Available KV cache memory: 11.42 GiB\n'
            'INFO ... GPU KV cache size: 46,752 tokens\n'
            'INFO ... Maximum concurrency for 32,768 tokens per request: 1.43x\n'
        )
        parsed = stage1_collect.parse_vllm_kv_from_logs(sample)
        self.assertEqual(parsed['log_kv_tokens'], 46752)
        self.assertAlmostEqual(parsed['log_kv_gib'], 11.42)
        self.assertAlmostEqual(parsed['log_max_concurrency'], 1.43)


if __name__ == '__main__':
    unittest.main()
