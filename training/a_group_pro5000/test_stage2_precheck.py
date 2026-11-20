"""Unit tests for Stage 2 pre-launch GPU & disk check."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stage2_precheck as p


class Stage2PrecheckTest(unittest.TestCase):
    def _clean_snap(self):
        gpu_csv = '\n'.join(
            f'{i}, GPU-{i}, NVIDIA RTX PRO 5000, 5, 73410, 73415, {100 if i == 6 else 0}, 2617, 35, 30.0'
            for i in range(8)
        )
        return p.parse_gpu_snapshot(gpu_csv, '')

    def test_core_gpus_stop_on_app_or_high_util(self):
        snap = self._clean_snap()
        ok, issues = p.core_gpus_clean(snap, snap)
        self.assertTrue(ok)
        self.assertEqual(issues, [])

        busy_snap = dict(snap)
        busy_snap[4] = dict(snap[4], apps=[{'pid': 999, 'used_mib': 10000}], used_mib=10000, util=50)
        ok, issues = p.core_gpus_clean(snap, busy_snap)
        self.assertFalse(ok)
        self.assertIn('GPU 4', issues[0])

    def test_decide_layout_7_vs_8(self):
        rc, v, extra = p.decide_layout(True, True, [], 150.0, 145.0)
        self.assertEqual((rc, v), (0, 'PRECHECK_PASS_8GPU'))
        self.assertGreaterEqual(extra['gpu6_to_gpu0_ratio'], 0.9)

        rc, v, _ = p.decide_layout(True, True, [], 150.0, 120.0)
        self.assertEqual((rc, v), (0, 'PRECHECK_PASS_7GPU'))

        rc, v, _ = p.decide_layout(True, True, [{'pid': 1, 'used_mib': 100}], 150.0, 150.0)
        self.assertEqual((rc, v), (0, 'PRECHECK_PASS_7GPU'))

        rc, v, _ = p.decide_layout(True, False, [], 150.0, 150.0)
        self.assertEqual((rc, v), (3, 'PRECHECK_STOP_DISK_LOW'))


if __name__ == '__main__':
    unittest.main()
