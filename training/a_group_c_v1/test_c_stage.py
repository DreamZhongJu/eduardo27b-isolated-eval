import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import c_stage as c


class StageCTest(unittest.TestCase):
    def test_child_exit_and_order_with_real_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            cb = c.Callbacks(work, 's', 'local-placeholder', work, work, lambda: {})
            # Run the exact guard plumbing with CPU stand-ins. The service flags
            # here are ignored by the stand-in applications.
            with patch.object(c.pilot, 'PYTHON', Path(sys.executable)):
                try:
                    for role in ('judge','student'):
                        cb.spawn(role, [sys.executable,'-c','import time; time.sleep(60)'], cb.base)
                    proc = cb.spawn('trainer',[sys.executable,'-c','import time; time.sleep(0.7)'],cb.base)
                    self.assertEqual(proc.wait(timeout=8),0)
                    c.pilot.wait_for_training(cb,Mock(scan=lambda:False),'C')
                finally:
                    cb.stop_started()
            for role in ('judge','student','trainer'):
                self.assertTrue(c.confirm_exit(work,role,role=='trainer'))
            rows = [json.loads((work / f'exit-{role}.json').read_text()) for role in ('trainer','student','judge')]
            self.assertEqual([r['exit_code'] for r in rows], [0, -15, -15])
            self.assertEqual(sorted(r['observed_ns'] for r in rows), [r['observed_ns'] for r in rows])

    def test_fallback_skips_merge(self):
        cb = Mock()
        with self.assertRaises(c.CFailure):
            c.finish(cb, Mock(scan=lambda: True), Mock())
        cb.spawn.assert_not_called()

    def test_automatic_merge_and_aggregate_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            ck = work / 'verl_ckpts' / c.c_config.EXPERIMENT
            ck.mkdir(parents=True)
            (ck / 'latest_checkpointed_iteration.txt').write_text('1')
            cb = Mock(work=work, base={})
            cb.spawn.return_value.wait.return_value = 0
            update = Mock()
            with patch.object(c,'confirm_exit',return_value=True), patch.object(c.ckpt,'inspect_input',return_value=({},5)), \
                 patch.object(c.ckpt,'inspect_export',return_value=(1,724)), patch.object(c.c_metrics,'report',return_value=0):
                c.finish(cb, Mock(scan=lambda: False), update)
            args=cb.spawn.call_args.args
            self.assertEqual(args[0],'cpu_merge')
            self.assertIn('verl.model_merger',args[1])
            self.assertEqual(args[2]['CUDA_VISIBLE_DEVICES'],'')
            result=json.loads((work/'c_acceptance.json').read_text())
            self.assertEqual((result['merge_verified'],result['metrics_exit']), (True,0))
            self.assertEqual(update.call_args.args[0], '待用户确认')

if __name__ == '__main__':
    unittest.main()
