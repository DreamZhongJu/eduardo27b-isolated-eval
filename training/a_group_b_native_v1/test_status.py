import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import status


class FinishStatusTests(unittest.TestCase):
    def test_real_global_steps_and_timings_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work/'trainer.log').write_text(
                '\x1b[36m(TaskRunner pid=8)\x1b[0m step:1 - training/global_step:1 - timing_s/step:336.12\n'
                '(TaskRunner pid=8) step:2 - timing_s/step:449.75 - training/global_step:2\n'
                'training/global_step:900 - example text\n'
                'step:5 - training/global_step:4 - timing_s/step:123\n'
                '(TaskRunner pid=8) \'Final validation metrics: None\'\n'
            )
            self.assertEqual(status.finish_metrics(work), ({1:336.12, 2:449.75}, True))

    def test_finish_only_preserves_missing_exit_codes_and_sanitizes_log(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work/'trainer.log').write_text('do not emit this line\n')
            (work/'orchestration.json').write_text(json.dumps({
                'phase':'失败', 'reason':'本任务服务异常退出',
                'role_processes':{'trainer':False, 'student':True, 'judge':True},
                'role_exit_codes':{'student':None, 'judge':137},
            }))
            marker = work/'verl_ckpts/b-functional/latest_checkpointed_iteration.txt'
            marker.parent.mkdir(parents=True)
            marker.write_text('2\n')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status.report(work, finish_only=True)
            text = out.getvalue()
            self.assertIn('| 训练器 | 已退出 | 无记录 |', text)
            self.assertIn('| 学生 | 仍在运行 | 无记录 |', text)
            self.assertIn('| 最近检查点步号文件 | 2 |', text)
            self.assertIn('| CPU 合并启动痕迹 | 未发现 |', text)
            self.assertNotIn('do not emit this line', text)

    def test_absent_and_empty_logs_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            self.assertEqual(status.finish_metrics(work), (None, None))
            (work/'trainer.log').touch()
            self.assertEqual(status.finish_metrics(work), ({}, False))


if __name__ == '__main__':
    unittest.main()
