import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import c_status


class TestStatus(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def write(self, name, obj):
        (self.work / name).write_text(json.dumps(obj))

    def test_exit_observation_order_guard_and_child_disagree(self):
        for role, obs, code, req in (("trainer", 1000000000, 0, False), ("student", 3000000000, -15, True),
                                     ("judge", 2000000000, 0, True), ("cpu_merge", 3000000000, 0, False)):
            self.write(f"exit-{role}.json", dict(exit_code=code, observed_ns=obs, requested=req))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            c_status.exit_report(self.work, {"role_exit_codes": {"student": 0}})
        text = out.getvalue()
        self.assertIn("| 3 | 学生 | 未知 | 0 | -15 |", text)
        self.assertIn("| 3 | CPU 合并 |", text)
        self.assertLess(text.index("1 | 训练器"), text.index("2 | 裁判"))
        self.assertLess(text.index("2 | 裁判"), text.index("3 | 学生"))

    def test_report_unknown_peak_role_and_missing_report(self):
        self.write("orchestration.json", {"bindings": {"tutor": [0, 3], "student": [7], "judge": [1]}})
        self.write("gpu_peaks.json", {"0": None, "1": 72123, "3": 123, "7": None})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            c_status.report(self.work)
        text = out.getvalue()
        self.assertIn("| 1 | 裁判 | 72123 |", text)
        self.assertIn("| 7 | 学生 | 未知 |", text)
        self.assertIn("| 自动 CPU 合并与权重读取 | 未知 |", text)
        self.assertNotIn("B 收尾", text)

    def test_controller_identity_and_recycled_pid(self):
        folder = self.work / "proc/350"
        folder.mkdir(parents=True)
        work = self.work / "pilots/c_paper_v1"
        args = [b"python", os.fsencode(Path(c_status.__file__).resolve().with_name("c_stage.py")),
                b"--run", b"--name", b"c_paper_v1"]
        (folder / "cmdline").write_bytes(b"\0".join(args))
        (folder / "stat").write_text("350 (python) R 1 1 1")
        self.assertEqual(c_status.controller_state(work, 350, folder.parent), "运行中")
        (folder / "cmdline").write_bytes(b"\0".join(args[:-1] + [b"different"]))
        self.assertEqual(c_status.controller_state(work, 350, folder.parent), "未发现匹配控制器")

    def test_complete_and_historical_default(self):
        self.write("c_acceptance.json", dict(merge_verified=True, merge_shards=1, merge_keys=724, metrics_exit=0))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            c_status.report(self.work)
        self.assertIn("| 已核验张量键数 | 724 |", out.getvalue())
        self.assertIn("| C 统计验收退出码 | 0 |", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            c_status.base.report(self.work)
        self.assertIn("B 收尾", out.getvalue())


if __name__ == "__main__":
    unittest.main()
