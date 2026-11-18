import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from b_numbers import disk_lines, gate_count, metrics


class TestBNumbers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.checkpoints = [self.work / f"global_step_{i}" for i in (1, 2)]
        for i, ckpt in enumerate(self.checkpoints, 1):
            ckpt.mkdir()
            path = ckpt / "data.pt"
            path.write_bytes(b"ok")
            os.utime(path, (i * 100, i * 100))
        self.logdir = self.work / "eduardo_conversations"
        self.logdir.mkdir()
        self.rows = [dict(uuid=f"fake-{i}", gated=int(i not in (0, 14)), note="ok",
                          judge_pass=int(i in (0, 14)), problem=f"source-{i // 4}",
                          ts=10 + i + (100 if i >= 8 else 0)) for i in range(16)]

    def write(self):
        for i in range(2):
            path = self.logdir / f"conversations_{i}.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in reversed(self.rows[i::2])))

    def test_gate_with_original_checkpoint_time_and_two_groups(self):
        self.write()
        self.assertEqual(gate_count(self.work, self.checkpoints, {1, 2}), "1 / 8")

    def test_gate_empty_boundaries_and_duplicate_uuid(self):
        self.rows[1]["ts"] = 100
        self.write()
        self.assertEqual(gate_count(self.work, self.checkpoints, {1, 2}), "未知（0–2 / 8）")
        self.rows[1]["uuid"] = self.rows[0]["uuid"]
        self.write()
        self.assertEqual(gate_count(self.work, self.checkpoints, {1, 2}), "未知")

    def test_gate_disjoint_source_check(self):
        for row in self.rows:
            if row["problem"] == "source-2":
                row["problem"] = "source-1"
        self.write()
        self.assertEqual(gate_count(self.work, self.checkpoints, {1, 2}), "未知（0–2 / 8）")

    def test_metrics_step_binding_wrapped_scalar_and_conflicts(self):
        path = self.work / "trainer.log"
        path.write_text("step:1 - training/global_step:1 - actor/grad_norm:np.float64(0.18) - actor/pg_loss:-1.2e-7 - actor/lr:1e-5\n"
                        "step:2 - training/global_step:2 - actor/grad_norm:0.61\n"
                        "step:2 - training/global_step:1 - actor/grad_norm:100\n"
                        "step:1 - training/global_step:1 - actor/pg_loss:-2.2e-7\n")
        self.assertEqual(metrics(path), ({1, 2}, {"actor/grad_norm": 0.18, "actor/lr": 1e-5}))

    def test_usage_existing_and_missing_directory(self):
        dirs = self.checkpoints + [self.logdir, self.work / "export"]
        dirs[-1].mkdir()
        for missing in (False, True):
            if missing:
                dirs[-1].rmdir()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                disk_lines(self.work, dirs)
            self.assertEqual("未知" in output.getvalue(), missing)
            self.assertIn("df -h", output.getvalue())


if __name__ == "__main__":
    unittest.main()
