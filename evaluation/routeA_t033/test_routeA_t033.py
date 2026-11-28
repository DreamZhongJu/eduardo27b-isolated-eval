#!/usr/bin/env python3
"""Unit tests for T-033 Route A Step-100 evaluation package (no GPU or API calls)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import common  # noqa: E402
import check_overlap_t033  # noqa: E402
import precision_compare_selfbuilt  # noqa: E402
import preflight_t033  # noqa: E402
import run_eval_routea_author494  # noqa: E402
import run_post_eval_routea  # noqa: E402
import run_t033_dev_sim  # noqa: E402


class TestT033Package(unittest.TestCase):
    def test_frozen_modules_sha256(self) -> None:
        res = preflight_t033.verify_frozen_modules(REPO_ROOT)
        self.assertEqual(res["agnes_judge.py"], common.AGNES_JUDGE_SHA256)
        self.assertEqual(res["hybrid_grader.py"], common.HYBRID_GRADER_SHA256)
        self.assertEqual(res["paired_gain.py"], common.PAIRED_GAIN_SHA256)

    def test_step100_manifest_lock_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            export_dir = root / "step_100"
            export_dir.mkdir()
            lock = {}
            lines = []
            for fname in common.ROUTEA_STEP100_LOCK:
                fp = export_dir / fname
                fp.write_text(f"dummy-{fname}\n", encoding="utf-8")
                dig = common.sha256_file(fp)
                lock[fname] = dig
                lines.append(f"{dig}  {fname}\n")
            sha_file = root / "export_step100_sha256.txt"
            sha_file.write_text("".join(lines), encoding="utf-8")

            verified = common.verify_dir_against_lock(export_dir, lock, label="test", manifest_file=sha_file)
            self.assertEqual(len(verified), 7)

            (export_dir / "model.safetensors").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                common.verify_dir_against_lock(export_dir, lock, label="test", manifest_file=sha_file)

    def test_gpu_selection_handles_gpu6_util_quirk(self) -> None:
        gpu_csv = "\n".join([
            "0, UUID-0, 10, 73405, 73415, 0",
            "6, UUID-6, 12, 73403, 73415, 100",
            "2, UUID-2, 45000, 28415, 73415, 50",
        ])
        app_csv = "UUID-2, 9999, 45000\n"
        s1 = common.parse_gpu_snapshot(gpu_csv, app_csv)
        s2 = common.parse_gpu_snapshot(gpu_csv, app_csv)
        free = common.select_free_gpus(s1, s2)
        self.assertEqual(free, [0, 6])

    def test_overlap_checker_passes_disjoint_and_blocks_overlap(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()

            def _row_pq(sid: str, prob: str, var_prob: str) -> dict:
                return {
                    "extra_info": {
                        "source_id": sid,
                        "problem": prob,
                        "variants": json.dumps([{"variant_id": "v1", "problem": var_prob}]),
                    }
                }

            train_p = root / "train.parquet"
            val_p = root / "val.parquet"
            a494_p = root / "test.parquet"
            pd.DataFrame([_row_pq("tr1", "Train P1", "Train V1")]).to_parquet(train_p)
            pd.DataFrame([_row_pq("va1", "Val P1", "Val V1")]).to_parquet(val_p)
            pd.DataFrame([{"extra_info": {"problem": "Author P1", "transfer_problem": "Author T1"}}]).to_parquet(a494_p)

            # Build minimal valid t021_bundle
            sb_entry = {
                "difficulty": "easy",
                "tier_zh": "简单",
                "source_id": "sb1",
                "source_key": common.source_key("sb1"),
                "row_index": 0,
                "row": {
                    "source_id": "sb1",
                    "problem": "Selfbuilt P1",
                    "answer": "1",
                    "transfer_problem": "Selfbuilt T1",
                    "transfer_answer": "2",
                },
            }
            inputs_p = bundle_dir / "selfbuilt_eval_inputs_1489.jsonl"
            inputs_p.write_text(json.dumps(sb_entry, ensure_ascii=False) + "\n", encoding="utf-8")

            import csv as csv_mod
            counts_p = bundle_dir / "ref27b_counts.csv"
            cols = ["difficulty", "source_key", "source_id", "row_index"]
            row_c = {"difficulty": "easy", "source_key": sb_entry["source_key"], "source_id": "sb1", "row_index": "0"}
            for g in ("agnes", "ar"):
                for t in ("orig", "trans"):
                    for ph in ("pre", "control", "post"):
                        cols.extend([f"{g}_{t}_{ph}_correct", f"{g}_{t}_{ph}_incorrect", f"{g}_{t}_{ph}_unknown"])
                        row_c[f"{g}_{t}_{ph}_correct"] = "8"
                        row_c[f"{g}_{t}_{ph}_incorrect"] = "8"
                        row_c[f"{g}_{t}_{ph}_unknown"] = "0"
            with counts_p.open("w", encoding="utf-8", newline="") as f:
                w = csv_mod.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerow(row_c)

            man = {
                "schema": "t021_a100_export_v1",
                "eligible_count": 1,
                "skipped_count": 0,
                "files_sha256": {
                    inputs_p.name: common.sha256_file(inputs_p),
                    counts_p.name: common.sha256_file(counts_p),
                },
            }
            (bundle_dir / "manifest_export.json").write_text(json.dumps(man), encoding="utf-8")

            # Use require_exact_totals=False by patching load_and_verify_bundle call via monkeypatch
            orig_loader = check_overlap_t033.load_and_verify_bundle
            try:
                check_overlap_t033.load_and_verify_bundle = lambda d, require_exact_totals=True: orig_loader(
                    d, require_exact_totals=False
                )
                rep = check_overlap_t033.compute_overlap_report(
                    train_parquet=train_p,
                    val_parquet=val_p,
                    author494_parquet=a494_p,
                    selfbuilt_bundle_dir=bundle_dir,
                    require_exact_counts=False,
                )
                self.assertTrue(rep["passed"])
                self.assertEqual(rep["overlap_total"], 0)

                # Now inject text overlap (variant vs author494 with different whitespace/case)
                pd.DataFrame([_row_pq("tr1", "Train P1", "  author   t1 ")]).to_parquet(train_p)
                rep_bad = check_overlap_t033.compute_overlap_report(
                    train_parquet=train_p,
                    val_parquet=val_p,
                    author494_parquet=a494_p,
                    selfbuilt_bundle_dir=bundle_dir,
                    require_exact_counts=False,
                )
                self.assertFalse(rep_bad["passed"])
                self.assertGreater(rep_bad["overlap_total"], 0)
                self.assertNotIn("Train P1", json.dumps(rep_bad))
            finally:
                check_overlap_t033.load_and_verify_bundle = orig_loader

    def test_precision_compare_math_and_privacy(self) -> None:
        entries = []
        fp16_rates = {}
        bf16_rates = {}
        for d in ("easy", "medium", "hard"):
            for i in range(4):
                sid = f"{d}_{i}"
                entries.append(
                    {
                        "difficulty": d,
                        "source_id": sid,
                        "source_key": common.source_key(sid),
                        "ref_counts_ar": {
                            "原题": {"pre": {"correct": 8}, "control": {"correct": 8}},
                            "迁移题": {"pre": {"correct": 6}, "control": {"correct": 6}},
                        },
                    }
                )
                fp16_rates[(d, sid)] = {"orig_pre_rate": 0.5, "trans_pre_rate": 0.375}
                bf16_rates[(d, sid)] = {"orig_pre_rate": 0.5625, "trans_pre_rate": 0.4375}

        with tempfile.TemporaryDirectory() as td:
            csv_p = Path(td) / "prec.csv"
            res = precision_compare_selfbuilt.analyze_precision_compare(
                entries,
                fp16_rates,
                bf16_rates,
                fp16_stu_fail=0,
                bf16_stu_fail=0,
                draws=200,
                seed=42,
                out_csv=csv_p,
            )
            self.assertTrue(res["passed"])
            self.assertAlmostEqual(res["overall_all_32"]["diff_bf16_minus_fp16"], 0.0625, places=6)
            self.assertTrue(csv_p.is_file())

    def test_author494_wrapper_allow_abbrev_false(self) -> None:
        w_args, pass_argv = run_eval_routea_author494.strip_t033_wrapper_args(
            ["--max-agnes-window", "12", "--student-model", "Llama-3.1-8B", "--student-model-dir", "/tmp/stu"]
        )
        self.assertEqual(w_args.max_agnes_window, 12)
        self.assertEqual(w_args.student_model_dir, "/tmp/stu")
        self.assertIn("--student-model", pass_argv)
        self.assertIn("Llama-3.1-8B", pass_argv)

    def test_selfbuilt_smoke16_selection(self) -> None:
        eligible = [
            {"difficulty": d, "source_id": f"{d}_{i}", "source_key": common.source_key(f"{d}_{i}")}
            for d in ("easy", "medium", "hard")
            for i in range(10)
        ]
        chosen = run_post_eval_routea.select_smoke_entries({"eligible": eligible}, smoke_total=16, seed=42)
        self.assertEqual(len(chosen), 16)
        counts = {d: sum(1 for e in chosen if e["difficulty"] == d) for d in ("easy", "medium", "hard")}
        self.assertEqual(counts, {"easy": 6, "medium": 5, "hard": 5})

    def test_dev_sim_reward_and_unconditional_theta(self) -> None:
        class _DummyTok:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return list(range(max(1, len(text) // 4)))

        transcript = [
            {"role": "user", "content": "Solve x+1=2"},
            {"role": "assistant", "content": "<think>short thought</think>Try subtracting 1."},
        ]
        climb_res = {
            "climbs": [
                {
                    "climb_index": i,
                    "steps": [
                        {"b": -1.0, "correct": True, "tech_failed": False},
                        {"b": 0.5, "correct": i < 8, "tech_failed": False},
                    ],
                }
                for i in range(16)
            ]
        }
        m_ok = run_t033_dev_sim.compute_dialogue_routea_metrics(
            transcript=transcript, climb_res=climb_res, no_leak="OK", teaching="OK", tokenizer=_DummyTok()
        )
        self.assertEqual(m_ok["gated"], 0)
        self.assertIsNotNone(m_ok["k16"]["theta_hat"])
        self.assertGreater(m_ok["k16"]["reward"], 0.0)

        m_gated = run_t033_dev_sim.compute_dialogue_routea_metrics(
            transcript=transcript, climb_res=climb_res, no_leak="REJECT", teaching="OK", tokenizer=_DummyTok()
        )
        self.assertEqual(m_gated["gated"], 1)
        self.assertAlmostEqual(m_gated["k16"]["reward"], -0.2, places=9)
        # theta_hat must STILL be computed even when gated!
        self.assertEqual(m_gated["k16"]["theta_hat"], m_ok["k16"]["theta_hat"])

    def test_wait_for_free_gpus_stdout_is_clean(self) -> None:
        import contextlib
        import io
        from unittest import mock

        gpu_csv = "\n".join([
            "0, UUID-0, 10, 73405, 73415, 0",
            "1, UUID-1, 10, 73405, 73415, 0",
            "2, UUID-2, 10, 73405, 73415, 0",
            "3, UUID-3, 10, 73405, 73415, 0",
        ])
        snap = common.parse_gpu_snapshot(gpu_csv, "")
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with (
            mock.patch.object(common, "snapshot_gpus", return_value=snap),
            mock.patch.object(common.time, "sleep", return_value=None),
            contextlib.redirect_stdout(out_buf),
            contextlib.redirect_stderr(err_buf),
        ):
            picked = common.wait_for_free_gpus(4, poll_seconds=1)
        self.assertEqual(picked, [0, 1, 2, 3])
        self.assertEqual(out_buf.getvalue(), "")
        self.assertIn("[t033_gpu]", err_buf.getvalue())

    def test_shell_scripts_bash_syntax(self) -> None:
        for sh in sorted(HERE.glob("*.sh")):
            proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed on {sh.name}: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
