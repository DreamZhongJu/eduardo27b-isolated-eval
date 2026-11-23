#!/usr/bin/env python3
"""Unit tests for Pro 5000 author-494 evaluation tools (synthetic data only)."""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRED_DIR = HERE / "paired_gain"
for p in (HERE, PAIRED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_paired_csv as mpc  # noqa: E402
import paired_gain as pg  # noqa: E402
import summarize_author494_pro5000 as s  # noqa: E402
import wait_gpus_author494_pro5000 as wg  # noqa: E402


def _find_author_repo() -> Path:
    candidates = []
    if os.environ.get("EDUARDO_REPO"):
        candidates.append(Path(os.environ["EDUARDO_REPO"]))
    candidates.extend([
        HERE.parents[2] / "EduardoRL_A组源码_20261117",
        Path("/workspace/A_group_repro_20261117/source/EduardoRL"),
    ])
    for cand in candidates:
        if (cand / "eval" / "run_env_test_eval.py").is_file():
            return cand
    raise RuntimeError("Could not locate upstream EduardoRL repo; set EDUARDO_REPO")


AUTHOR_REPO = _find_author_repo()
if str(AUTHOR_REPO) not in sys.path:
    sys.path.insert(0, str(AUTHOR_REPO))

from eval.run_env_test_eval import compute_summary as author_compute_summary  # noqa: E402
from eduardo.reward_function import _normalized_delta as author_normalized_delta  # noqa: E402


def _make_record(
    row_index: int,
    *,
    pre_same: float = 0.25,
    pre_transfer: float = 0.25,
    post_same: float = 0.50,
    post_transfer: float = 0.375,
    leaked: int = 0,
    teaching_rejected: int = 0,
    no_leak_reasoning: str = "[Vote 1: OK] ok1 | [Vote 2: OK] ok2 | [Vote 3: OK] ok3",
    teaching_reasoning: str = "[Vote 1: OK] ok1 | [Vote 2: OK] ok2 | [Vote 3: OK] ok3",
    pedagogy: float | None = 3.5,
    correctness: float | None = 4.0,
    blanked_rate: float = 0.1,
    end_reason: str = "end_of_conversation",
    eoc_used: bool = True,
    thinking_tokens: list[int] | None = None,
    n_student_calls_failed: int = 0,
    assume_thinking: bool = True,
) -> dict:
    dt = author_normalized_delta(post_transfer, pre_transfer)
    ds = author_normalized_delta(post_same, pre_same)
    return {
        "row_index": row_index,
        "sample_index": 0,
        "problem": f"SYNTHETIC_SECRET_PROBLEM_{row_index}",
        "answer": f"SYNTHETIC_SECRET_ANSWER_{row_index}",
        "messages": [{"role": "teacher", "content": f"SYNTHETIC_SECRET_DIALOGUE_{row_index}"}],
        "pre_solve_rate": pre_same,
        "pre_solve_rate_transfer": pre_transfer,
        "post_solve_transfer": post_transfer,
        "post_solve_same": post_same,
        "delta_transfer": dt,
        "delta_same": ds,
        "delta_transfer_raw": post_transfer - pre_transfer,
        "delta_same_raw": post_same - pre_same,
        "delta_transfer_gated": 0.0 if leaked else dt,
        "delta_same_gated": 0.0 if leaked else ds,
        "transfer_gap": ds - dt,
        "leaked": leaked,
        "teaching_rejected": teaching_rejected,
        "judge_reasoning": {
            "no_leak": no_leak_reasoning,
            "teaching": teaching_reasoning,
        },
        "eval_judge_pedagogy": pedagogy,
        "eval_judge_correctness": correctness,
        "eval_judge_ok": int(pedagogy is not None and correctness is not None),
        "blanked_teacher_turn_rate": blanked_rate,
        "end_reason": end_reason,
        "eoc_used": eoc_used,
        "num_teacher_turns": 2,
        "thinking_tokens": [300, 400] if thinking_tokens is None else thinking_tokens,
        "used_transfer_problem": 1,
        "n_student_calls_failed": n_student_calls_failed,
        "assume_thinking": assume_thinking,
    }


def _make_doc(records: list[dict], *, tag: str = "synthetic", num_attempts: int = 16) -> dict:
    return {
        "model": "/synthetic/model",
        "tag": tag,
        "judge_model": "agnes-3.0-flash",
        "args": {"judge_votes": 3, "num_student_attempts": num_attempts},
        "metrics": author_compute_summary(records),
        "conversations": records,
    }


class TestAuthor494Pro5000(unittest.TestCase):
    def test_make_paired_csv_alignment_pre_mismatch_and_delta_verification(self):
        base_recs = [
            _make_record(101, pre_same=0.25, pre_transfer=0.125, post_same=0.375, post_transfer=0.25),
            _make_record(202, pre_same=0.50, pre_transfer=0.25, post_same=0.50, post_transfer=0.125),
            _make_record(303, pre_same=0.125, pre_transfer=0.0, post_same=0.25, post_transfer=0.25),
        ]
        ag_recs = [
            _make_record(303, pre_same=0.125, pre_transfer=0.0, post_same=0.625, post_transfer=0.50),
            _make_record(101, pre_same=0.25, pre_transfer=0.125, post_same=0.75, post_transfer=0.625),
            _make_record(202, pre_same=0.50, pre_transfer=0.25, post_same=0.875, post_transfer=0.50),
        ]
        base_doc = _make_doc(base_recs, tag="base")
        ag_doc = _make_doc(ag_recs, tag="a_group")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "paired.csv"
            info = mpc.build_and_verify_paired_csv(
                base_doc=base_doc, a_group_doc=ag_doc, out_csv=out_csv
            )
            self.assertEqual(info["n_problems"], 3)
            self.assertEqual(info["n_csv_rows"], 12)
            for chk in info["delta_checks"].values():
                self.assertLess(chk["abs_diff"], 1e-9)

            with out_csv.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # Check exact row_index alignment and 16-sample post counts.
            r101_ag_t = [
                r for r in rows
                if r["unit_id"] == "101" and r["condition"] == "a_group" and r["group"] == "author494_transfer"
            ][0]
            self.assertEqual(r101_ag_t["post_correct"], "10")  # 0.625 * 16 = 10
            self.assertEqual(r101_ag_t["post_total"], "16")
            self.assertEqual(r101_ag_t["post_unknown"], "0")

            # Mismatched pre_solve_rate_transfer must raise and NOT expose row_index.
            bad_ag = copy.deepcopy(ag_doc)
            bad_ag["conversations"][1]["pre_solve_rate_transfer"] = 0.50  # row 101 mismatch (was 0.125)
            bad_ag["conversations"][1]["delta_transfer"] = author_normalized_delta(0.625, 0.50)
            bad_ag["conversations"][1]["delta_transfer_raw"] = 0.625 - 0.50
            bad_ag["conversations"][1]["delta_transfer_gated"] = bad_ag["conversations"][1]["delta_transfer"]
            bad_ag["conversations"][1]["transfer_gap"] = (
                bad_ag["conversations"][1]["delta_same"] - bad_ag["conversations"][1]["delta_transfer"]
            )
            bad_ag["metrics"] = author_compute_summary(bad_ag["conversations"])
            bad_csv = Path(tmpdir) / "bad_pre.csv"
            with self.assertRaises(ValueError) as ctx:
                mpc.build_and_verify_paired_csv(base_doc=base_doc, a_group_doc=bad_ag, out_csv=bad_csv)
            self.assertNotIn("101", str(ctx.exception))
            self.assertFalse(bad_csv.exists())

            # Tampered JSON metrics (mismatch with CSV recomputed delta) must raise.
            tampered_ag = copy.deepcopy(ag_doc)
            tampered_ag["metrics"]["delta_transfer"] = 0.9999
            tampered_csv = Path(tmpdir) / "tampered.csv"
            with self.assertRaises(ValueError):
                mpc.build_and_verify_paired_csv(base_doc=base_doc, a_group_doc=tampered_ag, out_csv=tampered_csv)
            self.assertFalse(tampered_csv.exists())

    def test_verdict_a_and_c_exact_boundaries(self):
        # Verdict A against Paper Eduardo-4B: dt=0.25 [0.21, 0.29], ds=0.37 [0.33, 0.42]
        v_ok, _ = s.judge_repro_verdict(0.29, (0.21, 0.35), 0.33, (0.28, 0.33))
        self.assertEqual(v_ok, "对上")  # diff exactly 0.04, interval touching at 0.33
        v_mid, _ = s.judge_repro_verdict(0.33, (0.28, 0.38), 0.37, (0.33, 0.42))
        self.assertEqual(v_mid, "部分对上")  # diff exactly 0.08, overlapping
        v_mid2, _ = s.judge_repro_verdict(0.2901, (0.21, 0.35), 0.37, (0.33, 0.42))
        self.assertEqual(v_mid2, "部分对上")
        v_bad_diff, _ = s.judge_repro_verdict(0.3301, (0.28, 0.38), 0.37, (0.33, 0.42))
        self.assertEqual(v_bad_diff, "对不上")  # diff > 0.08
        v_bad_ci, _ = s.judge_repro_verdict(0.25, (0.2901, 0.35), 0.37, (0.33, 0.42))
        self.assertEqual(v_bad_ci, "对不上")  # CI disjoint

        # Verdict C against A100 base T-018: dt=0.1247 [0.0888, 0.1596], ds=0.1769 [0.1377, 0.2158]
        c_ok, _ = s.judge_cross_machine_verdict(0.1647, (0.1596, 0.20), 0.1369, (0.10, 0.1377))
        self.assertEqual(c_ok, "一致")  # diff exactly 0.04, intervals touching
        c_bad_diff, _ = s.judge_cross_machine_verdict(0.1648, (0.12, 0.20), 0.1769, (0.1377, 0.2158))
        self.assertEqual(c_bad_diff, "不一致")
        c_bad_ci, _ = s.judge_cross_machine_verdict(0.1247, (0.1597, 0.20), 0.1769, (0.1377, 0.2158))
        self.assertEqual(c_bad_ci, "不一致")

    def test_verdict_b_boundaries(self):
        self.assertEqual(s.judge_paired_verdict(0.001, 0.10), "A 组更好")
        self.assertEqual(s.judge_paired_verdict(0.0, 0.10), "看不出差别")
        self.assertEqual(s.judge_paired_verdict(-0.02, 0.05), "看不出差别")
        self.assertEqual(s.judge_paired_verdict(-0.10, -0.001), "更差")

    def test_full_comparison_output_contains_no_sensitive_strings(self):
        base_recs = []
        ag_recs = []
        for i in range(12):
            rid = 8800 + i
            base_recs.append(
                _make_record(
                    rid,
                    pre_same=0.25,
                    pre_transfer=0.25,
                    post_same=0.3125,
                    post_transfer=0.3125,
                    no_leak_reasoning=f"[Vote 1: OK] UNIQUE_BASE_NL_{rid} | [Vote 2: OK] ok | [Vote 3: OK] ok",
                    teaching_reasoning=f"[Vote 1: OK] UNIQUE_BASE_TEACH_{rid} | [Vote 2: OK] ok | [Vote 3: OK] ok",
                )
            )
            ag_recs.append(
                _make_record(
                    rid,
                    pre_same=0.25,
                    pre_transfer=0.25,
                    post_same=0.625,
                    post_transfer=0.5625,
                    no_leak_reasoning=f"[Vote 1: OK] UNIQUE_AG_NL_{rid} | [Vote 2: OK] ok | [Vote 3: OK] ok",
                    teaching_reasoning=f"[Vote 1: OK] UNIQUE_AG_TEACH_{rid} | [Vote 2: OK] ok | [Vote 3: OK] ok",
                )
            )
        base_doc = _make_doc(base_recs, tag="base_pro5000")
        ag_doc = _make_doc(ag_recs, tag="a_group_step100")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "paired.csv"
            mpc.build_and_verify_paired_csv(base_doc=base_doc, a_group_doc=ag_doc, out_csv=csv_path)
            comp = s.build_full_comparison(
                a_group_doc=ag_doc, base_doc=base_doc, paired_csv_path=csv_path, seed=42, draws=200
            )
            self.assertEqual(comp["verdict_b"]["verdict"], "A 组更好")
            rendered = s.format_full_comparison(comp) + "\n" + json.dumps(comp, ensure_ascii=False)
            for i in range(12):
                rid = 8800 + i
                for forbidden in (
                    str(rid),
                    f"SYNTHETIC_SECRET_PROBLEM_{rid}",
                    f"SYNTHETIC_SECRET_ANSWER_{rid}",
                    f"SYNTHETIC_SECRET_DIALOGUE_{rid}",
                    f"UNIQUE_BASE_NL_{rid}",
                    f"UNIQUE_AG_TEACH_{rid}",
                ):
                    self.assertNotIn(forbidden, rendered)

    def test_pro5000_gpu_waiter_and_safe_log_filters(self):
        # Pro 5000 73415 MiB GPUs; GPU 6 has idle sensor quirk (util=100, used=0, apps=[]).
        # Should pick 0,1,2,3 (util=0) before GPU 6, and use GPU 6 if needed.
        csv_8 = "\n".join([
            "0, GPU-0, 72800, 73415, 0",
            "1, GPU-1, 72800, 73415, 0",
            "2, GPU-2, 72800, 73415, 0",
            "3, GPU-3, 10000, 73415, 95",
            "6, GPU-6, 73300, 73415, 100",
        ])
        snap = wg.parse_gpu_csv(csv_8, "GPU-3, 111, 60000\n")
        chosen = wg.select_confirmed_gpus(snap, snap, num_gpus=4, min_free_mib=70000)
        self.assertEqual(chosen, [0, 1, 2, 6])

        # Verify WARNING / unquoted exception message never leaks.
        fake_log = "\n".join([
            "2026-11-23 11:00:00,100 [INFO] eduardo.transfer_eval: Thinking mode AUTO: HAS <think>",
            "2026-11-23 11:01:00,200 [WARNING] eduardo.metrics.judge_quality: Scoring RAW_JUDGE_TEXT",
            "Traceback (most recent call last):",
            '  File "/data/EduardoRL/eval/run_env_test_eval.py", line 900, in main',
            "    x = SECRET_CODE_LINE",
            "RuntimeError: UNQUOTED_SECRET_PROBLEM_IN_EXCEPTION",
        ])
        excerpt = s.format_eval_log_excerpt(fake_log)
        fail_sum = s.extract_safe_failure_summary(fake_log)
        self.assertNotIn("RAW_JUDGE_TEXT", excerpt)
        self.assertNotIn("RAW_JUDGE_TEXT", fail_sum)
        self.assertNotIn("SECRET_CODE_LINE", fail_sum)
        self.assertNotIn("UNQUOTED_SECRET_PROBLEM_IN_EXCEPTION", fail_sum)
        self.assertEqual(
            fail_sum,
            'File "/data/EduardoRL/eval/run_env_test_eval.py", line 900, in main\nRuntimeError',
        )


if __name__ == "__main__":
    unittest.main()
