#!/usr/bin/env python3
"""Unit tests for Pro 5000 author-494 evaluation v2 tools (synthetic data only)."""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PAIRED_DIR = HERE / "paired_gain"
for p in (HERE, PAIRED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import compare_stack as cs  # noqa: E402
import export_t018_table as ext  # noqa: E402
import make_paired_csv as mpc  # noqa: E402
import run_control_author494_pro5000 as rc_mod  # noqa: E402
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
from eduardo.create_transfer_dataset import _solve_rate as author_solve_rate  # noqa: E402
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


def _make_stage3_doc(records: list[dict]) -> dict:
    n = len(records)
    return {
        "benchmark": "transfer_untutored_control",
        "tag": "synthetic_ctl",
        "k": 16,
        "num_problems": n,
        "metrics": {
            "ctl1_same": sum(r["ctl1_same"] for r in records) / n,
            "ctl1_transfer": sum(r["ctl1_transfer"] for r in records) / n,
            "ctl2_same": sum(r["ctl2_same"] for r in records) / n,
            "ctl2_transfer": sum(r["ctl2_transfer"] for r in records) / n,
            "calls_expected": n * 64,
            "calls_scored": n * 64,
            "calls_failed": 0,
            "fail_rate": 0.0,
            "exceeds_1pct": False,
        },
        "records": records,
    }


class TestAuthor494Pro5000V2(unittest.TestCase):
    def test_export_t018_table_and_make_paired_csv_alignment_and_deltas(self):
        row_ids = [101, 202, 303, 404]
        base_recs = [
            _make_record(101, pre_same=0.25, pre_transfer=0.125, post_same=0.375, post_transfer=0.25, leaked=1),
            _make_record(202, pre_same=0.50, pre_transfer=0.25, post_same=0.50, post_transfer=0.125, blanked_rate=0.0, end_reason="token_budget", eoc_used=False, pedagogy=None),
            _make_record(303, pre_same=0.125, pre_transfer=0.0, post_same=0.25, post_transfer=0.25, teaching_rejected=1),
            _make_record(404, pre_same=0.25, pre_transfer=0.25, post_same=0.375, post_transfer=0.375),
        ]
        s3_recs = [
            {
                "row_index": r["row_index"],
                "pre_solve_rate": r["pre_solve_rate"],
                "pre_solve_rate_transfer": r["pre_solve_rate_transfer"],
                "ctl1_same": 0.25,
                "ctl1_same_scored": 16,
                "ctl1_same_failed": 0,
                "ctl1_transfer": 0.125,
                "ctl1_transfer_scored": 16,
                "ctl1_transfer_failed": 0,
                "ctl2_same": 0.25,
                "ctl2_same_scored": 16,
                "ctl2_same_failed": 0,
                "ctl2_transfer": 0.125,
                "ctl2_transfer_scored": 16,
                "ctl2_transfer_failed": 0,
            }
            for r in base_recs
        ]
        ag_recs = [
            _make_record(404, pre_same=0.25, pre_transfer=0.25, post_same=0.75, post_transfer=0.625),
            _make_record(101, pre_same=0.25, pre_transfer=0.125, post_same=0.75, post_transfer=0.625),
            _make_record(303, pre_same=0.125, pre_transfer=0.0, post_same=0.625, post_transfer=0.50),
            _make_record(202, pre_same=0.50, pre_transfer=0.25, post_same=0.875, post_transfer=0.50),
        ]

        s2_doc = _make_doc(base_recs, tag="base_t018")
        s3_doc = _make_stage3_doc(s3_recs)
        ag_doc = _make_doc(ag_recs, tag="a_group_step100")

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            s2_p = td / "s2.json"
            s3_p = td / "s3.json"
            pq_p = td / "test.parquet"
            s2_p.write_text(json.dumps(s2_doc), encoding="utf-8")
            s3_p.write_text(json.dumps(s3_doc), encoding="utf-8")
            pq_p.write_bytes(b"SYNTHETIC_PARQUET_BYTES")

            bundle_dir = td / "bundle"
            exp_res = ext.export_t018_bundle(
                stage2_json=s2_p,
                stage3_json=s3_p,
                test_parquet=pq_p,
                out_dir=bundle_dir,
                expected_rows=len(row_ids),
            )
            csv_table = bundle_dir / "t018_per_problem.csv"
            manifest_p = bundle_dir / "t018_manifest.json"
            self.assertTrue(csv_table.is_file())
            self.assertTrue((bundle_dir / "t018_table_bundle.tar.gz").is_file())

            # Table must contain NO problem text, answers, or dialogues.
            table_txt = csv_table.read_text(encoding="utf-8")
            for forbidden in ("SYNTHETIC_SECRET_PROBLEM", "SYNTHETIC_SECRET_ANSWER", "SYNTHETIC_SECRET_DIALOGUE"):
                self.assertNotIn(forbidden, table_txt)

            # Recomputed metrics from table match Stage 2 / Stage 3 within < 1e-9.
            rc = exp_res["manifest"]["recomputed_checks"]
            self.assertLess(abs(rc["delta_transfer"] - s2_doc["metrics"]["delta_transfer"]), 1e-9)
            self.assertLess(abs(rc["delta_same"] - s2_doc["metrics"]["delta_same"]), 1e-9)
            self.assertLess(abs(rc["delta_transfer_gated"] - s2_doc["metrics"]["delta_transfer_gated"]), 1e-9)
            self.assertLess(abs(rc["ctl1_same_mean"] - s3_doc["metrics"]["ctl1_same"]), 1e-9)

            # Build paired CSV from t018_per_problem.csv + a_group JSON.
            t018_rows = ext.load_t018_table(csv_table)
            out_csv = td / "paired.csv"
            info = mpc.build_and_verify_paired_csv(
                t018_by_row=t018_rows,
                a_group_doc=ag_doc,
                out_csv=out_csv,
                t018_manifest=json.loads(manifest_p.read_text(encoding="utf-8")),
            )
            self.assertEqual(info["n_problems"], 4)
            for chk in info["delta_checks"].values():
                self.assertLess(chk["abs_diff"], 1e-9)

            # Pre-solve mismatch between T-018 table and A-group must raise without exposing row_index.
            bad_ag = copy.deepcopy(ag_doc)
            bad_ag["conversations"][1]["pre_solve_rate_transfer"] = 0.50  # row 101 mismatch
            bad_ag["conversations"][1]["delta_transfer"] = author_normalized_delta(0.625, 0.50)
            bad_ag["conversations"][1]["delta_transfer_raw"] = 0.625 - 0.50
            bad_ag["conversations"][1]["delta_transfer_gated"] = bad_ag["conversations"][1]["delta_transfer"]
            bad_ag["conversations"][1]["transfer_gap"] = (
                bad_ag["conversations"][1]["delta_same"] - bad_ag["conversations"][1]["delta_transfer"]
            )
            bad_ag["metrics"] = author_compute_summary(bad_ag["conversations"])
            bad_csv = td / "bad_pre.csv"
            with self.assertRaises(ValueError) as ctx:
                mpc.build_and_verify_paired_csv(t018_by_row=t018_rows, a_group_doc=bad_ag, out_csv=bad_csv)
            self.assertNotIn("101", str(ctx.exception))
            self.assertFalse(bad_csv.exists())

            # Full summary report must contain no sensitive strings.
            full = s.build_full_comparison(
                a_group_doc=ag_doc,
                t018_by_row=s._load_t018_csv_rows(csv_table),
                paired_csv_path=out_csv,
                seed=42,
                draws=200,
            )
            rendered = s.format_full_comparison(full) + "\n" + json.dumps(full, ensure_ascii=False)
            for rid in row_ids:
                for forbidden in (
                    str(rid),
                    f"SYNTHETIC_SECRET_PROBLEM_{rid}",
                    f"SYNTHETIC_SECRET_ANSWER_{rid}",
                    f"SYNTHETIC_SECRET_DIALOGUE_{rid}",
                ):
                    self.assertNotIn(forbidden, rendered)

    def test_verdict_a_and_compare_stack_boundaries(self):
        # Verdict A against Paper Eduardo-4B: dt=0.25 [0.21, 0.29], ds=0.37 [0.33, 0.42]
        v_ok, _ = s.judge_repro_verdict(0.29, (0.21, 0.35), 0.33, (0.28, 0.33))
        self.assertEqual(v_ok, "对上")
        v_mid, _ = s.judge_repro_verdict(0.33, (0.28, 0.38), 0.37, (0.33, 0.42))
        self.assertEqual(v_mid, "部分对上")
        v_bad_diff, _ = s.judge_repro_verdict(0.3301, (0.28, 0.38), 0.37, (0.33, 0.42))
        self.assertEqual(v_bad_diff, "对不上")
        v_bad_ci, _ = s.judge_repro_verdict(0.25, (0.2901, 0.35), 0.37, (0.33, 0.42))
        self.assertEqual(v_bad_ci, "对不上")

        # Verdict B boundaries
        self.assertEqual(s.judge_paired_verdict(0.001, 0.10), "A 组更好")
        self.assertEqual(s.judge_paired_verdict(0.0, 0.10), "看不出差别")
        self.assertEqual(s.judge_paired_verdict(-0.10, -0.001), "更差")

        # compare_stack exact boundaries: mean diff <= 0.02, r_transfer >= 0.834, r_same >= 0.817,
        # smoke empty rate within 0.7014 +/- 0.15, assume_thinking=True, calls_failed=0.
        t018_rows = {}
        ctl_recs = []
        vals = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875]
        for i, v in enumerate(vals):
            t018_rows[i] = {
                "row_index": i,
                "pre_solve_rate": 0.25,
                "pre_solve_rate_transfer": 0.25,
                "ctl1_same_correct": int(round(v * 16)),
                "ctl1_same_total": 16,
                "ctl1_transfer_correct": int(round(v * 16)),
                "ctl1_transfer_total": 16,
            }
            # Shift by +0.02 exactly (r = 1.0, |diff| = 0.02)
            ctl_recs.append({
                "row_index": i,
                "pre_solve_rate": 0.25,
                "pre_solve_rate_transfer": 0.25,
                "ctl_same": v + 0.02,
                "ctl_same_scored": 16,
                "ctl_same_failed": 0,
                "ctl_transfer": v + 0.02,
                "ctl_transfer_scored": 16,
                "ctl_transfer_failed": 0,
            })
        ctl_doc = rc_mod.build_control_doc(
            ctl_recs, tag="ctl_ok", student_model="s", k=16,
            student_temperature=0.6, student_max_tokens=1024, elapsed_seconds=1.0, timestamp="t"
        )
        smoke_ok_doc = _make_doc([_make_record(0, blanked_rate=0.8514, assume_thinking=True)], tag="sm_ok")
        res_ok = cs.evaluate_stack_equivalence(
            pro5000_ctl_doc=ctl_doc, t018_by_row=t018_rows, pro5000_smoke_doc=smoke_ok_doc
        )
        self.assertTrue(res_ok["overall_passed"])
        self.assertEqual(res_ok["verdict"], "通过")

        # Mean diff 0.0201 -> student side fails
        bad_mean_doc = copy.deepcopy(ctl_doc)
        for r in bad_mean_doc["records"]:
            r["ctl_same"] += 0.001
        res_bad_mean = cs.evaluate_stack_equivalence(
            pro5000_ctl_doc=bad_mean_doc, t018_by_row=t018_rows, pro5000_smoke_doc=smoke_ok_doc
        )
        self.assertFalse(res_bad_mean["student_side"]["passed"])
        self.assertFalse(res_bad_mean["overall_passed"])

        # Smoke empty rate outside +/- 0.15 -> tutor side fails
        smoke_bad_empty = _make_doc([_make_record(0, blanked_rate=0.8515, assume_thinking=True)], tag="sm_bad")
        res_bad_smoke = cs.evaluate_stack_equivalence(
            pro5000_ctl_doc=ctl_doc, t018_by_row=t018_rows, pro5000_smoke_doc=smoke_bad_empty
        )
        self.assertFalse(res_bad_smoke["tutor_side"]["passed"])
        self.assertFalse(res_bad_smoke["overall_passed"])

        # assume_thinking=False -> tutor side fails
        smoke_bad_think = _make_doc([_make_record(0, blanked_rate=0.70, assume_thinking=False)], tag="sm_no_think")
        res_bad_think = cs.evaluate_stack_equivalence(
            pro5000_ctl_doc=ctl_doc, t018_by_row=t018_rows, pro5000_smoke_doc=smoke_bad_think
        )
        self.assertFalse(res_bad_think["tutor_side"]["passed"])

    def test_run_control_calls_author_solve_rate_and_pro5000_gpu_waiter(self):
        calls = []

        class _FakeCompletions:
            async def create(self, *, model, messages, temperature, max_tokens, extra_body):
                prompt = messages[0]["content"]
                calls.append({"temp": temperature, "max_tokens": max_tokens, "extra_body": extra_body})
                ans = "SECRET_ANS_ORIG" if "SECRET_ORIG_PROB" in prompt else "SECRET_ANS_TRANS"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=f"Steps...\n\\boxed{{{ans}}}"))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        rows = [{
            "row_index": 555,
            "problem": "SECRET_ORIG_PROB",
            "answer": "SECRET_ANS_ORIG",
            "transfer_problem": "SECRET_TRANS_PROB",
            "transfer_answer": "SECRET_ANS_TRANS",
            "pre_solve_rate": 0.5,
            "pre_solve_rate_transfer": 0.25,
        }]
        records = asyncio.run(
            rc_mod.run_control_round(
                rows,
                client=fake_client,
                solve_rate_fn=author_solve_rate,
                student_model="Llama-3.1-8B-Instruct-test",
                student_temperature=0.6,
                student_max_tokens=1024,
                k=4,
            )
        )
        self.assertEqual(len(calls), 8)  # 1 round x 2 problems x 4 attempts
        self.assertTrue(all(c["temp"] == 0.6 and c["max_tokens"] == 1024 for c in calls))
        self.assertTrue(all(c["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} for c in calls))
        self.assertEqual(records[0]["ctl_same"], 1.0)
        self.assertEqual(records[0]["ctl_transfer"], 1.0)

        # GPU 6 quirk check on Pro 5000
        csv_8 = "\n".join([
            "0, GPU-0, 72800, 73415, 0",
            "1, GPU-1, 72800, 73415, 0",
            "2, GPU-2, 72800, 73415, 0",
            "3, GPU-3, 10000, 73415, 95",
            "6, GPU-6, 73300, 73415, 100",
        ])
        snap = wg.parse_gpu_csv(csv_8, "GPU-3, 111, 60000\n")
        self.assertEqual(wg.select_confirmed_gpus(snap, snap, num_gpus=4, min_free_mib=70000), [0, 1, 2, 6])

        # Safe failure summary leak check
        fake_fail_log = "\n".join([
            "2026-11-23 11:01:00,200 [WARNING] eduardo.metrics.judge_quality: Scoring RAW_JUDGE_TEXT",
            "Traceback (most recent call last):",
            '  File "/data/EduardoRL/eval/run_env_test_eval.py", line 900, in main',
            "    x = SECRET_CODE_LINE",
            "RuntimeError: UNQUOTED_SECRET_PROBLEM_IN_EXCEPTION",
        ])
        fail_sum = s.extract_safe_failure_summary(fake_fail_log)
        self.assertNotIn("RAW_JUDGE_TEXT", fail_sum)
        self.assertNotIn("SECRET_CODE_LINE", fail_sum)
        self.assertNotIn("UNQUOTED_SECRET_PROBLEM_IN_EXCEPTION", fail_sum)


if __name__ == "__main__":
    unittest.main()
