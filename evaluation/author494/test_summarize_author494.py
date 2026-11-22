#!/usr/bin/env python3
"""Unit tests for summarize_author494.py using synthetic data only."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import summarize_author494 as s  # noqa: E402
import agnes_ping_author494 as ping  # noqa: E402
import wait_gpus_author494 as wg  # noqa: E402
import run_control_author494 as rc_mod  # noqa: E402


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


def _make_record(
    row_index: int,
    *,
    delta_transfer: float = 0.12,
    delta_same: float = 0.14,
    leaked: int = 0,
    teaching_rejected: int = 0,
    no_leak_reasoning: str = "[Vote 1: OK] ok1 | [Vote 2: OK] ok2 | [Vote 3: OK] ok3",
    teaching_reasoning: str = "[Vote 1: OK] ok1 | [Vote 2: OK] ok2 | [Vote 3: OK] ok3",
    pedagogy: float | None = 2.5,
    correctness: float | None = 3.0,
    blanked_rate: float = 0.5,
    end_reason: str = "end_of_conversation",
    eoc_used: bool = True,
    thinking_tokens: list[int] | None = None,
    n_student_calls_failed: int = 0,
    assume_thinking: bool = True,
) -> dict:
    return {
        "row_index": row_index,
        "sample_index": 0,
        "problem": f"SYNTHETIC_SECRET_PROBLEM_{row_index}",
        "answer": f"SYNTHETIC_SECRET_ANSWER_{row_index}",
        "messages": [{"role": "teacher", "content": f"SYNTHETIC_SECRET_DIALOGUE_{row_index}"}],
        "pre_solve_rate": 0.25,
        "pre_solve_rate_transfer": 0.20,
        "post_solve_transfer": 0.30,
        "post_solve_same": 0.35,
        "delta_transfer": delta_transfer,
        "delta_same": delta_same,
        "delta_transfer_raw": 0.10,
        "delta_same_raw": 0.10,
        "delta_transfer_gated": 0.0 if leaked else delta_transfer,
        "delta_same_gated": 0.0 if leaked else delta_same,
        "transfer_gap": delta_same - delta_transfer,
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


def _make_doc(records: list[dict], judge_votes: int = 3) -> dict:
    return {
        "model": "/synthetic/Qwen3.5-4B",
        "tag": "synthetic_test",
        "judge_model": "agnes-3.0-flash",
        "args": {"judge_votes": judge_votes},
        "metrics": author_compute_summary(records),
        "conversations": records,
    }


class TestSummarizeAuthor494(unittest.TestCase):
    def test_recomputed_matches_author_compute_summary(self):
        records = [
            _make_record(0, delta_transfer=0.10, delta_same=0.12, leaked=0, blanked_rate=0.0, end_reason="end_of_conversation", eoc_used=True),
            _make_record(1, delta_transfer=0.20, delta_same=0.05, leaked=1, no_leak_reasoning="[Vote 1: REJECT] leaked | [Vote 2: REJECT] leaked | [Vote 3: OK] ok", blanked_rate=1.0, end_reason="token_budget", eoc_used=False, pedagogy=None, correctness=2.0),
            _make_record(2, delta_transfer=0.00, delta_same=0.18, leaked=0, teaching_rejected=1, teaching_reasoning="[Vote 1: REJECT] bad | [Vote 2: REJECT] bad | [Vote 3: OK] ok", blanked_rate=0.5, end_reason="max_teacher_turns", eoc_used=False, thinking_tokens=[600]),
        ]
        author_metrics = author_compute_summary(records)
        recomputed = s.recompute_author_summary(records)
        s.verify_author_summary_match(recomputed, author_metrics)

        doc = _make_doc(records)
        summary = s.build_summary(doc, seed=42, draws=200)
        self.assertAlmostEqual(summary["metrics"]["delta_transfer"]["mean"], author_metrics["delta_transfer"], places=12)
        self.assertAlmostEqual(summary["metrics"]["delta_same"]["mean"], author_metrics["delta_same"], places=12)
        self.assertAlmostEqual(summary["metrics"]["eval_judge_pedagogy"], author_metrics["eval_judge_pedagogy"], places=12)
        self.assertAlmostEqual(summary["metrics"]["eval_judge_correctness"], author_metrics["eval_judge_correctness"], places=12)

    def test_reproducible_bootstrap_same_seed(self):
        records = [_make_record(i, delta_transfer=0.05 + 0.01 * i, delta_same=0.06 + 0.01 * i, leaked=i % 2) for i in range(20)]
        doc = _make_doc(records)
        s1 = s.build_summary(doc, seed=42, draws=1000)
        s2 = s.build_summary(doc, seed=42, draws=1000)
        self.assertEqual(s1["metrics"]["delta_transfer"], s2["metrics"]["delta_transfer"])
        self.assertEqual(s1["metrics"]["delta_same"], s2["metrics"]["delta_same"])
        self.assertEqual(s1["metrics"]["leak_rate_author"], s2["metrics"]["leak_rate_author"])

    def test_verdict_three_tiers_and_exact_boundaries(self):
        # Exact diff == 0.04 and touching upper CI boundary [0.02, 0.06] vs [0.06, 0.13] -> 对上
        v, _ = s.judge_alignment_verdict(0.14, (0.06, 0.18), 0.15, (0.08, 0.20))
        self.assertEqual(v, "对上")
        v_touch, _ = s.judge_alignment_verdict(0.06, (0.02, 0.06), 0.15, (0.15, 0.20))
        self.assertEqual(v_touch, "对上")

        # Exact diff == 0.08 and overlapping -> 部分对上
        v_mid, _ = s.judge_alignment_verdict(0.18, (0.12, 0.22), 0.11, (0.08, 0.15))
        self.assertEqual(v_mid, "部分对上")
        v_mid2, _ = s.judge_alignment_verdict(0.141, (0.06, 0.18), 0.11, (0.08, 0.15))
        self.assertEqual(v_mid2, "部分对上")

        # Diff > 0.08 -> 对不上
        v_bad_diff, _ = s.judge_alignment_verdict(0.181, (0.12, 0.22), 0.11, (0.08, 0.15))
        self.assertEqual(v_bad_diff, "对不上")

        # Disjoint interval even with small point diff -> 对不上
        v_disjoint, _ = s.judge_alignment_verdict(0.12, (0.1301, 0.16), 0.11, (0.08, 0.15))
        self.assertEqual(v_disjoint, "对不上")

    def test_separate_tech_and_parse_failures_and_sensitivity_leak(self):
        # 100 conversations -> 600 votes total (3 no_leak + 3 teaching per conv).
        # Conv 0: 1 retry-failed vote in no_leak (and leaked=1 because majority REJECT with 2 REJECTs)
        # Conv 1: 2 parse-failed votes in teaching (not a retry failure)
        # Conv 2..19: leaked=1 legitimately; Conv 20..99: leaked=0
        records = []
        for i in range(100):
            no_leak_r = "[Vote 1: OK] ok | [Vote 2: OK] ok | [Vote 3: OK] ok"
            teach_r = "[Vote 1: OK] ok | [Vote 2: OK] ok | [Vote 3: OK] ok"
            leaked = 0
            teach_rej = 0
            if i == 0:
                leaked = 1
                no_leak_r = "[Vote 1: REJECT] Judge call failed after retries | [Vote 2: REJECT] leaked | [Vote 3: OK] ok"
            elif i == 1:
                teach_rej = 1
                teach_r = "[Vote 1: REJECT] Parse failure | [Vote 2: REJECT] Parse failure | [Vote 3: OK] ok"
            elif 2 <= i < 20:
                leaked = 1
                no_leak_r = "[Vote 1: REJECT] leaked | [Vote 2: REJECT] leaked | [Vote 3: OK] ok"
            records.append(
                _make_record(i, leaked=leaked, teaching_rejected=teach_rej, no_leak_reasoning=no_leak_r, teaching_reasoning=teach_r)
            )

        doc = _make_doc(records)
        summary = s.build_summary(doc, seed=42, draws=200)
        diag = summary["agnes_diagnostics"]
        self.assertEqual(diag["total_votes"], 600)
        self.assertEqual(diag["retry_failed_votes"], 1)
        self.assertEqual(diag["parse_failed_votes"], 2)
        self.assertEqual(diag["conversations_with_retry_fail"], 1)
        self.assertTrue(diag["judge_available"])  # 1/600 <= 1%

        # Author leak rate: 19/100 = 0.19; excluding conv 0 (tech fail): 18/99
        self.assertAlmostEqual(summary["metrics"]["leak_rate_author"]["mean"], 0.19, places=12)
        self.assertAlmostEqual(summary["metrics"]["leak_rate_excl_tech_fail"]["mean"], 18 / 99, places=12)

        # Now push retry-failed votes > 1% (e.g. 7 out of 600 = 1.17% -> judge unavailable)
        bad_records = copy.deepcopy(records)
        for i in range(2, 8):
            bad_records[i]["judge_reasoning"]["no_leak"] = (
                "[Vote 1: REJECT] Judge call failed after retries | [Vote 2: REJECT] leaked | [Vote 3: OK] ok"
            )
        bad_doc = _make_doc(bad_records)
        bad_summary = s.build_summary(bad_doc, seed=42, draws=200)
        self.assertEqual(bad_summary["agnes_diagnostics"]["retry_failed_votes"], 7)
        self.assertFalse(bad_summary["agnes_diagnostics"]["judge_available"])
        rendered = s.format_report(bad_summary)
        self.assertIn("不可用", rendered)

    def test_output_contains_no_sensitive_strings(self):
        records = [
            _make_record(
                777,
                no_leak_reasoning="[Vote 1: OK] UNIQUE_REASON_ALPHA | [Vote 2: OK] ok | [Vote 3: OK] ok",
                teaching_reasoning="[Vote 1: OK] UNIQUE_REASON_BETA | [Vote 2: OK] ok | [Vote 3: OK] ok",
            )
        ]
        doc = _make_doc(records)
        summary = s.build_summary(doc, seed=42, draws=100)
        out = s.format_report(summary) + "\n" + json.dumps(summary, ensure_ascii=False)
        for forbidden in (
            "SYNTHETIC_SECRET_PROBLEM_777",
            "SYNTHETIC_SECRET_ANSWER_777",
            "SYNTHETIC_SECRET_DIALOGUE_777",
            "UNIQUE_REASON_ALPHA",
            "UNIQUE_REASON_BETA",
            "777",
        ):
            self.assertNotIn(forbidden, out)

    def test_missing_fields_raise_and_stop(self):
        records = [_make_record(0)]
        doc = _make_doc(records)

        bad_metrics = copy.deepcopy(doc)
        del bad_metrics["metrics"]["delta_transfer"]
        with self.assertRaises(ValueError):
            s.build_summary(bad_metrics)

        bad_rec = copy.deepcopy(doc)
        del bad_rec["conversations"][0]["row_index"]
        with self.assertRaises(ValueError):
            s.build_summary(bad_rec)

        bad_mismatch = copy.deepcopy(doc)
        bad_mismatch["metrics"]["delta_transfer"] = 0.9999
        with self.assertRaises(ValueError):
            s.build_summary(bad_mismatch)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.json"
            p.write_text(json.dumps(bad_rec), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = ["summarize_author494.py", str(p)]
                rc = s.main()
            finally:
                sys.argv = old_argv
            self.assertEqual(rc, 1)

    def test_pipe_in_reasoning_and_ping_json_validator(self):
        votes = s.parse_judge_votes(
            "[Vote 1: OK] a | b | c | [Vote 2: REJECT] Parse failure | [Vote 3: REJECT] Judge call failed after retries",
            expected_votes=3,
        )
        self.assertEqual(votes, [("OK", "a | b | c"), ("REJECT", "Parse failure"), ("REJECT", "Judge call failed after retries")])
        ok, detail = ping._strict_judge_json_ok('```json\n{"reasoning": "fine", "decision": "OK"}\n```')
        self.assertTrue(ok)
        self.assertEqual(detail, "OK")
        bad, _ = ping._strict_judge_json_ok('{"reasoning": "missing decision"}')
        self.assertFalse(bad)

    def test_extract_safe_eval_log_lines_excludes_warning_and_error(self):
        fake_log = "\n".join([
            "2026-11-21 21:00:01,100 [INFO] eduardo.transfer_eval: Loaded 20 rows from /tmp/test.parquet (20 carry a transfer variant)",
            "2026-11-21 21:00:02,200 [INFO] eduardo.transfer_eval: Thinking mode AUTO: chat template has NO <think> prefill -> assume_thinking=False",
            "2026-11-21 21:00:03,300 [INFO] eduardo.transfer_eval: Thinking mode FORCED: assume_thinking=True (--assume-thinking yes)",
            "2026-11-21 21:00:04,350 [INFO] eduardo.transfer_eval: Round 1: teacher turn for 20 conversations",
            "2026-11-21 21:05:00,400 [INFO] eduardo.transfer_eval: Rollout of 20 conversations took 298.0s",
            "2026-11-21 21:05:01,500 [INFO] eduardo.transfer_eval: Scoring 20 conversations: 16 student final attempts x 2",
            "2026-11-21 21:06:00,600 [WARNING] eduardo.metrics.judge_quality: Failed to parse judge response, defaulting to REJECT: Scoring RAW_JUDGE_RESPONSE_SHOULD_NOT_APPEAR",
            "2026-11-21 21:06:01,700 [WARNING] eduardo.transfer_eval: Scoring RAW_TRANSFER_WARNING_SHOULD_NOT_APPEAR",
            "2026-11-21 21:06:02,800 [ERROR] eduardo.transfer_eval: 70% of teacher turns strip to NOTHING Thinking mode AUTO: RAW_ERROR_SHOULD_NOT_APPEAR",
            "2026-11-21 21:10:00,900 [INFO] eduardo.transfer_eval: Scoring took 300.0s",
            "=" * 72,
            "Model:                        /workspace/models/Qwen3.5-4B",
            "DELTA_TRANSFER:             +0.1000 +/- 0.0200",
            "=" * 72,
            "Results written to: /tmp/out/transfer_eval_test.json",
        ])
        safe_lines = s.extract_safe_eval_log_lines(fake_log)
        safe_status_lines = s.extract_safe_eval_log_lines(fake_log, include_rounds=True)
        excerpt = s.format_eval_log_excerpt(fake_log)
        for forbidden in (
            "RAW_JUDGE_RESPONSE_SHOULD_NOT_APPEAR",
            "RAW_TRANSFER_WARNING_SHOULD_NOT_APPEAR",
            "RAW_ERROR_SHOULD_NOT_APPEAR",
            "[WARNING]",
            "[ERROR]",
            "Loaded 20 rows",
        ):
            self.assertNotIn(forbidden, "\n".join(safe_lines))
            self.assertNotIn(forbidden, "\n".join(safe_status_lines))
            self.assertNotIn(forbidden, excerpt)
        self.assertEqual(len(safe_lines), 6)
        self.assertEqual(len(safe_status_lines), 7)
        self.assertNotIn("Round 1:", excerpt)
        self.assertIn("Thinking mode AUTO:", excerpt)
        self.assertIn("Thinking mode FORCED:", excerpt)
        self.assertIn("Rollout of 20 conversations took 298.0s", excerpt)
        self.assertIn("Scoring 20 conversations:", excerpt)
        self.assertIn("Scoring took 300.0s", excerpt)
        self.assertIn("Results written to: /tmp/out/transfer_eval_test.json", excerpt)
        self.assertIn("DELTA_TRANSFER:             +0.1000 +/- 0.0200", excerpt)

    def test_wait_for_four_gpus_rejects_busy_and_confirms_two_snapshots(self):
        busy_csv = "\n".join([
            "0, GPU-0, 35000, 81920, 70",
            "1, GPU-1, 80000, 81920, 0",
            "2, GPU-2, 80000, 81920, 0",
            "3, GPU-3, 80000, 81920, 0",
            "4, GPU-4, 80000, 81920, 15",  # util != 0 -> not eligible
        ])
        flaky_csv = "\n".join([
            "0, GPU-0, 80000, 81920, 0",
            "1, GPU-1, 80000, 81920, 0",
            "2, GPU-2, 80000, 81920, 0",
            "3, GPU-3, 80000, 81920, 0",
            "4, GPU-4, 80000, 81920, 0",
        ])
        # Second snapshot loses GPU-0 and GPU-1 to apps, so only 3 confirmed -> must keep waiting.
        apps_snap2 = "GPU-0, 12345, 5000\nGPU-1, 12346, 5000\n"
        ready_csv = "\n".join([
            "0, GPU-0, 35000, 81920, 50",
            "1, GPU-1, 80000, 81920, 0",
            "2, GPU-2, 80000, 81920, 0",
            "3, GPU-3, 80000, 81920, 0",
            "4, GPU-4, 80000, 81920, 0",
        ])
        snaps = iter([
            wg.parse_gpu_csv(busy_csv, ""),
            wg.parse_gpu_csv(flaky_csv, ""),
            wg.parse_gpu_csv(flaky_csv, apps_snap2),
            wg.parse_gpu_csv(ready_csv, ""),
            wg.parse_gpu_csv(ready_csv, ""),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            st_file = Path(tmpdir) / "wait_status.json"
            lg_file = Path(tmpdir) / "wait.log"
            tutor, students = wg.wait_for_four_gpus(
                status_path=st_file,
                log_path=lg_file,
                port=18961,
                snapshot_fn=lambda: next(snaps),
                port_fn=lambda _p: True,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(tutor, 1)
            self.assertEqual(students, [2, 3, 4])
            status_doc = json.loads(st_file.read_text(encoding="utf-8"))
            self.assertEqual(status_doc["state"], "ACQUIRED")
            self.assertEqual(status_doc["polls"], 3)

    def test_find_replacement_tutor_and_safe_failure_summary(self):
        csv = "\n".join([
            "0, GPU-0, 45000, 81920, 10",
            "1, GPU-1, 7000, 81920, 0",
            "4, GPU-4, 7000, 81920, 0",
            "5, GPU-5, 7000, 81920, 0",
            "6, GPU-6, 81050, 81920, 0",
        ])
        snap = wg.parse_gpu_csv(csv, "")
        rep = wg.find_replacement_tutor(snap, snap, exclude={1, 4, 5})
        self.assertEqual(rep, 6)

        fake_fail_log = "\n".join([
            "2026-11-22 01:49:00 [WARNING] eduardo.metrics.judge_quality: Parse failure RAW_SECRET_SHOULD_NOT_APPEAR",
            "Traceback (most recent call last):",
            '  File "/workspace/source/eval/run_env_test_eval.py", line 942, in main',
            "    secret_var = UNQUOTED_CODE_LINE_SHOULD_NOT_APPEAR",
            "ValueError: UNQUOTED_PROBLEM_TEXT_IN_EXCEPTION_MESSAGE x + y = 999",
        ])
        fail_summary = s.extract_safe_failure_summary(fake_fail_log)
        self.assertNotIn("RAW_SECRET_SHOULD_NOT_APPEAR", fail_summary)
        self.assertNotIn("UNQUOTED_CODE_LINE_SHOULD_NOT_APPEAR", fail_summary)
        self.assertNotIn("UNQUOTED_PROBLEM_TEXT_IN_EXCEPTION_MESSAGE", fail_summary)
        self.assertNotIn("x + y = 999", fail_summary)
        self.assertEqual(
            fail_summary,
            'File "/workspace/source/eval/run_env_test_eval.py", line 942, in main\nValueError',
        )

    def test_stage3_hake_g_cohens_d_correlations_and_control_metrics(self):
        # Hand-calculated 4-problem dataset:
        # pre_transfer  = [0.2, 0.4, 0.6, 0.8] -> mean 0.5
        # post_transfer = [0.4, 0.6, 0.8, 1.0] -> mean 0.7
        # Hake g (transfer) = (0.7 - 0.5) / (1 - 0.5) = 0.4
        # Cohen d (transfer) = (0.7 - 0.5) / sqrt(0.2/3) = sqrt(0.6)
        pre_t = [0.2, 0.4, 0.6, 0.8]
        post_t = [0.4, 0.6, 0.8, 1.0]
        self.assertAlmostEqual(s.compute_hake_g(pre_t, post_t), 0.4, places=12)
        self.assertAlmostEqual(s.compute_cohens_d(pre_t, post_t), math.sqrt(0.6), places=12)

        # Correlation helpers: exact +1, uncorrelated 0, and Spearman with ties.
        self.assertAlmostEqual(s.pearson_r([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]), 1.0, places=12)
        self.assertAlmostEqual(s.pearson_r([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]), 0.0, places=12)
        self.assertAlmostEqual(
            s.spearman_rho([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 30.0, 40.0]),
            math.sqrt(0.9),
            places=12,
        )

        # Difficulty tier exact boundaries at 1/3 and 2/3.
        self.assertEqual(s.classify_difficulty_tier(0.0), "hard")
        self.assertEqual(s.classify_difficulty_tier(1.0 / 3.0 - 1e-9), "hard")
        self.assertEqual(s.classify_difficulty_tier(1.0 / 3.0), "medium")
        self.assertEqual(s.classify_difficulty_tier(2.0 / 3.0 - 1e-9), "medium")
        self.assertEqual(s.classify_difficulty_tier(2.0 / 3.0), "easy")
        self.assertEqual(s.classify_difficulty_tier(1.0), "easy")

        # Full end-to-end Stage 2 + Stage 3 control summary + reproducibility + retry exclusion.
        ctl1_t = [0.25, 0.50, 0.50, 0.75]
        ctl2_t = [0.20, 1.0 / 3.0, 2.0 / 3.0, 0.80]  # 1 hard, 1 medium (==1/3), 2 easy (==2/3, 0.8)
        pre_s = [0.25, 0.25, 0.50, 0.75]
        post_s = [0.50, 0.50, 0.75, 0.75]
        ctl1_s = [0.25, 0.50, 0.50, 0.75]
        ctl2_s = [0.25, 0.50, 0.60, 0.80]

        records = []
        ctl_records = []
        for i in range(4):
            dt = s.normalized_delta(post_t[i], pre_t[i])
            ds = s.normalized_delta(post_s[i], pre_s[i])
            no_leak_r = (
                "[Vote 1: REJECT] Judge call failed after retries | [Vote 2: OK] ok2 | [Vote 3: OK] ok3"
                if i == 0
                else "[Vote 1: OK] ok1 | [Vote 2: OK] ok2 | [Vote 3: OK] ok3"
            )
            rec = _make_record(
                9000 + i,
                delta_transfer=dt,
                delta_same=ds,
                leaked=1 if i == 3 else 0,
                teaching_rejected=1 if i in (1, 2) else 0,
                no_leak_reasoning=no_leak_r,
                pedagogy=float(i + 1),
                correctness=float(i + 1),
            )
            rec["pre_solve_rate"] = pre_s[i]
            rec["pre_solve_rate_transfer"] = pre_t[i]
            rec["post_solve_same"] = post_s[i]
            rec["post_solve_transfer"] = post_t[i]
            rec["delta_transfer_raw"] = post_t[i] - pre_t[i]
            rec["delta_same_raw"] = post_s[i] - pre_s[i]
            rec["delta_transfer_gated"] = 0.0 if rec["leaked"] else dt
            rec["delta_same_gated"] = 0.0 if rec["leaked"] else ds
            rec["transfer_gap"] = ds - dt
            records.append(rec)
            ctl_records.append({
                "row_index": 9000 + i,
                "pre_solve_rate": pre_s[i],
                "pre_solve_rate_transfer": pre_t[i],
                "ctl1_same": ctl1_s[i],
                "ctl1_same_scored": 16,
                "ctl1_same_failed": 0,
                "ctl1_transfer": ctl1_t[i],
                "ctl1_transfer_scored": 16,
                "ctl1_transfer_failed": 0,
                "ctl2_same": ctl2_s[i],
                "ctl2_same_scored": 16,
                "ctl2_same_failed": 0,
                "ctl2_transfer": ctl2_t[i],
                "ctl2_transfer_scored": 16,
                "ctl2_transfer_failed": 0,
            })

        doc = _make_doc(records)
        ctl_doc = rc_mod.build_control_doc(
            ctl_records,
            tag="synthetic_ctl",
            student_model="synthetic-student",
            k=16,
            student_temperature=0.6,
            student_max_tokens=1024,
            elapsed_seconds=1.0,
            timestamp="20261122_000000",
        )
        sum1 = s.build_summary(doc, seed=42, draws=300, control_doc=ctl_doc)
        sum2 = s.build_summary(doc, seed=42, draws=300, control_doc=ctl_doc)
        self.assertEqual(sum1, sum2)

        self.assertAlmostEqual(sum1["metrics"]["r2_hake_g_transfer"]["mean"], 0.4, places=12)
        self.assertAlmostEqual(sum1["metrics"]["r3_cohens_d_transfer"]["mean"], math.sqrt(0.6), places=12)

        # Conversation 0 (row 9000) has a retry fail vote -> excluded from process-outcome correlations (3 remain).
        poc = sum1["process_outcome_correlations"]
        self.assertEqual(poc["n_clean_retry_conversations"], 3)
        self.assertEqual(poc["pedagogy_vs_delta_transfer"]["n_conversations"], 3)

        ca = sum1["control_analysis"]
        exp_r4_r1_t = sum(s.normalized_delta(c, p) for p, c in zip(pre_t, ctl1_t)) / 4.0
        exp_re_r1_t = sum(s.normalized_delta(po, c) for c, po in zip(ctl1_t, post_t)) / 4.0
        self.assertAlmostEqual(ca["r4_untutored_baseline"]["transfer"]["r1"]["mean"], exp_r4_r1_t, places=12)
        self.assertAlmostEqual(ca["rebaselined_gain_ctl1_to_post"]["transfer"]["r1"]["mean"], exp_re_r1_t, places=12)
        self.assertAlmostEqual(
            ca["test_retest_reliability"]["transfer_pre_ctl1"]["mean"],
            s.pearson_r(pre_t, ctl1_t),
            places=12,
        )
        tiers = ca["difficulty_tiers_by_ctl2_transfer"]
        self.assertEqual(tiers["hard"]["n_problems"], 1)
        self.assertEqual(tiers["medium"]["n_problems"], 1)
        self.assertEqual(tiers["easy"]["n_problems"], 2)

        rendered = s.format_report(sum1) + "\n" + json.dumps(sum1, ensure_ascii=False)
        for forbidden in ("9000", "9001", "9002", "9003", "SYNTHETIC_SECRET_PROBLEM", "SYNTHETIC_SECRET_ANSWER"):
            self.assertNotIn(forbidden, rendered)

        # Control fail rate > 1% should mark control_available=False and control_analysis=None.
        bad_ctl_records = copy.deepcopy(ctl_records)
        bad_ctl_records[0]["ctl1_same_scored"] = 10
        bad_ctl_records[0]["ctl1_same_failed"] = 6  # 6/256 = 2.34% > 1%
        bad_ctl_doc = rc_mod.build_control_doc(
            bad_ctl_records,
            tag="bad_ctl",
            student_model="synthetic-student",
            k=16,
            student_temperature=0.6,
            student_max_tokens=1024,
            elapsed_seconds=1.0,
            timestamp="20261122_000001",
        )
        self.assertTrue(bad_ctl_doc["metrics"]["exceeds_1pct"])
        bad_sum = s.build_summary(doc, seed=42, draws=100, control_doc=bad_ctl_doc)
        self.assertFalse(bad_sum["control_diagnostics"]["control_available"])
        self.assertIsNone(bad_sum["control_analysis"])
        self.assertIn("不可用", s.format_report(bad_sum))

    def test_run_control_calls_author_solve_rate_with_fake_client(self):
        calls = []

        class _FakeCompletions:
            async def create(self, *, model, messages, temperature, max_tokens, extra_body):
                content_prompt = messages[0]["content"]
                calls.append({
                    "model": model,
                    "prompt": content_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_body": extra_body,
                })
                if "FAKE_FAIL_ONCE" in content_prompt and len(calls) == 1:
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r"\boxed{0}"))])
                ans = "SECRET_ANS_ORIG" if "SECRET_ORIG_PROB" in content_prompt else "SECRET_ANS_TRANS"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=f"Reasoning...\n\\boxed{{{ans}}}"))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        rows = [{
            "row_index": 555,
            "problem": "SECRET_ORIG_PROB FAKE_FAIL_ONCE",
            "answer": "SECRET_ANS_ORIG",
            "transfer_problem": "SECRET_TRANS_PROB",
            "transfer_answer": "SECRET_ANS_TRANS",
            "pre_solve_rate": 0.5,
            "pre_solve_rate_transfer": 0.25,
        }]
        records = asyncio.run(
            rc_mod.run_control_rounds(
                rows,
                client=fake_client,
                solve_rate_fn=author_solve_rate,
                student_model="Llama-3.1-8B-Instruct-test",
                student_temperature=0.6,
                student_max_tokens=1024,
                k=4,
                student_concurrency=8,
                row_concurrency=4,
            )
        )
        # 1 row x 2 rounds (ctl1, ctl2) x 2 problems (orig, trans) x k=4 = 16 calls
        self.assertEqual(len(calls), 16)
        self.assertTrue(all(c["temperature"] == 0.6 and c["max_tokens"] == 1024 for c in calls))
        self.assertTrue(all(c["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} for c in calls))
        self.assertEqual(records[0]["ctl1_same"], 0.75)  # first call returned \boxed{0}
        self.assertEqual(records[0]["ctl1_transfer"], 1.0)
        self.assertEqual(records[0]["ctl2_same"], 1.0)
        self.assertEqual(records[0]["ctl2_transfer"], 1.0)

        doc = rc_mod.build_control_doc(
            records,
            tag="fake",
            student_model="Llama-3.1-8B-Instruct-test",
            k=4,
            student_temperature=0.6,
            student_max_tokens=1024,
            elapsed_seconds=0.5,
            timestamp="20261122_000002",
        )
        safe_dump = json.dumps(doc["metrics"], ensure_ascii=False)
        for forbidden in ("SECRET_ORIG_PROB", "SECRET_ANS_ORIG", "SECRET_TRANS_PROB", "SECRET_ANS_TRANS", "555"):
            self.assertNotIn(forbidden, safe_dump)

    def test_wait_for_three_gpus_for_control(self):
        ready_csv = "\n".join([
            "0, GPU-0, 35000, 81920, 50",
            "2, GPU-2, 80000, 81920, 0",
            "5, GPU-5, 80000, 81920, 0",
            "7, GPU-7, 80000, 81920, 0",
        ])
        snaps = iter([wg.parse_gpu_csv(ready_csv, ""), wg.parse_gpu_csv(ready_csv, "")])
        with tempfile.TemporaryDirectory() as tmpdir:
            chosen = wg.wait_for_gpus(
                status_path=Path(tmpdir) / "wait_status.json",
                log_path=Path(tmpdir) / "wait.log",
                port=18961,
                num_gpus=3,
                snapshot_fn=lambda: next(snaps),
                port_fn=lambda _p: True,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(chosen, [2, 5, 7])


if __name__ == "__main__":
    unittest.main()
