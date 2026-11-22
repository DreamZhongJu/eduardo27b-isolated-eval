#!/usr/bin/env python3
"""Unit tests for summarize_author494.py using synthetic data only."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import summarize_author494 as s  # noqa: E402
import agnes_ping_author494 as ping  # noqa: E402
import wait_gpus_author494 as wg  # noqa: E402


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
            'ValueError: Free memory on device cuda:0 (45.49/79.15 GiB) on startup is less than desired',
        ])
        fail_summary = s.extract_safe_failure_summary(fake_fail_log)
        self.assertNotIn("RAW_SECRET_SHOULD_NOT_APPEAR", fail_summary)
        self.assertIn("run_env_test_eval.py", fail_summary)
        self.assertIn("ValueError: Free memory on device cuda:0", fail_summary)


if __name__ == "__main__":
    unittest.main()
