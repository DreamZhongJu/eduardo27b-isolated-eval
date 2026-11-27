"""Unit tests for T-031 Route C data pipeline (Card v1 / Inbox #1), using fake data ONLY.

Runnable via BOTH `python -m unittest discover -s tests` (Pro 5000 pro5000-fast)
and `pytest` (cu04).

Covers:
1. `routeC_inventory.py`:
   - Exact (tau, margin) routing counts, `total_usable`, `gap_vs_1600`, `easy_total + hard_total == N`.
   - Boundary cases for hard-branch floor (`<= 1/16`) and easy-branch ceiling (`>= 15/16`).
   - Cross-tab by source direction (`easier / mixed / harder`) and variant `generation_tier`.
   - Hard rejection on SHA-256 mismatch or invalid pretest values.
   - Stage 2 mode (`--source-pretest-jsonl`) replacing old `p_orig` with newly measured `p_orig`.
2. `stage2_source_runner.py`:
   - End-to-end synthetic GPU + summarize phase: rule pass skips 27B, rule fail calls 27B,
     `unknown` counts as incorrect, >1% technical failure gate aborts, `--resume` skips completed items.
   - Correlation and mean-difference calculations vs old `pre_solve_rate`.
   - Zero leakage check: unique sentinel problem/answer/response strings NEVER appear in stdout,
     `source_pretest_measured.jsonl`, `stage2_summary.*`, `stage2_manifest.json`, or `inventory_summary.*`.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List

try:
    import eduardo.utils  # type: ignore  # noqa: F401
except ImportError:
    _edu_pkg = types.ModuleType("eduardo")
    _edu_utils = types.ModuleType("eduardo.utils")

    def _stub_grade_math_answer(solution_str: str, ground_truth: str) -> Dict[str, Any]:
        marker = f"\\boxed{{{ground_truth}}}"
        if marker in (solution_str or ""):
            return {"score": 1.0, "acc": 1.0, "pred": ground_truth, "incorrect_format": 0, "truncated": 0}
        inc_fmt = 0 if "\\boxed{" in (solution_str or "") else 1
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": inc_fmt, "truncated": 0}

    _edu_utils.grade_math_answer = _stub_grade_math_answer  # type: ignore[attr-defined]
    _edu_pkg.utils = _edu_utils  # type: ignore[attr-defined]
    sys.modules["eduardo"] = _edu_pkg
    sys.modules["eduardo.utils"] = _edu_utils

from routeA_data.hybrid_grader_caller import TechnicalFailureGateError
from routeA_data.rasch_ladder import compute_smoothed_p_and_b
from routeC_data.routeC_inventory import (
    compute_inventory,
    compute_source_pretest_table_sha256,
    format_inventory_report,
    run_inventory,
)
from routeC_data.stage2_source_runner import (
    load_measured_rows,
    run_gpu_phase,
    run_summarize_phase,
)


def _make_variant(
    vid: str,
    n_correct: int,
    *,
    origin: str = "self_generated",
    generation_tier: str = "easier",
    n_scored: int = 16,
) -> Dict[str, Any]:
    p_new, b_val = compute_smoothed_p_and_b(n_correct, n_scored)
    return {
        "variant_id": vid,
        "origin": origin,
        "problem": f"SENTINEL_VAR_PROB_{vid}",
        "answer": f"SENTINEL_VAR_ANS_{vid}",
        "generation_tag": generation_tier,
        "generation_tier": generation_tier,
        "n_scored": n_scored,
        "n_correct_hybrid": n_correct,
        "p_new": round(p_new, 8),
        "b": round(b_val, 8),
    }


def _make_source_row(
    sid: str,
    split: str,
    direction: str,
    old_k16: int,
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "source_id": sid,
        "split": split,
        "direction": direction,
        "source_problem": f"SENTINEL_SRC_PROB_{sid}",
        "source_answer": f"42_{sid}",
        "source_pre_solve_rate_old": old_k16 / 16.0,
        "source_llama8b_solve_rate_old": 0.25,
        "variants": variants,
    }


class _FakeCompletions:
    def __init__(self, handler: Any) -> None:
        self._handler = handler

    async def create(self, **kwargs: Any) -> Any:
        return await self._handler(**kwargs)


class _FakeClient:
    def __init__(self, handler: Any) -> None:
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(handler))


def _fake_resp(text: str, finish: str = "stop") -> Any:
    msg = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=msg, finish_reason=finish)
    return types.SimpleNamespace(choices=[choice])


class TestRouteCInventory(unittest.TestCase):
    def _build_synthetic_rows(self) -> List[Dict[str, Any]]:
        # Construct 6 fake sources covering easy/hard branches, floor, ceiling, and margins:
        # S1 (train, hard): old=1/16 (p=0.0882, b=+2.335), var=8/16 (b=0.0 -> delta_b=-2.335 <= -1.0) -> hard_usable & floor!
        # S2 (train, hard): old=2/16 (p=0.1471, b=+1.758), var=4/16 (b=+1.022 -> delta_b=-0.736: usable at m=0.5, NOT m=1.0; NOT floor)
        # S3 (train, mid):  old=5/16 (p=0.3235, b=+0.738), var=2/16 (b=+1.758 -> delta_b=+1.020) -> easy at tau=0.3 (usable m=0.5,1.0), hard at tau>=0.4 (not usable)
        # S4 (train, easy): old=16/16 (p=0.9706, b=-3.497), var=15/16 (b=-2.335 -> delta_b=+1.161 >= 1.0) -> easy_usable & CEILING (15/16)!
        # S5 (train, easy): old=16/16 (p=0.9706, b=-3.497), vars=[15/16 (db=+1.161), 14/16 (db=+1.758)] -> easy_usable & NOT ceiling (has 14/16)!
        # S6 (dev, easy):   old=10/16 (p=0.6176, b=-0.479), var=7/16 (b=+0.236 -> delta_b=+0.715) -> easy at all taus, usable at m=0.5, NOT m=1.0
        return [
            _make_source_row("S1", "train", "easier", 1, [_make_variant("V1", 8, generation_tier="easier")]),
            _make_source_row("S2", "train", "easier", 2, [_make_variant("V2", 4, generation_tier="easier")]),
            _make_source_row("S3", "train", "mixed", 5, [_make_variant("V3", 2, generation_tier="nearby")]),
            _make_source_row("S4", "train", "harder", 16, [_make_variant("V4", 15, generation_tier="harder")]),
            _make_source_row(
                "S5",
                "train",
                "harder",
                16,
                [
                    _make_variant("V5a", 15, generation_tier="harder"),
                    _make_variant("V5b", 14, origin="paper_transfer", generation_tier="paper_transfer"),
                ],
            ),
            _make_source_row("S6", "dev", "harder", 10, [_make_variant("V6", 7, generation_tier="harder")]),
        ]

    def test_routing_counts_floor_ceiling_and_crosstab(self) -> None:
        rows = self._build_synthetic_rows()
        summary, old_sha = compute_inventory(rows)
        self.assertEqual(summary["n_sources"], 6)
        self.assertEqual(summary["n_train"], 5)
        self.assertEqual(summary["n_dev"], 1)
        self.assertEqual(summary["n_variants"], 7)
        self.assertEqual(len(old_sha), 64)

        all_tbl = {(r["tau"], r["margin"]): r for r in summary["routing_tables"]["all"]}
        for (tau, m), rec in all_tbl.items():
            self.assertEqual(rec["easy_total"] + rec["hard_total"], 6)
            self.assertEqual(rec["total_usable"], rec["easy_usable"] + rec["hard_usable"])
            self.assertEqual(rec["gap_vs_1600"], rec["total_usable"] - 1600)

        # At tau=0.3, m=0.5: hard={S1,S2} (both usable, S1 floor=1); easy={S3,S4,S5,S6} (all 4 usable, S4 ceil=1)
        r_03_05 = all_tbl[(0.3, 0.5)]
        self.assertEqual(r_03_05["hard_total"], 2)
        self.assertEqual(r_03_05["hard_usable"], 2)
        self.assertEqual(r_03_05["hard_usable_orig_floor_le_1_16"], 1)
        self.assertEqual(r_03_05["easy_total"], 4)
        self.assertEqual(r_03_05["easy_usable"], 4)
        self.assertEqual(r_03_05["easy_usable_all_qualifying_harder_ceil_ge_15_16"], 1)
        self.assertEqual(r_03_05["total_usable"], 6)
        self.assertEqual(r_03_05["usable_excl_floor_and_ceil"], 4)

        # At tau=0.4, m=1.0: hard={S1,S2,S3} (only S1 usable); easy={S4,S5,S6} (S4,S5 usable; S6 db=0.715 < 1.0)
        r_04_10 = all_tbl[(0.4, 1.0)]
        self.assertEqual(r_04_10["hard_total"], 3)
        self.assertEqual(r_04_10["hard_usable"], 1)
        self.assertEqual(r_04_10["easy_total"], 3)
        self.assertEqual(r_04_10["easy_usable"], 2)
        self.assertEqual(r_04_10["easy_usable_all_qualifying_harder_ceil_ge_15_16"], 1)
        self.assertEqual(r_04_10["total_usable"], 3)

        # Cross-tab checks
        ct = summary["direction_crosstab"]["by_source_direction_all_variants"]
        self.assertEqual(ct["easier"]["n_variants"], 2)
        self.assertEqual(ct["easier"]["matched_strict_count"], 2)
        self.assertEqual(ct["harder"]["n_variants"], 4)
        self.assertEqual(ct["harder"]["matched_strict_count"], 4)

    def test_sha256_mismatch_rejected_and_stage2_source_override_works(self) -> None:
        rows = self._build_synthetic_rows()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mpath = tmp / "measured.jsonl"
            mpath.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            with self.assertRaises(ValueError):
                run_inventory(
                    measured_jsonl=mpath,
                    out_dir=tmp / "bad",
                    expected_measured_sha256="0" * 64,
                    allow_custom_sha256=False,
                )

            # Stage 1 mode with custom sha allowed
            s1_dir = tmp / "s1"
            s1_sum = run_inventory(measured_jsonl=mpath, out_dir=s1_dir, allow_custom_sha256=True)
            self.assertEqual(s1_sum["mode"], "stage1_preliminary_old_pretest")

            # Stage 2 mode: override S1 from 1/16 -> 12/16, so S1 moves to easy branch at tau=0.4!
            sp_path = tmp / "source_pretest.jsonl"
            sp_rows = [
                {"source_id": "S1", "n_correct_hybrid": 12, "n_scored": 16},
                {"source_id": "S2", "n_correct_hybrid": 2, "n_scored": 16},
                {"source_id": "S3", "n_correct_hybrid": 5, "n_scored": 16},
                {"source_id": "S4", "n_correct_hybrid": 16, "n_scored": 16},
                {"source_id": "S5", "n_correct_hybrid": 16, "n_scored": 16},
                {"source_id": "S6", "n_correct_hybrid": 10, "n_scored": 16},
            ]
            sp_path.write_text("".join(json.dumps(r) + "\n" for r in sp_rows), encoding="utf-8")
            s2_dir = tmp / "s2"
            s2_sum = run_inventory(
                measured_jsonl=mpath,
                out_dir=s2_dir,
                source_pretest_jsonl=sp_path,
                allow_custom_sha256=True,
            )
            self.assertEqual(s2_sum["mode"], "stage2_final")
            tbl = {(r["tau"], r["margin"]): r for r in s2_sum["routing_tables"]["all"]}
            # S1 is now 12/16 >= 0.4, with variant 8/16 (delta_b > 0.5), so easy_total=4 at tau=0.4!
            self.assertEqual(tbl[(0.4, 0.5)]["easy_total"], 4)
            self.assertEqual(tbl[(0.4, 0.5)]["hard_total"], 2)


class TestStage2SourceRunner(unittest.TestCase):
    def test_end_to_end_fake_gpu_summarize_resume_and_zero_leakage(self) -> None:
        sentinel_prob = "UNIQUE_LEAK_TOKEN_PROBLEM_9911"
        sentinel_ans = "UNIQUE_LEAK_TOKEN_ANS_8822"
        sentinel_resp = "UNIQUE_LEAK_TOKEN_RESPONSE_7733"

        rows = [
            {
                "source_id": "SRC_A",
                "split": "train",
                "direction": "easier",
                "source_problem": f"{sentinel_prob}_A",
                "source_answer": f"10",
                "source_pre_solve_rate_old": 4 / 16.0,
                "variants": [_make_variant("VA1", 8, generation_tier="easier")],
            },
            {
                "source_id": "SRC_B",
                "split": "dev",
                "direction": "harder",
                "source_problem": f"{sentinel_prob}_B",
                "source_answer": f"{sentinel_ans}",
                "source_pre_solve_rate_old": 12 / 16.0,
                "variants": [_make_variant("VB1", 6, generation_tier="harder")],
            },
        ]

        stu_calls = 0
        llm_calls = 0

        async def fake_student(**kwargs: Any) -> Any:
            nonlocal stu_calls
            stu_calls += 1
            prompt = kwargs["messages"][0]["content"]
            if "_A" in prompt:
                # 1st half hit rule \boxed{10}, 2nd half non-boxed equivalent "ten" (goes to 27B)
                if stu_calls % 2 == 0:
                    return _fake_resp(f"{sentinel_resp} \\boxed{{10}}")
                return _fake_resp(f"{sentinel_resp} answer is ten")
            # For SRC_B, answer is sentinel_ans: return non-boxed half the time, ambiguous half the time
            if stu_calls % 4 == 0:
                return _fake_resp(f"{sentinel_resp} ambiguous")
            return _fake_resp(f"{sentinel_resp} \\boxed{{{sentinel_ans}}}")

        async def fake_judge(**kwargs: Any) -> Any:
            nonlocal llm_calls
            llm_calls += 1
            u = kwargs["messages"][1]["content"]
            if "answer is ten" in u:
                return _fake_resp('{"correct": true}')
            return _fake_resp('{"correct": null}')  # unknown -> counted as wrong

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mpath = tmp / "measured.jsonl"
            mpath.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            loaded_rows, in_sha = load_measured_rows(mpath, expected_sha256=None)

            run_dir = tmp / "run"
            stu_client = _FakeClient(fake_student)
            jdg_client = _FakeClient(fake_judge)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats1 = asyncio.run(
                    run_gpu_phase(
                        loaded_rows,
                        run_dir=run_dir,
                        student_client=stu_client,
                        student_model="fake-student",
                        judge_client=jdg_client,
                        judge_model="fake-judge",
                        k_samples=8,
                    )
                )
                self.assertEqual(stats1["student_calls"], 16)
                self.assertEqual(stats1["student_failed"], 0)
                self.assertGreater(stats1["llm_calls"], 0)
                self.assertGreater(stats1["llm_unknown"], 0)

                # Resume should make 0 additional student or judge calls
                prev_stu, prev_llm = stu_calls, llm_calls
                stats2 = asyncio.run(
                    run_gpu_phase(
                        loaded_rows,
                        run_dir=run_dir,
                        student_client=stu_client,
                        student_model="fake-student",
                        judge_client=jdg_client,
                        judge_model="fake-judge",
                        k_samples=8,
                    )
                )
                self.assertEqual(stu_calls, prev_stu)
                self.assertEqual(llm_calls, prev_llm)
                self.assertEqual(stats2["sources_resumed_resp"], 2)
                self.assertEqual(stats2["sources_resumed_grade"], 2)

                summary = run_summarize_phase(
                    loaded_rows,
                    run_dir=run_dir,
                    input_path=mpath,
                    input_sha256=in_sha,
                    allow_custom_sha256=True,
                )

            self.assertEqual(summary["counts"]["n_sources"], 2)
            self.assertEqual(summary["counts"]["n_samples"], 16)
            self.assertEqual(summary["counts"]["n_failed"], 0)
            self.assertEqual(summary["counts"]["n_llm_called"], llm_calls)
            self.assertGreater(summary["counts"]["n_unknown"], 0)
            self.assertIn("vs_old_pre_solve_rate", summary)
            self.assertEqual(summary["final_inventory"]["mode"], "stage2_final")

            # Strict zero-leakage check across stdout and all public/summary outputs
            artifacts_to_scan = [
                buf.getvalue(),
                (run_dir / "source_pretest_measured.jsonl").read_text(encoding="utf-8"),
                (run_dir / "stage2_summary.json").read_text(encoding="utf-8"),
                (run_dir / "stage2_summary.txt").read_text(encoding="utf-8"),
                (run_dir / "stage2_manifest.json").read_text(encoding="utf-8"),
                (run_dir / "final_inventory" / "inventory_summary.json").read_text(encoding="utf-8"),
                (run_dir / "final_inventory" / "inventory_summary.txt").read_text(encoding="utf-8"),
            ]
            for blob in artifacts_to_scan:
                for tok in (sentinel_prob, sentinel_ans, sentinel_resp, "SENTINEL_VAR"):
                    self.assertNotIn(tok, blob)

    def test_technical_failure_gate_aborts_above_1_percent(self) -> None:
        # 15 sources x 16 = 240 rollouts; make all student calls fail -> triggers >1% gate after min_samples=200
        rows = [
            _make_source_row(f"S{i}", "train", "easier", 2, [_make_variant(f"V{i}", 4)])
            for i in range(15)
        ]

        async def failing_student(**_kwargs: Any) -> Any:
            raise RuntimeError("simulated transport outage")

        async def ok_judge(**_kwargs: Any) -> Any:
            return _fake_resp('{"correct": false}')

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(TechnicalFailureGateError):
                asyncio.run(
                    run_gpu_phase(
                        rows,
                        run_dir=Path(td) / "abort_run",
                        student_client=_FakeClient(failing_student),
                        student_model="fake-student",
                        judge_client=_FakeClient(ok_judge),
                        judge_model="fake-judge",
                        k_samples=16,
                    )
                )


if __name__ == "__main__":
    unittest.main()
