"""Unit tests for T-031 Route C data pipeline (Card v1 / Inbox #2), using fake data ONLY.

Runnable via BOTH `python -m unittest discover -s tests` (Pro 5000 pro5000-fast)
and `pytest` (cu04).

Covers:
1. `routeC_inventory.py` (Inbox #2 rules):
   - T-026 ceiling group filtering on train only (dev preserved).
   - 3-way routing (`easy_routed`, `hard_routed`, `fallback_routeA`) summing to `n_sources`.
   - Decision 4: hard branch picks the variant with `delta_b <= -m` that has maximum `b` (closest to original).
   - Decision 2: hard-branch floor requires `orig` AND all untaught variants to ALL be `<= 1/16`.
   - Hard-branch post-test ladder level distribution (`1 / 2 / 3+`) summing to `hard_routed`.
   - Branch switching (`old_hard->new_easy`, `old_easy->new_hard`) and 3-state transitions when Stage 2 `source_pretest_jsonl` is provided.
   - SHA-256 mismatch rejection.
2. `stage2_source_runner.py`:
   - End-to-end synthetic GPU + summarize phase: rule pass skips 27B, rule fail calls 27B,
     `unknown` counts as incorrect, >1% technical failure gate aborts, `--resume` skips completed items.
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
    is_t026_ceiling_group,
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
        # S1 (train, hard): orig=1/16 (floor), vars=[8/16 (self), 1/16 (paper)]
        #    -> hard_routed, taught=8/16, rem=[1/16] -> level=2, whole post-test ladder <= 1/16 -> LADDER FLOOR!
        # S2 (train, hard): orig=1/16 (floor), vars=[10/16 (paper, b=-0.479), 5/16 (self, b=+0.738), 3/16 (paper, b=+1.358)]
        #    -> delta_b(orig=2.335): 10/16(-2.814), 5/16(-1.598), 3/16(-0.977)
        #    -> at m=0.5: taught=3/16 (max b, paper_transfer), rem=[10/16, 5/16] -> level=3+, NOT ladder floor!
        #    -> at m=1.0: taught=5/16 (max b in <=-1.0, self_generated), rem=[10/16, 3/16] -> level=3+, NOT ladder floor!
        # S3 (train, hard): orig=2/16 (not floor), var=[4/16] (db=-0.736) -> hard_routed at m=0.5 (level=1), fallback at m=1.0!
        # S4 (train, ceiling): orig=16/16, var=[15/16] -> T-026 train ceiling REMOVED from train_t026!
        # S5 (train, easy): orig=16/16, vars=[15/16, 14/16] -> NOT ceiling, easy_routed!
        # S6 (train, mid): orig=6/16 (p=0.3824), var=[6/16] (db=0) -> always fallback_routeA!
        # S7 (dev, ceiling): orig=16/16, var=[15/16] -> dev is NOT removed, easy_routed (ceil=1)!
        return [
            _make_source_row(
                "S1",
                "train",
                "easier",
                1,
                [
                    _make_variant("V1a", 8, origin="self_generated", generation_tier="easier"),
                    _make_variant("V1b", 1, origin="paper_transfer", generation_tier="paper_transfer"),
                ],
            ),
            _make_source_row(
                "S2",
                "train",
                "easier",
                1,
                [
                    _make_variant("V2a", 10, origin="paper_transfer", generation_tier="paper_transfer"),
                    _make_variant("V2b", 5, origin="self_generated", generation_tier="easier"),
                    _make_variant("V2c", 3, origin="paper_transfer", generation_tier="paper_transfer"),
                ],
            ),
            _make_source_row(
                "S3",
                "train",
                "easier",
                2,
                [_make_variant("V3a", 4, origin="self_generated", generation_tier="easier")],
            ),
            _make_source_row(
                "S4",
                "train",
                "harder",
                16,
                [_make_variant("V4a", 15, origin="self_generated", generation_tier="harder")],
            ),
            _make_source_row(
                "S5",
                "train",
                "harder",
                16,
                [
                    _make_variant("V5a", 15, origin="self_generated", generation_tier="harder"),
                    _make_variant("V5b", 14, origin="paper_transfer", generation_tier="paper_transfer"),
                ],
            ),
            _make_source_row(
                "S6",
                "train",
                "mixed",
                6,
                [_make_variant("V6a", 6, origin="self_generated", generation_tier="nearby")],
            ),
            _make_source_row(
                "S7",
                "dev",
                "harder",
                16,
                [_make_variant("V7a", 15, origin="self_generated", generation_tier="harder")],
            ),
        ]

    def test_inbox2_t026_pool_taught_closest_ladder_floor_and_levels(self) -> None:
        rows = self._build_synthetic_rows()
        self.assertTrue(is_t026_ceiling_group(rows[3]["variants"]))
        self.assertFalse(is_t026_ceiling_group(rows[4]["variants"]))

        summary, _ = compute_inventory(rows)
        pc = summary["pool_counts"]
        self.assertEqual(pc["train_t026_sources"], 5)  # S4 removed
        self.assertEqual(pc["removed_train_ceiling_sources"], 1)
        self.assertEqual(pc["dev_sources"], 1)  # S7 kept
        self.assertEqual(pc["t026_pool_sources"], 6)
        self.assertEqual(pc["all_2100_sources"], 7)

        tbl = {(r["tau"], r["margin"]): r for r in summary["routing_tables"]["train_t026"]}
        # At tau=0.4, m=0.5:
        #   S1,S2,S3 -> hard_routed=3 (S1 ladder_floor=1, S2 not floor; levels: S3=1, S1=2, S2=3+;
        #     taught origins: S1=self, S2(3/16)=paper, S3=self -> self:2, paper:1)
        #   S5 -> easy_routed=1 (ceil=0)
        #   S6 -> fallback_routeA=1 (hard_fallback=1)
        r04_05 = tbl[(0.4, 0.5)]
        self.assertEqual(r04_05["hard_routed"], 3)
        self.assertEqual(r04_05["easy_routed"], 1)
        self.assertEqual(r04_05["fallback_routeA"], 1)
        self.assertEqual(r04_05["hard_routed_orig_only_floor_le_1_16"], 2)  # S1 and S2 orig <= 1/16
        self.assertEqual(r04_05["hard_routed_ladder_floor_le_1_16"], 1)  # only S1 has whole ladder <= 1/16
        self.assertEqual(r04_05["hard_routed_posttest_levels"], {"1": 1, "2": 1, "3+": 1})
        self.assertEqual(r04_05["hard_routed_taught_origin"], {"paper_transfer": 1, "self_generated": 2})

        # At tau=0.4, m=1.0:
        #   S2 picks 5/16 (self_generated) instead of 3/16 (db=-0.977 > -1.0)
        #   S3 falls back to Route A
        r04_10 = tbl[(0.4, 1.0)]
        self.assertEqual(r04_10["hard_routed"], 2)
        self.assertEqual(r04_10["fallback_routeA"], 2)
        self.assertEqual(r04_10["hard_routed_taught_origin"], {"self_generated": 2})

    def test_sha256_mismatch_and_stage2_branch_switching(self) -> None:
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

            # Stage 2 pretest flips S1 (1/16 -> 12/16, hard->easy) and S5 (16/16 -> 2/16, easy->hard)
            sp_path = tmp / "source_pretest.jsonl"
            sp_rows = [
                {"source_id": "S1", "n_correct_hybrid": 12, "n_scored": 16},
                {"source_id": "S2", "n_correct_hybrid": 1, "n_scored": 16},
                {"source_id": "S3", "n_correct_hybrid": 2, "n_scored": 16},
                {"source_id": "S4", "n_correct_hybrid": 16, "n_scored": 16},
                {"source_id": "S5", "n_correct_hybrid": 2, "n_scored": 16},
                {"source_id": "S6", "n_correct_hybrid": 6, "n_scored": 16},
                {"source_id": "S7", "n_correct_hybrid": 16, "n_scored": 16},
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
            sw_tau04 = next(
                x
                for x in s2_sum["switching_stats"]["train_t026"]["by_tau_branch_switch"]
                if abs(x["tau"] - 0.4) < 1e-9
            )
            self.assertEqual(sw_tau04["old_hard_to_new_easy"], 1)  # S1
            self.assertEqual(sw_tau04["old_easy_to_new_hard"], 1)  # S5
            self.assertEqual(sw_tau04["total_branch_switched"], 2)


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
                "source_answer": "10",
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
                if stu_calls % 2 == 0:
                    return _fake_resp(f"{sentinel_resp} \\boxed{{10}}")
                return _fake_resp(f"{sentinel_resp} answer is ten")
            if stu_calls % 4 == 0:
                return _fake_resp(f"{sentinel_resp} ambiguous")
            return _fake_resp(f"{sentinel_resp} \\boxed{{{sentinel_ans}}}")

        async def fake_judge(**kwargs: Any) -> Any:
            nonlocal llm_calls
            llm_calls += 1
            u = kwargs["messages"][1]["content"]
            if "answer is ten" in u:
                return _fake_resp('{"correct": true}')
            return _fake_resp('{"correct": null}')

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
            self.assertIn("switching_stats", summary["final_inventory"])

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
