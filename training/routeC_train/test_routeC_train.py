"""Unit tests for T-034 Stage 1 Route C training (fake data only)."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
PRO5000_DIR = CODE_ROOT / "training" / "a_group_pro5000"
for p in (CODE_ROOT, PRO5000_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from routeA_data.rasch_ladder import compute_smoothed_p_and_b
from training.routeA_reward.hybrid_grader import HybridGrader, verify_hybrid_grader_sha256
from training.routeA_reward.ladder import extract_ladder_variants
from training.routeA_reward.reward_function import (
    close_step_metric_sinks,
    route_a_reward_func,
    summarize_by_branch,
    summarize_gate_breakdown,
    summarize_metrics_window,
    summarize_step_metrics,
)
from training.routeA_train.prepare_parquet import prepare_routea_parquets
from training.routeC_train.preflight_stage1 import check_t026_step1_3_source_ids
from training.routeC_train.prepare_parquet import (
    prepare_routec_parquets,
    route_and_build_parquet_row,
)

EXPECTED_HYBRID_SHA256 = (
    "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
)


def _run(coro):
    return asyncio.run(coro)


def _float_bits(x: float) -> bytes:
    return struct.pack("!d", float(x))


class _FakeCompletion:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]


class _FakeOpenAIClient:
    def __init__(self, responder):
        self.calls = []

        async def _create(**kwargs):
            idx = len(self.calls)
            self.calls.append(kwargs)
            return responder(kwargs, idx)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def _make_dialogue(
    problem: str = "ORIG_X",
    teacher_think: str = "plan",
    teacher_visible: str = "try factoring",
    student_reply: str = "I get x = 2.",
):
    return [
        {"role": "system", "content": f"Tutor system prompt: {problem}"},
        {"role": "user", "content": f"This is the problem I am trying to understand: {problem}"},
        {"role": "assistant", "content": f"<think>{teacher_think}</think>\n{teacher_visible}"},
        {"role": "user", "content": student_reply},
    ]


def _make_variant(
    vid: str,
    n_correct: int,
    *,
    problem: str | None = None,
    answer: str | None = None,
    origin: str = "self_generated",
    generation_tier: str = "nearby",
    n_scored: int = 16,
) -> dict:
    p_val, b_val = compute_smoothed_p_and_b(n_correct, n_scored)
    return {
        "variant_id": vid,
        "problem": problem if problem is not None else f"PROB_{vid}",
        "answer": answer if answer is not None else f"ANS_{vid}",
        "n_correct_hybrid": n_correct,
        "n_scored": n_scored,
        "p_new": p_val,
        "b": b_val,
        "ladder_rank": 1,
        "origin": origin,
        "generation_tier": generation_tier,
    }


def _make_source_row(
    sid: str,
    variants: list[dict],
    *,
    split: str = "train",
    direction: str = "mixed",
    old_rate: float = 0.25,
    sentinel: str = "SENTINEL",
) -> dict:
    return {
        "source_id": sid,
        "split": split,
        "direction": direction,
        "source_problem": f"SRC_PROB_{sentinel}_{sid}",
        "source_answer": f"SRC_ANS_{sentinel}_{sid}",
        "source_pre_solve_rate_old": old_rate,
        "variants": variants,
    }


class RouteCTrainStage1Tests(unittest.TestCase):
    def test_hybrid_grader_sha256_preserved(self):
        self.assertEqual(
            verify_hybrid_grader_sha256(
                CODE_ROOT / "training" / "routeA_reward" / "hybrid_grader.py",
                EXPECTED_HYBRID_SHA256,
            ),
            EXPECTED_HYBRID_SHA256,
        )
        self.assertEqual(
            verify_hybrid_grader_sha256(
                CODE_ROOT / "routeA_data" / "hybrid_grader.py",
                EXPECTED_HYBRID_SHA256,
            ),
            EXPECTED_HYBRID_SHA256,
        )

    def test_routing_boundaries_exact(self):
        # n_correct=8/16 -> p_var=0.5, b_var=0.0 exactly!
        v_mid = _make_variant("v_mid", 8)
        v_drop = _make_variant("v_drop", 12)  # p=12.5/17 -> b = ln(4.5/12.5) ~ -1.02165
        v_keep_near = _make_variant("v_keep", 11)  # p=11.5/17 -> b = ln(5.5/11.5) ~ -0.7376

        # 1) p_orig == 0.4 exactly -> EASY branch (never hard); delta_b == +1.0 exactly qualifies!
        sp_exact_easy = {
            "s_bound_easy": {
                "source_id": "s_bound_easy",
                "n_correct": 6,
                "n_scored": 16,
                "p_orig": 0.4,
                "b_orig": -1.0,  # v_mid b=0.0 -> delta_b = +1.0 exactly
            }
        }
        row_easy = _make_source_row("s_bound_easy", [v_mid, v_drop])
        # Force v_drop delta_b <= -1.0 relative to b_orig=-1.0? Wait: -1.02165 - (-1.0) = -0.02165 > -1.0!
        # Give another variant with n_correct=15 -> p=15.5/17 -> b = ln(1.5/15.5) = -2.335 -> delta_b = -1.335 <= -1.0
        v_much_easier = _make_variant("v_much_easier", 15)
        row_easy2 = _make_source_row("s_bound_easy", [v_mid, v_keep_near, v_much_easier])
        out_easy, meta_easy = route_and_build_parquet_row(
            row_easy2,
            source_pretest=sp_exact_easy,
            teacher_system_prompt="Sys {problem}",
            parquet_split="train",
        )
        self.assertEqual(meta_easy["route_branch"], "easy")
        self.assertEqual(out_easy["extra_info"]["route_branch"], "easy")
        self.assertEqual(out_easy["extra_info"]["taught_id"], "s_bound_easy")
        self.assertEqual(out_easy["extra_info"]["problem"], row_easy2["source_problem"])
        # v_much_easier (delta_b=-1.335 <= -1.0) dropped; v_keep_near (delta_b=+0.262) & v_mid kept
        kept_ids = [v.variant_id for v in extract_ladder_variants(out_easy["extra_info"])]
        self.assertNotIn("v_much_easier", kept_ids)
        self.assertEqual(set(kept_ids), {"v_keep", "v_mid"})

        # 2) Easy post-test drops delta_b == -1.0 exactly, keeps delta_b > -1.0
        v_harder = _make_variant("v_harder", 1)  # p=1.5/17 -> b = +2.335
        v_exact_m1 = _make_variant("v_exact_m1", 8)  # b=0.0 exactly
        v_gt_m1 = _make_variant("v_gt_m1", 4)  # p=4.5/17 -> b = +1.02165 > 0.0
        sp_drop_m1 = {
            "s_drop_m1": {
                "source_id": "s_drop_m1",
                "n_correct": 12,
                "n_scored": 16,
                "p_orig": 0.7,
                "b_orig": 1.0,  # v_exact_m1 delta_b = 0.0 - 1.0 = -1.0 exactly; v_harder delta_b = +1.335 >= 1.0
            }
        }
        out_dm1, _ = route_and_build_parquet_row(
            _make_source_row("s_drop_m1", [v_harder, v_exact_m1, v_gt_m1]),
            source_pretest=sp_drop_m1,
            teacher_system_prompt="Sys {problem}",
            parquet_split="train",
        )
        self.assertEqual(out_dm1["extra_info"]["route_branch"], "easy")
        kept_dm1 = [v.variant_id for v in extract_ladder_variants(out_dm1["extra_info"])]
        self.assertNotIn("v_exact_m1", kept_dm1)
        self.assertEqual(set(kept_dm1), {"v_gt_m1", "v_harder"})

        # 3) Hard branch: p_orig < 0.4 and delta_b == -1.0 exactly qualifies!
        sp_hard_m1 = {
            "s_hard_m1": {
                "source_id": "s_hard_m1",
                "n_correct": 4,
                "n_scored": 16,
                "p_orig": 0.3999,
                "b_orig": 1.0,  # v_exact_m1 b=0.0 -> delta_b = -1.0 exactly
            }
        }
        out_hm1, _ = route_and_build_parquet_row(
            _make_source_row("s_hard_m1", [v_exact_m1]),
            source_pretest=sp_hard_m1,
            teacher_system_prompt="Sys {problem}",
            parquet_split="train",
        )
        self.assertEqual(out_hm1["extra_info"]["route_branch"], "hard")
        self.assertEqual(out_hm1["extra_info"]["taught_id"], "v_exact_m1")

    def test_hard_branch_max_b_tiebreak_and_ladder_contents(self):
        # Source p_orig = 2.5/17 = 0.147 -> b_orig = ln(14.5/2.5) = +1.7579
        p_orig, b_orig = compute_smoothed_p_and_b(2, 16)
        # v_too_close: n=4 -> b=+1.0217 -> delta_b = -0.7362 (does NOT qualify <= -1.0)
        v_too_close = _make_variant("v_c", 4)
        # v_b & v_a: n=8 -> b=0.0 -> delta_b = -1.7579 <= -1.0 (tie on max b among qualifying!)
        v_tie_b = _make_variant("v_b", 8, problem="PROB_TIE_B", answer="ANS_TIE_B")
        v_tie_a = _make_variant("v_a", 8, problem="PROB_TIE_A", answer="ANS_TIE_A")
        # v_easiest: n=14 -> b=-1.576 -> delta_b = -3.33 <= -1.0 (smaller b than v_a/v_b)
        v_easiest = _make_variant("v_d_easiest", 14)

        sp = {
            "src_hard": {
                "source_id": "src_hard",
                "n_correct": 2,
                "n_scored": 16,
                "p_orig": p_orig,
                "b_orig": b_orig,
            }
        }
        row = _make_source_row("src_hard", [v_too_close, v_tie_b, v_tie_a, v_easiest])
        out_row, meta = route_and_build_parquet_row(
            row,
            source_pretest=sp,
            teacher_system_prompt="Tutor: {problem}",
            parquet_split="train",
        )

        self.assertEqual(meta["route_branch"], "hard")
        # Must pick max b among delta_b <= -1.0 (0.0 > -1.576) and smallest variant_id ("v_a" < "v_b")
        self.assertEqual(out_row["extra_info"]["taught_id"], "v_a")
        self.assertEqual(out_row["extra_info"]["problem"], "PROB_TIE_A")
        self.assertEqual(out_row["extra_info"]["answer"], "ANS_TIE_A")
        self.assertEqual(out_row["reward_model"]["ground_truth"], "ANS_TIE_A")
        self.assertIn("PROB_TIE_A", out_row["prompt"][0]["content"])
        self.assertIn("PROB_TIE_A", out_row["prompt"][1]["content"])
        self.assertEqual(out_row["extra_info"]["interaction_kwargs"]["problem"], "PROB_TIE_A")
        self.assertEqual(out_row["extra_info"]["interaction_kwargs"]["answer"], "ANS_TIE_A")

        # Post-test ladder must exclude taught v_a, include original + v_easiest + v_tie_b + v_too_close
        ladder = extract_ladder_variants(out_row["extra_info"])
        ladder_ids = [v.variant_id for v in ladder]
        self.assertNotIn("v_a", ladder_ids)
        self.assertIn("src_hard", ladder_ids)
        self.assertEqual(set(ladder_ids), {"v_d_easiest", "v_b", "v_c", "src_hard"})
        # Sorted from easiest (lowest b / highest p) to hardest (highest b = orig)
        self.assertEqual(ladder_ids[0], "v_d_easiest")
        self.assertEqual(ladder_ids[-1], "src_hard")
        self.assertEqual([v.ladder_rank for v in ladder], [1, 2, 3, 4])

    def test_missing_variant_field_raises(self):
        p_orig, b_orig = compute_smoothed_p_and_b(10, 16)
        sp = {
            "s1": {
                "source_id": "s1",
                "n_correct": 10,
                "n_scored": 16,
                "p_orig": p_orig,
                "b_orig": b_orig,
            }
        }
        bad_prob = _make_source_row("s1", [_make_variant("v1", 2, problem="  ")])
        with self.assertRaises(ValueError):
            route_and_build_parquet_row(
                bad_prob,
                source_pretest=sp,
                teacher_system_prompt="Sys {problem}",
                parquet_split="train",
            )
        bad_ans = _make_source_row("s1", [_make_variant("v1", 2, answer="")])
        with self.assertRaises(ValueError):
            route_and_build_parquet_row(
                bad_ans,
                source_pretest=sp,
                teacher_system_prompt="Sys {problem}",
                parquet_split="train",
            )

    def test_fallback_matches_t026_field_by_field(self):
        import pandas as pd

        sentinel = "PRIVACY_SENTINEL_XYZ"
        # Build 4 rows: 1 train ceiling (removed), 1 train easy, 1 train hard, 1 train fallback, 1 dev fallback
        p8, b8 = compute_smoothed_p_and_b(8, 16)
        p2, b2 = compute_smoothed_p_and_b(2, 16)
        p12, b12 = compute_smoothed_p_and_b(12, 16)

        rows = [
            _make_source_row("ceil1", [_make_variant("c1", 15)], split="train", sentinel=sentinel),
            _make_source_row(
                "easy1",
                [_make_variant("e_v1", 1), _make_variant("e_v2", 14)],
                split="train",
                old_rate=0.75,
                sentinel=sentinel,
            ),
            _make_source_row(
                "hard1",
                [_make_variant("h_v1", 12), _make_variant("h_v2", 4)],
                split="train",
                old_rate=0.125,
                sentinel=sentinel,
            ),
            _make_source_row(
                "fb1",
                [_make_variant("fb_v1", 7), _make_variant("fb_v2", 9)],
                split="train",
                old_rate=0.5,
                sentinel=sentinel,
            ),
            _make_source_row(
                "dev_fb",
                [_make_variant("dfb_v1", 8)],
                split="dev",
                old_rate=0.5,
                sentinel=sentinel,
            ),
        ]
        # Give each source pretest matching easy / hard / fallback
        pretest_rows = [
            {"source_id": "ceil1", "n_correct_hybrid": 15, "n_scored": 16},
            {"source_id": "easy1", "n_correct_hybrid": 12, "n_scored": 16},  # p=12.5/17=0.735 >=0.4, e_v1 b=+2.33 -> easy
            {"source_id": "hard1", "n_correct_hybrid": 2, "n_scored": 16},   # p=2.5/17=0.147 <0.4, h_v1 delta_b=-2.78 -> hard
            {"source_id": "fb1", "n_correct_hybrid": 8, "n_scored": 16},     # p=0.5 >=0.4, v1/v2 delta_b < 1.0 -> fallback
            {"source_id": "dev_fb", "n_correct_hybrid": 8, "n_scored": 16},  # p=0.5 >=0.4, dfb_v1 delta_b=0 -> fallback
        ]
        # Sanity check b values
        self.assertLess(b12 - b2, -1.0)
        self.assertGreater(abs(b8 - b8), -1.0)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            t023_path = tmp / "t023.jsonl"
            t031_path = tmp / "t031.jsonl"
            t026_dir = tmp / "t026"
            routec_dir = tmp / "routec"

            t023_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            t031_path.write_text("\n".join(json.dumps(r) for r in pretest_rows) + "\n", encoding="utf-8")

            prepare_routea_parquets(
                input_jsonl=t023_path,
                out_dir=t026_dir,
                expected_sha256=None,
                teacher_system_prompt="Tutor: {problem}",
            )
            summary = prepare_routec_parquets(
                input_jsonl=t023_path,
                source_pretest_jsonl=t031_path,
                out_dir=routec_dir,
                t026_train_parquet=t026_dir / "train.parquet",
                t026_val_parquet=t026_dir / "val.parquet",
                expected_t023_sha256=None,
                expected_source_pretest_sha256=None,
                expected_t026_train_sha256=None,
                expected_t026_val_sha256=None,
                teacher_system_prompt="Tutor: {problem}",
                enforce_t031_counts=False,
            )

            self.assertEqual(summary["train"]["branch_counts"], {"easy": 1, "hard": 1, "fallback": 1})
            self.assertEqual(summary["val"]["branch_counts"], {"easy": 0, "hard": 0, "fallback": 1})
            self.assertTrue(summary["t026_parity"]["train"]["order_match"])
            self.assertTrue(summary["t026_parity"]["train"]["fallback_exact_match"])
            self.assertTrue(summary["t026_parity"]["val"]["fallback_exact_match"])
            self.assertNotIn(sentinel, (routec_dir / "dataset_summary.json").read_text(encoding="utf-8"))

            df_a = pd.read_parquet(t026_dir / "train.parquet")
            df_c = pd.read_parquet(routec_dir / "train.parquet")
            self.assertEqual(
                [r["source_id"] for r in df_a["extra_info"]],
                [r["source_id"] for r in df_c["extra_info"]],
            )
            fb_a = df_a.iloc[2].to_dict()
            fb_c = df_c.iloc[2].to_dict()
            self.assertEqual(fb_c["extra_info"]["route_branch"], "fallback")
            self.assertEqual(fb_c["extra_info"]["variants"], fb_a["extra_info"]["variants"])
            self.assertEqual(fb_c["extra_info"]["problem"], fb_a["extra_info"]["problem"])
            self.assertEqual(fb_c["extra_info"]["answer"], fb_a["extra_info"]["answer"])

    def test_reward_bit_identical_and_branch_gate_logging_no_text_leak(self):
        sentinels = [
            "SENTINEL_PROB_111",
            "SENTINEL_ANS_222",
            "SENTINEL_V_PROB_333",
            "SENTINEL_V_ANS_444",
            "SENTINEL_TEACH_THINK_555",
            "SENTINEL_TEACH_VIS_666",
            "SENTINEL_STUD_DLG_777",
            "SENTINEL_STUD_TRY_888",
            "SENTINEL_JUDGE_REASON_999",
        ]
        dialogue = _make_dialogue(
            problem=sentinels[0],
            teacher_think=sentinels[4],
            teacher_visible=sentinels[5],
            student_reply=sentinels[6],
        )

        async def _llm_false(_msgs):
            return '{"correct": false}', "stop"

        def _run_reward(
            *,
            extra: dict,
            leak_dec: str,
            teach_dec: str,
            student_ans: str,
            metrics_dir: str = "",
        ) -> dict:
            jc = _FakeOpenAIClient(
                lambda kw, _idx: _FakeCompletion(
                    json.dumps(
                        {
                            "reasoning": sentinels[8],
                            "decision": leak_dec if "leak" in kw["messages"][0]["content"].lower() else teach_dec,
                        }
                    )
                )
            )
            sc = _FakeOpenAIClient(lambda _kw, _idx: _FakeCompletion(student_ans))
            grader = HybridGrader(llm_caller=_llm_false, backoff_base=0.0)
            with mock.patch(
                "training.routeA_reward.reward_function.get_async_openai_client",
                side_effect=lambda base_u, _env: jc if "judge" in base_u else sc,
            ):
                return _run(
                    route_a_reward_func(
                        data_source="bigmath",
                        ground_truth=sentinels[1],
                        extra_info=extra,
                        mode="ladder_rasch",
                        judge_api_base="http://judge.local/v1",
                        student_api_base="http://student.local/v1",
                        judge_model="judge-local",
                        student_model="student-local",
                        hybrid_grader=grader,
                        ladder_num_climbs=4,
                        theta_scale=0.25,
                        enable_batch_logging=False,
                        routeA_metrics_dir=metrics_dir,
                    )
                )

        base_extra = {
            "problem": sentinels[0],
            "answer": sentinels[1],
            "split": "train",
            "messages": dialogue,
            "variants": [
                {
                    "variant_id": "v1",
                    "problem": sentinels[2],
                    "answer": sentinels[3],
                    "b": 0.5,
                    "ladder_rank": 1,
                }
            ],
        }

        # 1. Bit-for-bit identical reward with vs without route_branch
        for leak_d, teach_d, ans in (
            ("OK", "OK", rf"\boxed{{{sentinels[3]}}}"),
            ("OK", "OK", rf"\boxed{{{sentinels[7]}}}"),
            ("REJECT", "OK", rf"\boxed{{{sentinels[3]}}}"),
            ("OK", "REJECT", rf"\boxed{{{sentinels[3]}}}"),
        ):
            r_no_branch = _run_reward(extra=dict(base_extra), leak_dec=leak_d, teach_dec=teach_d, student_ans=ans)
            r_hard = _run_reward(
                extra={**base_extra, "route_branch": "hard"},
                leak_dec=leak_d,
                teach_dec=teach_d,
                student_ans=ans,
            )
            for k in ("score", "rasch_theta", "rasch_gain_g", "r_gate", "r_length", "r_think_length", "post_solve_rate"):
                self.assertEqual(_float_bits(r_hard[k]), _float_bits(r_no_branch[k]), msg=f"key={k}")

        # 2. Emit mixed branches + all 4 gate outcomes into metrics_dir and aggregate
        with tempfile.TemporaryDirectory() as td:
            mdir = Path(td) / "metrics"
            try:
                t_start = __import__("time").monotonic_ns()
                # easy: 1 pass (pos), 1 teaching_only
                _run_reward(
                    extra={**base_extra, "route_branch": "easy", "split": "train"},
                    leak_dec="OK",
                    teach_dec="OK",
                    student_ans=rf"\boxed{{{sentinels[3]}}}",
                    metrics_dir=str(mdir),
                )
                _run_reward(
                    extra={**base_extra, "route_branch": "easy", "split": "train"},
                    leak_dec="OK",
                    teach_dec="REJECT",
                    student_ans=rf"\boxed{{{sentinels[7]}}}",
                    metrics_dir=str(mdir),
                )
                # hard: 1 pass (neg), 1 leak_only, 1 both
                _run_reward(
                    extra={**base_extra, "route_branch": "hard", "split": "train"},
                    leak_dec="OK",
                    teach_dec="OK",
                    student_ans=rf"\boxed{{{sentinels[7]}}}",
                    metrics_dir=str(mdir),
                )
                _run_reward(
                    extra={**base_extra, "route_branch": "hard", "split": "train"},
                    leak_dec="REJECT",
                    teach_dec="OK",
                    student_ans=rf"\boxed{{{sentinels[7]}}}",
                    metrics_dir=str(mdir),
                )
                _run_reward(
                    extra={**base_extra, "route_branch": "hard", "split": "train"},
                    leak_dec="REJECT",
                    teach_dec="REJECT",
                    student_ans=rf"\boxed{{{sentinels[7]}}}",
                    metrics_dir=str(mdir),
                )
                # fallback: 1 pass (pos)
                _run_reward(
                    extra={**base_extra, "route_branch": "fallback", "split": "train"},
                    leak_dec="OK",
                    teach_dec="OK",
                    student_ans=rf"\boxed{{{sentinels[3]}}}",
                    metrics_dir=str(mdir),
                )
                # val: 1 hard pass
                _run_reward(
                    extra={**base_extra, "route_branch": "hard", "split": "val"},
                    leak_dec="OK",
                    teach_dec="OK",
                    student_ans=rf"\boxed{{{sentinels[3]}}}",
                    metrics_dir=str(mdir),
                )
                t_end = __import__("time").monotonic_ns()
            finally:
                close_step_metric_sinks()

            # Verify zero sentinel leakage in JSONL sink
            sink_files = list(mdir.glob("routeA-metrics-*.jsonl"))
            self.assertEqual(len(sink_files), 1)
            sink_text = sink_files[0].read_text(encoding="ascii")
            for s in sentinels:
                self.assertNotIn(s, sink_text)

            train_agg = summarize_metrics_window(mdir, t_start, t_end, split="train")
            self.assertEqual(train_agg["n_rollouts"], 6)
            gb = train_agg["gate_breakdown"]
            self.assertEqual(gb["n_total"], 6)
            self.assertEqual(gb["n_gated"], 3)
            self.assertEqual(gb["teaching_only_count"], 1)
            self.assertEqual(gb["leak_only_count"], 1)
            self.assertEqual(gb["both_count"], 1)
            self.assertEqual(gb["teaching_any_count"], 2)
            self.assertEqual(gb["leak_any_count"], 2)

            bb = train_agg["by_branch"]
            self.assertEqual(bb["easy"]["count"], 2)
            self.assertEqual(bb["easy"]["gated_count"], 1)
            self.assertEqual(bb["easy"]["teaching_only_count"], 1)
            self.assertAlmostEqual(bb["easy"]["gated_rate"], 0.5, places=12)
            self.assertGreater(bb["easy"]["compliant_g_median"], 0.0)

            self.assertEqual(bb["hard"]["count"], 3)
            self.assertEqual(bb["hard"]["gated_count"], 2)
            self.assertEqual(bb["hard"]["leak_only_count"], 1)
            self.assertEqual(bb["hard"]["both_count"], 1)
            self.assertLess(bb["hard"]["compliant_g_median"], 0.0)

            self.assertEqual(bb["fallback"]["count"], 1)
            self.assertEqual(bb["fallback"]["gated_count"], 0)

            # Flattened keys in summarize_step_metrics
            m = train_agg["metrics"]
            self.assertEqual(m["eduardo/routeC_gate_fail_teaching_only_count"], 1.0)
            self.assertEqual(m["eduardo/routeC_gate_fail_leak_only_count"], 1.0)
            self.assertEqual(m["eduardo/routeC_gate_fail_both_count"], 1.0)
            self.assertEqual(m["eduardo/routeC_hard_count"], 3.0)
            self.assertAlmostEqual(m["eduardo/routeC_hard_gated_rate"], 2.0 / 3.0, places=12)

            val_agg = summarize_metrics_window(mdir, t_start, t_end, split="val")
            self.assertEqual(val_agg["n_rollouts"], 1)
            self.assertEqual(val_agg["by_branch"]["hard"]["count"], 1)
            self.assertEqual(val_agg["by_branch"]["easy"]["count"], 0)

    def test_summarize_helpers_empty_and_t026_log_probe(self):
        self.assertEqual(summarize_gate_breakdown([])["n_total"], 0)
        self.assertEqual(summarize_by_branch([])["easy"]["count"], 0)
        self.assertEqual(summarize_step_metrics([]), {})
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            self.assertEqual(check_t026_step1_3_source_ids(base), "unverified_t026_run_dir_not_found")
            sdir = base / "runs" / "t026_stage3_formal_20261126_140508" / "step-statistics"
            sdir.mkdir(parents=True)
            for s in (1, 2, 3):
                (sdir / f"{s}.json").write_text('{"step": ' + str(s) + '}', encoding="utf-8")
            self.assertEqual(check_t026_step1_3_source_ids(base), "unverified_no_source_id_in_t026_logs")


if __name__ == "__main__":
    unittest.main()
