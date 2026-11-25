"""Unit tests for Stage 2 dev-set simulation and checks (a)-(g) (T-024 v4)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parent.parent
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from eduardo.utils import grade_math_answer, strip_teacher_controls
from training.routeA_reward.stage2_analysis import (
    analyze_stage2,
    choose_k_from_split_half,
    classify_source_stratum,
    paired_bootstrap,
    score_dialogue_from_climbs,
    spearman_brown,
)
from training.routeA_reward.stage2_dev_sim import (
    EXPECTED_HYBRID_GRADER_SHA256,
    allocate_gpu_layout,
    file_sha256,
    parse_gpu_snapshot,
    rollout_one_dialogue,
    select_free_gpus,
)


def _make_climb(steps: list[tuple[float, float, bool, bool]]) -> dict:
    return {
        "steps": [
            {"b": b, "p_new": p, "correct": ok, "tech_failed": tf}
            for b, p, ok, tf in steps
        ]
    }


class TestStage2Analysis(unittest.TestCase):
    def test_hybrid_grader_sha256_locked_to_t023(self):
        self.assertEqual(file_sha256(HERE / "hybrid_grader.py"), EXPECTED_HYBRID_GRADER_SHA256)

    def test_paired_bootstrap_reproducible_and_sign(self):
        diffs = [0.2 + 0.01 * i for i in range(20)]
        r1 = paired_bootstrap(diffs, draws=2000, seed=42)
        r2 = paired_bootstrap(diffs, draws=2000, seed=42)
        self.assertEqual(r1, r2)
        self.assertGreater(r1["ci95_low"], 0.0)
        self.assertEqual(r1["p_gt_0"], 1.0)
        self.assertGreater(r1["cohen_dz"], 0.0)

    def test_spearman_brown_k_decision_rule(self):
        # r=0.72 -> k=8
        c8 = choose_k_from_split_half(0.72)
        self.assertEqual(c8["chosen_k"], 8)
        self.assertAlmostEqual(c8["reliability_k8"], 0.72, places=12)

        # r=0.62 -> rel(8)=0.62 < 0.7, rel(12)=1.5*0.62/(1+0.5*0.62)=0.7099 >= 0.7 -> k=12
        c12 = choose_k_from_split_half(0.62)
        self.assertEqual(c12["chosen_k"], 12)
        self.assertGreaterEqual(c12["reliability_k12"], 0.7)

        # r=0.50 -> rel(12)=0.60 < 0.7 -> k=16
        c16 = choose_k_from_split_half(0.50)
        self.assertEqual(c16["chosen_k"], 16)
        self.assertAlmostEqual(c16["reliability_k16"], spearman_brown(0.50, 2.0), places=12)

    def test_classify_four_strata(self):
        self.assertEqual(
            classify_source_stratum([{"n_correct_hybrid": 0, "b": 3.5}, {"n_correct_hybrid": 1, "b": 2.3}]),
            "all_le_1_16",
        )
        self.assertEqual(
            classify_source_stratum([{"n_correct_hybrid": 15, "b": -2.3}, {"n_correct_hybrid": 16, "b": -3.5}]),
            "all_ge_15_16",
        )
        self.assertEqual(
            classify_source_stratum([{"n_correct_hybrid": 8, "b": 0.0}, {"n_correct_hybrid": 9, "b": -0.2}]),
            "narrow_span_lt_0_5",
        )
        self.assertEqual(
            classify_source_stratum([{"n_correct_hybrid": 5, "b": 0.7}, {"n_correct_hybrid": 11, "b": -0.7}]),
            "other",
        )

    def test_full_analyze_stage2_checks_a_to_g_and_privacy(self):
        # 4 synthetic source problems covering all 4 strata of check (g)
        source_meta = {
            "s_floor": {
                "variants": [{"variant_id": "v1", "n_correct_hybrid": 1, "p_new": 1.5 / 17, "b": 2.33, "ladder_rank": 1}]
            },
            "s_ceil": {
                "variants": [{"variant_id": "v2", "n_correct_hybrid": 15, "p_new": 15.5 / 17, "b": -2.33, "ladder_rank": 1}]
            },
            "s_narrow": {
                "variants": [
                    {"variant_id": "v3", "n_correct_hybrid": 8, "p_new": 8.5 / 17, "b": 0.0, "ladder_rank": 1},
                    {"variant_id": "v4", "n_correct_hybrid": 7, "p_new": 7.5 / 17, "b": 0.24, "ladder_rank": 2},
                ]
            },
            "s_other": {
                "variants": [
                    {"variant_id": "v5", "n_correct_hybrid": 12, "p_new": 12.5 / 17, "b": -1.02, "ladder_rank": 1},
                    {"variant_id": "v6", "n_correct_hybrid": 4, "p_new": 4.5 / 17, "b": 1.02, "ladder_rank": 2},
                ]
            },
        }

        records = []
        for sid, meta in source_meta.items():
            vs = meta["variants"]
            for tutor, pass_rate in (("base", 0.25), ("a_group", 0.75)):
                for d_idx in (0, 1):
                    climbs = []
                    n_pass = int(16 * pass_rate) + (d_idx if tutor == "a_group" else 0)
                    for c_idx in range(16):
                        if c_idx < n_pass:
                            steps = [(v["b"], v["p_new"], True, False) for v in vs]
                        else:
                            steps = [(vs[0]["b"], vs[0]["p_new"], False, False)]
                        climbs.append(_make_climb(steps))
                    records.append(
                        {
                            "tutor": tutor,
                            "source_id": sid,
                            "dialogue_idx": d_idx,
                            "climbs": climbs,
                            "student_calls": sum(len(c["steps"]) for c in climbs),
                            "student_failed": 0,
                            "grader_rule_correct": 10,
                            "grader_llm_calls": 6,
                            "grader_llm_unknown": 0,
                            "grader_tech_failed": 0,
                            "grader_llm_time_s": 0.6,
                            "grader_llm_latencies_ms": [100.0] * 6,
                        }
                    )

        out = analyze_stage2(records, source_meta, bootstrap_draws=500, bootstrap_seed=42)
        self.assertEqual(out["n_sources"], 4)
        self.assertEqual(out["n_dialogues_total"], 16)
        # (a) a_group > base with positive CI lower bound
        self.assertTrue(out["check_a_discrimination"]["passed_ci_low_gt_0"])
        self.assertGreater(out["check_a_discrimination"]["mean"], 0.0)
        # (b) chosen_k in {8, 12, 16}
        self.assertIn(out["check_b_reliability_k"]["chosen_k"], (8, 12, 16))
        # (c) variance decomposition populated
        self.assertGreaterEqual(out["check_c_variance_signal"]["pooled"]["within_source_obs_var"], 0.0)
        # (d) scale populated
        self.assertEqual(out["check_d_reward_scale"]["all"]["n"], 16)
        # (e) all 3 alternatives populated with cohen_dz
        for key in ("alt1_one_minus_p", "alt2_max_passed_b", "alt3_discrete_ladder_frac"):
            self.assertGreater(out["check_e_alternatives"][key]["mean"], 0.0)
        # (f) projection populated
        self.assertGreater(out["check_f_cost_and_load"]["step_projection_k8"]["llm_27b_calls_per_step_256"], 0.0)
        # (g) all 4 strata have exact source count 1
        for st_name in ("all_le_1_16", "all_ge_15_16", "narrow_span_lt_0_5", "other"):
            self.assertEqual(out["check_g_low_disc_strata"][st_name]["n_sources"], 1)

        # Zero NaN/Inf in JSON serialization
        blob = json.dumps(out, allow_nan=False)
        self.assertIn("check_g_low_disc_strata", blob)


class TestStage2Infra(unittest.TestCase):
    def test_gpu_selection_handles_gpu6_idle_sensor_quirk(self):
        # GPU 6 reports util=100% when idle, but used=10 MiB and 0 apps -> MUST be free!
        gpu_csv = "\n".join(
            [
                "0, GPU-0, 20000, 53000, 73415, 45",  # busy
                "1, GPU-1, 10, 73000, 73415, 0",      # free
                "2, GPU-2, 12, 73000, 73415, 0",      # free
                "3, GPU-3, 15, 73000, 73415, 0",      # free
                "4, GPU-4, 15, 73000, 73415, 0",      # free
                "6, GPU-6, 10, 73000, 73415, 100",    # idle sensor quirk -> free!
            ]
        )
        app_csv = "GPU-0, 12345, 19500\n"
        snap = parse_gpu_snapshot(gpu_csv, app_csv)
        free = select_free_gpus(snap, snap)
        self.assertEqual(free, [1, 2, 3, 4, 6])
        layout5 = allocate_gpu_layout(free)
        self.assertEqual(layout5["num_gpus"], 5)
        self.assertEqual(layout5["tutor_gpu"], 1)
        self.assertEqual(layout5["student_gpus"], [2, 3])
        self.assertEqual(layout5["grader_gpus"], [4, 6])

        layout4 = allocate_gpu_layout([1, 2, 3, 4])
        self.assertEqual(layout4["num_gpus"], 4)
        self.assertEqual(layout4["student_gpus"], [2])
        self.assertEqual(layout4["grader_gpus"], [3, 4])
        self.assertIsNone(allocate_gpu_layout([1, 2, 3]))

    def test_rollout_eoc_honored_only_after_min_prior_turns(self):
        turn_counter = {"t": 0}

        class FakeClient:
            def __init__(self, fn):
                self._fn = fn
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            async def _create(self, **kwargs):
                return self._fn(kwargs)

        def tutor_fn(_kw):
            turn_counter["t"] += 1
            # Emit EoC on every turn; should be ignored on turns 1, 2, 3 (prior=0,1,2) and honored on turn 4 (prior=3)!
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="think</think>hint <end_of_conversation>"))],
                usage=SimpleNamespace(total_tokens=200),
            )

        def student_fn(_kw):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="student step"))],
                usage=SimpleNamespace(total_tokens=100),
            )

        transcript, meta = asyncio.run(
            rollout_one_dialogue(
                problem="P",
                tutor_client=FakeClient(tutor_fn),
                tutor_model="t",
                student_client=FakeClient(student_fn),
                student_model="s",
                teacher_system_prompt="Teach {problem}",
                student_system_prompt="Solve {problem}",
                strip_teacher_controls_fn=strip_teacher_controls,
                tutor_semaphore=asyncio.Semaphore(4),
                student_semaphore=asyncio.Semaphore(4),
            )
        )
        self.assertEqual(meta["teacher_turns"], 4)
        self.assertEqual(meta["student_turns"], 3)
        self.assertTrue(meta["eoc_used"])
        self.assertEqual(len(transcript), 1 + 4 + 3)

    def test_real_math_verify_0_8_0_when_installed(self):
        if importlib.util.find_spec("math_verify") is None:
            self.skipTest("math_verify not installed on cu04; runs in pro5000-fast")
        # Symbolic equivalence that exact string match cannot pass: 1/2 vs \frac{2}{4}
        res_eq = grade_math_answer(r"Therefore \boxed{\frac{2}{4}}.", r"\frac{1}{2}")
        self.assertEqual(res_eq["acc"], 1.0)
        # math-verify==0.8.0 strict percent behavior (28% != 28 in 0.8.0, changed in 0.9.0)
        res_pct = grade_math_answer(r"\boxed{28\%}", "28")
        self.assertEqual(res_pct["acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
