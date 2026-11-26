"""Unit tests for Route A reward module (T-024 Stage 1).

Uses ONLY fake student / fake judge / fake 27B grader stubs (zero real network
calls, zero real problem/dataset reads). Covers every acceptance item in T-024
Stage 1:
  - Ladder stop rule & masked-history isolation (no leakage across rungs)
  - 1-, 2-, and 3-variant source problems & ladder_rank ordering
  - Rasch MAP boundaries (all-correct, all-wrong, single-obs, empty/invalid),
    grid-search agreement, and difficulty monotonicity
  - Technical failures (student & 27B grader) excluded from Rasch & not wrong
  - Gate rejection -> -0.2 directly with zero post-test / grader calls
  - ``author`` mode bit-for-bit identical to upstream ``eduardo_reward_func``
  - ``author_hybrid`` middle-variant selection & hybrid grading
  - Hybrid grader policy (rule-pass skips 27B, rule-fail calls 27B, unknown ->
    wrong, 27B technical failure retries & is not a verdict, AIMD window)
  - ``hybrid_grader.py`` SHA-256 verification helper & prompt match vs Agnes v3
  - Zero problem/answer/response/transcript text in logs
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch

# Ensure worktree root and author upstream repo are importable when invoked directly.
HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parent.parent
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from eduardo import reward_function as author_rf
from eduardo import utils as author_utils
from eduardo.metrics import correctness as author_correctness
from eduardo.metrics import judge_quality as author_judge_quality

from training.routeA_reward.hybrid_grader import (
    AGNES_JUDGE_REFERENCE_SHA256,
    EQUIV_JUDGE_SYSTEM_PROMPT,
    AsyncAdaptiveWindow,
    GradeResult,
    HybridGrader,
    check_technical_failure_rate,
    file_sha256,
    parse_equiv_verdict,
    verify_hybrid_grader_sha256,
)
from training.routeA_reward.ladder import (
    LadderVariant,
    climb_ladder_batch,
    extract_ladder_variants,
    select_middle_variant,
)
from training.routeA_reward.rasch import (
    difficulty_from_rate,
    difficulty_from_solve_count,
    rasch_grid_search,
    rasch_log_posterior,
    rasch_map_estimate,
    sigmoid,
)
from training.routeA_reward.reward_function import (
    reset_batch_stats,
    resolve_reward_mode,
    route_a_reward_func,
    summarize_step_metrics,
)


def _run(coro):
    return asyncio.run(coro)


def _float_bits(x: float) -> bytes:
    return struct.pack("!d", float(x))


def _fake_completion(content: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeOpenAIClient:
    """Deterministic async OpenAI client stub with call recording."""

    def __init__(self, handler):
        self.calls: list[dict[str, Any]] = []
        self._handler = handler
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._handler(kwargs, len(self.calls) - 1)
        if asyncio.iscoroutine(result):
            result = await result
        return result


class TransientHTTPStatusError(Exception):
    def __init__(self, status_code: int, message: str = "transient"):
        super().__init__(message)
        self.status_code = status_code


def _make_dialogue(
    *,
    problem: str = "ORIG_PROBLEM_ALPHA",
    teacher_think: str = "PRIVATE_THINK_ALPHA",
    teacher_visible: str = "GUIDING_HINT_ALPHA",
    student_reply: str = "STUDENT_THOUGHT_ALPHA",
    eoc: bool = False,
) -> list[dict[str, str]]:
    eoc_tag = " <end_of_conversation>" if eoc else ""
    return [
        {"role": "system", "content": f"System for {problem}"},
        {"role": "user", "content": f"This is the problem I am trying to understand: {problem}"},
        {"role": "assistant", "content": f"{teacher_think}</think>{teacher_visible}{eoc_tag}"},
        {"role": "user", "content": student_reply},
    ]


class TestRaschMAP(unittest.TestCase):
    def test_difficulty_from_solve_count_formula(self):
        p0, b0 = difficulty_from_solve_count(0, 16)
        p8, b8 = difficulty_from_solve_count(8, 16)
        p16, b16 = difficulty_from_solve_count(16, 16)
        self.assertAlmostEqual(p0, 0.5 / 17.0, places=12)
        self.assertAlmostEqual(p8, 8.5 / 17.0, places=12)
        self.assertAlmostEqual(p16, 16.5 / 17.0, places=12)
        self.assertAlmostEqual(b8, 0.0, places=12)
        self.assertAlmostEqual(b0, -b16, places=12)
        self.assertGreater(b0, 3.0)
        self.assertAlmostEqual(difficulty_from_rate(p0), b0, places=12)

    def test_boundaries_and_grid_search_agreement(self):
        cases = [
            # All correct (ceiling -> pulled by N(0,1) prior)
            ([( -1.0, 1), (0.0, 1), (1.0, 1)] * 4, True, True, False),
            # All wrong (floor -> pulled by N(0,1) prior)
            ([(-1.0, 0)] * 8, True, False, True),
            # Single observation: correct
            ([(0.7, 1)], True, True, False),
            # Single observation: wrong
            ([(-0.4, 0)], True, False, True),
            # Mixed ladder outcomes (not prior-pulled)
            ([(-1.2, 1), (0.0, 1), (1.3, 0), (-1.2, 1), (0.0, 0), (-1.2, 0)], False, False, False),
        ]
        for obs, expect_pulled, expect_all_c, expect_all_w in cases:
            est = rasch_map_estimate(obs)
            self.assertTrue(est.valid)
            self.assertTrue(est.converged)
            self.assertIsNotNone(est.theta)
            self.assertEqual(est.n_obs, len(obs))
            self.assertEqual(est.prior_pulled, expect_pulled)
            self.assertEqual(est.all_correct, expect_all_c)
            self.assertEqual(est.all_wrong, expect_all_w)

            # First-order optimality: grad(theta_hat) == 0
            grad = -est.theta + sum(y - sigmoid(est.theta - b) for b, y in obs)
            self.assertLess(abs(grad), 1e-9)

            # Agreement with brute-force grid search
            grid_theta = rasch_grid_search(obs, low=-10.0, high=10.0, num_points=40001)
            self.assertAlmostEqual(est.theta, grid_theta, delta=1e-3)
            self.assertGreaterEqual(
                rasch_log_posterior(est.theta, obs),
                rasch_log_posterior(grid_theta, obs) - 1e-12,
            )

    def test_empty_observations_returns_invalid(self):
        est = rasch_map_estimate([])
        self.assertFalse(est.valid)
        self.assertIsNone(est.theta)
        self.assertEqual(est.n_obs, 0)
        self.assertFalse(est.prior_pulled)

    def test_higher_difficulty_correct_gives_larger_theta(self):
        # Single correct observation at increasing difficulty
        thetas_single = [rasch_map_estimate([(b, 1)]).theta for b in (-2.0, -0.5, 0.0, 0.5, 2.0)]
        for a, b in zip(thetas_single, thetas_single[1:]):
            self.assertLess(a, b)

        # Identical pattern [1, 1, 0] shifted to harder items -> larger theta_hat
        easy = rasch_map_estimate([(-2.0, 1), (-1.0, 1), (0.0, 0)]).theta
        hard = rasch_map_estimate([(0.0, 1), (1.0, 1), (2.0, 0)]).theta
        self.assertGreater(hard, easy)

    def test_extreme_difficulties_numerically_stable(self):
        est_pos = rasch_map_estimate([(50.0, 1), (40.0, 1)])
        est_neg = rasch_map_estimate([(-50.0, 0), (-40.0, 0)])
        self.assertTrue(est_pos.valid and math.isfinite(est_pos.theta))
        self.assertTrue(est_neg.valid and math.isfinite(est_neg.theta))
        self.assertGreater(est_pos.theta, 0.0)
        self.assertLess(est_neg.theta, 0.0)


class TestHybridGrader(unittest.TestCase):
    def test_prompt_matches_agnes_eval_v3_byte_for_byte(self):
        agnes_path = WORKTREE_ROOT / "evaluation" / "agnes_eval_v3" / "agnes_judge.py"
        self.assertTrue(agnes_path.is_file())
        self.assertEqual(file_sha256(agnes_path), AGNES_JUDGE_REFERENCE_SHA256)
        spec = importlib.util.spec_from_file_location("agnes_judge_ref", agnes_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(EQUIV_JUDGE_SYSTEM_PROMPT, mod.SYSTEM)

    def test_rule_correct_skips_27b(self):
        llm_calls = 0

        async def fake_llm(_msgs):
            nonlocal llm_calls
            llm_calls += 1
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=fake_llm, backoff_base=0.0)
        res = _run(grader.grade("Compute 2+2.", "4", r"Thus \boxed{4}."))
        self.assertTrue(res.correct)
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.source, "rule")
        self.assertTrue(res.rule_correct)
        self.assertFalse(res.llm_called)
        self.assertEqual(llm_calls, 0)

    def test_rule_incorrect_and_unboxed_call_27b(self):
        seen_payloads: list[dict[str, str]] = []

        async def fake_llm(msgs):
            seen_payloads.append(json.loads(msgs[1]["content"]))
            if "four" in msgs[1]["content"]:
                return '{"correct": true}', "stop"
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=fake_llm, backoff_base=0.0)
        # Missing \boxed{} -> rule fails, 27B promotes to correct
        res1 = _run(grader.grade("Compute 2+2.", "4", "The answer is four."))
        self.assertTrue(res1.correct)
        self.assertEqual(res1.status, "correct")
        self.assertEqual(res1.source, "llm")
        self.assertFalse(res1.rule_correct)
        self.assertEqual(res1.incorrect_format, 1)
        self.assertTrue(res1.llm_called)
        self.assertFalse(res1.is_unknown)
        self.assertFalse(res1.technical_failure)

        # Wrong \boxed{} -> rule fails, 27B confirms incorrect
        res2 = _run(grader.grade("Compute 2+2.", "4", r"\boxed{5}"))
        self.assertFalse(res2.correct)
        self.assertEqual(res2.status, "incorrect")
        self.assertEqual(res2.incorrect_format, 0)
        self.assertTrue(res2.llm_called)
        self.assertFalse(res2.is_unknown)
        self.assertFalse(res2.technical_failure)
        self.assertEqual(len(seen_payloads), 2)

    def test_unknown_and_bad_format_count_as_wrong(self):
        # Explicit {"correct": null} -> unknown immediately, counts as wrong
        null_calls = 0

        async def null_llm(_msgs):
            nonlocal null_calls
            null_calls += 1
            return '{"correct": null}', "stop"

        g_null = HybridGrader(llm_caller=null_llm, max_attempts=4, backoff_base=0.0)
        r_null = _run(g_null.grade("Q", "4", r"\boxed{?}"))
        self.assertFalse(r_null.correct)
        self.assertEqual(r_null.status, "unknown")
        self.assertTrue(r_null.is_unknown)
        self.assertFalse(r_null.technical_failure)
        self.assertEqual(null_calls, 1)

        # Persistent malformed output -> retries max_attempts, then unknown (counts as wrong)
        bad_calls = 0

        async def bad_llm(_msgs):
            nonlocal bad_calls
            bad_calls += 1
            return "not valid json", "stop"

        g_bad = HybridGrader(llm_caller=bad_llm, max_attempts=3, backoff_base=0.0)
        r_bad = _run(g_bad.grade("Q", "4", r"\boxed{?}"))
        self.assertFalse(r_bad.correct)
        self.assertEqual(r_bad.status, "unknown")
        self.assertTrue(r_bad.is_unknown)
        self.assertFalse(r_bad.technical_failure)
        self.assertEqual(bad_calls, 3)

    def test_27b_technical_failure_retries_and_is_not_verdict(self):
        # Transient errors followed by success
        attempts = 0

        async def flaky_llm(_msgs):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise TransientHTTPStatusError(503, "server busy")
            return '```json\n{"correct": true}\n```', "stop"

        win = AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0)
        g_flaky = HybridGrader(
            llm_caller=flaky_llm,
            max_attempts=4,
            window=win,
            backoff_base=0.0,
        )
        r_ok = _run(g_flaky.grade("Q", "4", "four"))
        self.assertTrue(r_ok.correct)
        self.assertFalse(r_ok.technical_failure)
        self.assertEqual(r_ok.attempts, 3)
        self.assertEqual(attempts, 3)

        # Exhausted transient errors -> technical_failure=True (NOT a verdict)
        hard_attempts = 0

        async def dead_llm(_msgs):
            nonlocal hard_attempts
            hard_attempts += 1
            raise TimeoutError("timed out")

        win2 = AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0)
        g_dead = HybridGrader(
            llm_caller=dead_llm,
            max_attempts=3,
            window=win2,
            backoff_base=0.0,
        )
        r_dead = _run(g_dead.grade("Q", "4", "four"))
        self.assertFalse(r_dead.correct)
        self.assertEqual(r_dead.status, "technical_error")
        self.assertTrue(r_dead.technical_failure)
        self.assertFalse(r_dead.is_unknown)
        self.assertEqual(r_dead.attempts, 3)
        self.assertEqual(hard_attempts, 3)

    def test_aimd_window_and_tech_failure_rate(self):
        async def exercise():
            win = AsyncAdaptiveWindow(max_window=4, max_cooldown=0.0)
            self.assertEqual(win.window, 1)
            for expected in (2, 3, 4, 4):
                await win.acquire()
                await win.release(True)
                self.assertEqual(win.window, expected)
            # Non-verdict HTTP 200 leaves window unchanged
            await win.acquire()
            await win.release(None)
            self.assertEqual(win.window, 4)
            # Error halves window
            await win.acquire()
            await win.release(False)
            self.assertEqual(win.window, 2)
            await win.acquire()
            await win.release(False)
            self.assertEqual(win.window, 1)
            await win.acquire()
            await win.release(False)
            self.assertEqual(win.window, 1)

        _run(exercise())
        self.assertFalse(check_technical_failure_rate(0, 0))
        self.assertFalse(check_technical_failure_rate(1, 100))
        self.assertTrue(check_technical_failure_rate(2, 100))
        self.assertTrue(check_technical_failure_rate(1, 99))

    def test_verify_hybrid_grader_sha256_helper(self):
        grader_path = HERE / "hybrid_grader.py"
        expected = file_sha256(grader_path)
        self.assertEqual(verify_hybrid_grader_sha256(grader_path, expected), expected)
        with tempfile.TemporaryDirectory() as td:
            copy_path = Path(td) / "hybrid_grader_t023.py"
            copy_path.write_bytes(grader_path.read_bytes())
            self.assertEqual(verify_hybrid_grader_sha256(copy_path, expected), expected)
            copy_path.write_bytes(grader_path.read_bytes() + b"\n# drift\n")
            with self.assertRaises(ValueError):
                verify_hybrid_grader_sha256(copy_path, expected)

    def test_parse_equiv_verdict_edge_cases(self):
        self.assertEqual(parse_equiv_verdict('{"correct": true}', "stop"), (True, None))
        self.assertEqual(parse_equiv_verdict('{"correct": false}', "stop"), (False, None))
        self.assertEqual(parse_equiv_verdict('{"correct": null}', "stop"), (None, None))
        self.assertEqual(parse_equiv_verdict("true", "stop"), (True, None))
        self.assertEqual(parse_equiv_verdict("null", "stop"), (None, None))
        self.assertEqual(parse_equiv_verdict('{"correct": true}', "length"), (None, "response_truncated"))
        self.assertEqual(
            parse_equiv_verdict('{"correct": true, "correct": false}', "stop"),
            (None, "response_duplicate_keys"),
        )
        self.assertEqual(
            parse_equiv_verdict('{"correct": "yes"}', "stop"),
            (None, "response_schema_invalid"),
        )


class TestLadderPostTest(unittest.TestCase):
    def test_stop_rule_and_masked_history_isolation(self):
        dialogue = _make_dialogue(
            problem="ORIG_PROB_SECRET",
            teacher_think="TEACHER_PRIVATE_THOUGHT",
            teacher_visible="TEACHER_VISIBLE_HINT",
            student_reply="STUDENT_OWN_STEP",
        )
        variants = [
            LadderVariant("v1", "VARIANT_ONE_TEXT", "11", b=-1.0, ladder_rank=1),
            LadderVariant("v2", "VARIANT_TWO_TEXT", "22", b=0.0, ladder_rank=2),
            LadderVariant("v3", "VARIANT_THREE_TEXT", "33", b=1.0, ladder_rank=3),
        ]

        # 4 climbs:
        # Level 0 (4 calls): 3 pass, 1 fails
        # Level 1 (3 calls): 2 pass, 1 fails
        # Level 2 (2 calls): 1 passes, 1 fails
        level_queues = {
            "VARIANT_ONE_TEXT": [r"\boxed{11}", r"\boxed{11}", r"\boxed{11}", r"\boxed{999}"],
            "VARIANT_TWO_TEXT": [r"\boxed{22}", r"\boxed{22}", r"\boxed{999}"],
            "VARIANT_THREE_TEXT": [r"\boxed{33}", r"\boxed{999}"],
        }
        seen_msgs: list[list[dict[str, str]]] = []

        async def fake_sampler(msgs):
            seen_msgs.append(msgs)
            final_turn = msgs[-1]["content"]
            for key, queue in level_queues.items():
                if key in final_turn:
                    return queue.pop(0)
            raise AssertionError("unexpected prompt")

        async def always_false_llm(_msgs):
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=always_false_llm, backoff_base=0.0)
        res = _run(
            climb_ladder_batch(
                messages=dialogue,
                problem="ORIG_PROB_SECRET",
                variants=variants,
                student_client=None,
                student_model="fake-student",
                grader=grader,
                num_climbs=4,
                omit_teacher_turns=True,
                student_sampler=fake_sampler,
            )
        )

        # Total student calls: 4 + 3 + 2 = 9
        self.assertEqual(res.student_calls, 9)
        self.assertEqual(res.reached_counts, (4, 3, 2))
        self.assertEqual(res.passed_counts, (3, 2, 1))
        self.assertEqual(res.student_failed, 0)
        self.assertEqual(len(res.observations), 9)
        self.assertEqual(
            res.observations,
            (
                (-1.0, 1), (-1.0, 1), (-1.0, 1), (-1.0, 0),
                (0.0, 1), (0.0, 1), (0.0, 0),
                (1.0, 1), (1.0, 0),
            ),
        )

        # Verify masking & cross-level isolation on EVERY student call
        self.assertEqual(len(seen_msgs), 9)
        for idx, msgs in enumerate(seen_msgs):
            joined = "\n".join(m["content"] for m in msgs)
            # System prompt keeps original problem, teacher turns masked as (hidden)
            self.assertIn("ORIG_PROB_SECRET", msgs[0]["content"])
            self.assertNotIn("TEACHER_PRIVATE_THOUGHT", joined)
            self.assertNotIn("TEACHER_VISIBLE_HINT", joined)
            self.assertIn("(hidden)", joined)
            self.assertIn("STUDENT_OWN_STEP", joined)
            # Match author's _build_final_attempt_messages identically
            expected_variant = variants[0 if idx < 4 else (1 if idx < 7 else 2)]
            expected_msgs = author_correctness._build_final_attempt_messages(
                dialogue,
                "ORIG_PROB_SECRET",
                transfer_problem=expected_variant.problem,
                omit_teacher_turns=True,
            )
            self.assertEqual(msgs, expected_msgs)
            # No other level's variant text appears anywhere
            for other in variants:
                if other.variant_id != expected_variant.variant_id:
                    self.assertNotIn(other.problem, joined)

    def test_one_two_three_variants_and_rank_sorting(self):
        dialogue = _make_dialogue()

        async def always_false_llm(_msgs):
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=always_false_llm, backoff_base=0.0)

        for n_vars in (1, 2, 3):
            # Pass variants intentionally out of rank order to test sorting
            raw_list = [
                {
                    "variant_id": f"v{r}",
                    "problem": f"VPROB_{r}",
                    "answer": str(10 * r),
                    "b": float(r - 2),
                    "ladder_rank": r,
                }
                for r in range(n_vars, 0, -1)
            ]
            # Test both list and JSON string forms
            variants = extract_ladder_variants({"variants": json.dumps(raw_list)})
            self.assertEqual([v.ladder_rank for v in variants], list(range(1, n_vars + 1)))

            async def perfect_sampler(msgs):
                for v in variants:
                    if v.problem in msgs[-1]["content"]:
                        return rf"\boxed{{{v.answer}}}"
                raise AssertionError("unknown variant")

            res = _run(
                climb_ladder_batch(
                    messages=dialogue,
                    problem="ORIG_PROBLEM_ALPHA",
                    variants=variants,
                    student_client=None,
                    student_model="fake-student",
                    grader=grader,
                    num_climbs=3,
                    student_sampler=perfect_sampler,
                )
            )
            self.assertEqual(res.num_variants, n_vars)
            self.assertEqual(res.reached_counts, tuple([3] * n_vars))
            self.assertEqual(res.passed_counts, tuple([3] * n_vars))
            self.assertEqual(res.student_calls, 3 * n_vars)
            self.assertEqual(len(res.observations), 3 * n_vars)

    def test_student_and_grader_tech_failures_excluded_from_rasch(self):
        dialogue = _make_dialogue()
        variants = [
            LadderVariant("v1", "V1", "10", b=-0.5, ladder_rank=1),
            LadderVariant("v2", "V2", "20", b=0.5, ladder_rank=2),
        ]
        # 4 climbs on Level 0:
        # climb 0: student returns None (student tech fail -> stop, no obs)
        # climb 1: student returns wrong-rule text, 27B exhausts retries (grader tech fail -> stop, no obs)
        # climb 2: student returns correct \boxed{10} -> advances to Level 1, fails Level 1
        # climb 3: student returns \boxed{99} (27B says false) -> obs (-0.5, 0), stops
        lvl0_responses = [None, "TRIGGER_GRADER_TECH_FAIL", r"\boxed{10}", r"\boxed{99}"]
        lvl1_responses = [r"\boxed{99}"]

        async def fake_sampler(msgs):
            if "V1" in msgs[-1]["content"]:
                return lvl0_responses.pop(0)
            return lvl1_responses.pop(0)

        async def selective_llm(msgs):
            if "TRIGGER_GRADER_TECH_FAIL" in msgs[1]["content"]:
                raise TimeoutError("27B timeout")
            return '{"correct": false}', "stop"

        grader = HybridGrader(
            llm_caller=selective_llm,
            max_attempts=2,
            window=AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0),
            backoff_base=0.0,
        )
        res = _run(
            climb_ladder_batch(
                messages=dialogue,
                problem="ORIG_PROBLEM_ALPHA",
                variants=variants,
                student_client=None,
                student_model="fake-student",
                grader=grader,
                num_climbs=4,
                student_sampler=fake_sampler,
            )
        )
        self.assertEqual(res.student_calls, 5)
        self.assertEqual(res.student_failed, 1)
        self.assertEqual(res.grader_tech_failed, 1)
        self.assertEqual(res.reached_counts, (4, 1))
        self.assertEqual(res.passed_counts, (1, 0))
        # Only climb 2 (L0=1, L1=0) and climb 3 (L0=0) enter Rasch -> 3 observations total
        self.assertEqual(res.observations, ((-0.5, 1), (-0.5, 0), (0.5, 0)))
        self.assertEqual(res.climb_outcomes[0].stop_reason, "student_tech_failure")
        self.assertEqual(res.climb_outcomes[1].stop_reason, "grader_tech_failure")
        self.assertEqual(res.climb_outcomes[2].stop_reason, "incorrect")
        self.assertEqual(res.climb_outcomes[3].stop_reason, "incorrect")


class TestRouteARewardFunction(unittest.TestCase):
    def setUp(self):
        reset_batch_stats()

    def _base_kwargs(self, mode: str = "ladder_rasch", **overrides) -> dict[str, Any]:
        kw = {
            "mode": mode,
            "judge_api_base": "http://fake-judge/v1",
            "judge_api_key_env": "FAKE_API_KEY",
            "judge_model": "judge-local",
            "judge_votes": 3,
            "judge_temperature": 0.6,
            "judge_max_tokens": 4096,
            "judge_hard_gate": True,
            "gate_penalty": 0.2,
            "eoc_bonus": 0.0,
            "think_reward": 0.0,
            "enable_thinking": True,
            "max_tokens_per_turn": 150,
            "max_thinking_tokens_per_turn": 512,
            "student_api_base": "http://fake-student/v1",
            "student_api_key_env": "FAKE_API_KEY",
            "student_model": "student-local",
            "student_temperature": 0.6,
            "num_student_attempts": 16,
            "ladder_num_climbs": 4,
            "omit_teacher_turns": True,
            "test_on_transfer": True,
            "eval_judge_model": "",
            "student_concurrency": 0,
            "judge_concurrency": 0,
            "num_retries": 0,
            "enable_batch_logging": False,
        }
        kw.update(overrides)
        return kw

    def test_gate_rejected_returns_minus_0_2_and_skips_posttest(self):
        for leak_dec, teach_dec in (("REJECT", "OK"), ("OK", "REJECT")):
            judge_client = FakeOpenAIClient(
                lambda kw, _idx, ld=leak_dec, td=teach_dec: _fake_completion(
                    json.dumps(
                        {
                            "reasoning": "ok",
                            "decision": ld if "leak" in kw["messages"][0]["content"].lower() else td,
                        }
                    )
                )
            )
            student_client = FakeOpenAIClient(lambda _kw, _idx: (_ for _ in ()).throw(AssertionError("student should not be called")))
            grader_calls = 0

            async def fail_llm(_msgs):
                nonlocal grader_calls
                grader_calls += 1
                raise AssertionError("grader should not be called")

            grader = HybridGrader(llm_caller=fail_llm, backoff_base=0.0)
            dialogue = _make_dialogue()
            extra = {
                "problem": "ORIG_PROBLEM_ALPHA",
                "answer": "42",
                "messages": dialogue,
                "variants": [
                    {"variant_id": "v1", "problem": "V1", "answer": "10", "b": 0.0, "ladder_rank": 1}
                ],
            }

            def fake_client_factory(api_base, _key_env):
                return judge_client if "judge" in api_base else student_client

            with patch.object(
                sys.modules["training.routeA_reward.reward_function"],
                "get_async_openai_client",
                side_effect=fake_client_factory,
            ):
                res = _run(
                    route_a_reward_func(
                        data_source="routeA",
                        ground_truth="42",
                        extra_info=extra,
                        **self._base_kwargs("ladder_rasch", hybrid_grader=grader),
                    )
                )

            self.assertEqual(res["score"], -0.2)
            self.assertEqual(res["gated"], 1)
            self.assertEqual(res["r_gate"], -0.2)
            self.assertEqual(res["ladder_student_calls"], 0)
            self.assertEqual(len(student_client.calls), 0)
            self.assertEqual(grader_calls, 0)

    def test_formula5_decay_on_positive_gain_and_no_decay_on_negative_gain(self):
        # Long teacher turn (>150 visible tokens and >512 thinking tokens) so both decays < 1.0
        long_think = " ".join(["thinkword"] * 800)
        long_vis = " ".join(["visword"] * 400)
        dialogue = _make_dialogue(teacher_think=long_think, teacher_visible=long_vis)

        class CharLenTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False):
                return text.split()

        tok = CharLenTokenizer()
        expected_vis_decay = author_rf._token_weighted_decay([400], 150.0)
        expected_think_decay = author_rf._token_weighted_decay([800], 512.0)
        self.assertLess(expected_vis_decay, 1.0)
        self.assertLess(expected_think_decay, 1.0)

        judge_client = FakeOpenAIClient(
            lambda _kw, _idx: _fake_completion('{"reasoning": "good", "decision": "OK"}')
        )

        # Case 1: all correct -> theta > 0 -> decayed by gamma_eff
        student_pos = FakeOpenAIClient(lambda _kw, _idx: _fake_completion(r"\boxed{10}"))
        # Case 2: all wrong -> theta < 0 -> NOT decayed
        student_neg = FakeOpenAIClient(lambda _kw, _idx: _fake_completion(r"\boxed{999}"))

        async def false_llm(_msgs):
            return '{"correct": false}', "stop"

        extra = {
            "problem": "ORIG_PROBLEM_ALPHA",
            "answer": "42",
            "messages": dialogue,
            "variants": [
                {"variant_id": "v1", "problem": "V1", "answer": "10", "b": 0.5, "ladder_rank": 1}
            ],
        }

        for student_stub, expect_positive in ((student_pos, True), (student_neg, False)):
            grader = HybridGrader(llm_caller=false_llm, backoff_base=0.0)

            def fake_factory(api_base, _key_env, s=student_stub):
                return judge_client if "judge" in api_base else s

            with patch.object(
                sys.modules["training.routeA_reward.reward_function"],
                "get_async_openai_client",
                side_effect=fake_factory,
            ):
                res = _run(
                    route_a_reward_func(
                        data_source="routeA",
                        ground_truth="42",
                        extra_info=extra,
                        tokenizer=tok,
                        **self._base_kwargs("ladder_rasch", hybrid_grader=grader),
                    )
                )

            theta = res["rasch_theta"]
            self.assertEqual(res["rasch_valid"], 1)
            self.assertEqual(res["rasch_prior_pulled"], 1)
            self.assertAlmostEqual(res["rasch_theta_scale"], 0.25, places=12)
            expected_g = theta * 0.25
            self.assertAlmostEqual(res["rasch_gain_g"], expected_g, places=12)
            if expect_positive:
                self.assertGreater(theta, 0.0)
                self.assertAlmostEqual(
                    res["score"],
                    expected_g * expected_vis_decay * expected_think_decay,
                    places=12,
                )
            else:
                self.assertLess(theta, 0.0)
                self.assertAlmostEqual(res["score"], expected_g, places=12)
                self.assertEqual(res["r_length"], 0.0)
                self.assertEqual(res["r_think_length"], 0.0)

    def test_all_climbs_tech_failed_returns_invalid_zero_score(self):
        dialogue = _make_dialogue()
        judge_client = FakeOpenAIClient(
            lambda _kw, _idx: _fake_completion('{"reasoning": "ok", "decision": "OK"}')
        )
        student_client = FakeOpenAIClient(
            lambda _kw, _idx: (_ for _ in ()).throw(asyncio.TimeoutError("student down"))
        )
        grader = HybridGrader(
            llm_caller=lambda _m: asyncio.sleep(0, result=('{"correct": false}', "stop")),
            backoff_base=0.0,
        )
        extra = {
            "problem": "ORIG_PROBLEM_ALPHA",
            "answer": "42",
            "messages": dialogue,
            "variants": [
                {"variant_id": "v1", "problem": "V1", "answer": "10", "b": 0.0, "ladder_rank": 1}
            ],
        }
        with patch.object(
            sys.modules["training.routeA_reward.reward_function"],
            "get_async_openai_client",
            side_effect=lambda base, _env: judge_client if "judge" in base else student_client,
        ):
            res = _run(
                route_a_reward_func(
                    data_source="routeA",
                    ground_truth="42",
                    extra_info=extra,
                    **self._base_kwargs("ladder_rasch", hybrid_grader=grader),
                )
            )
        self.assertEqual(res["rasch_valid"], 0)
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["ladder_student_failed"], 4)

    def test_author_mode_bit_for_bit_identical_to_upstream(self):
        scenarios = [
            # (judge_decision, student_answers, pre_transfer, long_teacher)
            ("OK", [r"\boxed{77}"] * 12 + [r"\boxed{0}"] * 4, 0.25, True),
            ("OK", [r"\boxed{77}"] * 2 + [r"\boxed{0}"] * 14, 0.50, False),
            ("REJECT", [r"\boxed{77}"] * 16, 0.25, False),
        ]

        for judge_dec, answers, pre_tr, long_t in scenarios:
            vis = " ".join(["w"] * 220) if long_t else "short hint"
            think = " ".join(["t"] * 600) if long_t else "short think"
            dialogue = _make_dialogue(teacher_think=think, teacher_visible=vis)
            extra = {
                "problem": "ORIG_P",
                "answer": "12",
                "messages": dialogue,
                "pre_solve_rate": 0.30,
                "transfer_problem": "TRANS_P",
                "transfer_answer": "77",
                "pre_solve_rate_transfer_problem": pre_tr,
            }

            async def run_one(use_route_a: bool) -> dict[str, Any]:
                ans_iter = list(answers)
                jc = FakeOpenAIClient(
                    lambda _kw, _idx, d=judge_dec: _fake_completion(
                        json.dumps({"reasoning": "r", "decision": d})
                    )
                )
                sc = FakeOpenAIClient(
                    lambda _kw, idx, arr=ans_iter: _fake_completion(arr[idx % len(arr)])
                )
                author_kw = {
                    "judge_api_base": "http://fake-judge/v1",
                    "judge_api_key_env": "FAKE_KEY",
                    "judge_model": "judge-local",
                    "judge_votes": 3,
                    "judge_temperature": 0.6,
                    "judge_max_tokens": 4096,
                    "judge_hard_gate": True,
                    "gate_penalty": 0.2,
                    "eoc_bonus": 0.0,
                    "think_reward": 0.0,
                    "enable_thinking": True,
                    "max_tokens_per_turn": 150,
                    "max_thinking_tokens_per_turn": 512,
                    "student_api_base": "http://fake-student/v1",
                    "student_api_key_env": "FAKE_KEY",
                    "student_model": "student-local",
                    "student_temperature": 0.6,
                    "num_student_attempts": 16,
                    "omit_teacher_turns": True,
                    "test_on_transfer": True,
                    "eval_judge_model": "",
                    "student_concurrency": 0,
                    "judge_concurrency": 0,
                    "num_retries": 0,
                }
                with patch.object(
                    author_rf,
                    "get_async_openai_client",
                    side_effect=lambda base, _env: jc if "judge" in base else sc,
                ), patch.dict(os.environ, {"EDUARDO_CONV_DIR": ""}, clear=False):
                    if use_route_a:
                        return await route_a_reward_func(
                            data_source="bigmath",
                            ground_truth="12",
                            extra_info=extra,
                            mode="author",
                            ladder_num_climbs=8,
                            **author_kw,
                        )
                    return await author_rf.eduardo_reward_func(
                        data_source="bigmath",
                        ground_truth="12",
                        extra_info=extra,
                        **author_kw,
                    )

            ref = _run(run_one(False))
            got = _run(run_one(True))
            self.assertEqual(set(got.keys()), set(ref.keys()))
            for k in ref:
                if isinstance(ref[k], float):
                    self.assertEqual(
                        _float_bits(got[k]),
                        _float_bits(ref[k]),
                        msg=f"float bit mismatch on key={k}: got={got[k]!r} ref={ref[k]!r}",
                    )
                else:
                    self.assertEqual(got[k], ref[k], msg=f"mismatch on key={k}")

    def test_author_hybrid_middle_variant_and_hybrid_promotion(self):
        dialogue = _make_dialogue()
        seen_prompts: list[str] = []
        # 4 attempts: 2 rule-exact \boxed{20}, 1 unboxed "twenty" (27B promotes!), 1 wrong \boxed{99}
        responses = [r"\boxed{20}", r"\boxed{20}", "The final answer is twenty.", r"\boxed{99}"]

        judge_client = FakeOpenAIClient(
            lambda _kw, _idx: _fake_completion('{"reasoning": "ok", "decision": "OK"}')
        )

        def student_handler(kw, idx):
            seen_prompts.append(kw["messages"][-1]["content"])
            return _fake_completion(responses[idx])

        student_client = FakeOpenAIClient(student_handler)

        async def promo_llm(msgs):
            if "twenty" in msgs[1]["content"]:
                return '{"correct": true}', "stop"
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=promo_llm, backoff_base=0.0)
        extra = {
            "problem": "ORIG_PROBLEM_ALPHA",
            "answer": "42",
            "messages": dialogue,
            "variants": [
                {"variant_id": "v1", "problem": "EASY_V1", "answer": "10", "p_new": 0.7, "b": difficulty_from_rate(0.7), "ladder_rank": 1},
                {"variant_id": "v2", "problem": "MID_V2", "answer": "20", "p_new": 0.25, "b": difficulty_from_rate(0.25), "ladder_rank": 2},
                {"variant_id": "v3", "problem": "HARD_V3", "answer": "30", "p_new": 0.1, "b": difficulty_from_rate(0.1), "ladder_rank": 3},
            ],
        }

        with patch.object(
            sys.modules["training.routeA_reward.reward_function"],
            "get_async_openai_client",
            side_effect=lambda base, _env: judge_client if "judge" in base else student_client,
        ):
            res = _run(
                route_a_reward_func(
                    data_source="routeA",
                    ground_truth="42",
                    extra_info=extra,
                    **self._base_kwargs(
                        "author_hybrid",
                        num_student_attempts=4,
                        hybrid_grader=grader,
                    ),
                )
            )

        # Must have tested ONLY the middle variant MID_V2
        self.assertEqual(len(seen_prompts), 4)
        for p in seen_prompts:
            self.assertIn("MID_V2", p)
            self.assertNotIn("EASY_V1", p)
            self.assertNotIn("HARD_V3", p)

        # 3/4 = 0.75 post rate, pre = 0.25 -> normalized delta = (0.75 - 0.25) / (1 - 0.25) = 2/3
        self.assertAlmostEqual(res["post_solve_rate"], 0.75, places=12)
        self.assertAlmostEqual(res["pre_solve_rate"], 0.25, places=12)
        self.assertAlmostEqual(res["score"], (0.75 - 0.25) / (1.0 - 0.25), places=12)
        self.assertEqual(res["grader_rule_correct"], 2)
        self.assertEqual(res["grader_llm_calls"], 2)
        self.assertEqual(res["grader_llm_unknown"], 0)

        # Check middle selection for 1, 2, 3 variants
        v1 = LadderVariant("a", "P1", "1", -1.0, 1)
        v2 = LadderVariant("b", "P2", "2", 0.0, 2)
        v3 = LadderVariant("c", "P3", "3", 1.0, 3)
        self.assertEqual(select_middle_variant([v1]).variant_id, "a")
        self.assertEqual(select_middle_variant([v1, v2], even_policy="lower").variant_id, "a")
        self.assertEqual(select_middle_variant([v1, v2], even_policy="upper").variant_id, "b")
        self.assertEqual(select_middle_variant([v1, v2, v3]).variant_id, "b")

    def test_author_hybrid_matches_author_when_27b_never_promotes(self):
        dialogue = _make_dialogue(
            teacher_think=" ".join(["t"] * 600),
            teacher_visible=" ".join(["v"] * 200),
        )
        answers = [r"\boxed{77}"] * 10 + [r"\boxed{1}"] * 6
        extra = {
            "problem": "ORIG_P",
            "answer": "12",
            "messages": dialogue,
            "pre_solve_rate": 0.2,
            "transfer_problem": "TRANS_P",
            "transfer_answer": "77",
            "pre_solve_rate_transfer_problem": 0.25,
        }

        async def run_mode(mode_name: str) -> dict[str, Any]:
            jc = FakeOpenAIClient(
                lambda _kw, _idx: _fake_completion('{"reasoning": "ok", "decision": "OK"}')
            )
            sc = FakeOpenAIClient(
                lambda _kw, idx: _fake_completion(answers[idx % len(answers)])
            )

            async def always_false(_msgs):
                return '{"correct": false}', "stop"

            grader = HybridGrader(llm_caller=always_false, backoff_base=0.0)
            kw = self._base_kwargs(
                mode_name,
                num_student_attempts=16,
                hybrid_grader=grader,
            )
            with patch.object(
                author_rf,
                "get_async_openai_client",
                side_effect=lambda base, _env: jc if "judge" in base else sc,
            ), patch.object(
                sys.modules["training.routeA_reward.reward_function"],
                "get_async_openai_client",
                side_effect=lambda base, _env: jc if "judge" in base else sc,
            ), patch.dict(os.environ, {"EDUARDO_CONV_DIR": ""}, clear=False):
                return await route_a_reward_func(
                    data_source="bigmath",
                    ground_truth="12",
                    extra_info=extra,
                    **kw,
                )

        r_author = _run(run_mode("author"))
        r_hybrid = _run(run_mode("author_hybrid"))
        for key in ("score", "post_solve_rate", "pre_solve_rate", "delta_solve", "r_length", "r_think_length"):
            self.assertEqual(_float_bits(r_hybrid[key]), _float_bits(r_author[key]), msg=f"mismatch on {key}")

    def test_no_problem_answer_or_response_in_logs_and_step_summary(self):
        sentinels = [
            "SENTINEL_ORIG_PROB_9911",
            "SENTINEL_ORIG_ANS_8822",
            "SENTINEL_VAR1_PROB_7733",
            "SENTINEL_VAR1_ANS_6644",
            "SENTINEL_VAR2_PROB_5566",
            "SENTINEL_VAR2_ANS_4455",
            "SENTINEL_TEACHER_THINK_3344",
            "SENTINEL_TEACHER_VIS_2233",
            "SENTINEL_STUDENT_DIALOG_1122",
            "SENTINEL_STUDENT_ATTEMPT_0099",
            "SENTINEL_JUDGE_REASON_8877",
        ]
        dialogue = _make_dialogue(
            problem=sentinels[0],
            teacher_think=sentinels[6],
            teacher_visible=sentinels[7],
            student_reply=sentinels[8],
        )

        judge_client = FakeOpenAIClient(
            lambda _kw, _idx: _fake_completion(
                json.dumps({"reasoning": sentinels[10], "decision": "OK"})
            )
        )
        # Alternate exact-correct on V1 and unboxed text (triggering 27B null/unknown) on V2
        student_client = FakeOpenAIClient(
            lambda kw, idx: _fake_completion(
                rf"\boxed{{{sentinels[3]}}}"
                if sentinels[2] in kw["messages"][-1]["content"]
                else f"My guess is {sentinels[9]}"
            )
        )

        async def unknown_llm(_msgs):
            return '{"correct": null}', "stop"

        extra = {
            "problem": sentinels[0],
            "answer": sentinels[1],
            "messages": dialogue,
            "variants": [
                {"variant_id": "v1", "problem": sentinels[2], "answer": sentinels[3], "b": -0.5, "ladder_rank": 1},
                {"variant_id": "v2", "problem": sentinels[4], "answer": sentinels[5], "b": 0.5, "ladder_rank": 2},
            ],
        }

        captured_records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(self.format(record))

        handler = ListHandler()
        target_loggers = [
            logging.getLogger(),
            logging.getLogger("routeA_reward"),
            logging.getLogger("routeA_reward.hybrid_grader"),
        ]
        old_levels = [lg.level for lg in target_loggers]
        for lg in target_loggers:
            lg.addHandler(handler)
            lg.setLevel(logging.DEBUG)

        try:
            with patch.object(
                sys.modules["training.routeA_reward.reward_function"],
                "get_async_openai_client",
                side_effect=lambda base, _env: judge_client if "judge" in base else student_client,
            ):
                grader = HybridGrader(llm_caller=unknown_llm, backoff_base=0.0)
                r1 = _run(
                    route_a_reward_func(
                        data_source="routeA",
                        ground_truth=sentinels[1],
                        extra_info=extra,
                        **self._base_kwargs(
                            "ladder_rasch",
                            ladder_num_climbs=2,
                            hybrid_grader=grader,
                            enable_batch_logging=True,
                            batch_log_interval=2,
                        ),
                    )
                )
                # Empty transcript triggers warning + batch flush at interval=2
                r2 = _run(
                    route_a_reward_func(
                        data_source="routeA",
                        ground_truth=sentinels[1],
                        extra_info={"problem": sentinels[0], "answer": sentinels[1]},
                        **self._base_kwargs(
                            "ladder_rasch",
                            ladder_num_climbs=2,
                            hybrid_grader=grader,
                            enable_batch_logging=True,
                            batch_log_interval=2,
                        ),
                    )
                )
        finally:
            for lg, lvl in zip(target_loggers, old_levels):
                lg.removeHandler(handler)
                lg.setLevel(lvl)

        # Ensure logs were actually emitted (not vacuously empty)
        self.assertGreaterEqual(len(captured_records), 3)
        all_logs = "\n".join(captured_records)
        result_blob = json.dumps([r1, r2], ensure_ascii=False)
        for s in sentinels:
            self.assertNotIn(s, all_logs, msg=f"leaked sentinel {s} into logs")
            self.assertNotIn(s, result_blob, msg=f"leaked sentinel {s} into result dict")

        # Verify step-level summary metrics
        summary = summarize_step_metrics([r1, r2])
        self.assertAlmostEqual(summary["eduardo/routeA_theta_mean"], r1["rasch_theta"], places=12)
        self.assertAlmostEqual(summary["eduardo/routeA_theta_std"], 0.0, places=12)
        self.assertAlmostEqual(summary["eduardo/routeA_theta_valid_rate"], 0.5, places=12)
        self.assertAlmostEqual(summary["eduardo/routeA_reach_l1_rate"], 0.5, places=12)
        self.assertAlmostEqual(summary["eduardo/routeA_reach_l2_rate"], 0.5, places=12)
        self.assertAlmostEqual(summary["eduardo/routeA_grader_llm_calls_total"], 2.0, places=12)
        self.assertAlmostEqual(summary["eduardo/routeA_grader_llm_unknown_rate"], 1.0, places=12)

    def test_resolve_reward_mode(self):
        self.assertEqual(resolve_reward_mode({}), "ladder_rasch")
        self.assertEqual(resolve_reward_mode({"mode": "author"}), "author")
        self.assertEqual(resolve_reward_mode({"reward_mode": "author_hybrid"}), "author_hybrid")
        self.assertEqual(resolve_reward_mode({"reward": {"mode": "author"}}), "author")
        with self.assertRaises(ValueError):
            resolve_reward_mode({"mode": "invalid_mode"})


if __name__ == "__main__":
    unittest.main()
