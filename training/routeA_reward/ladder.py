"""Ladder post-test runner for Route A reward.

Given a completed tutoring dialogue on an original problem and 1 to 3 transfer
variants ordered by ``ladder_rank`` (easiest -> hardest), runs ``k`` independent
ladder climbs concurrently level-by-level:
    - Level 0 (easiest): fire ``k`` student requests concurrently;
    - Each climb sees ONLY the masked original dialogue (via author's
      ``eduardo.metrics.correctness._build_final_attempt_messages``) plus the
      current level's variant — never any previous level's variant or attempt;
    - Correct climbs advance to the next level; wrong climbs stop;
    - Student or grader technical failures stop that climb WITHOUT recording a
      wrong observation in Rasch MAP and are counted separately.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from eduardo.metrics import correctness as author_correctness
from eduardo.utils import call_openai_with_retries

from .rasch import difficulty_from_rate, difficulty_from_solve_count

DEFAULT_LADDER_CLIMBS = 8
DEFAULT_STUDENT_FINAL_MAX_TOKENS = 800
DEFAULT_STUDENT_TEMPERATURE = 0.6


class AsyncGraderProtocol(Protocol):
    async def grade(
        self,
        problem: str,
        reference_answer: str,
        student_response: str,
    ) -> Any: ...


@dataclass(frozen=True)
class LadderVariant:
    """One rung on the transfer ladder (never the original problem)."""

    variant_id: str
    problem: str
    answer: str
    b: float
    ladder_rank: int
    p_new: Optional[float] = None


@dataclass(frozen=True)
class ClimbOutcome:
    """Metadata-only outcome of one climb (contains zero raw problem/answer text)."""

    climb_index: int
    levels_attempted: int
    levels_passed: int
    stop_reason: str  # "completed" | "incorrect" | "student_tech_failure" | "grader_tech_failure"


@dataclass(frozen=True)
class LadderBatchResult:
    """Aggregated outcome of ``k`` independent ladder climbs for one dialogue."""

    observations: tuple[tuple[float, int], ...]
    num_climbs: int
    num_variants: int
    reached_counts: tuple[int, ...]
    passed_counts: tuple[int, ...]
    student_calls: int
    student_failed: int
    student_incorrect_format: int
    grader_total_calls: int
    grader_rule_correct: int
    grader_llm_calls: int
    grader_llm_unknown: int
    grader_tech_failed: int
    climb_outcomes: tuple[ClimbOutcome, ...]


def _parse_variant_b_and_p(raw: dict[str, Any]) -> tuple[float, Optional[float]]:
    p_val: Optional[float] = None
    for p_key in ("p_new", "pre_solve_rate"):
        if raw.get(p_key) is not None:
            p_val = float(raw[p_key])
            break

    if raw.get("b") is not None:
        b_val = float(raw["b"])
        if not math.isfinite(b_val):
            raise ValueError("variant difficulty b must be finite")
        return b_val, p_val

    if p_val is not None:
        return difficulty_from_rate(p_val), p_val

    if raw.get("n_correct_hybrid") is not None:
        n_total = int(raw.get("n_total", 16))
        p_calc, b_calc = difficulty_from_solve_count(int(raw["n_correct_hybrid"]), n_total)
        return b_calc, p_calc

    raise ValueError("variant missing difficulty 'b' (and no p_new / n_correct_hybrid)")


def extract_ladder_variants(extra_info: dict[str, Any]) -> list[LadderVariant]:
    """Parse and sort 1..3 ladder variants from ``extra_info['variants']``.

    Original problem is never placed on the ladder. Variants are sorted ascending
    by ``ladder_rank`` (easiest first).
    """
    raw_variants = extra_info.get("variants")
    if isinstance(raw_variants, str):
        raw_variants = json.loads(raw_variants)
    if not isinstance(raw_variants, (list, tuple)) or len(raw_variants) == 0:
        raise ValueError("extra_info['variants'] must be a non-empty list of variants")

    parsed: list[tuple[int, int, LadderVariant]] = []
    for idx, item in enumerate(raw_variants):
        if not isinstance(item, dict):
            raise ValueError(f"variant at index {idx} must be a dict")
        v_prob = item.get("problem") or item.get("variant_problem") or item.get("question") or ""
        v_ans = item.get("answer") or item.get("reference_answer") or item.get("variant_answer") or ""
        if not isinstance(v_prob, str) or not v_prob.strip():
            raise ValueError(f"variant at index {idx} missing non-empty problem")
        if not isinstance(v_ans, str) or not v_ans.strip():
            raise ValueError(f"variant at index {idx} missing non-empty answer")
        if item.get("ladder_rank") is None or isinstance(item.get("ladder_rank"), bool):
            raise ValueError(f"variant at index {idx} missing integer ladder_rank")
        rank = int(item["ladder_rank"])
        b_val, p_val = _parse_variant_b_and_p(item)
        vid = str(item.get("variant_id") or f"v{idx}")
        parsed.append(
            (
                rank,
                idx,
                LadderVariant(
                    variant_id=vid,
                    problem=v_prob,
                    answer=v_ans,
                    b=b_val,
                    ladder_rank=rank,
                    p_new=p_val,
                ),
            )
        )

    parsed.sort(key=lambda t: (t[0], t[1]))
    return [v for _, _, v in parsed]


def select_middle_variant(
    variants: Sequence[LadderVariant],
    *,
    even_policy: str = "lower",
) -> LadderVariant:
    """Select the middle-rank variant for ``author_hybrid`` (ablation A1).

    For 1 variant -> index 0; for 3 variants -> index 1.
    For 2 variants -> ``even_policy='lower'`` picks index 0 (easier);
    ``even_policy='upper'`` picks index 1 (harder).
    """
    n = len(variants)
    if n == 0:
        raise ValueError("variants must be non-empty")
    if n % 2 == 1:
        return variants[n // 2]
    if even_policy == "lower":
        return variants[(n - 1) // 2]
    if even_policy == "upper":
        return variants[n // 2]
    raise ValueError(f"unknown even_policy: {even_policy!r}")


async def climb_ladder_batch(
    *,
    messages: list[dict[str, Any]],
    problem: str,
    variants: Sequence[LadderVariant],
    student_client: Any,
    student_model: str,
    grader: AsyncGraderProtocol,
    num_climbs: int = DEFAULT_LADDER_CLIMBS,
    temperature: float = DEFAULT_STUDENT_TEMPERATURE,
    max_tokens: int = DEFAULT_STUDENT_FINAL_MAX_TOKENS,
    omit_teacher_turns: bool = True,
    retry_kwargs: Optional[dict[str, Any]] = None,
    student_semaphore: Optional[asyncio.Semaphore] = None,
    student_sampler: Optional[
        Callable[[list[dict[str, Any]]], Awaitable[Optional[str]]]
    ] = None,
) -> LadderBatchResult:
    """Execute ``num_climbs`` independent ladder climbs level-by-level.

    Calls author's ``correctness._build_final_attempt_messages`` directly so
    teacher masking and the transfer final-attempt prompt stay identical to
    upstream EduardoRL.
    """
    if num_climbs < 1:
        raise ValueError(f"num_climbs must be >= 1, got {num_climbs}")
    if not variants:
        raise ValueError("variants must be non-empty")

    retry_kw = dict(retry_kwargs or {})
    num_variants = len(variants)
    reached_counts = [0] * num_variants
    passed_counts = [0] * num_variants
    observations: list[tuple[float, int]] = []

    student_calls = 0
    student_failed = 0
    student_incorrect_format = 0
    grader_total_calls = 0
    grader_rule_correct = 0
    grader_llm_calls = 0
    grader_llm_unknown = 0
    grader_tech_failed = 0

    levels_attempted = [0] * num_climbs
    levels_passed = [0] * num_climbs
    stop_reasons = ["completed"] * num_climbs

    active_climbs = list(range(num_climbs))

    for lvl, variant in enumerate(variants):
        if not active_climbs:
            break

        # Build freshly from the original dialogue: never leaks prior ladder rungs.
        final_msgs = author_correctness._build_final_attempt_messages(
            messages,
            problem,
            transfer_problem=variant.problem,
            omit_teacher_turns=omit_teacher_turns,
        )

        async def _sample_one(msgs_payload: list[dict[str, Any]] = final_msgs) -> Optional[str]:
            if student_sampler is not None:
                return await student_sampler(msgs_payload)
            return await call_openai_with_retries(
                author_correctness._sample_once,
                student_client,
                student_model,
                msgs_payload,
                temperature,
                max_tokens,
                label="student_final",
                use_fallback=True,
                fallback=None,
                semaphore=student_semaphore,
                **retry_kw,
            )

        reached_counts[lvl] = len(active_climbs)
        student_calls += len(active_climbs)
        for cid in active_climbs:
            levels_attempted[cid] = lvl + 1

        # Fire all active climbs for this level concurrently.
        responses = await asyncio.gather(*[_sample_one() for _ in active_climbs])

        # Grade non-failed responses concurrently.
        gradable_indices: list[int] = []
        grade_tasks: list[Awaitable[Any]] = []
        for idx_in_lvl, text in enumerate(responses):
            cid = active_climbs[idx_in_lvl]
            if text is None:
                student_failed += 1
                stop_reasons[cid] = "student_tech_failure"
            else:
                gradable_indices.append(idx_in_lvl)
                grade_tasks.append(grader.grade(variant.problem, variant.answer, text))

        grade_results = await asyncio.gather(*grade_tasks) if grade_tasks else []

        next_active: list[int] = []
        for idx_in_lvl, g_res in zip(gradable_indices, grade_results):
            cid = active_climbs[idx_in_lvl]
            grader_total_calls += 1
            student_incorrect_format += int(getattr(g_res, "incorrect_format", 0))
            if getattr(g_res, "rule_correct", False):
                grader_rule_correct += 1
            if getattr(g_res, "llm_called", False):
                grader_llm_calls += 1
            if getattr(g_res, "is_unknown", False):
                grader_llm_unknown += 1

            if getattr(g_res, "technical_failure", False):
                # Technical failure is NOT a verdict: stop climb, exclude from Rasch.
                grader_tech_failed += 1
                stop_reasons[cid] = "grader_tech_failure"
                continue

            is_correct = bool(getattr(g_res, "correct", False))
            observations.append((variant.b, 1 if is_correct else 0))
            if is_correct:
                passed_counts[lvl] += 1
                levels_passed[cid] = lvl + 1
                next_active.append(cid)
            else:
                stop_reasons[cid] = "incorrect"

        active_climbs = next_active

    outcomes = tuple(
        ClimbOutcome(
            climb_index=cid,
            levels_attempted=levels_attempted[cid],
            levels_passed=levels_passed[cid],
            stop_reason=stop_reasons[cid],
        )
        for cid in range(num_climbs)
    )

    return LadderBatchResult(
        observations=tuple(observations),
        num_climbs=num_climbs,
        num_variants=num_variants,
        reached_counts=tuple(reached_counts),
        passed_counts=tuple(passed_counts),
        student_calls=student_calls,
        student_failed=student_failed,
        student_incorrect_format=student_incorrect_format,
        grader_total_calls=grader_total_calls,
        grader_rule_correct=grader_rule_correct,
        grader_llm_calls=grader_llm_calls,
        grader_llm_unknown=grader_llm_unknown,
        grader_tech_failed=grader_tech_failed,
        climb_outcomes=outcomes,
    )
