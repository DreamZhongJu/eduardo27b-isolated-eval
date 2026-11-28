"""veRL custom reward function for Route A (ladder post-test + Rasch MAP gain).

Reuses author EduardoRL functions directly without modifying any author files:
- Gates: ``eduardo.metrics.judge_quality.majority_vote``, ``strip_teacher_controls``
- Efficiency decay & teacher stats: ``eduardo.reward_function._analyze_teacher_turn``,
  ``_count_tokens_per_turn``, ``_token_weighted_decay``, ``_is_substantive_visible``
- Masking & final prompt: ``eduardo.metrics.correctness._build_final_attempt_messages``,
  ``_sample_once``
- Normalized gain (author_hybrid): ``eduardo.reward_function._normalized_delta``
- Author mode: direct delegation to ``eduardo.reward_function.eduardo_reward_func``
  (guaranteed bit-for-bit identical).

Switch ``reward.mode`` (via ``reward_kwargs.mode`` / ``reward_kwargs.reward_mode`` /
``reward_kwargs.reward.mode``):
- ``ladder_rasch``: Route A ladder post-test + Rasch MAP ability gain + hybrid grader
- ``author``: exact upstream ``eduardo.reward_function.eduardo_reward_func``
- ``author_hybrid``: author formula (4) on one variant (middle rank for Route A data)
  with hybrid grading (for ablation A1)

Privacy guarantee:
    This module NEVER logs problem text, answers, student responses, transcripts,
    or judge reasoning text. Only numeric counts and summary metrics are logged.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional, Sequence
import weakref

from eduardo import reward_function as author_rf
from eduardo.metrics import correctness as author_correctness
from eduardo.metrics import judge_quality as author_judge_quality
from eduardo.utils import (
    call_openai_with_retries,
    get_async_openai_client,
    get_concurrency_semaphore,
    strip_teacher_controls,
)

from .hybrid_grader import (
    AsyncAdaptiveWindow,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_WINDOW,
    HybridGrader,
    _default_openai_llm_call,
)
from .ladder import (
    DEFAULT_LADDER_CLIMBS,
    DEFAULT_STUDENT_FINAL_MAX_TOKENS,
    DEFAULT_STUDENT_TEMPERATURE,
    climb_ladder_batch,
    extract_ladder_variants,
    select_middle_variant,
)
from .rasch import rasch_map_estimate

logger = logging.getLogger("routeA_reward")

VALID_REWARD_MODES = ("ladder_rasch", "author", "author_hybrid")
DEFAULT_THETA_SCALE = 0.25

# Kwargs consumed only by Route A and stripped before delegating to author mode.
_ROUTE_A_ONLY_KWARGS = frozenset(
    {
        "mode",
        "reward_mode",
        "reward",
        "theta_scale",
        "ladder_num_climbs",
        "num_climbs",
        "student_final_max_tokens",
        "hybrid_grader",
        "grader_api_base",
        "grader_api_key_env",
        "grader_model",
        "grader_max_concurrency",
        "grader_max_attempts",
        "grader_temperature",
        "grader_max_tokens",
        "grader_extra_body",
        "grader_use_vllm_no_thinking",
        "grader_backoff_base",
        "grader_independent_semaphore",
        "middle_variant_policy",
        "batch_log_interval",
        "enable_batch_logging",
        "routeA_metrics_dir",
    }
)

# Process-local shared adaptive congestion windows for local 27B hybrid grading.
_GRADER_WINDOWS: dict[str, AsyncAdaptiveWindow] = {}
_GRADER_WINDOW_LOCK = threading.Lock()

# Thread-safe periodic batch summary accumulator (numeric fields only).
_BATCH_LOCK = threading.Lock()
_BATCH_ROWS: list[dict[str, Any]] = []
_BATCH_START_TIME: Optional[float] = None

# Thread-safe numeric-only per-rollout step metrics sink (ZERO text).
_SINK_LOCK = threading.Lock()
_SINK_FILES: dict[str, Any] = {}
_VALID_ROUTE_BRANCHES = ("easy", "hard", "fallback")
_VALID_GATE_FAIL_TYPES = ("pass", "teaching_only", "leak_only", "both")
_METRIC_SINK_KEYS: tuple[str, ...] = (
    "score",
    "r_gate",
    "gated",
    "judge_no_leak_ok",
    "judge_teaching_ok",
    "gate_fail_teaching_only",
    "gate_fail_leak_only",
    "gate_fail_both",
    "rasch_theta",
    "rasch_gain_g",
    "rasch_theta_scale",
    "rasch_valid",
    "rasch_prior_pulled",
    "rasch_all_correct",
    "rasch_all_wrong",
    "compliant_g_lt_neg02",
    "ladder_num_variants",
    "ladder_num_climbs",
    "ladder_student_calls",
    "ladder_student_failed",
    "ladder_reach_l1_rate",
    "ladder_reach_l2_rate",
    "ladder_reach_l3_rate",
    "ladder_pass_l1_rate",
    "ladder_pass_l2_rate",
    "ladder_pass_l3_rate",
    "grader_total_calls",
    "grader_rule_correct",
    "grader_llm_calls",
    "grader_llm_unknown",
    "grader_tech_failed",
)


def _normalize_route_branch(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    return val if val in _VALID_ROUTE_BRANCHES else "unknown"


def _normalize_gate_fail_type(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    return val if val in _VALID_GATE_FAIL_TYPES else "pass"


def _median_float(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _get_grader_window(key: str, max_window: int) -> AsyncAdaptiveWindow:
    with _GRADER_WINDOW_LOCK:
        existing = _GRADER_WINDOWS.get(key)
        if existing is None:
            existing = AsyncAdaptiveWindow(max_window=max_window)
            _GRADER_WINDOWS[key] = existing
        return existing


def reset_batch_stats() -> None:
    """Reset module-level numeric batch accumulator (useful in tests)."""
    global _BATCH_ROWS, _BATCH_START_TIME
    with _BATCH_LOCK:
        _BATCH_ROWS = []
        _BATCH_START_TIME = None


def resolve_reward_mode(kwargs: dict[str, Any]) -> str:
    """Resolve ``reward.mode`` from kwargs. Defaults to ``'ladder_rasch'``."""
    raw: Any = None
    if kwargs.get("reward_mode") is not None:
        raw = kwargs["reward_mode"]
    elif kwargs.get("mode") is not None:
        raw = kwargs["mode"]
    elif isinstance(kwargs.get("reward"), dict) and kwargs["reward"].get("mode") is not None:
        raw = kwargs["reward"]["mode"]
    else:
        raw = "ladder_rasch"

    mode = str(raw).strip()
    if mode not in VALID_REWARD_MODES:
        raise ValueError(
            f"invalid reward.mode={mode!r}; expected one of {VALID_REWARD_MODES}"
        )
    return mode


def _record_hybrid_grader_timing(**kwargs: Any) -> None:
    try:
        import queue_metrics  # type: ignore

        queue_metrics.record_custom_timing('hybrid_grader', **kwargs)
    except Exception:
        return


_GRADER_SEM_TIMING: contextvars.ContextVar[tuple[int, int]] = contextvars.ContextVar(
    "_GRADER_SEM_TIMING", default=(0, 0)
)
_GRADER_SEM_TOKEN: contextvars.ContextVar[Optional[contextvars.Token]] = contextvars.ContextVar(
    "_GRADER_SEM_TOKEN", default=None
)


class _TimedGraderSemaphore:
    """Wrap an asyncio.Semaphore to capture per-task queue wait and acquired time."""

    def __init__(self, inner: asyncio.Semaphore) -> None:
        self._inner = inner

    async def __aenter__(self) -> Any:
        t0 = time.monotonic_ns()
        res = await self._inner.__aenter__()
        now = time.monotonic_ns()
        tok = _GRADER_SEM_TIMING.set((max(0, now - t0), now))
        _GRADER_SEM_TOKEN.set(tok)
        return res

    async def __aexit__(self, *exc: Any) -> Any:
        tok = _GRADER_SEM_TOKEN.get()
        if tok is not None:
            _GRADER_SEM_TIMING.reset(tok)
            _GRADER_SEM_TOKEN.set(None)
        return await self._inner.__aexit__(*exc)


def _build_hybrid_grader(
    kwargs: dict[str, Any],
    *,
    judge_semaphore: Optional[asyncio.Semaphore],
) -> Any:
    injected = kwargs.get("hybrid_grader")
    if injected is not None:
        return injected

    api_base = str(kwargs.get("grader_api_base") or kwargs["judge_api_base"])
    api_key_env = str(
        kwargs.get("grader_api_key_env") or kwargs.get("judge_api_key_env", "SERVING_API_KEY")
    )
    model = str(kwargs.get("grader_model") or kwargs["judge_model"])
    max_window = int(kwargs.get("grader_max_concurrency", DEFAULT_MAX_WINDOW))
    max_attempts = int(kwargs.get("grader_max_attempts", DEFAULT_MAX_ATTEMPTS))
    temperature = float(kwargs.get("grader_temperature", 0.0))
    max_tokens = int(kwargs.get("grader_max_tokens", 1024))
    extra_body = kwargs.get("grader_extra_body")
    use_vllm_no_thinking = bool(kwargs.get("grader_use_vllm_no_thinking", True))
    backoff_base = float(kwargs.get("grader_backoff_base", 1.0))

    independent_sem = bool(kwargs.get("grader_independent_semaphore", True))
    raw_sem = (
        get_concurrency_semaphore("hybrid_grader", max_window)
        if independent_sem
        else judge_semaphore
    )

    client = get_async_openai_client(api_base, api_key_env)
    window = _get_grader_window(f"{api_base}::{model}", max_window=max_window)

    use_queue_metrics = bool(os.environ.get("EDU_QUEUE_METRICS_DIR", "").strip())
    if use_queue_metrics and raw_sem is not None:
        timed_sem = _TimedGraderSemaphore(raw_sem)
        resolved_extra_body = (
            dict(extra_body)
            if extra_body is not None
            else ({"chat_template_kwargs": {"enable_thinking": False}} if use_vllm_no_thinking else None)
        )

        async def _timed_llm_caller(messages: list[dict[str, str]]) -> tuple[Optional[str], Optional[str]]:
            svc_start = time.monotonic_ns()
            attempt_err = 0
            try:
                return await _default_openai_llm_call(
                    client,
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=resolved_extra_body,
                )
            except BaseException:
                attempt_err = 1
                raise
            finally:
                end_ns = time.monotonic_ns()
                queued_ns, acquired_ns = _GRADER_SEM_TIMING.get()
                svc_ns = max(0, end_ns - svc_start)
                held_ns = max(svc_ns, end_ns - acquired_ns) if acquired_ns > 0 else svc_ns
                start_ns = max(0, acquired_ns - queued_ns) if acquired_ns > 0 else svc_start
                _record_hybrid_grader_timing(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    queued_ns=queued_ns,
                    service_ns=svc_ns,
                    held_ns=held_ns,
                    attempts=1,
                    attempt_errors=attempt_err,
                    fallback=0,
                    call_error=attempt_err,
                )

        return HybridGrader(
            llm_caller=_timed_llm_caller,
            model=model,
            max_attempts=max_attempts,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            use_vllm_no_thinking=use_vllm_no_thinking,
            window=window,
            semaphore=timed_sem,  # type: ignore[arg-type]
            backoff_base=backoff_base,
        )

    return HybridGrader(
        client=client,
        model=model,
        max_attempts=max_attempts,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        use_vllm_no_thinking=use_vllm_no_thinking,
        window=window,
        semaphore=raw_sem,
        backoff_base=backoff_base,
    )


def _compute_teacher_stats(
    msgs: list[dict[str, Any]],
    *,
    tokenizer: Any,
    enable_thinking: bool,
    target_tokens: float,
    think_target_tokens: float,
) -> dict[str, Any]:
    """Compute teacher token budgets and decays using author functions directly."""
    teacher_contents = [
        (m.get("content") or "") for m in msgs if m.get("role") == "assistant"
    ]
    num_turns = len(teacher_contents)
    turn_analyses = [
        author_rf._analyze_teacher_turn(c, enable_thinking) for c in teacher_contents
    ]
    thinking_token_counts = author_rf._count_tokens_per_turn(
        [thinking for thinking, _, _ in turn_analyses], tokenizer
    )
    visible_token_counts = author_rf._count_tokens_per_turn(
        [visible for _, visible, _ in turn_analyses], tokenizer
    )
    thinking_tokens_total = sum(thinking_token_counts)
    visible_tokens_total = sum(visible_token_counts)
    teacher_tokens_total = thinking_tokens_total + visible_tokens_total
    teacher_tokens_mean = teacher_tokens_total / num_turns if num_turns > 0 else 0.0
    thinking_tokens_mean = thinking_tokens_total / num_turns if num_turns > 0 else 0.0
    visible_tokens_mean = visible_tokens_total / num_turns if num_turns > 0 else 0.0

    student_contents: list[str] = []
    seen_teacher = False
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            seen_teacher = True
        elif role == "user" and seen_teacher:
            student_contents.append(m.get("content") or "")
    student_tokens_total = sum(
        author_rf._count_tokens_per_turn(student_contents, tokenizer)
    )
    shared_tokens = student_tokens_total + visible_tokens_total
    student_token_share = (
        student_tokens_total / shared_tokens if shared_tokens > 0 else 0.0
    )

    length_decay = author_rf._token_weighted_decay(visible_token_counts, target_tokens)
    thinking_decay = author_rf._token_weighted_decay(
        thinking_token_counts, think_target_tokens
    )

    eoc_used = any(
        "<end_of_conversation>" in (m.get("content") or "")
        and m.get("role") == "assistant"
        for m in msgs
    )

    n_think_violations = sum(v for _, _, v in turn_analyses)
    n_missing_think = sum(
        1
        for thinking, visible, _ in turn_analyses
        if not thinking.strip() and author_rf._is_substantive_visible(visible)
    )
    n_correct_think_turns = sum(
        1
        for thinking, visible, violations in turn_analyses
        if violations == 0
        and (bool(thinking.strip()) or not author_rf._is_substantive_visible(visible))
    )
    correct_think_rate = (
        n_correct_think_turns / num_turns if num_turns > 0 else 0.0
    )

    return {
        "num_turns": num_turns,
        "teacher_tokens_total": teacher_tokens_total,
        "teacher_tokens_mean": teacher_tokens_mean,
        "thinking_tokens_total": thinking_tokens_total,
        "thinking_tokens_mean": thinking_tokens_mean,
        "visible_tokens_mean": visible_tokens_mean,
        "student_tokens_total": student_tokens_total,
        "student_token_share": student_token_share,
        "length_decay": length_decay,
        "thinking_decay": thinking_decay,
        "eoc_used": eoc_used,
        "n_think_violations": n_think_violations,
        "n_missing_think": n_missing_think,
        "n_correct_think_turns": n_correct_think_turns,
        "correct_think_rate": correct_think_rate,
    }


def _empty_route_a_reward(
    *,
    num_climbs: int = DEFAULT_LADDER_CLIMBS,
    omit_teacher_turns: bool = True,
    theta_scale: float = DEFAULT_THETA_SCALE,
    route_branch: str = "unknown",
) -> dict[str, Any]:
    """Fixed-schema numeric reward dict for empty transcripts (matches veRL contract)."""
    return {
        "score": 0.0,
        "acc": 0.0,
        "r_correctness": 0.0,
        "r_judge": 0.0,
        "r_gate": 0.0,
        "gated": 0,
        "gate_fail_teaching_only": 0,
        "gate_fail_leak_only": 0,
        "gate_fail_both": 0,
        "r_length": 0.0,
        "length_decay": 1.0,
        "r_think_length": 0.0,
        "thinking_decay": 1.0,
        "r_eoc": 0.0,
        "student_token_share": 0.0,
        "student_tokens_total": 0,
        "r_think": 0.0,
        "correct_think_rate": 0.0,
        "n_correct_think_turns": 0,
        "n_think_violations": 0,
        "n_missing_think": 0,
        "pre_solve_rate": 0.0,
        "post_solve_rate": 0.0,
        "delta_solve": 0.0,
        "delta_solve_raw": 0.0,
        "used_transfer_problem": 0,
        "pre_solve_rate_original": 0.0,
        "omit_teacher_turns": int(omit_teacher_turns),
        "student_n_incorrect_format": 0,
        "student_n_failed": 0,
        "judge_no_leak_ok": 0,
        "leak_rate": 1,
        "judge_teaching_ok": 0,
        "num_turns": 0,
        "teacher_tokens_total": 0,
        "teacher_tokens_mean": 0.0,
        "thinking_tokens_total": 0,
        "thinking_tokens_mean": 0.0,
        "visible_tokens_mean": 0.0,
        "eoc_used": 0,
        "judge_pass": 0,
        "leak_rate_early": 0,
        **author_rf._EVAL_JUDGE_BLANK,
        # Route A numeric telemetry (uniform keys on every rollout for veRL batching).
        "rasch_theta": 0.0,
        "rasch_theta_sq": 0.0,
        "rasch_gain_g": 0.0,
        "rasch_theta_scale": float(theta_scale),
        "rasch_valid": 0,
        "rasch_prior_pulled": 0,
        "rasch_all_correct": 0,
        "rasch_all_wrong": 0,
        "compliant_g_lt_neg02": 0,
        "rasch_n_obs": 0,
        "ladder_num_variants": 0,
        "ladder_num_climbs": int(num_climbs),
        "ladder_student_calls": 0,
        "ladder_student_failed": 0,
        "ladder_reach_l1_rate": 0.0,
        "ladder_reach_l2_rate": 0.0,
        "ladder_reach_l3_rate": 0.0,
        "ladder_pass_l1_rate": 0.0,
        "ladder_pass_l2_rate": 0.0,
        "ladder_pass_l3_rate": 0.0,
        "grader_total_calls": 0,
        "grader_rule_correct": 0,
        "grader_llm_calls": 0,
        "grader_llm_unknown": 0,
        "grader_llm_unknown_rate": 0.0,
        "grader_tech_failed": 0,
        # Non-numeric status labels only (zero reasoning/text logged).
        "judge_no_leak": "REJECT",
        "judge_teaching": "REJECT",
        "route_branch": _normalize_route_branch(route_branch),
        "gate_fail_type": "pass",
    }


def _row_gate_fail_type(row: dict[str, Any]) -> str:
    if int(row.get("gated", 0)) == 0:
        return "pass"
    raw_type = row.get("gate_fail_type")
    if raw_type in ("teaching_only", "leak_only", "both"):
        return str(raw_type)
    if int(row.get("gate_fail_both", 0)) == 1:
        return "both"
    if int(row.get("gate_fail_teaching_only", 0)) == 1:
        return "teaching_only"
    if int(row.get("gate_fail_leak_only", 0)) == 1:
        return "leak_only"
    if "judge_no_leak_ok" in row and "judge_teaching_ok" in row:
        leak_bad = int(row.get("judge_no_leak_ok", 1)) == 0
        teach_bad = int(row.get("judge_teaching_ok", 1)) == 0
        if leak_bad and teach_bad:
            return "both"
        if teach_bad:
            return "teaching_only"
        if leak_bad:
            return "leak_only"
    return "unknown"


def summarize_gate_breakdown(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize gate failure counts & rates by gate type (numeric-only)."""
    n_total = len(results)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_gated": 0,
            "gated_rate": 0.0,
            "teaching_only_count": 0,
            "teaching_only_rate": 0.0,
            "leak_only_count": 0,
            "leak_only_rate": 0.0,
            "both_count": 0,
            "both_rate": 0.0,
            "teaching_any_count": 0,
            "leak_any_count": 0,
        }
    n_gated = sum(1 for r in results if int(r.get("gated", 0)) == 1)
    types = [_row_gate_fail_type(r) for r in results]
    teaching_only = sum(1 for t in types if t == "teaching_only")
    leak_only = sum(1 for t in types if t == "leak_only")
    both = sum(1 for t in types if t == "both")
    return {
        "n_total": n_total,
        "n_gated": n_gated,
        "gated_rate": n_gated / n_total,
        "teaching_only_count": teaching_only,
        "teaching_only_rate": teaching_only / n_total,
        "leak_only_count": leak_only,
        "leak_only_rate": leak_only / n_total,
        "both_count": both,
        "both_rate": both / n_total,
        "teaching_any_count": teaching_only + both,
        "leak_any_count": leak_only + both,
    }


def summarize_by_branch(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize rollout metrics grouped by ``route_branch`` (numeric-only)."""
    grouped: dict[str, list[dict[str, Any]]] = {b: [] for b in _VALID_ROUTE_BRANCHES}
    grouped["unknown"] = []
    for r in results:
        b = _normalize_route_branch(r.get("route_branch"))
        grouped[b].append(r)

    branches_to_emit = list(_VALID_ROUTE_BRANCHES)
    if grouped["unknown"]:
        branches_to_emit.append("unknown")

    out: dict[str, dict[str, Any]] = {}
    for b in branches_to_emit:
        rows = grouped[b]
        cnt = len(rows)
        reward_mean = (
            sum(float(r.get("score", 0.0)) for r in rows) / cnt if cnt > 0 else 0.0
        )
        valid_compliant = [
            r
            for r in rows
            if int(r.get("gated", 0)) == 0 and int(r.get("rasch_valid", 0)) == 1
        ]
        thetas = [float(r["rasch_theta"]) for r in valid_compliant]
        gs = [
            float(r.get("rasch_gain_g", r.get("delta_solve", 0.0)))
            for r in valid_compliant
        ]
        gb = summarize_gate_breakdown(rows)
        out[b] = {
            "count": cnt,
            "reward_mean": reward_mean,
            "theta_mean": (sum(thetas) / len(thetas)) if thetas else 0.0,
            "gated_count": gb["n_gated"],
            "gated_rate": gb["gated_rate"],
            "compliant_valid_count": len(valid_compliant),
            "compliant_g_mean": (sum(gs) / len(gs)) if gs else 0.0,
            "compliant_g_median": _median_float(gs) if gs else 0.0,
            "teaching_only_count": gb["teaching_only_count"],
            "teaching_only_rate": gb["teaching_only_rate"],
            "leak_only_count": gb["leak_only_count"],
            "leak_only_rate": gb["leak_only_rate"],
            "both_count": gb["both_count"],
            "both_rate": gb["both_rate"],
            "teaching_any_count": gb["teaching_any_count"],
            "leak_any_count": gb["leak_any_count"],
        }
    return out


def summarize_step_metrics(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Compute step-level Route A metrics (numeric-only, zero text).

    Keys use the ``eduardo/`` prefix so existing trainers (e.g.
    ``pro5000_trainer_entry.extract_numeric_metrics``) capture them automatically.
    """
    n = len(results)
    if n == 0:
        return {}

    scores = [float(r.get("score", 0.0)) for r in results]
    valid_compliant = [
        r
        for r in results
        if int(r.get("gated", 0)) == 0 and int(r.get("rasch_valid", 0)) == 1
    ]
    valid_thetas = [float(r["rasch_theta"]) for r in valid_compliant]
    valid_gs = [
        float(r.get("rasch_gain_g", r.get("delta_solve", 0.0)))
        for r in valid_compliant
    ]
    n_valid = len(valid_thetas)
    if n_valid > 0:
        theta_mean = sum(valid_thetas) / n_valid
        theta_var = sum((t - theta_mean) ** 2 for t in valid_thetas) / n_valid
        theta_std = math.sqrt(max(0.0, theta_var))
        g_mean = sum(valid_gs) / n_valid
        n_g_lt_neg02 = sum(1 for g in valid_gs if g < -0.2)
    else:
        theta_mean = 0.0
        theta_std = 0.0
        g_mean = 0.0
        n_g_lt_neg02 = 0

    score_mean = sum(scores) / n
    score_var = sum((s - score_mean) ** 2 for s in scores) / n
    score_std = math.sqrt(max(0.0, score_var))
    score_min = min(scores)

    n_gated = sum(1 for r in results if int(r.get("gated", 0)) == 1)
    n_prior_pulled = sum(
        1 for r in valid_compliant if int(r.get("rasch_prior_pulled", 0)) == 1
    )
    n_prior_all_correct = sum(
        1
        for r in valid_compliant
        if int(r.get("rasch_prior_pulled", 0)) == 1
        and int(r.get("rasch_all_correct", 0)) == 1
    )
    n_prior_all_wrong = sum(
        1
        for r in valid_compliant
        if int(r.get("rasch_prior_pulled", 0)) == 1
        and int(r.get("rasch_all_wrong", 0)) == 1
    )
    total_student_calls = sum(int(r.get("ladder_student_calls", 0)) for r in results)
    total_student_failed = sum(int(r.get("ladder_student_failed", 0)) for r in results)
    total_llm_calls = sum(int(r.get("grader_llm_calls", 0)) for r in results)
    total_llm_unknown = sum(int(r.get("grader_llm_unknown", 0)) for r in results)
    total_grader_tech_failed = sum(int(r.get("grader_tech_failed", 0)) for r in results)

    non_gated = [r for r in results if int(r.get("gated", 0)) == 0]
    n_ng = len(non_gated)
    compliant_student_calls = sum(int(r.get("ladder_student_calls", 0)) for r in non_gated)

    def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / len(rows) if rows else 0.0

    gb = summarize_gate_breakdown(results)
    metrics: dict[str, float] = {
        "eduardo/routeA_reward_mean": score_mean,
        "eduardo/routeA_reward_std": score_std,
        "eduardo/routeA_reward_min": score_min,
        "eduardo/routeA_theta_mean": theta_mean,
        "eduardo/routeA_theta_std": theta_std,
        "eduardo/routeA_g_mean": g_mean,
        "eduardo/routeA_compliant_g_lt_neg02_rate": (
            n_g_lt_neg02 / n_valid if n_valid else 0.0
        ),
        "eduardo/routeA_theta_valid_rate": n_valid / n,
        "eduardo/routeA_prior_pulled_rate": (n_prior_pulled / n_valid) if n_valid else 0.0,
        "eduardo/routeA_prior_pulled_all_correct_rate": (
            n_prior_all_correct / n_valid if n_valid else 0.0
        ),
        "eduardo/routeA_prior_pulled_all_wrong_rate": (
            n_prior_all_wrong / n_valid if n_valid else 0.0
        ),
        "eduardo/routeA_prior_pulled_all_rate": n_prior_pulled / n,
        "eduardo/routeA_prior_pulled_all_correct_all_rate": n_prior_all_correct / n,
        "eduardo/routeA_prior_pulled_all_wrong_all_rate": n_prior_all_wrong / n,
        "eduardo/routeA_gated_rate": n_gated / n,
        "eduardo/routeA_reach_l1_rate": _mean(non_gated, "ladder_reach_l1_rate") if n_ng else 0.0,
        "eduardo/routeA_reach_l2_rate": _mean(non_gated, "ladder_reach_l2_rate") if n_ng else 0.0,
        "eduardo/routeA_reach_l3_rate": _mean(non_gated, "ladder_reach_l3_rate") if n_ng else 0.0,
        "eduardo/routeA_pass_l1_rate": _mean(non_gated, "ladder_pass_l1_rate") if n_ng else 0.0,
        "eduardo/routeA_pass_l2_rate": _mean(non_gated, "ladder_pass_l2_rate") if n_ng else 0.0,
        "eduardo/routeA_pass_l3_rate": _mean(non_gated, "ladder_pass_l3_rate") if n_ng else 0.0,
        "eduardo/routeA_student_calls_mean": total_student_calls / n,
        "eduardo/routeA_student_calls_compliant_mean": (
            compliant_student_calls / n_ng if n_ng else 0.0
        ),
        "eduardo/routeA_student_calls_total": float(total_student_calls),
        "eduardo/routeA_student_failed_total": float(total_student_failed),
        "eduardo/routeA_grader_llm_calls_total": float(total_llm_calls),
        "eduardo/routeA_grader_llm_calls_mean": total_llm_calls / n,
        "eduardo/routeA_grader_llm_unknown_total": float(total_llm_unknown),
        "eduardo/routeA_grader_llm_unknown_rate": (
            total_llm_unknown / total_llm_calls if total_llm_calls > 0 else 0.0
        ),
        "eduardo/routeA_grader_tech_failed_total": float(total_grader_tech_failed),
        "eduardo/routeA_grader_tech_failed_rate": (
            total_grader_tech_failed / total_llm_calls if total_llm_calls > 0 else 0.0
        ),
        "eduardo/routeC_compliant_g_median": _median_float(valid_gs) if valid_gs else 0.0,
        "eduardo/routeC_gate_fail_teaching_only_count": float(gb["teaching_only_count"]),
        "eduardo/routeC_gate_fail_teaching_only_rate": gb["teaching_only_rate"],
        "eduardo/routeC_gate_fail_leak_only_count": float(gb["leak_only_count"]),
        "eduardo/routeC_gate_fail_leak_only_rate": gb["leak_only_rate"],
        "eduardo/routeC_gate_fail_both_count": float(gb["both_count"]),
        "eduardo/routeC_gate_fail_both_rate": gb["both_rate"],
    }
    if any(_normalize_route_branch(r.get("route_branch")) in _VALID_ROUTE_BRANCHES for r in results):
        by_branch = summarize_by_branch(results)
        for b in _VALID_ROUTE_BRANCHES:
            st = by_branch[b]
            metrics[f"eduardo/routeC_{b}_count"] = float(st["count"])
            metrics[f"eduardo/routeC_{b}_reward_mean"] = float(st["reward_mean"])
            metrics[f"eduardo/routeC_{b}_theta_mean"] = float(st["theta_mean"])
            metrics[f"eduardo/routeC_{b}_gated_rate"] = float(st["gated_rate"])
            metrics[f"eduardo/routeC_{b}_compliant_g_mean"] = float(st["compliant_g_mean"])
            metrics[f"eduardo/routeC_{b}_compliant_g_median"] = float(st["compliant_g_median"])
            metrics[f"eduardo/routeC_{b}_gate_teaching_only_rate"] = float(st["teaching_only_rate"])
            metrics[f"eduardo/routeC_{b}_gate_leak_only_rate"] = float(st["leak_only_rate"])
            metrics[f"eduardo/routeC_{b}_gate_both_rate"] = float(st["both_rate"])
    return metrics


def close_step_metric_sinks() -> None:
    """Close any open per-rollout step metric file handles (useful in tests)."""
    with _SINK_LOCK:
        for fobj in _SINK_FILES.values():
            try:
                fobj.close()
            except Exception:
                pass
        _SINK_FILES.clear()


def _emit_step_metric_row(
    result: dict[str, Any],
    *,
    split: str,
    metrics_dir: str,
    route_branch: str = "unknown",
    gate_fail_type: str = "pass",
) -> None:
    """Append a strictly numeric/enum rollout snapshot to per-PID JSONL (no text)."""
    raw_dir = metrics_dir.strip()
    if not raw_dir:
        return
    try:
        dpath = Path(raw_dir)
        if not dpath.is_absolute() or dpath.is_symlink():
            return
        dpath.mkdir(mode=0o700, parents=True, exist_ok=True)
        split_tag = split if split in ("train", "val") else "unknown"
        payload: dict[str, Any] = {
            "ts_ns": time.monotonic_ns(),
            "split": split_tag,
            "route_branch": _normalize_route_branch(route_branch),
            "gate_fail_type": _normalize_gate_fail_type(gate_fail_type),
        }
        for k in _METRIC_SINK_KEYS:
            val = result.get(k, 0)
            if isinstance(val, bool):
                payload[k] = int(val)
            elif isinstance(val, (int, float)) and math.isfinite(float(val)):
                payload[k] = val
            else:
                payload[k] = 0
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        key = f"{raw_dir}::{os.getpid()}"
        with _SINK_LOCK:
            fobj = _SINK_FILES.get(key)
            if fobj is None or fobj.closed:
                fpath = dpath / f"routeA-metrics-{os.getpid()}.jsonl"
                fobj = fpath.open("a", encoding="ascii", buffering=1)
                _SINK_FILES[key] = fobj
                weakref.finalize(fobj, fobj.close)
            fobj.write(line)
    except Exception:
        return


def summarize_metrics_window(
    directory: str | Path,
    start_ns: int,
    end_ns: int,
    *,
    split: str = "train",
) -> dict[str, Any]:
    """Aggregate per-rollout numeric JSONL metrics within [start_ns, end_ns]."""
    dpath = Path(directory)
    if not dpath.is_dir() or dpath.is_symlink():
        return {
            "n_rollouts": 0,
            "malformed_rows": 0,
            "metrics": {},
            "by_branch": {},
            "gate_breakdown": summarize_gate_breakdown([]),
        }
    matched: list[dict[str, Any]] = []
    malformed = 0
    for path in sorted(dpath.glob("routeA-metrics-*.jsonl")):
        if path.is_symlink():
            continue
        try:
            with path.open("r", encoding="ascii") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        malformed += 1
                        continue
                    if not isinstance(row, dict) or not isinstance(row.get("ts_ns"), int):
                        malformed += 1
                        continue
                    ts = row["ts_ns"]
                    if start_ns <= ts <= end_ns and row.get("split") == split:
                        matched.append(row)
        except (OSError, UnicodeError):
            malformed += 1
    return {
        "n_rollouts": len(matched),
        "malformed_rows": malformed,
        "metrics": summarize_step_metrics(matched),
        "by_branch": summarize_by_branch(matched),
        "gate_breakdown": summarize_gate_breakdown(matched),
    }


def _record_and_maybe_log_batch(
    result: dict[str, Any],
    *,
    interval: int,
) -> Optional[dict[str, float]]:
    """Accumulate numeric rollout metrics and log a sanitized batch summary."""
    global _BATCH_ROWS, _BATCH_START_TIME
    if interval <= 0:
        return None
    numeric_snapshot = {
        k: v for k, v in result.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    with _BATCH_LOCK:
        if _BATCH_START_TIME is None:
            _BATCH_START_TIME = time.time()
        _BATCH_ROWS.append(numeric_snapshot)
        if len(_BATCH_ROWS) < interval:
            return None
        rows = _BATCH_ROWS
        elapsed = time.time() - _BATCH_START_TIME
        _BATCH_ROWS = []
        _BATCH_START_TIME = None

    summary = summarize_step_metrics(rows)
    logger.info(
        "routeA_batch n=%d elapsed=%.1fs reward=%.3f±%.3f theta=%.3f±%.3f "
        "valid=%.3f prior_pulled=%.3f gated=%.3f reach=(L1:%.3f,L2:%.3f,L3:%.3f) "
        "student_calls=%.2f llm_calls=%d llm_unknown_rate=%.4f grader_tech_fail=%d",
        len(rows),
        elapsed,
        summary["eduardo/routeA_reward_mean"],
        summary["eduardo/routeA_reward_std"],
        summary["eduardo/routeA_theta_mean"],
        summary["eduardo/routeA_theta_std"],
        summary["eduardo/routeA_theta_valid_rate"],
        summary["eduardo/routeA_prior_pulled_rate"],
        summary["eduardo/routeA_gated_rate"],
        summary["eduardo/routeA_reach_l1_rate"],
        summary["eduardo/routeA_reach_l2_rate"],
        summary["eduardo/routeA_reach_l3_rate"],
        summary["eduardo/routeA_student_calls_mean"],
        int(summary["eduardo/routeA_grader_llm_calls_total"]),
        summary["eduardo/routeA_grader_llm_unknown_rate"],
        int(summary["eduardo/routeA_grader_tech_failed_total"]),
    )
    return summary


async def _run_author_hybrid_posttest(
    *,
    msgs: list[dict[str, Any]],
    problem: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    student_client: Any,
    student_model: str,
    grader: Any,
    num_attempts: int,
    temperature: float,
    max_tokens: int,
    omit_teacher_turns: bool,
    test_on_transfer: bool,
    middle_variant_policy: str,
    retry_kwargs: dict[str, Any],
    student_semaphore: Optional[asyncio.Semaphore],
) -> dict[str, Any]:
    """Run author-style single-variant post-test with hybrid grading (ablation A1)."""
    transfer_problem = extra_info.get("transfer_problem") or ""
    transfer_answer = extra_info.get("transfer_answer") or ""
    pre_solve_rate_original = author_rf._pre_rate(
        extra_info, "pre_solve_rate", "solve_rate", "llama8b_solve_rate"
    )

    use_transfer = False
    test_problem = problem
    test_answer = ground_truth
    pre_solve_rate = pre_solve_rate_original

    if test_on_transfer and transfer_problem and transfer_answer:
        use_transfer = True
        test_problem = transfer_problem
        test_answer = transfer_answer
        pre_solve_rate = author_rf._pre_rate(
            extra_info, "pre_solve_rate_transfer_problem"
        )
    elif test_on_transfer and extra_info.get("variants"):
        variants = extract_ladder_variants(extra_info)
        chosen = select_middle_variant(variants, even_policy=middle_variant_policy)
        use_transfer = True
        test_problem = chosen.problem
        test_answer = chosen.answer
        if chosen.p_new is not None:
            pre_solve_rate = float(chosen.p_new)
        else:
            # Recover p from b = ln((1 - p) / p) -> p = 1 / (1 + exp(b))
            pre_solve_rate = 1.0 / (1.0 + math.exp(chosen.b))

    final_msgs = author_correctness._build_final_attempt_messages(
        msgs,
        problem,
        transfer_problem=test_problem if use_transfer else None,
        omit_teacher_turns=omit_teacher_turns,
    )

    async def _one() -> Optional[str]:
        return await call_openai_with_retries(
            author_correctness._sample_once,
            student_client,
            student_model,
            final_msgs,
            temperature,
            max_tokens,
            label="student_final",
            use_fallback=True,
            fallback=None,
            semaphore=student_semaphore,
            **retry_kwargs,
        )

    texts = await asyncio.gather(*[_one() for _ in range(num_attempts)])

    n_student_failed = 0
    gradable: list[str] = []
    for t in texts:
        if t is None:
            n_student_failed += 1
        else:
            gradable.append(t)

    grade_results = (
        await asyncio.gather(
            *[grader.grade(test_problem, test_answer, t) for t in gradable]
        )
        if gradable
        else []
    )

    n_correct = 0
    n_scored = 0
    n_format = 0
    n_rule_correct = 0
    n_llm_calls = 0
    n_llm_unknown = 0
    n_grader_tech_failed = 0

    for g_res in grade_results:
        n_format += int(getattr(g_res, "incorrect_format", 0))
        if getattr(g_res, "rule_correct", False):
            n_rule_correct += 1
        if getattr(g_res, "llm_called", False):
            n_llm_calls += 1
        if getattr(g_res, "is_unknown", False):
            n_llm_unknown += 1
        if getattr(g_res, "technical_failure", False):
            n_grader_tech_failed += 1
            continue
        n_scored += 1
        if bool(getattr(g_res, "correct", False)):
            n_correct += 1

    post_rate = (n_correct / n_scored) if n_scored > 0 else 0.0
    delta_solve = author_rf._normalized_delta(float(post_rate), pre_solve_rate)
    delta_solve_raw = float(post_rate) - pre_solve_rate

    return {
        "post_solve_rate": float(post_rate),
        "pre_solve_rate": float(pre_solve_rate),
        "pre_solve_rate_original": float(pre_solve_rate_original),
        "delta_solve": float(delta_solve),
        "delta_solve_raw": float(delta_solve_raw),
        "used_transfer_problem": int(use_transfer),
        "student_calls": num_attempts,
        "student_failed": n_student_failed,
        "student_incorrect_format": n_format,
        "grader_total_calls": len(grade_results),
        "grader_rule_correct": n_rule_correct,
        "grader_llm_calls": n_llm_calls,
        "grader_llm_unknown": n_llm_unknown,
        "grader_tech_failed": n_grader_tech_failed,
    }


async def route_a_reward_func(
    data_source: str,
    solution_str: str | list[dict] | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    *,
    messages: list[dict] | None = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """veRL-compatible async reward function supporting Route A and author modes."""
    mode = resolve_reward_mode(kwargs)

    # Exact bit-for-bit delegation to upstream author reward function.
    if mode == "author":
        author_kwargs = {
            k: v for k, v in kwargs.items() if k not in _ROUTE_A_ONLY_KWARGS
        }
        return await author_rf.eduardo_reward_func(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            messages=messages,
            tokenizer=tokenizer,
            **author_kwargs,
        )

    if extra_info is None:
        extra_info = {}
    problem: str = extra_info.get("problem") or extra_info.get("source_problem") or ""
    if ground_truth is None:
        ground_truth = extra_info.get("answer") or extra_info.get("source_answer") or ""

    msgs: list[dict[str, Any]] = messages or extra_info.get("messages") or []
    if not msgs and isinstance(solution_str, list):
        msgs = solution_str

    omit_teacher_turns = bool(kwargs.get("omit_teacher_turns", True))
    num_climbs = int(
        kwargs.get("ladder_num_climbs", kwargs.get("num_climbs", DEFAULT_LADDER_CLIMBS))
    )
    theta_scale = float(kwargs.get("theta_scale", DEFAULT_THETA_SCALE))
    if not math.isfinite(theta_scale) or theta_scale <= 0.0:
        raise ValueError(f"invalid theta_scale={theta_scale!r}; must be finite > 0")
    metrics_dir = str(
        kwargs.get("routeA_metrics_dir")
        or os.environ.get("ROUTEA_STEP_METRICS_DIR", "")
    )
    split_tag = str(extra_info.get("split") or "unknown")
    route_branch = _normalize_route_branch(extra_info.get("route_branch"))
    batch_log_interval = (
        int(kwargs.get("batch_log_interval", 16))
        if bool(kwargs.get("enable_batch_logging", True))
        else 0
    )

    if not msgs:
        logger.warning(
            "route_a_reward_func: empty transcript mode=%s data_source=%s",
            mode,
            data_source,
        )
        empty = _empty_route_a_reward(
            num_climbs=num_climbs,
            omit_teacher_turns=omit_teacher_turns,
            theta_scale=theta_scale,
            route_branch=route_branch,
        )
        _emit_step_metric_row(
            empty,
            split=split_tag,
            metrics_dir=metrics_dir,
            route_branch=route_branch,
            gate_fail_type="pass",
        )
        _record_and_maybe_log_batch(empty, interval=batch_log_interval)
        return empty

    retry_kwargs = {
        "num_retries": kwargs.get("num_retries", 3),
        "backoff_base": kwargs.get("backoff_base", 2.0),
        "backoff_cap": kwargs.get("backoff_cap", 30.0),
        "jitter": kwargs.get("jitter", True),
    }

    student_semaphore = get_concurrency_semaphore(
        "student", int(kwargs.get("student_concurrency", 64))
    )
    judge_semaphore = get_concurrency_semaphore(
        "judge", int(kwargs.get("judge_concurrency", 128))
    )

    judge_client = get_async_openai_client(
        kwargs["judge_api_base"], kwargs.get("judge_api_key_env", "SERVING_API_KEY")
    )
    student_client = get_async_openai_client(
        kwargs["student_api_base"], kwargs.get("student_api_key_env", "SERVING_API_KEY")
    )

    # 1. Author binary gates on sanitized transcript (no_leak & teaching quality).
    judge_msgs = [
        {**m, "content": strip_teacher_controls(m.get("content") or "")}
        if m.get("role") == "assistant"
        else m
        for m in msgs
    ]
    (
        (no_leak, teaching, _judge_reasoning),
        eval_judge_metrics,
    ) = await asyncio.gather(
        author_judge_quality.majority_vote(
            messages=judge_msgs,
            problem=problem,
            answer=ground_truth,
            client=judge_client,
            model=kwargs["judge_model"],
            num_votes=kwargs.get("judge_votes", 3),
            temperature=kwargs.get("judge_temperature", 0.6),
            max_tokens=int(kwargs.get("judge_max_tokens", 4096)),
            retry_kwargs=retry_kwargs,
            semaphore=judge_semaphore,
        ),
        author_rf._maybe_eval_judge(
            judge_msgs, problem, ground_truth, extra_info, kwargs, retry_kwargs
        ),
    )

    hard_gate = bool(kwargs.get("judge_hard_gate", True))
    pedagogy_violated = (no_leak == "REJECT") or (teaching == "REJECT")
    gated = hard_gate and pedagogy_violated

    judge_pen = float(kwargs.get("judge_penalty", 0.75))
    r_judge = -judge_pen if (pedagogy_violated and not hard_gate) else 0.0

    gate_pen = float(kwargs.get("gate_penalty", 0.2))
    r_gate = -gate_pen if gated else 0.0

    # 2. Teacher efficiency decay & turn stats via author helpers.
    enable_thinking = bool(kwargs.get("enable_thinking", True))
    target_tokens = float(kwargs.get("max_tokens_per_turn", 150))
    think_target_tokens = float(
        kwargs.get("max_thinking_tokens_per_turn", 512)
    )
    tstats = _compute_teacher_stats(
        msgs,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        target_tokens=target_tokens,
        think_target_tokens=think_target_tokens,
    )

    # Default telemetry values (populated if not gated).
    rasch_theta = 0.0
    rasch_gain_g = 0.0
    rasch_valid = 0
    rasch_prior_pulled = 0
    rasch_all_correct = 0
    rasch_all_wrong = 0
    rasch_n_obs = 0
    ladder_num_variants = 0
    ladder_student_calls = 0
    ladder_student_failed = 0
    student_incorrect_format = 0
    reach_rates = [0.0, 0.0, 0.0]
    pass_rates = [0.0, 0.0, 0.0]
    grader_total_calls = 0
    grader_rule_correct = 0
    grader_llm_calls = 0
    grader_llm_unknown = 0
    grader_tech_failed = 0

    pre_solve_rate_original = author_rf._pre_rate(
        extra_info, "pre_solve_rate", "solve_rate", "llama8b_solve_rate"
    )
    pre_solve_rate = 0.0
    post_solve_rate = 0.0
    delta_solve = 0.0
    delta_solve_raw = 0.0
    used_transfer = 0

    # 3. Post-test (skipped when gated to save student/grader compute).
    reward_invalid = False
    gain = 0.0

    if not gated:
        grader = _build_hybrid_grader(kwargs, judge_semaphore=judge_semaphore)
        student_temp = float(
            kwargs.get("student_temperature", DEFAULT_STUDENT_TEMPERATURE)
        )
        student_max_tokens = int(
            kwargs.get("student_final_max_tokens", DEFAULT_STUDENT_FINAL_MAX_TOKENS)
        )

        if mode == "ladder_rasch":
            variants = extract_ladder_variants(extra_info)
            ladder_num_variants = len(variants)
            used_transfer = 1
            batch_res = await climb_ladder_batch(
                messages=msgs,
                problem=problem,
                variants=variants,
                student_client=student_client,
                student_model=kwargs["student_model"],
                grader=grader,
                num_climbs=num_climbs,
                temperature=student_temp,
                max_tokens=student_max_tokens,
                omit_teacher_turns=omit_teacher_turns,
                retry_kwargs=retry_kwargs,
                student_semaphore=student_semaphore,
            )
            ladder_student_calls = batch_res.student_calls
            ladder_student_failed = batch_res.student_failed
            student_incorrect_format = batch_res.student_incorrect_format
            grader_total_calls = batch_res.grader_total_calls
            grader_rule_correct = batch_res.grader_rule_correct
            grader_llm_calls = batch_res.grader_llm_calls
            grader_llm_unknown = batch_res.grader_llm_unknown
            grader_tech_failed = batch_res.grader_tech_failed

            for lvl_idx in range(min(3, batch_res.num_variants)):
                reach_rates[lvl_idx] = batch_res.reached_counts[lvl_idx] / num_climbs
                pass_rates[lvl_idx] = batch_res.passed_counts[lvl_idx] / num_climbs

            est = rasch_map_estimate(batch_res.observations)
            rasch_n_obs = est.n_obs
            if est.valid and est.theta is not None:
                rasch_valid = 1
                rasch_theta = float(est.theta)
                rasch_prior_pulled = int(est.prior_pulled)
                rasch_all_correct = int(est.all_correct)
                rasch_all_wrong = int(est.all_wrong)
                gain = rasch_theta * theta_scale
                rasch_gain_g = float(gain)
                delta_solve = gain
                delta_solve_raw = rasch_theta
                post_solve_rate = est.n_correct / est.n_obs
            else:
                # All climbs failed technically: invalid, no score awarded.
                reward_invalid = True
                gain = 0.0
        else:
            # mode == "author_hybrid"
            ah = await _run_author_hybrid_posttest(
                msgs=msgs,
                problem=problem,
                ground_truth=ground_truth,
                extra_info=extra_info,
                student_client=student_client,
                student_model=kwargs["student_model"],
                grader=grader,
                num_attempts=int(kwargs.get("num_student_attempts", 16)),
                temperature=student_temp,
                max_tokens=student_max_tokens,
                omit_teacher_turns=omit_teacher_turns,
                test_on_transfer=bool(kwargs.get("test_on_transfer", True)),
                middle_variant_policy=str(kwargs.get("middle_variant_policy", "lower")),
                retry_kwargs=retry_kwargs,
                student_semaphore=student_semaphore,
            )
            post_solve_rate = ah["post_solve_rate"]
            pre_solve_rate = ah["pre_solve_rate"]
            pre_solve_rate_original = ah["pre_solve_rate_original"]
            delta_solve = ah["delta_solve"]
            delta_solve_raw = ah["delta_solve_raw"]
            used_transfer = ah["used_transfer_problem"]
            ladder_student_calls = ah["student_calls"]
            ladder_student_failed = ah["student_failed"]
            student_incorrect_format = ah["student_incorrect_format"]
            grader_total_calls = ah["grader_total_calls"]
            grader_rule_correct = ah["grader_rule_correct"]
            grader_llm_calls = ah["grader_llm_calls"]
            grader_llm_unknown = ah["grader_llm_unknown"]
            grader_tech_failed = ah["grader_tech_failed"]
            gain = delta_solve

    # 4. Combine reward via author formula (5).
    length_decay = float(tstats["length_decay"])
    thinking_decay = float(tstats["thinking_decay"])
    eoc_used = bool(tstats["eoc_used"])
    correct_think_rate = float(tstats["correct_think_rate"])

    if gated:
        r_correct = 0.0
        r_length = 0.0
        r_think_length = 0.0
        r_eoc = 0.0
        r_think = 0.0
        total = r_gate
    elif reward_invalid:
        r_correct = 0.0
        r_length = 0.0
        r_think_length = 0.0
        r_eoc = 0.0
        r_think = 0.0
        total = 0.0
    else:
        r_correct = gain
        if gain > 0.0:
            r_length = gain * (length_decay - 1.0)
            r_think_length = gain * length_decay * (thinking_decay - 1.0)
        else:
            r_length = 0.0
            r_think_length = 0.0
        r_eoc = (
            float(kwargs.get("eoc_bonus", 0.0))
            if (eoc_used and gain > 0.0)
            else 0.0
        )
        r_think = float(kwargs.get("think_reward", 0.0)) * correct_think_rate
        total = (
            r_correct
            + r_judge
            + r_gate
            + r_length
            + r_think_length
            + r_eoc
            + r_think
        )

    judge_pass = (no_leak == "OK") and (teaching == "OK")
    leak_rate_early = (no_leak == "REJECT") and (tstats["num_turns"] <= 1)
    llm_unknown_rate = (
        grader_llm_unknown / grader_llm_calls if grader_llm_calls > 0 else 0.0
    )
    gate_fail_teaching_only = int(gated and (teaching == "REJECT") and (no_leak == "OK"))
    gate_fail_leak_only = int(gated and (no_leak == "REJECT") and (teaching == "OK"))
    gate_fail_both = int(gated and (no_leak == "REJECT") and (teaching == "REJECT"))
    if not gated:
        gate_fail_type = "pass"
    elif gate_fail_both:
        gate_fail_type = "both"
    elif gate_fail_teaching_only:
        gate_fail_type = "teaching_only"
    else:
        gate_fail_type = "leak_only"

    result = {
        "score": float(total),
        "acc": float(post_solve_rate),
        "r_correctness": float(r_correct),
        "r_judge": float(r_judge),
        "r_gate": float(r_gate),
        "gated": int(gated),
        "gate_fail_teaching_only": int(gate_fail_teaching_only),
        "gate_fail_leak_only": int(gate_fail_leak_only),
        "gate_fail_both": int(gate_fail_both),
        "r_length": float(r_length),
        "length_decay": float(length_decay),
        "r_think_length": float(r_think_length),
        "thinking_decay": float(thinking_decay),
        "r_eoc": float(r_eoc),
        "student_token_share": float(tstats["student_token_share"]),
        "student_tokens_total": int(tstats["student_tokens_total"]),
        "r_think": float(r_think),
        "correct_think_rate": float(correct_think_rate),
        "n_correct_think_turns": int(tstats["n_correct_think_turns"]),
        "n_think_violations": int(tstats["n_think_violations"]),
        "n_missing_think": int(tstats["n_missing_think"]),
        "pre_solve_rate": float(pre_solve_rate),
        "post_solve_rate": float(post_solve_rate),
        "delta_solve": float(delta_solve),
        "delta_solve_raw": float(delta_solve_raw),
        "used_transfer_problem": int(used_transfer),
        "pre_solve_rate_original": float(pre_solve_rate_original),
        "omit_teacher_turns": int(omit_teacher_turns),
        "student_n_incorrect_format": int(student_incorrect_format),
        "student_n_failed": int(ladder_student_failed),
        "judge_no_leak_ok": int(no_leak == "OK"),
        "leak_rate": int(no_leak == "REJECT"),
        "judge_teaching_ok": int(teaching == "OK"),
        "num_turns": int(tstats["num_turns"]),
        "teacher_tokens_total": int(tstats["teacher_tokens_total"]),
        "teacher_tokens_mean": float(tstats["teacher_tokens_mean"]),
        "thinking_tokens_total": int(tstats["thinking_tokens_total"]),
        "thinking_tokens_mean": float(tstats["thinking_tokens_mean"]),
        "visible_tokens_mean": float(tstats["visible_tokens_mean"]),
        "eoc_used": int(eoc_used),
        "judge_pass": int(judge_pass),
        "leak_rate_early": int(leak_rate_early),
        **eval_judge_metrics,
        # Route A specific metrics
        "rasch_theta": float(rasch_theta),
        "rasch_theta_sq": float(rasch_theta * rasch_theta) if rasch_valid else 0.0,
        "rasch_gain_g": float(rasch_gain_g),
        "rasch_theta_scale": float(theta_scale),
        "rasch_valid": int(rasch_valid),
        "rasch_prior_pulled": int(rasch_prior_pulled),
        "rasch_all_correct": int(rasch_all_correct),
        "rasch_all_wrong": int(rasch_all_wrong),
        "compliant_g_lt_neg02": int(
            (not gated) and rasch_valid == 1 and rasch_gain_g < -0.2
        ),
        "rasch_n_obs": int(rasch_n_obs),
        "ladder_num_variants": int(ladder_num_variants),
        "ladder_num_climbs": int(num_climbs),
        "ladder_student_calls": int(ladder_student_calls),
        "ladder_student_failed": int(ladder_student_failed),
        "ladder_reach_l1_rate": float(reach_rates[0]),
        "ladder_reach_l2_rate": float(reach_rates[1]),
        "ladder_reach_l3_rate": float(reach_rates[2]),
        "ladder_pass_l1_rate": float(pass_rates[0]),
        "ladder_pass_l2_rate": float(pass_rates[1]),
        "ladder_pass_l3_rate": float(pass_rates[2]),
        "grader_total_calls": int(grader_total_calls),
        "grader_rule_correct": int(grader_rule_correct),
        "grader_llm_calls": int(grader_llm_calls),
        "grader_llm_unknown": int(grader_llm_unknown),
        "grader_llm_unknown_rate": float(llm_unknown_rate),
        "grader_tech_failed": int(grader_tech_failed),
        # Non-numeric gate status only (no text / reasoning).
        "judge_no_leak": no_leak,
        "judge_teaching": teaching,
        "route_branch": route_branch,
        "gate_fail_type": gate_fail_type,
    }

    # Sanitized rollout debug log: numeric & status fields only.
    logger.debug(
        "routeA_rollout mode=%s branch=%s score=%.4f theta=%.4f valid=%d prior_pulled=%d "
        "gated=%d gate_fail=%s student_calls=%d llm_calls=%d llm_unknown=%d tech_fail=%d",
        mode,
        route_branch,
        result["score"],
        result["rasch_theta"],
        result["rasch_valid"],
        result["rasch_prior_pulled"],
        result["gated"],
        gate_fail_type,
        result["ladder_student_calls"],
        result["grader_llm_calls"],
        result["grader_llm_unknown"],
        result["grader_tech_failed"],
    )
    _emit_step_metric_row(
        result,
        split=split_tag,
        metrics_dir=metrics_dir,
        route_branch=route_branch,
        gate_fail_type=gate_fail_type,
    )
    _record_and_maybe_log_batch(result, interval=batch_log_interval)
    return result
