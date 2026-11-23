#!/usr/bin/env python3
"""Summarize and verify Eduardo author-494 evaluation results (aggregate only).

Reads the JSON output from ``eval/run_env_test_eval.py``, verifies schema and
exact agreement with the author's summary metrics, separates Agnes retry-exhausted
technical failures from parse failures, runs problem-clustered 10,000x bootstrap
CIs (seed=42), and evaluates the pre-registered comparison verdict against the
paper's Qwen3.5-4B base numbers.

Never prints problem IDs, problem text, answers, dialogues, or judge reasoning.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# Paper baselines for Qwen3.5-4B base (n=494, Table 1, 7, 8).
PAPER_BASE: dict[str, Any] = {
    "label": "Paper (base 4B)",
    "delta_transfer": 0.10,
    "delta_transfer_ci": (0.06, 0.13),
    "delta_same": 0.11,
    "delta_same_ci": (0.08, 0.15),
    "leak_rate": 0.168,
    "leak_rate_ci": (0.136, 0.202),
    "eval_judge_pedagogy": 2.26,
    "eval_judge_correctness": 2.88,
    "blanked_teacher_turn_rate": 0.700,
    "token_budget_truncated_rate": 0.312,
    "eoc_rate": 0.688,
    "thinking_tokens_avg_per_turn": 596.0,
}

# Paper baselines for Eduardo-27B (n=494, Table 1, 7, 8).
PAPER_EDUARDO_27B: dict[str, Any] = {
    "label": "Paper (Eduardo-27B)",
    "delta_transfer": 0.29,
    "delta_transfer_ci": (0.25, 0.32),
    "delta_same": 0.46,
    "delta_same_ci": (0.42, 0.51),
    "leak_rate": 0.093,
    "leak_rate_ci": (0.069, 0.119),
    "eval_judge_pedagogy": 4.50,
    "eval_judge_correctness": 4.56,
    "blanked_teacher_turn_rate": 0.004,
    "token_budget_truncated_rate": 0.000,
    "eoc_rate": 0.534,
    "thinking_tokens_avg_per_turn": 57.0,
}

PAPER_BASELINES: dict[str, dict[str, Any]] = {
    "base-4b": PAPER_BASE,
    "eduardo-27b": PAPER_EDUARDO_27B,
}

REQUIRED_METRICS_KEYS = (
    "num_conversations",
    "num_problems",
    "samples_per_problem",
    "delta_transfer",
    "delta_same",
    "delta_transfer_raw",
    "delta_same_raw",
    "delta_transfer_gated",
    "delta_same_gated",
    "transfer_gap",
    "post_solve_transfer",
    "post_solve_same",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "leak_rate",
    "teaching_reject_rate",
    "eval_judge_pedagogy",
    "eval_judge_correctness",
    "eval_judge_ok",
    "blanked_teacher_turn_rate",
    "token_budget_truncated_rate",
    "eoc_rate",
    "num_teacher_turns",
    "thinking_tokens_avg_per_turn",
    "transfer_tested_rate",
    "n_student_calls_failed",
    "assume_thinking",
)

REQUIRED_RECORD_KEYS = (
    "row_index",
    "delta_transfer",
    "delta_same",
    "delta_transfer_raw",
    "delta_same_raw",
    "delta_transfer_gated",
    "delta_same_gated",
    "transfer_gap",
    "post_solve_transfer",
    "post_solve_same",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "leaked",
    "teaching_rejected",
    "judge_reasoning",
    "eval_judge_pedagogy",
    "eval_judge_correctness",
    "eval_judge_ok",
    "blanked_teacher_turn_rate",
    "end_reason",
    "eoc_used",
    "num_teacher_turns",
    "thinking_tokens",
    "used_transfer_problem",
    "n_student_calls_failed",
    "assume_thinking",
)

_MEAN_KEYS = (
    "delta_transfer",
    "delta_same",
    "delta_transfer_raw",
    "delta_same_raw",
    "delta_transfer_gated",
    "delta_same_gated",
    "transfer_gap",
    "post_solve_transfer",
    "post_solve_same",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "leaked",
    "teaching_rejected",
    "used_transfer_problem",
    "eoc_used",
    "num_teacher_turns",
    "eval_judge_ok",
    "blanked_teacher_turn_rate",
)

_VOTE_HEADER_RE = re.compile(r"(?:^| \| )\[Vote (\d+): (OK|REJECT)\] ")
_RETRY_FAIL_REASON = "Judge call failed after retries"
_PARSE_FAIL_REASON = "Parse failure"
_REGEX_RECOVERED_REASON = "<unparsed JSON>"
_EPS = 1e-12

_SAFE_INFO_LOG_RE = re.compile(
    r"^(?:"
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] eduardo\.transfer_eval: "
    r"(?:Thinking mode (?:AUTO|FORCED):|Rollout of |Scoring (?:\d+|took ))"
    r"|(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] eduardo27b\.agnes_window: )?"
    r"\[27b_agnes_window\] (?:STARTUP|SUMMARY|TUTOR_CONFIG) "
    r")"
)
_SAFE_ROUND_INFO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] eduardo\.transfer_eval: Round \d+: "
)
_RESULTS_WRITTEN_RE = re.compile(r"^Results written to: ")
_AUTHOR_SUMMARY_BLOCK_RE = re.compile(r"^={72}\nModel:\s+.*?^={72}$", re.M | re.S)


def extract_safe_eval_log_lines(text: str, *, include_rounds: bool = False) -> list[str]:
    """Return only allowlisted eduardo.transfer_eval INFO lines and Results written line.

    WARNING and ERROR lines (which may contain raw judge/model text) are never returned.
    """
    out: list[str] = []
    for line in text.splitlines():
        if _SAFE_INFO_LOG_RE.match(line) or _RESULTS_WRITTEN_RE.match(line):
            out.append(line)
        elif include_rounds and _SAFE_ROUND_INFO_RE.match(line):
            out.append(line)
    return out


def format_eval_log_excerpt(text: str) -> str:
    """Format the author print_summary block plus safe allowlisted status lines."""
    parts: list[str] = []
    match = _AUTHOR_SUMMARY_BLOCK_RE.search(text)
    if match:
        parts.append(match.group(0))
    safe_lines = extract_safe_eval_log_lines(text)
    if safe_lines:
        parts.extend(safe_lines)
    return "\n".join(parts)


_TRACEBACK_FRAME_RE = re.compile(
    r'^(?:.*\]\s+)?\s*File "([^"]+)", line (\d+), in ([A-Za-z0-9_<>.-]+)\s*$'
)
_EXC_CLASS_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|SystemExit|KeyboardInterrupt|StopIteration))\s*(?::.*)?$"
)


def extract_safe_failure_summary(text: str, *, max_frames: int = 20) -> str:
    """Extract only stack frame locations (file, line, func) and exception class names.

    Never emits exception messages (quoted or unquoted), WARNING/INFO lines, or payloads.
    """
    frames: list[str] = []
    exc_classes: list[str] = []
    for line in text.splitlines():
        if "[WARNING]" in line or "[INFO]" in line:
            continue
        frame_m = _TRACEBACK_FRAME_RE.match(line)
        if frame_m:
            frames.append(
                f'File "{frame_m.group(1)}", line {frame_m.group(2)}, in {frame_m.group(3)}'
            )
            continue
        exc_m = _EXC_CLASS_RE.match(line.strip())
        if exc_m:
            exc_classes.append(exc_m.group(1))
    out = frames[-max_frames:] + exc_classes[-5:]
    return "\n".join(out)


def normalized_delta(post: float, pre: float) -> float:
    """Author normalized gain formula: (post - pre)/(1 - pre) if post > pre else post - pre."""
    if post <= pre or pre >= 1.0:
        return post - pre
    return (post - pre) / (1.0 - pre)


def compute_hake_g(pres: list[float], posts: list[float]) -> float | None:
    """R2 Hake normalized gain across problems: (<post> - <pre>) / (1 - <pre>)."""
    if not pres or len(pres) != len(posts):
        return None
    mean_pre = statistics.fmean(pres)
    mean_post = statistics.fmean(posts)
    denom = 1.0 - mean_pre
    if abs(denom) <= _EPS:
        return None
    return (mean_post - mean_pre) / denom


def compute_cohens_d(pres: list[float], posts: list[float]) -> float | None:
    """R3 Cohen's d across problems: (<post> - <pre>) / sqrt((s_pre^2 + s_post^2) / 2)."""
    if len(pres) < 2 or len(pres) != len(posts):
        return None
    mean_pre = statistics.fmean(pres)
    mean_post = statistics.fmean(posts)
    s_pre = statistics.stdev(pres)
    s_post = statistics.stdev(posts)
    s_pooled = math.sqrt((s_pre * s_pre + s_post * s_post) / 2.0)
    if s_pooled <= _EPS:
        return 0.0 if abs(mean_post - mean_pre) <= _EPS else None
    return (mean_post - mean_pre) / s_pooled


def _average_ranks(vals: list[float]) -> list[float]:
    """1-based average ranks with tie handling for Spearman correlation."""
    indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient; returns None if n < 2 or zero variance."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    num = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.fsum((x - mx) ** 2 for x in xs)
    dy = math.fsum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx * dy)
    if denom <= _EPS:
        return None
    return max(-1.0, min(1.0, num / denom))


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation with average-rank tie handling."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson_r(_average_ranks(xs), _average_ranks(ys))


def classify_difficulty_tier(ctl2_transfer_rate: float) -> str:
    """Classify difficulty tier by ctl2_transfer rate with 1/3 and 2/3 cutoffs:
    hard=[0, 1/3), medium=[1/3, 2/3), easy=[2/3, 1].
    """
    if ctl2_transfer_rate < (1.0 / 3.0):
        return "hard"
    if ctl2_transfer_rate < (2.0 / 3.0):
        return "medium"
    return "easy"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_results_doc(doc: Any) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Strictly validate the result JSON structure; raise ValueError on missing/invalid fields."""
    if not isinstance(doc, dict):
        raise ValueError("Result JSON root must be a dict")
    if "metrics" not in doc or not isinstance(doc["metrics"], dict):
        raise ValueError("Result JSON missing 'metrics' dict")
    if "conversations" not in doc or not isinstance(doc["conversations"], list) or not doc["conversations"]:
        raise ValueError("Result JSON missing non-empty 'conversations' list")

    metrics = doc["metrics"]
    for key in REQUIRED_METRICS_KEYS:
        if key not in metrics:
            raise ValueError(f"Result JSON metrics missing required field: {key}")

    records = doc["conversations"]
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"Conversation record #{idx} is not a dict")
        for key in REQUIRED_RECORD_KEYS:
            if key not in rec:
                raise ValueError(f"Conversation record #{idx} missing required field: {key}")
        if isinstance(rec["row_index"], bool) or not isinstance(rec["row_index"], int):
            raise ValueError(f"Conversation record #{idx} has invalid 'row_index'")
        if not isinstance(rec["judge_reasoning"], dict):
            raise ValueError(f"Conversation record #{idx} has invalid 'judge_reasoning'")
        for crit in ("no_leak", "teaching"):
            if not isinstance(rec["judge_reasoning"].get(crit), str):
                raise ValueError(f"Conversation record #{idx} missing judge_reasoning['{crit}']")
        if not isinstance(rec["thinking_tokens"], list) or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) for t in rec["thinking_tokens"]
        ):
            raise ValueError(f"Conversation record #{idx} has invalid 'thinking_tokens'")
        for num_key in (
            "delta_transfer",
            "delta_same",
            "delta_transfer_gated",
            "delta_same_gated",
            "blanked_teacher_turn_rate",
        ):
            if not _is_number(rec[num_key]):
                raise ValueError(f"Conversation record #{idx} has non-numeric '{num_key}'")
        for crit in ("eval_judge_pedagogy", "eval_judge_correctness"):
            val = rec[crit]
            if val is not None and not _is_number(val):
                raise ValueError(f"Conversation record #{idx} has invalid '{crit}'")

    judge_votes = 3
    if isinstance(doc.get("args"), dict) and "judge_votes" in doc["args"]:
        jv = doc["args"]["judge_votes"]
        if isinstance(jv, int) and not isinstance(jv, bool) and jv >= 1:
            judge_votes = jv
        else:
            raise ValueError("Result JSON args['judge_votes'] is invalid")
    return metrics, records, judge_votes


def recompute_author_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the author's compute_summary fields from conversation records."""
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        return statistics.fmean(vals) if vals else 0.0

    summary: dict[str, Any] = {"num_conversations": len(records)}
    summary.update({k: mean(k) for k in _MEAN_KEYS})
    summary["leak_rate"] = summary.pop("leaked")
    summary["teaching_reject_rate"] = summary.pop("teaching_rejected")
    summary["eoc_rate"] = summary.pop("eoc_used")
    summary["transfer_tested_rate"] = summary.pop("used_transfer_problem")

    for crit in ("pedagogy", "correctness"):
        vals = [
            float(r[f"eval_judge_{crit}"])
            for r in records
            if r.get(f"eval_judge_{crit}") is not None
        ]
        summary[f"eval_judge_{crit}"] = statistics.fmean(vals) if vals else 0.0

    summary["n_student_calls_failed"] = sum(
        int(r["n_student_calls_failed"]) for r in records
    )
    summary["assume_thinking"] = bool(records[0]["assume_thinking"])

    groups: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["row_index"], []).append(r)
    summary["num_problems"] = len(groups)
    summary["samples_per_problem"] = len(records) / len(groups)

    per_turn = [t for r in records for t in r["thinking_tokens"]]
    summary["thinking_tokens_avg_per_turn"] = statistics.fmean(per_turn) if per_turn else 0.0
    summary["token_budget_truncated_rate"] = (
        sum(1 for r in records if r["end_reason"] == "token_budget") / len(records)
    )
    return summary


def verify_author_summary_match(recomputed: dict[str, Any], recorded: dict[str, Any]) -> None:
    """Verify recomputed summary matches recorded JSON metrics exactly."""
    for key, exp_val in recomputed.items():
        act_val = recorded[key]
        if isinstance(exp_val, bool):
            if bool(act_val) is not exp_val:
                raise ValueError(f"Summary mismatch on '{key}': {act_val!r} != {exp_val!r}")
        elif isinstance(exp_val, int):
            if int(act_val) != exp_val:
                raise ValueError(f"Summary mismatch on '{key}': {act_val!r} != {exp_val!r}")
        else:
            if not _is_number(act_val) or not math.isclose(
                float(act_val), float(exp_val), rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"Summary mismatch on '{key}': {act_val!r} != {exp_val!r}")


def parse_judge_votes(text: str, expected_votes: int | None = None) -> list[tuple[str, str]]:
    """Parse ``[Vote i: OK|REJECT] reason`` entries from judge_reasoning string."""
    matches = list(_VOTE_HEADER_RE.finditer(text))
    if not matches:
        raise ValueError("Unrecognized judge_reasoning format")
    votes: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        vote_num = int(match.group(1))
        if vote_num != idx + 1:
            raise ValueError(f"Non-sequential vote index in judge_reasoning: got {vote_num}, expected {idx + 1}")
        decision = match.group(2).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        votes.append((decision, text[start:end].strip()))
    if expected_votes is not None and len(votes) != expected_votes:
        raise ValueError(f"Expected {expected_votes} votes in judge_reasoning, got {len(votes)}")
    return votes


def analyze_judge_failures(records: list[dict[str, Any]], expected_votes: int) -> dict[str, Any]:
    """Count retry-exhausted technical failures and parse failures separately."""
    total_votes = 0
    retry_failed_votes = 0
    parse_failed_votes = 0
    regex_recovered_votes = 0
    by_criterion = {
        "no_leak": {"total": 0, "retry_failed": 0, "parse_failed": 0, "regex_recovered": 0},
        "teaching": {"total": 0, "retry_failed": 0, "parse_failed": 0, "regex_recovered": 0},
    }
    convs_with_retry_fail: list[bool] = []
    convs_with_any_fail: list[bool] = []
    eval_pedagogy_missing = 0
    eval_correctness_missing = 0

    for rec in records:
        conv_retry_fail = False
        conv_any_fail = False
        for crit in ("no_leak", "teaching"):
            votes = parse_judge_votes(rec["judge_reasoning"][crit], expected_votes=expected_votes)
            for _decision, reason in votes:
                total_votes += 1
                by_criterion[crit]["total"] += 1
                if reason == _RETRY_FAIL_REASON:
                    retry_failed_votes += 1
                    by_criterion[crit]["retry_failed"] += 1
                    conv_retry_fail = True
                    conv_any_fail = True
                elif reason == _PARSE_FAIL_REASON:
                    parse_failed_votes += 1
                    by_criterion[crit]["parse_failed"] += 1
                    conv_any_fail = True
                elif reason == _REGEX_RECOVERED_REASON:
                    regex_recovered_votes += 1
                    by_criterion[crit]["regex_recovered"] += 1
        convs_with_retry_fail.append(conv_retry_fail)
        convs_with_any_fail.append(conv_any_fail)
        if rec["eval_judge_pedagogy"] is None:
            eval_pedagogy_missing += 1
        if rec["eval_judge_correctness"] is None:
            eval_correctness_missing += 1

    clean_retry_records = [r for r, bad in zip(records, convs_with_retry_fail) if not bad]
    clean_any_records = [r for r, bad in zip(records, convs_with_any_fail) if not bad]

    retry_fail_rate = (retry_failed_votes / total_votes) if total_votes else 0.0
    parse_fail_rate = (parse_failed_votes / total_votes) if total_votes else 0.0

    return {
        "total_votes": total_votes,
        "retry_failed_votes": retry_failed_votes,
        "parse_failed_votes": parse_failed_votes,
        "regex_recovered_votes": regex_recovered_votes,
        "retry_fail_rate": retry_fail_rate,
        "parse_fail_rate": parse_fail_rate,
        "by_criterion": by_criterion,
        "conversations_with_retry_fail": sum(convs_with_retry_fail),
        "conversations_with_any_fail": sum(convs_with_any_fail),
        "clean_retry_records": clean_retry_records,
        "clean_any_records": clean_any_records,
        "judge_available": retry_fail_rate <= 0.01 + _EPS,
        "eval_pedagogy_missing": eval_pedagogy_missing,
        "eval_correctness_missing": eval_correctness_missing,
    }


def _percentile(sorted_xs: list[float], q: float) -> float:
    rank = q * (len(sorted_xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (rank - lo)


def problem_values(records: list[dict[str, Any]], key: str) -> list[float]:
    """Compute per-problem mean values in sorted row_index order."""
    grouped: dict[int, list[float]] = {}
    for rec in records:
        val = rec.get(key)
        if val is not None:
            grouped.setdefault(rec["row_index"], []).append(float(val))
    return [statistics.fmean(grouped[idx]) for idx in sorted(grouped)]


def bootstrap_ci(values: list[float], seed: int = 42, draws: int = 10000) -> tuple[float, float, float, int]:
    """Nonparametric problem-level bootstrap mean and [2.5%, 97.5%] percentile CI."""
    if not values:
        raise ValueError("Cannot bootstrap empty value list")
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    n = len(values)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    return statistics.fmean(values), _percentile(samples, 0.025), _percentile(samples, 0.975), n


def intervals_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """Closed-interval overlap check with float precision tolerance."""
    return max(ci_a[0], ci_b[0]) <= min(ci_a[1], ci_b[1]) + _EPS


def judge_alignment_verdict(
    delta_transfer: float,
    ci_transfer: tuple[float, float],
    delta_same: float,
    ci_same: tuple[float, float],
    *,
    baseline: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply the pre-registered 3-tier decision rule on Δ_transfer and Δ_same."""
    ref = baseline or PAPER_BASE
    diff_t = abs(delta_transfer - ref["delta_transfer"])
    diff_s = abs(delta_same - ref["delta_same"])
    overlap_t = intervals_overlap(ci_transfer, ref["delta_transfer_ci"])
    overlap_s = intervals_overlap(ci_same, ref["delta_same_ci"])

    if (
        diff_t > 0.08 + _EPS
        or diff_s > 0.08 + _EPS
        or not overlap_t
        or not overlap_s
    ):
        verdict = "对不上"
    elif (
        diff_t <= 0.04 + _EPS
        and diff_s <= 0.04 + _EPS
        and overlap_t
        and overlap_s
    ):
        verdict = "对上"
    else:
        verdict = "部分对上"

    details = {
        "diff_transfer": diff_t,
        "diff_same": diff_s,
        "overlap_transfer": overlap_t,
        "overlap_same": overlap_s,
    }
    return verdict, details


def bootstrap_paired_stat(
    pairs: list[tuple[float, float]],
    stat_fn: Any,
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Bootstrap a paired problem-level statistic such as Hake g, Cohen's d, or Pearson r."""
    if not pairs:
        return {"mean": None, "ci": None, "n_problems": 0, "valid_draws": 0}
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    point = stat_fn(xs, ys)
    if point is None:
        return {"mean": None, "ci": None, "n_problems": len(pairs), "valid_draws": 0}
    n = len(pairs)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        sampled = rng.choices(pairs, k=n)
        val = stat_fn([p[0] for p in sampled], [p[1] for p in sampled])
        if val is not None and math.isfinite(val):
            samples.append(float(val))
    ci = None
    if samples:
        samples.sort()
        ci = [_percentile(samples, 0.025), _percentile(samples, 0.975)]
    return {"mean": float(point), "ci": ci, "n_problems": n, "valid_draws": len(samples)}


def bootstrap_corr_by_problem(
    records: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    method: str,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute per-conversation correlation with problem-clustered bootstrap CI."""
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    corr_fn = pearson_r if method == "pearson" else spearman_rho
    grouped: dict[int, list[tuple[float, float]]] = {}
    for rec in records:
        xv = rec.get(x_key)
        yv = rec.get(y_key)
        if xv is not None and yv is not None and _is_number(xv) and _is_number(yv):
            grouped.setdefault(rec["row_index"], []).append((float(xv), float(yv)))
    prob_ids = sorted(grouped)
    all_pairs = [xy for pid in prob_ids for xy in grouped[pid]]
    if len(all_pairs) < 2:
        return {
            "mean": None,
            "ci": None,
            "n_conversations": len(all_pairs),
            "n_problems": len(prob_ids),
            "method": method,
            "valid_draws": 0,
        }
    point = corr_fn([p[0] for p in all_pairs], [p[1] for p in all_pairs])
    if point is None:
        return {
            "mean": None,
            "ci": None,
            "n_conversations": len(all_pairs),
            "n_problems": len(prob_ids),
            "method": method,
            "valid_draws": 0,
        }
    rng = random.Random(seed)
    n_prob = len(prob_ids)
    samples: list[float] = []
    for _ in range(draws):
        sampled_ids = rng.choices(prob_ids, k=n_prob)
        boot_pairs = [xy for pid in sampled_ids for xy in grouped[pid]]
        val = corr_fn([p[0] for p in boot_pairs], [p[1] for p in boot_pairs])
        if val is not None and math.isfinite(val):
            samples.append(float(val))
    ci = None
    if samples:
        samples.sort()
        ci = [_percentile(samples, 0.025), _percentile(samples, 0.975)]
    return {
        "mean": float(point),
        "ci": ci,
        "n_conversations": len(all_pairs),
        "n_problems": n_prob,
        "method": method,
        "valid_draws": len(samples),
    }


def _problem_paired_values(
    records: list[dict[str, Any]], key_a: str, key_b: str
) -> list[tuple[float, float]]:
    grouped_a: dict[int, list[float]] = {}
    grouped_b: dict[int, list[float]] = {}
    for rec in records:
        va = rec.get(key_a)
        vb = rec.get(key_b)
        if va is not None and vb is not None:
            grouped_a.setdefault(rec["row_index"], []).append(float(va))
            grouped_b.setdefault(rec["row_index"], []).append(float(vb))
    common_ids = sorted(set(grouped_a) & set(grouped_b))
    return [
        (statistics.fmean(grouped_a[idx]), statistics.fmean(grouped_b[idx]))
        for idx in common_ids
    ]


REQUIRED_CONTROL_METRICS_KEYS = (
    "calls_expected",
    "calls_scored",
    "calls_failed",
    "fail_rate",
    "exceeds_1pct",
)

REQUIRED_CONTROL_RECORD_KEYS = (
    "row_index",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "ctl1_same",
    "ctl1_same_scored",
    "ctl1_same_failed",
    "ctl1_transfer",
    "ctl1_transfer_scored",
    "ctl1_transfer_failed",
    "ctl2_same",
    "ctl2_same_scored",
    "ctl2_same_failed",
    "ctl2_transfer",
    "ctl2_transfer_scored",
    "ctl2_transfer_failed",
)


def validate_control_doc(
    control_doc: Any, stage2_records: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Validate untutored control JSON and alignment with Stage 2 problem rows."""
    if not isinstance(control_doc, dict):
        raise ValueError("Control JSON root must be a dict")
    if "metrics" not in control_doc or not isinstance(control_doc["metrics"], dict):
        raise ValueError("Control JSON missing 'metrics' dict")
    if "records" not in control_doc or not isinstance(control_doc["records"], list) or not control_doc["records"]:
        raise ValueError("Control JSON missing non-empty 'records' list")

    cm = control_doc["metrics"]
    for key in REQUIRED_CONTROL_METRICS_KEYS:
        if key not in cm:
            raise ValueError(f"Control JSON metrics missing required field: {key}")

    if "k" not in control_doc or isinstance(control_doc["k"], bool) or not isinstance(control_doc["k"], int):
        raise ValueError("Control JSON missing integer 'k'")
    k = int(control_doc["k"])
    if k <= 0:
        raise ValueError(f"Control JSON invalid k={k}")

    by_row: dict[int, dict[str, Any]] = {}
    total_scored = 0
    total_failed = 0
    for idx, rec in enumerate(control_doc["records"]):
        if not isinstance(rec, dict):
            raise ValueError(f"Control record #{idx} is not a dict")
        for key in REQUIRED_CONTROL_RECORD_KEYS:
            if key not in rec:
                raise ValueError(f"Control record #{idx} missing required field: {key}")
        ridx = rec["row_index"]
        if isinstance(ridx, bool) or not isinstance(ridx, int):
            raise ValueError(f"Control record #{idx} has invalid row_index")
        if ridx in by_row:
            raise ValueError(f"Control JSON has duplicate row_index={ridx}")
        for num_k in (
            "pre_solve_rate",
            "pre_solve_rate_transfer",
            "ctl1_same",
            "ctl1_transfer",
            "ctl2_same",
            "ctl2_transfer",
        ):
            if not _is_number(rec[num_k]):
                raise ValueError(f"Control record #{idx} has non-numeric '{num_k}'")
        for rnd in ("ctl1", "ctl2"):
            for kind in ("same", "transfer"):
                sc = int(rec[f"{rnd}_{kind}_scored"])
                fl = int(rec[f"{rnd}_{kind}_failed"])
                if sc + fl != k or sc < 0 or fl < 0:
                    raise ValueError(
                        f"Control record #{idx} {rnd}_{kind} counts ({sc}+{fl}) != k={k}"
                    )
                total_scored += sc
                total_failed += fl
        by_row[ridx] = rec

    total_expected = len(control_doc["records"]) * 4 * k
    if int(cm["calls_expected"]) != total_expected:
        raise ValueError("Control JSON calls_expected mismatch")
    if int(cm["calls_scored"]) != total_scored or int(cm["calls_failed"]) != total_failed:
        raise ValueError("Control JSON call scored/failed sum mismatch")
    fail_rate = (total_failed / total_expected) if total_expected else 0.0
    exceeds_1pct = fail_rate > 0.01 + _EPS
    if bool(cm["exceeds_1pct"]) is not exceeds_1pct:
        raise ValueError("Control JSON exceeds_1pct mismatch")

    stage2_by_row: dict[int, list[dict[str, Any]]] = {}
    for rec in stage2_records:
        stage2_by_row.setdefault(rec["row_index"], []).append(rec)

    for ridx, recs in sorted(stage2_by_row.items()):
        if ridx not in by_row:
            raise ValueError(f"Control JSON missing row_index={ridx} present in Stage 2 results")
        ctl_rec = by_row[ridx]
        s2_pre_s = statistics.fmean(float(r["pre_solve_rate"]) for r in recs)
        s2_pre_t = statistics.fmean(float(r["pre_solve_rate_transfer"]) for r in recs)
        if not math.isclose(s2_pre_s, float(ctl_rec["pre_solve_rate"]), abs_tol=1e-9):
            raise ValueError(f"Pre-solve rate mismatch on original problem at row_index={ridx}")
        if not math.isclose(s2_pre_t, float(ctl_rec["pre_solve_rate_transfer"]), abs_tol=1e-9):
            raise ValueError(f"Pre-solve rate mismatch on transfer problem at row_index={ridx}")

    diag = {
        "num_control_problems": len(by_row),
        "k": k,
        "calls_expected": total_expected,
        "calls_scored": total_scored,
        "calls_failed": total_failed,
        "fail_rate": fail_rate,
        "exceeds_1pct": exceeds_1pct,
        "control_available": not exceeds_1pct,
    }
    return by_row, diag


def _bootstrap_r1_ci(values: list[float], *, seed: int, draws: int) -> dict[str, Any] | None:
    if not values:
        return None
    mean_v, lo_v, hi_v, n = bootstrap_ci(values, seed=seed, draws=draws)
    return {
        "mean": mean_v,
        "ci": [lo_v, hi_v],
        "n_problems": n,
        "excludes_zero": (lo_v > _EPS) or (hi_v < -_EPS),
    }


def build_control_analysis(
    stage2_records: list[dict[str, Any]],
    control_by_row: dict[int, dict[str, Any]],
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute Stage 3 Section C untutored control metrics aligned by problem."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for rec in stage2_records:
        grouped.setdefault(rec["row_index"], []).append(rec)

    aligned: list[dict[str, float]] = []
    for ridx in sorted(grouped):
        recs = grouped[ridx]
        ctl = control_by_row[ridx]
        aligned.append(
            {
                "row_index": float(ridx),
                "pre_same": statistics.fmean(float(r["pre_solve_rate"]) for r in recs),
                "pre_transfer": statistics.fmean(float(r["pre_solve_rate_transfer"]) for r in recs),
                "post_same": statistics.fmean(float(r["post_solve_same"]) for r in recs),
                "post_transfer": statistics.fmean(float(r["post_solve_transfer"]) for r in recs),
                "ctl1_same": float(ctl["ctl1_same"]),
                "ctl1_transfer": float(ctl["ctl1_transfer"]),
                "ctl2_same": float(ctl["ctl2_same"]),
                "ctl2_transfer": float(ctl["ctl2_transfer"]),
                "r1_pre_same": statistics.fmean(float(r["delta_same"]) for r in recs),
                "r1_pre_transfer": statistics.fmean(float(r["delta_transfer"]) for r in recs),
            }
        )

    def summarize_pair(pre_vals: list[float], post_vals: list[float]) -> dict[str, Any]:
        r1_vals = [normalized_delta(po, pr) for pr, po in zip(pre_vals, post_vals)]
        pairs = list(zip(pre_vals, post_vals))
        return {
            "pre_mean": statistics.fmean(pre_vals),
            "post_mean": statistics.fmean(post_vals),
            "r1": _bootstrap_r1_ci(r1_vals, seed=seed, draws=draws),
            "r2_hake_g": bootstrap_paired_stat(pairs, compute_hake_g, seed=seed, draws=draws),
            "r3_cohens_d": bootstrap_paired_stat(pairs, compute_cohens_d, seed=seed, draws=draws),
        }

    pre_t = [a["pre_transfer"] for a in aligned]
    pre_s = [a["pre_same"] for a in aligned]
    ctl1_t = [a["ctl1_transfer"] for a in aligned]
    ctl1_s = [a["ctl1_same"] for a in aligned]
    ctl2_t = [a["ctl2_transfer"] for a in aligned]
    ctl2_s = [a["ctl2_same"] for a in aligned]
    post_t = [a["post_transfer"] for a in aligned]
    post_s = [a["post_same"] for a in aligned]

    r4_baseline = {
        "transfer": summarize_pair(pre_t, ctl1_t),
        "same": summarize_pair(pre_s, ctl1_s),
    }
    rebaselined_gain = {
        "transfer": summarize_pair(ctl1_t, post_t),
        "same": summarize_pair(ctl1_s, post_s),
    }

    retest_reliability = {
        "same_pre_ctl1": bootstrap_paired_stat(list(zip(pre_s, ctl1_s)), pearson_r, seed=seed, draws=draws),
        "same_ctl1_ctl2": bootstrap_paired_stat(list(zip(ctl1_s, ctl2_s)), pearson_r, seed=seed, draws=draws),
        "transfer_pre_ctl1": bootstrap_paired_stat(list(zip(pre_t, ctl1_t)), pearson_r, seed=seed, draws=draws),
        "transfer_ctl1_ctl2": bootstrap_paired_stat(list(zip(ctl1_t, ctl2_t)), pearson_r, seed=seed, draws=draws),
    }

    tiers: dict[str, list[dict[str, float]]] = {"hard": [], "medium": [], "easy": []}
    for a in aligned:
        tiers[classify_difficulty_tier(a["ctl2_transfer"])].append(a)

    tier_summary: dict[str, Any] = {}
    for tier_name in ("hard", "medium", "easy"):
        items = tiers[tier_name]
        if not items:
            tier_summary[tier_name] = {"n_problems": 0}
            continue
        t_r1_pre_t = [x["r1_pre_transfer"] for x in items]
        t_r1_ctl1_t = [normalized_delta(x["post_transfer"], x["ctl1_transfer"]) for x in items]
        t_r1_pre_s = [x["r1_pre_same"] for x in items]
        t_r1_ctl1_s = [normalized_delta(x["post_same"], x["ctl1_same"]) for x in items]
        tier_summary[tier_name] = {
            "n_problems": len(items),
            "pre_transfer_mean": statistics.fmean(x["pre_transfer"] for x in items),
            "ctl1_transfer_mean": statistics.fmean(x["ctl1_transfer"] for x in items),
            "ctl2_transfer_mean": statistics.fmean(x["ctl2_transfer"] for x in items),
            "post_transfer_mean": statistics.fmean(x["post_transfer"] for x in items),
            "r1_pre_transfer": _bootstrap_r1_ci(t_r1_pre_t, seed=seed, draws=draws),
            "r1_ctl1_transfer": _bootstrap_r1_ci(t_r1_ctl1_t, seed=seed, draws=draws),
            "r1_pre_same": _bootstrap_r1_ci(t_r1_pre_s, seed=seed, draws=draws),
            "r1_ctl1_same": _bootstrap_r1_ci(t_r1_ctl1_s, seed=seed, draws=draws),
        }

    return {
        "n_problems": len(aligned),
        "r4_untutored_baseline": r4_baseline,
        "rebaselined_gain_ctl1_to_post": rebaselined_gain,
        "test_retest_reliability": retest_reliability,
        "difficulty_tiers_by_ctl2_transfer": tier_summary,
    }


def build_summary(
    doc: dict[str, Any],
    seed: int = 42,
    draws: int = 10000,
    control_doc: dict[str, Any] | None = None,
    paper_baseline: str = "base-4b",
    agnes_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, recompute, bootstrap, and assemble a completely sanitized summary dict."""
    recorded_metrics, records, judge_votes = validate_results_doc(doc)
    recomputed = recompute_author_summary(records)
    verify_author_summary_match(recomputed, recorded_metrics)

    judge_diag = analyze_judge_failures(records, expected_votes=judge_votes)

    dt_mean, dt_lo, dt_hi, n_prob = bootstrap_ci(problem_values(records, "delta_transfer"), seed=seed, draws=draws)
    ds_mean, ds_lo, ds_hi, _ = bootstrap_ci(problem_values(records, "delta_same"), seed=seed, draws=draws)
    dt_g_mean, dt_g_lo, dt_g_hi, _ = bootstrap_ci(problem_values(records, "delta_transfer_gated"), seed=seed, draws=draws)
    ds_g_mean, ds_g_lo, ds_g_hi, _ = bootstrap_ci(problem_values(records, "delta_same_gated"), seed=seed, draws=draws)
    gap_mean, gap_lo, gap_hi, _ = bootstrap_ci(problem_values(records, "transfer_gap"), seed=seed, draws=draws)
    leak_mean, leak_lo, leak_hi, _ = bootstrap_ci(problem_values(records, "leaked"), seed=seed, draws=draws)

    clean_retry_records = judge_diag["clean_retry_records"]
    if clean_retry_records:
        leak_sens_mean, leak_sens_lo, leak_sens_hi, leak_sens_n = bootstrap_ci(
            problem_values(clean_retry_records, "leaked"), seed=seed, draws=draws
        )
    else:
        leak_sens_mean, leak_sens_lo, leak_sens_hi, leak_sens_n = None, None, None, 0

    r2_transfer = bootstrap_paired_stat(
        _problem_paired_values(records, "pre_solve_rate_transfer", "post_solve_transfer"),
        compute_hake_g,
        seed=seed,
        draws=draws,
    )
    r2_same = bootstrap_paired_stat(
        _problem_paired_values(records, "pre_solve_rate", "post_solve_same"),
        compute_hake_g,
        seed=seed,
        draws=draws,
    )
    r3_transfer = bootstrap_paired_stat(
        _problem_paired_values(records, "pre_solve_rate_transfer", "post_solve_transfer"),
        compute_cohens_d,
        seed=seed,
        draws=draws,
    )
    r3_same = bootstrap_paired_stat(
        _problem_paired_values(records, "pre_solve_rate", "post_solve_same"),
        compute_cohens_d,
        seed=seed,
        draws=draws,
    )

    process_outcome_correlations = {
        "n_clean_retry_conversations": len(clean_retry_records),
        "leaked_vs_delta_transfer": bootstrap_corr_by_problem(
            clean_retry_records, "leaked", "delta_transfer", method="pearson", seed=seed, draws=draws
        ),
        "leaked_vs_delta_same": bootstrap_corr_by_problem(
            clean_retry_records, "leaked", "delta_same", method="pearson", seed=seed, draws=draws
        ),
        "teaching_rejected_vs_delta_transfer": bootstrap_corr_by_problem(
            clean_retry_records, "teaching_rejected", "delta_transfer", method="pearson", seed=seed, draws=draws
        ),
        "teaching_rejected_vs_delta_same": bootstrap_corr_by_problem(
            clean_retry_records, "teaching_rejected", "delta_same", method="pearson", seed=seed, draws=draws
        ),
        "pedagogy_vs_delta_transfer": bootstrap_corr_by_problem(
            clean_retry_records, "eval_judge_pedagogy", "delta_transfer", method="spearman", seed=seed, draws=draws
        ),
        "pedagogy_vs_delta_same": bootstrap_corr_by_problem(
            clean_retry_records, "eval_judge_pedagogy", "delta_same", method="spearman", seed=seed, draws=draws
        ),
        "correctness_vs_delta_transfer": bootstrap_corr_by_problem(
            clean_retry_records, "eval_judge_correctness", "delta_transfer", method="spearman", seed=seed, draws=draws
        ),
        "correctness_vs_delta_same": bootstrap_corr_by_problem(
            clean_retry_records, "eval_judge_correctness", "delta_same", method="spearman", seed=seed, draws=draws
        ),
    }

    if paper_baseline not in PAPER_BASELINES:
        raise ValueError(f"Unknown paper_baseline: {paper_baseline!r}")
    baseline_ref = PAPER_BASELINES[paper_baseline]
    verdict, verdict_details = judge_alignment_verdict(
        dt_mean, (dt_lo, dt_hi), ds_mean, (ds_lo, ds_hi), baseline=baseline_ref
    )

    summary_out: dict[str, Any] = {
        "tag": str(doc.get("tag") or ""),
        "judge_model": str(doc.get("judge_model") or ""),
        "paper_baseline": paper_baseline,
        "num_conversations": recomputed["num_conversations"],
        "num_problems": n_prob,
        "samples_per_problem": recomputed["samples_per_problem"],
        "assume_thinking": recomputed["assume_thinking"],
        "transfer_tested_rate": recomputed["transfer_tested_rate"],
        "n_student_calls_failed": recomputed["n_student_calls_failed"],
        "seed": seed,
        "draws": draws,
        "metrics": {
            "delta_transfer": {"mean": dt_mean, "ci": [dt_lo, dt_hi]},
            "delta_same": {"mean": ds_mean, "ci": [ds_lo, ds_hi]},
            "delta_transfer_gated": {"mean": dt_g_mean, "ci": [dt_g_lo, dt_g_hi]},
            "delta_same_gated": {"mean": ds_g_mean, "ci": [ds_g_lo, ds_g_hi]},
            "transfer_gap": {"mean": gap_mean, "ci": [gap_lo, gap_hi]},
            "r2_hake_g_transfer": r2_transfer,
            "r2_hake_g_same": r2_same,
            "r3_cohens_d_transfer": r3_transfer,
            "r3_cohens_d_same": r3_same,
            "leak_rate_author": {"mean": leak_mean, "ci": [leak_lo, leak_hi], "n_problems": n_prob},
            "leak_rate_excl_tech_fail": {
                "mean": leak_sens_mean,
                "ci": [leak_sens_lo, leak_sens_hi] if leak_sens_mean is not None else None,
                "n_problems": leak_sens_n,
                "n_conversations": len(clean_retry_records),
            },
            "teaching_reject_rate": recomputed["teaching_reject_rate"],
            "eval_judge_pedagogy": recomputed["eval_judge_pedagogy"],
            "eval_judge_correctness": recomputed["eval_judge_correctness"],
            "eval_judge_ok": recomputed["eval_judge_ok"],
            "blanked_teacher_turn_rate": recomputed["blanked_teacher_turn_rate"],
            "token_budget_truncated_rate": recomputed["token_budget_truncated_rate"],
            "eoc_rate": recomputed["eoc_rate"],
            "thinking_tokens_avg_per_turn": recomputed["thinking_tokens_avg_per_turn"],
            "pre_solve_rate": recomputed["pre_solve_rate"],
            "pre_solve_rate_transfer": recomputed["pre_solve_rate_transfer"],
            "post_solve_same": recomputed["post_solve_same"],
            "post_solve_transfer": recomputed["post_solve_transfer"],
        },
        "process_outcome_correlations": process_outcome_correlations,
        "agnes_diagnostics": {
            "total_votes": judge_diag["total_votes"],
            "retry_failed_votes": judge_diag["retry_failed_votes"],
            "parse_failed_votes": judge_diag["parse_failed_votes"],
            "regex_recovered_votes": judge_diag["regex_recovered_votes"],
            "retry_fail_rate": judge_diag["retry_fail_rate"],
            "parse_fail_rate": judge_diag["parse_fail_rate"],
            "by_criterion": judge_diag["by_criterion"],
            "conversations_with_retry_fail": judge_diag["conversations_with_retry_fail"],
            "conversations_with_any_fail": judge_diag["conversations_with_any_fail"],
            "judge_available": judge_diag["judge_available"],
            "eval_pedagogy_missing": judge_diag["eval_pedagogy_missing"],
            "eval_correctness_missing": judge_diag["eval_correctness_missing"],
        },
        "verdict": verdict,
        "verdict_details": verdict_details,
    }
    if agnes_stats is not None:
        summary_out["agnes_window_stats"] = {
            k: agnes_stats[k]
            for k in (
                "window_init",
                "window_max",
                "inner_max_retries",
                "outer_num_retries",
                "max_http_requests_per_vote",
                "backoff_base_s",
                "backoff_cap_s",
                "logical_calls",
                "http_attempts",
                "success_calls",
                "transient_retries",
                "ambiguous_failures",
                "non_retriable_failures",
                "max_limit_reached",
                "final_limit",
                "window_drops",
                "congestion_events",
            )
            if k in agnes_stats
        }

    if control_doc is not None:
        control_by_row, control_diag = validate_control_doc(control_doc, records)
        summary_out["control_diagnostics"] = control_diag
        if control_diag["control_available"]:
            summary_out["control_analysis"] = build_control_analysis(
                records, control_by_row, seed=seed, draws=draws
            )
        else:
            summary_out["control_analysis"] = None

    return summary_out


def _fmt_stat_ci(val: dict[str, Any] | None, *, pct: bool = False, signed: bool = True) -> str:
    if not val or val.get("mean") is None:
        return "N/A"
    mean_v = float(val["mean"])
    ci = val.get("ci")
    if pct:
        if ci is None:
            return f"{mean_v*100:.2f}% [N/A]"
        return f"{mean_v*100:.2f}% [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]"
    fmt = "+.4f" if signed else ".4f"
    if ci is None:
        return f"{format(mean_v, fmt)} [N/A]"
    return f"{format(mean_v, fmt)} [{format(ci[0], fmt)}, {format(ci[1], fmt)}]"


def format_report(summary: dict[str, Any]) -> str:
    """Format a human-readable aggregate summary table (no sensitive fields)."""
    m = summary["metrics"]
    diag = summary["agnes_diagnostics"]
    vd = summary["verdict_details"]
    judge_ok = diag["judge_available"]
    ref = PAPER_BASELINES.get(summary.get("paper_baseline", "base-4b"), PAPER_BASE)
    ref_label = ref.get("label", "Paper (base 4B)")
    ref_leak_lo, ref_leak_hi = ref["leak_rate_ci"]
    ref_dt_lo, ref_dt_hi = ref["delta_transfer_ci"]
    ref_ds_lo, ref_ds_hi = ref["delta_same_ci"]

    leak_author_str = _fmt_stat_ci(m["leak_rate_author"], pct=True) if judge_ok else "不可用 (>1% 技术失败)"
    leak_sens_str = _fmt_stat_ci(m["leak_rate_excl_tech_fail"], pct=True) if judge_ok else "不可用 (>1% 技术失败)"
    teach_str = f"{m['teaching_reject_rate']*100:.2f}%" if judge_ok else "不可用 (>1% 技术失败)"
    ped_str = f"{m['eval_judge_pedagogy']:.2f}" if judge_ok else "不可用 (>1% 技术失败)"
    fact_str = f"{m['eval_judge_correctness']:.2f}" if judge_ok else "不可用 (>1% 技术失败)"

    lines = [
        "=" * 78,
        f"Author-494 Eval Summary (tag={summary['tag'] or 'N/A'}, judge={summary['judge_model'] or 'N/A'})",
        f"Conversations: {summary['num_conversations']}  Problems: {summary['num_problems']}  "
        f"Bootstrap: {summary['draws']}x (seed={summary['seed']})",
        f"assume_thinking={summary['assume_thinking']}  "
        f"transfer_tested_rate={summary['transfer_tested_rate']:.2%}  "
        f"student_calls_failed={summary['n_student_calls_failed']}",
        "-" * 78,
        "Agnes Gate Diagnostics (no_leak + teaching):",
        f"  Total votes:           {diag['total_votes']}",
        f"  Retry-failed votes:    {diag['retry_failed_votes']} ({diag['retry_fail_rate']:.2%})  "
        f"[convs={diag['conversations_with_retry_fail']}]",
        f"  Parse-failed votes:    {diag['parse_failed_votes']} ({diag['parse_fail_rate']:.2%})  "
        f"[regex_recovered={diag['regex_recovered_votes']}]",
        f"  Judge status (<=1%):   {'可用' if judge_ok else '不可用（技术失败票超过 1%，停下来报告）'}",
        f"  1-5 Eval judge missing: pedagogy={diag['eval_pedagogy_missing']}, "
        f"correctness={diag['eval_correctness_missing']} (eval_judge_ok={m['eval_judge_ok']:.2%})",
    ]
    ws = summary.get("agnes_window_stats")
    if isinstance(ws, dict):
        lines.extend([
            "Agnes Congestion Window Diagnostics:",
            f"  logical_calls={ws.get('logical_calls')} http_attempts={ws.get('http_attempts')} "
            f"success_calls={ws.get('success_calls')} transient_retries={ws.get('transient_retries')} "
            f"ambiguous_failures={ws.get('ambiguous_failures')}",
            f"  window=[init={ws.get('window_init')}, max={ws.get('window_max')}, "
            f"reached={ws.get('max_limit_reached')}, final={ws.get('final_limit')}] "
            f"drops={ws.get('window_drops')} congestion_events={ws.get('congestion_events')} "
            f"max_reqs_per_vote={ws.get('max_http_requests_per_vote')}",
        ])
    lines.extend([
        "-" * 78,
        f"{'Metric':<30} | {'Ours (A100 + Agnes)':<30} | {ref_label:<22}",
        "-" * 78,
        f"{'Δ_transfer (primary)':<30} | {_fmt_stat_ci(m['delta_transfer']):<30} | {ref['delta_transfer']:+.4f} [{ref_dt_lo:+.4f}, {ref_dt_hi:+.4f}]",
        f"{'Δ_same (primary)':<30} | {_fmt_stat_ci(m['delta_same']):<30} | {ref['delta_same']:+.4f} [{ref_ds_lo:+.4f}, {ref_ds_hi:+.4f}]",
        f"{'Δ_transfer (gated)':<30} | {_fmt_stat_ci(m['delta_transfer_gated']):<30} | —",
        f"{'Δ_same (gated)':<30} | {_fmt_stat_ci(m['delta_same_gated']):<30} | —",
        f"{'Transfer gap (same-transfer)':<30} | {_fmt_stat_ci(m['transfer_gap']):<30} | —",
        f"{'R2 Hake g (transfer)':<30} | {_fmt_stat_ci(m['r2_hake_g_transfer']):<30} | —",
        f"{'R2 Hake g (same)':<30} | {_fmt_stat_ci(m['r2_hake_g_same']):<30} | —",
        f"{'R3 Cohen d (transfer)':<30} | {_fmt_stat_ci(m['r3_cohens_d_transfer']):<30} | —",
        f"{'R3 Cohen d (same)':<30} | {_fmt_stat_ci(m['r3_cohens_d_same']):<30} | —",
        f"{'Leak rate (author)':<30} | {leak_author_str:<30} | {ref['leak_rate']*100:.2f}% [{ref_leak_lo*100:.2f}%, {ref_leak_hi*100:.2f}%]",
        f"{'Leak rate (excl tech-fail)':<30} | {leak_sens_str:<30} | —",
        f"{'Teaching reject rate':<30} | {teach_str:<30} | —",
        f"{'Ped. (1-5)':<30} | {ped_str:<30} | {ref['eval_judge_pedagogy']:.2f}",
        f"{'Fact. (1-5)':<30} | {fact_str:<30} | {ref['eval_judge_correctness']:.2f}",
        f"{'Empty% (blanked turns)':<30} | {m['blanked_teacher_turn_rate']*100:>6.2f}%{'':<23} | {ref['blanked_teacher_turn_rate']*100:.2f}%",
        f"{'Seq. trunc.%':<30} | {m['token_budget_truncated_rate']*100:>6.2f}%{'':<23} | {ref['token_budget_truncated_rate']*100:.2f}%",
        f"{'EoC rate':<30} | {m['eoc_rate']*100:>6.2f}%{'':<23} | {ref['eoc_rate']*100:.2f}%",
        f"{'Thinking tok/turn':<30} | {m['thinking_tokens_avg_per_turn']:>6.1f}{'':<24} | {ref['thinking_tokens_avg_per_turn']:.1f}",
        "-" * 78,
        f"判定结论 (仅看主口径 Δ_transfer & Δ_same): {summary['verdict']}",
        f"  |Δ_transfer - {ref['delta_transfer']:.2f}| = {vd['diff_transfer']:.4f}, CI重叠={vd['overlap_transfer']}",
        f"  |Δ_same - {ref['delta_same']:.2f}|     = {vd['diff_same']:.4f}, CI重叠={vd['overlap_same']}",
    ])

    poc = summary["process_outcome_correlations"]
    lines.extend([
        "-" * 78,
        f"Process-Outcome Correlations (clean_retry_convs={poc['n_clean_retry_conversations']}, excl. retry-failed):",
    ])
    if not judge_ok:
        lines.append("  Process-Outcome Correlations: 不可用 (>1% 技术失败)")
    else:
        for label, key in (
            ("Pearson r(leaked, Δ_transfer)", "leaked_vs_delta_transfer"),
            ("Pearson r(leaked, Δ_same)", "leaked_vs_delta_same"),
            ("Pearson r(teaching_rej, Δ_transfer)", "teaching_rejected_vs_delta_transfer"),
            ("Pearson r(teaching_rej, Δ_same)", "teaching_rejected_vs_delta_same"),
            ("Spearman ρ(Pedagogy, Δ_transfer)", "pedagogy_vs_delta_transfer"),
            ("Spearman ρ(Pedagogy, Δ_same)", "pedagogy_vs_delta_same"),
            ("Spearman ρ(Correctness, Δ_transfer)", "correctness_vs_delta_transfer"),
            ("Spearman ρ(Correctness, Δ_same)", "correctness_vs_delta_same"),
        ):
            item = poc[key]
            lines.append(f"  {label:<38}: {_fmt_stat_ci(item)} (n_conv={item['n_conversations']})")

    if "control_diagnostics" in summary:
        cd = summary["control_diagnostics"]
        lines.extend([
            "-" * 78,
            f"Untutored Control Diagnostics: problems={cd['num_control_problems']} k={cd['k']} "
            f"calls={cd['calls_scored']}/{cd['calls_expected']} failed={cd['calls_failed']} "
            f"({cd['fail_rate']:.4%}) status={'可用' if cd['control_available'] else '不可用（>1% 学生调用失败，停下来报告）'}",
        ])
        ca = summary.get("control_analysis")
        if ca is not None:
            r4 = ca["r4_untutored_baseline"]
            rb = ca["rebaselined_gain_ctl1_to_post"]
            rel = ca["test_retest_reliability"]
            lines.extend([
                "R4 Untutored Baseline (pre -> ctl1) & Rebaselined Gain (ctl1 -> post):",
                f"  Transfer | pre={r4['transfer']['pre_mean']:.4f} ctl1={r4['transfer']['post_mean']:.4f} post={rb['transfer']['post_mean']:.4f}",
                f"    R4 (pre->ctl1):  R1={_fmt_stat_ci(r4['transfer']['r1'])} (excl_0={r4['transfer']['r1']['excludes_zero']}) | "
                f"R2={_fmt_stat_ci(r4['transfer']['r2_hake_g'])} | R3={_fmt_stat_ci(r4['transfer']['r3_cohens_d'])}",
                f"    R1 (pre->post):  {_fmt_stat_ci(m['delta_transfer'])}",
                f"    R1 (ctl1->post): {_fmt_stat_ci(rb['transfer']['r1'])}",
                f"  Original | pre={r4['same']['pre_mean']:.4f} ctl1={r4['same']['post_mean']:.4f} post={rb['same']['post_mean']:.4f}",
                f"    R4 (pre->ctl1):  R1={_fmt_stat_ci(r4['same']['r1'])} (excl_0={r4['same']['r1']['excludes_zero']}) | "
                f"R2={_fmt_stat_ci(r4['same']['r2_hake_g'])} | R3={_fmt_stat_ci(r4['same']['r3_cohens_d'])}",
                f"    R1 (pre->post):  {_fmt_stat_ci(m['delta_same'])}",
                f"    R1 (ctl1->post): {_fmt_stat_ci(rb['same']['r1'])}",
                "Test-Retest Reliability (Pearson r [95% CI]):",
                f"  Transfer pre  vs ctl1: {_fmt_stat_ci(rel['transfer_pre_ctl1'])} | ctl1 vs ctl2: {_fmt_stat_ci(rel['transfer_ctl1_ctl2'])}",
                f"  Original pre  vs ctl1: {_fmt_stat_ci(rel['same_pre_ctl1'])} | ctl1 vs ctl2: {_fmt_stat_ci(rel['same_ctl1_ctl2'])}",
                "Difficulty Tiers by ctl2_transfer (hard=[0,1/3), medium=[1/3,2/3), easy=[2/3,1]):",
            ])
            tiers = ca["difficulty_tiers_by_ctl2_transfer"]
            for tname, label in (("hard", "Hard   [0, 1/3) "), ("medium", "Medium [1/3,2/3)"), ("easy", "Easy   [2/3, 1] ")):
                td = tiers[tname]
                if td["n_problems"] == 0:
                    lines.append(f"  {label} | n=0")
                else:
                    lines.append(
                        f"  {label} | n={td['n_problems']:>3} | "
                        f"Trans R1(pre->post)={_fmt_stat_ci(td['r1_pre_transfer'])} R1(ctl1->post)={_fmt_stat_ci(td['r1_ctl1_transfer'])} | "
                        f"Same R1(pre->post)={_fmt_stat_ci(td['r1_pre_same'])} R1(ctl1->post)={_fmt_stat_ci(td['r1_ctl1_same'])}"
                    )

    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Eduardo author-494 eval JSON safely.")
    ap.add_argument("result_json", type=Path, help="Path to transfer_eval_*.json")
    ap.add_argument("--control-json", type=Path, default=None, help="Optional Stage 3 control_eval_*.json")
    ap.add_argument("--agnes-stats", type=Path, default=None, help="Optional agnes_window_stats.json")
    ap.add_argument(
        "--paper-baseline",
        choices=sorted(PAPER_BASELINES),
        default="base-4b",
        help="Paper reference baseline for verdict and comparison table",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path to write aggregate summary JSON")
    args = ap.parse_args()

    try:
        doc = json.loads(args.result_json.read_text(encoding="utf-8"))
        control_doc = json.loads(args.control_json.read_text(encoding="utf-8")) if args.control_json else None
        agnes_stats = json.loads(args.agnes_stats.read_text(encoding="utf-8")) if args.agnes_stats else None
        summary = build_summary(
            doc,
            seed=args.seed,
            draws=args.draws,
            control_doc=control_doc,
            paper_baseline=args.paper_baseline,
            agnes_stats=agnes_stats,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(format_report(summary))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
