#!/usr/bin/env python3
"""Summarize and verify Pro 5000 author-494 evaluation results (aggregate only).

Supports:
1. Single-run mode (e.g., Stage 1 20-problem base smoke test):
   summarize_author494_pro5000.py <result_json> [--paper-baseline base|eduardo4b]
2. Full T-020 comparison mode (Verdict A, Verdict B paired with A100 base,
   cross-machine stack check table, and secondary metrics including ctl1 rebaselined):
   summarize_author494_pro5000.py --a-group-json <path> --t018-table <path> --paired-csv <path> \
     [--compare-stack-json <path>] [--pro5000-ctl-json <path>]

Never prints problem IDs, row_index, unit_id, problem text, answers, dialogues,
or judge reasoning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PAIRED_DIR = HERE / "paired_gain"
if str(PAIRED_DIR) not in sys.path:
    sys.path.insert(0, str(PAIRED_DIR))

import numpy as np  # noqa: E402
import paired_gain as pg  # noqa: E402

PAPER_EDUARDO_4B: dict[str, Any] = {
    "delta_transfer": 0.25,
    "delta_transfer_ci": (0.21, 0.29),
    "delta_same": 0.37,
    "delta_same_ci": (0.33, 0.42),
    "leak_rate": 0.306,
    "leak_rate_ci": (0.267, 0.346),
    "eval_judge_pedagogy": 3.66,
    "eval_judge_correctness": 4.19,
    "blanked_teacher_turn_rate": 0.037,
    "token_budget_truncated_rate": 0.107,
    "eoc_rate": 0.696,
    "thinking_tokens_avg_per_turn": 544.0,
}

PAPER_BASE_4B: dict[str, Any] = {
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

# T-018 A100 Qwen3.5-4B base reference (from 组织/结果/T-018.md) for dialogue-level diagnostics
# not present in the per-problem table (Empty%, EoC rate, Thinking tok/turn).
A100_T018_BASE: dict[str, Any] = {
    "delta_transfer": 0.1247,
    "delta_transfer_ci": (0.0888, 0.1596),
    "delta_same": 0.1769,
    "delta_same_ci": (0.1377, 0.2158),
    "blanked_teacher_turn_rate": 0.7108,
    "token_budget_truncated_rate": 0.3057,
    "eoc_rate": 0.6943,
    "thinking_tokens_avg_per_turn": 599.7,
}

T018_SMOKE_EMPTY_RATE = 0.7014

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
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] eduardo\.transfer_eval: "
    r"(?:Thinking mode (?:AUTO|FORCED):|Rollout of |Scoring (?:\d+|took ))"
)
_SAFE_ROUND_INFO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] eduardo\.transfer_eval: Round \d+: "
)
_RESULTS_WRITTEN_RE = re.compile(r"^Results written to: ")
_AUTHOR_SUMMARY_BLOCK_RE = re.compile(r"^={72}\nModel:\s+.*?^={72}$", re.M | re.S)

_TRACEBACK_FRAME_RE = re.compile(
    r'^(?:.*\]\s+)?\s*File "([^"]+)", line (\d+), in ([A-Za-z0-9_<>.-]+)\s*$'
)
_EXC_CLASS_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|SystemExit|KeyboardInterrupt|StopIteration))\s*(?::.*)?$"
)


def extract_safe_eval_log_lines(text: str, *, include_rounds: bool = False) -> list[str]:
    """Return only allowlisted eduardo.transfer_eval INFO lines and Results written line."""
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


def extract_safe_failure_summary(text: str, *, max_frames: int = 20) -> str:
    """Extract only stack frame locations (file, line, func) and exception class names."""
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def compute_hake_g(pres: list[float], posts: list[float]) -> float | None:
    if not pres or len(pres) != len(posts):
        return None
    mean_pre = statistics.fmean(pres)
    mean_post = statistics.fmean(posts)
    denom = 1.0 - mean_pre
    if abs(denom) <= _EPS:
        return None
    return (mean_post - mean_pre) / denom


def compute_cohens_d(pres: list[float], posts: list[float]) -> float | None:
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


def validate_results_doc(doc: Any) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
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
            "pre_solve_rate",
            "pre_solve_rate_transfer",
            "post_solve_transfer",
            "post_solve_same",
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

    summary["n_student_calls_failed"] = sum(int(r["n_student_calls_failed"]) for r in records)
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
    total_votes = 0
    retry_failed_votes = 0
    parse_failed_votes = 0
    regex_recovered_votes = 0
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
                if reason == _RETRY_FAIL_REASON:
                    retry_failed_votes += 1
                    conv_retry_fail = True
                    conv_any_fail = True
                elif reason == _PARSE_FAIL_REASON:
                    parse_failed_votes += 1
                    conv_any_fail = True
                elif reason == _REGEX_RECOVERED_REASON:
                    regex_recovered_votes += 1
        convs_with_retry_fail.append(conv_retry_fail)
        convs_with_any_fail.append(conv_any_fail)
        if rec["eval_judge_pedagogy"] is None:
            eval_pedagogy_missing += 1
        if rec["eval_judge_correctness"] is None:
            eval_correctness_missing += 1

    clean_retry_records = [r for r, bad in zip(records, convs_with_retry_fail) if not bad]
    retry_fail_rate = (retry_failed_votes / total_votes) if total_votes else 0.0
    parse_fail_rate = (parse_failed_votes / total_votes) if total_votes else 0.0

    return {
        "total_votes": total_votes,
        "retry_failed_votes": retry_failed_votes,
        "parse_failed_votes": parse_failed_votes,
        "regex_recovered_votes": regex_recovered_votes,
        "retry_fail_rate": retry_fail_rate,
        "parse_fail_rate": parse_fail_rate,
        "conversations_with_retry_fail": sum(convs_with_retry_fail),
        "conversations_with_any_fail": sum(convs_with_any_fail),
        "clean_retry_records": clean_retry_records,
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
    grouped: dict[int, list[float]] = {}
    for rec in records:
        val = rec.get(key)
        if val is not None:
            grouped.setdefault(rec["row_index"], []).append(float(val))
    return [statistics.fmean(grouped[idx]) for idx in sorted(grouped)]


def _problem_paired_values(records: list[dict[str, Any]], key_a: str, key_b: str) -> list[tuple[float, float]]:
    grouped_a: dict[int, list[float]] = {}
    grouped_b: dict[int, list[float]] = {}
    for rec in records:
        va = rec.get(key_a)
        vb = rec.get(key_b)
        if va is not None and vb is not None:
            grouped_a.setdefault(rec["row_index"], []).append(float(va))
            grouped_b.setdefault(rec["row_index"], []).append(float(vb))
    common_ids = sorted(set(grouped_a) & set(grouped_b))
    return [(statistics.fmean(grouped_a[idx]), statistics.fmean(grouped_b[idx])) for idx in common_ids]


def bootstrap_ci(values: list[float], seed: int = 42, draws: int = 10000) -> tuple[float, float, float, int]:
    if not values:
        raise ValueError("Cannot bootstrap empty value list")
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    n = len(values)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    return statistics.fmean(values), _percentile(samples, 0.025), _percentile(samples, 0.975), n


def bootstrap_paired_stat(
    pairs: list[tuple[float, float]],
    stat_fn: Any,
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    if not pairs:
        return {"mean": None, "ci": None, "n_problems": 0, "valid_draws": 0}
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    point = stat_fn([p[0] for p in pairs], [p[1] for p in pairs])
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


def intervals_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """Closed-interval overlap check with float precision tolerance."""
    return max(ci_a[0], ci_b[0]) <= min(ci_a[1], ci_b[1]) + _EPS


def judge_repro_verdict(
    delta_transfer: float,
    ci_transfer: tuple[float, float],
    delta_same: float,
    ci_same: tuple[float, float],
    *,
    paper_ref: dict[str, Any] = PAPER_EDUARDO_4B,
) -> tuple[str, dict[str, Any]]:
    """Apply T-020 Verdict A (or T-018) 3-tier decision rule on Δ_transfer and Δ_same."""
    diff_t = abs(delta_transfer - float(paper_ref["delta_transfer"]))
    diff_s = abs(delta_same - float(paper_ref["delta_same"]))
    overlap_t = intervals_overlap(ci_transfer, tuple(paper_ref["delta_transfer_ci"]))
    overlap_s = intervals_overlap(ci_same, tuple(paper_ref["delta_same_ci"]))

    if diff_t > 0.08 + _EPS or diff_s > 0.08 + _EPS or not overlap_t or not overlap_s:
        verdict = "对不上"
    elif diff_t <= 0.04 + _EPS and diff_s <= 0.04 + _EPS and overlap_t and overlap_s:
        verdict = "对上"
    else:
        verdict = "部分对上"

    return verdict, {
        "diff_transfer": diff_t,
        "diff_same": diff_s,
        "overlap_transfer": overlap_t,
        "overlap_same": overlap_s,
    }


def judge_paired_verdict(diff_ci_low: float, diff_ci_high: float) -> str:
    """Apply T-020 Verdict B rule on paired Δ_transfer CI."""
    if diff_ci_low > _EPS:
        return "A 组更好"
    if diff_ci_high < -_EPS:
        return "更差"
    return "看不出差别"


def summarize_single_run(
    doc: dict[str, Any],
    *,
    seed: int = 42,
    draws: int = 10000,
    paper_ref: dict[str, Any] = PAPER_BASE_4B,
) -> dict[str, Any]:
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

    verdict, verdict_details = judge_repro_verdict(
        dt_mean, (dt_lo, dt_hi), ds_mean, (ds_lo, ds_hi), paper_ref=paper_ref
    )

    return {
        "tag": str(doc.get("tag") or ""),
        "judge_model": str(doc.get("judge_model") or ""),
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
        "agnes_diagnostics": {
            "total_votes": judge_diag["total_votes"],
            "retry_failed_votes": judge_diag["retry_failed_votes"],
            "parse_failed_votes": judge_diag["parse_failed_votes"],
            "regex_recovered_votes": judge_diag["regex_recovered_votes"],
            "retry_fail_rate": judge_diag["retry_fail_rate"],
            "parse_fail_rate": judge_diag["parse_fail_rate"],
            "conversations_with_retry_fail": judge_diag["conversations_with_retry_fail"],
            "conversations_with_any_fail": judge_diag["conversations_with_any_fail"],
            "judge_available": judge_diag["judge_available"],
            "eval_pedagogy_missing": judge_diag["eval_pedagogy_missing"],
            "eval_correctness_missing": judge_diag["eval_correctness_missing"],
        },
        "verdict": verdict,
        "verdict_details": verdict_details,
    }


def _summarize_t018_table(
    t018_by_row: dict[int, dict[str, Any]],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """Compute base metrics from the sanitized A100 T-018 per-problem table."""
    sorted_rows = sorted(t018_by_row)
    pre_t: list[float] = []
    pre_s: list[float] = []
    post_t: list[float] = []
    post_s: list[float] = []
    ctl1_t: list[float] = []
    ctl1_s: list[float] = []
    dt_vals: list[float] = []
    ds_vals: list[float] = []
    dt_g_vals: list[float] = []
    ds_g_vals: list[float] = []
    gap_vals: list[float] = []
    leaks: list[float] = []
    teach_rejs: list[float] = []
    peds: list[float] = []
    facts: list[float] = []
    truncs: list[float] = []

    for ridx in sorted_rows:
        r = t018_by_row[ridx]
        pt = r["post_transfer_correct"] / r["post_transfer_total"]
        ps = r["post_same_correct"] / r["post_same_total"]
        prt = float(r["pre_solve_rate_transfer"])
        prs = float(r["pre_solve_rate"])
        c1t = r["ctl1_transfer_correct"] / r["ctl1_transfer_total"]
        c1s = r["ctl1_same_correct"] / r["ctl1_same_total"]

        dt = pg.normalized_gain(prt, pt)
        ds = pg.normalized_gain(prs, ps)
        lk = int(r["leaked"])

        pre_t.append(prt)
        pre_s.append(prs)
        post_t.append(pt)
        post_s.append(ps)
        ctl1_t.append(c1t)
        ctl1_s.append(c1s)
        dt_vals.append(dt)
        ds_vals.append(ds)
        dt_g_vals.append(0.0 if lk else dt)
        ds_g_vals.append(0.0 if lk else ds)
        gap_vals.append(ds - dt)
        leaks.append(float(lk))
        teach_rejs.append(float(r["teaching_rejected"]))
        if r["eval_judge_pedagogy"] is not None:
            peds.append(float(r["eval_judge_pedagogy"]))
        if r["eval_judge_correctness"] is not None:
            facts.append(float(r["eval_judge_correctness"]))
        truncs.append(float(r["is_truncated"]))

    dt_m, dt_lo, dt_hi, n_prob = bootstrap_ci(dt_vals, seed=seed, draws=draws)
    ds_m, ds_lo, ds_hi, _ = bootstrap_ci(ds_vals, seed=seed, draws=draws)
    dtg_m, dtg_lo, dtg_hi, _ = bootstrap_ci(dt_g_vals, seed=seed, draws=draws)
    dsg_m, dsg_lo, dsg_hi, _ = bootstrap_ci(ds_g_vals, seed=seed, draws=draws)
    gap_m, gap_lo, gap_hi, _ = bootstrap_ci(gap_vals, seed=seed, draws=draws)
    lk_m, lk_lo, lk_hi, _ = bootstrap_ci(leaks, seed=seed, draws=draws)

    r2_t = bootstrap_paired_stat(list(zip(pre_t, post_t)), compute_hake_g, seed=seed, draws=draws)
    r2_s = bootstrap_paired_stat(list(zip(pre_s, post_s)), compute_hake_g, seed=seed, draws=draws)
    r3_t = bootstrap_paired_stat(list(zip(pre_t, post_t)), compute_cohens_d, seed=seed, draws=draws)
    r3_s = bootstrap_paired_stat(list(zip(pre_s, post_s)), compute_cohens_d, seed=seed, draws=draws)

    return {
        "num_problems": n_prob,
        "pre_solve_rate_transfer": statistics.fmean(pre_t),
        "pre_solve_rate": statistics.fmean(pre_s),
        "post_solve_transfer": statistics.fmean(post_t),
        "post_solve_same": statistics.fmean(post_s),
        "ctl1_transfer": statistics.fmean(ctl1_t),
        "ctl1_same": statistics.fmean(ctl1_s),
        "delta_transfer": {"mean": dt_m, "ci": [dt_lo, dt_hi]},
        "delta_same": {"mean": ds_m, "ci": [ds_lo, ds_hi]},
        "delta_transfer_gated": {"mean": dtg_m, "ci": [dtg_lo, dtg_hi]},
        "delta_same_gated": {"mean": dsg_m, "ci": [dsg_lo, dsg_hi]},
        "transfer_gap": {"mean": gap_m, "ci": [gap_lo, gap_hi]},
        "r2_hake_g_transfer": r2_t,
        "r2_hake_g_same": r2_s,
        "r3_cohens_d_transfer": r3_t,
        "r3_cohens_d_same": r3_s,
        "leak_rate_author": {"mean": lk_m, "ci": [lk_lo, lk_hi]},
        "teaching_reject_rate": statistics.fmean(teach_rejs),
        "eval_judge_pedagogy": statistics.fmean(peds) if peds else 0.0,
        "eval_judge_correctness": statistics.fmean(facts) if facts else 0.0,
        "token_budget_truncated_rate": statistics.fmean(truncs),
        "blanked_teacher_turn_rate": A100_T018_BASE["blanked_teacher_turn_rate"],
        "eoc_rate": A100_T018_BASE["eoc_rate"],
        "thinking_tokens_avg_per_turn": A100_T018_BASE["thinking_tokens_avg_per_turn"],
    }


def _load_t018_csv_rows(csv_path: Path) -> dict[int, dict[str, Any]]:
    by_row: dict[int, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            ridx = int(raw["row_index"])
            ped_s = raw["eval_judge_pedagogy"].strip()
            fact_s = raw["eval_judge_correctness"].strip()
            by_row[ridx] = {
                "row_index": ridx,
                "pre_solve_rate": float(raw["pre_solve_rate"]),
                "pre_solve_rate_transfer": float(raw["pre_solve_rate_transfer"]),
                "post_same_correct": int(raw["post_same_correct"]),
                "post_same_total": int(raw["post_same_total"]),
                "post_transfer_correct": int(raw["post_transfer_correct"]),
                "post_transfer_total": int(raw["post_transfer_total"]),
                "ctl1_same_correct": int(raw["ctl1_same_correct"]),
                "ctl1_same_total": int(raw["ctl1_same_total"]),
                "ctl1_transfer_correct": int(raw["ctl1_transfer_correct"]),
                "ctl1_transfer_total": int(raw["ctl1_transfer_total"]),
                "leaked": int(raw["leaked"]),
                "teaching_rejected": int(raw["teaching_rejected"]),
                "eval_judge_pedagogy": float(ped_s) if ped_s != "" else None,
                "eval_judge_correctness": float(fact_s) if fact_s != "" else None,
                "has_blanked_turn": int(raw["has_blanked_turn"]),
                "is_truncated": int(raw["is_truncated"]),
            }
    if not by_row:
        raise ValueError("Empty T-018 table CSV")
    return by_row


def _paired_bootstrap_arrays(
    base_vals: list[float],
    target_vals: list[float],
    *,
    seed: int,
    draws: int,
    group_label: str,
) -> dict[str, float]:
    sub_seed = pg.derive_seed(seed, group_label, True, False)
    return pg._bootstrap_stats_from_arrays(
        np.array(base_vals, dtype=np.float64),
        np.array(target_vals, dtype=np.float64),
        seed=sub_seed,
        draws=draws,
    )


def _fmt_stat_ci(val: dict[str, Any] | None, *, pct: bool = False, signed: bool = True) -> str:
    if not val or val.get("mean") is None:
        return "N/A"
    mean_v = float(val["mean"])
    ci = val.get("ci")
    if pct:
        if ci is None:
            return f"{mean_v * 100:.2f}% [N/A]"
        return f"{mean_v * 100:.2f}% [{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]"
    fmt = "+.4f" if signed else ".4f"
    if ci is None:
        return f"{format(mean_v, fmt)} [N/A]"
    return f"{format(mean_v, fmt)} [{format(ci[0], fmt)}, {format(ci[1], fmt)}]"


def format_single_report(summary: dict[str, Any], *, paper_label: str = "Paper (base 4B)") -> str:
    m = summary["metrics"]
    diag = summary["agnes_diagnostics"]
    vd = summary["verdict_details"]
    judge_ok = diag["judge_available"]
    is_eduardo = "Eduardo" in paper_label
    paper = PAPER_EDUARDO_4B if is_eduardo else PAPER_BASE_4B

    leak_author_str = _fmt_stat_ci(m["leak_rate_author"], pct=True) if judge_ok else "不可用 (>1% 技术失败)"
    leak_sens_str = _fmt_stat_ci(m["leak_rate_excl_tech_fail"], pct=True) if judge_ok else "不可用 (>1% 技术失败)"
    teach_str = f"{m['teaching_reject_rate'] * 100:.2f}%" if judge_ok else "不可用 (>1% 技术失败)"
    ped_str = f"{m['eval_judge_pedagogy']:.2f}" if judge_ok else "不可用 (>1% 技术失败)"
    fact_str = f"{m['eval_judge_correctness']:.2f}" if judge_ok else "不可用 (>1% 技术失败)"

    lines = [
        "=" * 82,
        f"Pro 5000 Author-494 Summary (tag={summary['tag'] or 'N/A'}, judge={summary['judge_model'] or 'N/A'})",
        f"Conversations: {summary['num_conversations']}  Problems: {summary['num_problems']}  "
        f"Bootstrap: {summary['draws']}x (seed={summary['seed']})",
        f"assume_thinking={summary['assume_thinking']}  "
        f"transfer_tested_rate={summary['transfer_tested_rate']:.2%}  "
        f"student_calls_failed={summary['n_student_calls_failed']}",
        "-" * 82,
        "Agnes Gate Diagnostics (no_leak + teaching):",
        f"  Total votes:           {diag['total_votes']}",
        f"  Retry-failed votes:    {diag['retry_failed_votes']} ({diag['retry_fail_rate']:.2%})  "
        f"[convs={diag['conversations_with_retry_fail']}]",
        f"  Parse-failed votes:    {diag['parse_failed_votes']} ({diag['parse_fail_rate']:.2%})  "
        f"[regex_recovered={diag['regex_recovered_votes']}]",
        f"  Judge status (<=1%):   {'可用' if judge_ok else '不可用（技术失败票超过 1%，停下来报告）'}",
        f"  1-5 Eval judge missing: pedagogy={diag['eval_pedagogy_missing']}, "
        f"correctness={diag['eval_correctness_missing']} (eval_judge_ok={m['eval_judge_ok']:.2%})",
        "-" * 82,
        f"{'Metric':<30} | {'Ours (Pro 5000 + Agnes)':<30} | {paper_label:<22}",
        "-" * 82,
        f"{'Δ_transfer (primary)':<30} | {_fmt_stat_ci(m['delta_transfer']):<30} | "
        f"{paper['delta_transfer']:+.4f} [{paper['delta_transfer_ci'][0]:+.4f}, {paper['delta_transfer_ci'][1]:+.4f}]",
        f"{'Δ_same (primary)':<30} | {_fmt_stat_ci(m['delta_same']):<30} | "
        f"{paper['delta_same']:+.4f} [{paper['delta_same_ci'][0]:+.4f}, {paper['delta_same_ci'][1]:+.4f}]",
        f"{'Δ_transfer (gated)':<30} | {_fmt_stat_ci(m['delta_transfer_gated']):<30} | —",
        f"{'Δ_same (gated)':<30} | {_fmt_stat_ci(m['delta_same_gated']):<30} | —",
        f"{'Transfer gap (same-transfer)':<30} | {_fmt_stat_ci(m['transfer_gap']):<30} | —",
        f"{'R2 Hake g (transfer)':<30} | {_fmt_stat_ci(m['r2_hake_g_transfer']):<30} | —",
        f"{'R2 Hake g (same)':<30} | {_fmt_stat_ci(m['r2_hake_g_same']):<30} | —",
        f"{'R3 Cohen d (transfer)':<30} | {_fmt_stat_ci(m['r3_cohens_d_transfer']):<30} | —",
        f"{'R3 Cohen d (same)':<30} | {_fmt_stat_ci(m['r3_cohens_d_same']):<30} | —",
        f"{'Leak rate (author)':<30} | {leak_author_str:<30} | "
        f"{paper['leak_rate']*100:.2f}% [{paper['leak_rate_ci'][0]*100:.2f}%, {paper['leak_rate_ci'][1]*100:.2f}%]",
        f"{'Leak rate (excl tech-fail)':<30} | {leak_sens_str:<30} | —",
        f"{'Teaching reject rate':<30} | {teach_str:<30} | —",
        f"{'Ped. (1-5)':<30} | {ped_str:<30} | {paper['eval_judge_pedagogy']:.2f}",
        f"{'Fact. (1-5)':<30} | {fact_str:<30} | {paper['eval_judge_correctness']:.2f}",
        f"{'Empty% (blanked turns)':<30} | {m['blanked_teacher_turn_rate'] * 100:>6.2f}%{'':<23} | "
        f"{paper['blanked_teacher_turn_rate']*100:.2f}% (T-018 smoke: {T018_SMOKE_EMPTY_RATE*100:.2f}%)",
        f"{'Seq. trunc.%':<30} | {m['token_budget_truncated_rate'] * 100:>6.2f}%{'':<23} | "
        f"{paper['token_budget_truncated_rate']*100:.2f}%",
        f"{'EoC rate':<30} | {m['eoc_rate'] * 100:>6.2f}%{'':<23} | {paper['eoc_rate']*100:.2f}%",
        f"{'Thinking tok/turn':<30} | {m['thinking_tokens_avg_per_turn']:>6.1f}{'':<24} | "
        f"{paper['thinking_tokens_avg_per_turn']:.1f}",
        "-" * 82,
        f"判定 (仅看主口径 Δ_transfer & Δ_same): {summary['verdict']} "
        f"(|diff_t|={vd['diff_transfer']:.4f}, overlap_t={vd['overlap_transfer']}; "
        f"|diff_s|={vd['diff_same']:.4f}, overlap_s={vd['overlap_same']})",
        "=" * 82,
    ]
    return "\n".join(lines)


def build_full_comparison(
    *,
    a_group_doc: dict[str, Any],
    t018_by_row: dict[int, dict[str, Any]],
    paired_csv_path: Path,
    seed: int = 42,
    draws: int = 10000,
    compare_stack_res: dict[str, Any] | None = None,
    pro5000_ctl_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build T-020 Verdicts A, B, stack-check table, and secondary metrics."""
    ag_sum = summarize_single_run(a_group_doc, seed=seed, draws=draws, paper_ref=PAPER_EDUARDO_4B)
    base_sum = _summarize_t018_table(t018_by_row, seed=seed, draws=draws)

    ag_m = ag_sum["metrics"]
    verdict_a, details_a = judge_repro_verdict(
        ag_m["delta_transfer"]["mean"],
        tuple(ag_m["delta_transfer"]["ci"]),
        ag_m["delta_same"]["mean"],
        tuple(ag_m["delta_same"]["ci"]),
        paper_ref=PAPER_EDUARDO_4B,
    )

    csv_rows = pg.load_csv(paired_csv_path)
    paired_report = pg.evaluate_dataset(
        rows=csv_rows,
        base_condition="base",
        target_condition="a_group",
        draws=draws,
        seed=seed,
        include_high_pre_filtered=False,
    )
    by_group = {
        r["group"]: r
        for r in paired_report["results"]
        if r["unknown_policy"] == "unknown_as_wrong" and r["source_filter"] == "all"
    }
    for req_grp in ("author494_transfer", "author494_same"):
        if req_grp not in by_group or by_group[req_grp]["stats"] is None:
            raise ValueError(f"Paired CSV missing valid group '{req_grp}'")
    tr_stats = by_group["author494_transfer"]["stats"]
    sm_stats = by_group["author494_same"]["stats"]
    verdict_b = judge_paired_verdict(tr_stats["diff_ci_low"], tr_stats["diff_ci_high"])

    # Secondary paired comparisons:
    # 1) Rebaselined R1 using A100 ctl1 as common baseline, and raw post - pre / post - ctl1 mean diffs.
    ag_by_row: dict[int, list[dict[str, Any]]] = {}
    for rec in a_group_doc["conversations"]:
        ag_by_row.setdefault(rec["row_index"], []).append(rec)

    sorted_rows = sorted(t018_by_row)
    if set(ag_by_row) != set(t018_by_row):
        raise ValueError("row_index sets differ between A-group JSON and T-018 table")

    pro_ctl_by_row: dict[int, dict[str, Any]] | None = None
    if pro5000_ctl_doc is not None:
        pro_ctl_by_row = {int(r["row_index"]): r for r in pro5000_ctl_doc.get("records", [])}
        if set(pro_ctl_by_row) != set(t018_by_row):
            raise ValueError("row_index sets differ between Pro 5000 ctl JSON and T-018 table")

    base_ctl1_r1_t: list[float] = []
    base_ctl1_r1_s: list[float] = []
    ag_ctl1_r1_t: list[float] = []
    ag_ctl1_r1_s: list[float] = []
    base_raw_pre_t: list[float] = []
    base_raw_pre_s: list[float] = []
    ag_raw_pre_t: list[float] = []
    ag_raw_pre_s: list[float] = []
    base_raw_ctl1_t: list[float] = []
    base_raw_ctl1_s: list[float] = []
    ag_raw_ctl1_t: list[float] = []
    ag_raw_ctl1_s: list[float] = []

    ag_own_ctl_r1_t: list[float] = []
    ag_own_ctl_r1_s: list[float] = []

    for ridx in sorted_rows:
        t_rec = t018_by_row[ridx]
        c1_t = t_rec["ctl1_transfer_correct"] / t_rec["ctl1_transfer_total"]
        c1_s = t_rec["ctl1_same_correct"] / t_rec["ctl1_same_total"]
        b_post_t = t_rec["post_transfer_correct"] / t_rec["post_transfer_total"]
        b_post_s = t_rec["post_same_correct"] / t_rec["post_same_total"]
        a_post_t = statistics.fmean(float(r["post_solve_transfer"]) for r in ag_by_row[ridx])
        a_post_s = statistics.fmean(float(r["post_solve_same"]) for r in ag_by_row[ridx])
        pre_t = float(t_rec["pre_solve_rate_transfer"])
        pre_s = float(t_rec["pre_solve_rate"])

        base_ctl1_r1_t.append(pg.normalized_gain(c1_t, b_post_t))
        base_ctl1_r1_s.append(pg.normalized_gain(c1_s, b_post_s))
        ag_ctl1_r1_t.append(pg.normalized_gain(c1_t, a_post_t))
        ag_ctl1_r1_s.append(pg.normalized_gain(c1_s, a_post_s))

        base_raw_pre_t.append(b_post_t - pre_t)
        base_raw_pre_s.append(b_post_s - pre_s)
        ag_raw_pre_t.append(a_post_t - pre_t)
        ag_raw_pre_s.append(a_post_s - pre_s)

        base_raw_ctl1_t.append(b_post_t - c1_t)
        base_raw_ctl1_s.append(b_post_s - c1_s)
        ag_raw_ctl1_t.append(a_post_t - c1_t)
        ag_raw_ctl1_s.append(a_post_s - c1_s)

        if pro_ctl_by_row is not None:
            pc = pro_ctl_by_row[ridx]
            ag_own_ctl_r1_t.append(pg.normalized_gain(float(pc["ctl_transfer"]), a_post_t))
            ag_own_ctl_r1_s.append(pg.normalized_gain(float(pc["ctl_same"]), a_post_s))

    secondary_paired = {
        "ctl1_rebaselined_r1_transfer": _paired_bootstrap_arrays(
            base_ctl1_r1_t, ag_ctl1_r1_t, seed=seed, draws=draws, group_label="ctl1_r1_transfer"
        ),
        "ctl1_rebaselined_r1_same": _paired_bootstrap_arrays(
            base_ctl1_r1_s, ag_ctl1_r1_s, seed=seed, draws=draws, group_label="ctl1_r1_same"
        ),
        "raw_post_minus_pre_transfer": _paired_bootstrap_arrays(
            base_raw_pre_t, ag_raw_pre_t, seed=seed, draws=draws, group_label="raw_pre_transfer"
        ),
        "raw_post_minus_pre_same": _paired_bootstrap_arrays(
            base_raw_pre_s, ag_raw_pre_s, seed=seed, draws=draws, group_label="raw_pre_same"
        ),
        "raw_post_minus_ctl1_transfer": _paired_bootstrap_arrays(
            base_raw_ctl1_t, ag_raw_ctl1_t, seed=seed, draws=draws, group_label="raw_ctl1_transfer"
        ),
        "raw_post_minus_ctl1_same": _paired_bootstrap_arrays(
            base_raw_ctl1_s, ag_raw_ctl1_s, seed=seed, draws=draws, group_label="raw_ctl1_same"
        ),
    }
    if pro_ctl_by_row is not None:
        secondary_paired["own_ctl_r1_transfer"] = _paired_bootstrap_arrays(
            base_ctl1_r1_t, ag_own_ctl_r1_t, seed=seed, draws=draws, group_label="own_ctl_r1_transfer"
        )
        secondary_paired["own_ctl_r1_same"] = _paired_bootstrap_arrays(
            base_ctl1_r1_s, ag_own_ctl_r1_s, seed=seed, draws=draws, group_label="own_ctl_r1_same"
        )

    return {
        "seed": seed,
        "draws": draws,
        "a_group": ag_sum,
        "base_a100_t018": base_sum,
        "verdict_a": {
            "verdict": verdict_a,
            "details": details_a,
            "paper_eduardo4b": PAPER_EDUARDO_4B,
        },
        "verdict_b": {
            "verdict": verdict_b,
            "author494_transfer": {
                "n_paired": by_group["author494_transfer"]["n_paired"],
                "n_unpaired_total": by_group["author494_transfer"]["n_unpaired_total"],
                **tr_stats,
                "paper_ref_diff": PAPER_EDUARDO_4B["delta_transfer"] - PAPER_BASE_4B["delta_transfer"],
            },
            "author494_same": {
                "n_paired": by_group["author494_same"]["n_paired"],
                "n_unpaired_total": by_group["author494_same"]["n_unpaired_total"],
                **sm_stats,
                "paper_ref_diff": PAPER_EDUARDO_4B["delta_same"] - PAPER_BASE_4B["delta_same"],
            },
        },
        "compare_stack": compare_stack_res,
        "secondary_paired": secondary_paired,
    }


def format_full_comparison(comp: dict[str, Any]) -> str:
    ag = comp["a_group"]
    bs = comp["base_a100_t018"]
    ag_m = ag["metrics"]
    va = comp["verdict_a"]
    vb = comp["verdict_b"]
    sp = comp["secondary_paired"]

    def _p_ci(st: dict[str, Any], prefix: str) -> str:
        return (
            f"{st[f'{prefix}_mean']:+.4f} "
            f"[{st[f'{prefix}_ci_low']:+.4f}, {st[f'{prefix}_ci_high']:+.4f}]"
        )

    lines = [
        "=" * 100,
        f"T-020 Pro 5000 A-Group vs A100 T-018 Base Report (Bootstrap {comp['draws']}x, seed={comp['seed']})",
        f"A-group (Pro 5000): tag={ag['tag']} n={ag['num_problems']} failed={ag['n_student_calls_failed']} "
        f"assume_thinking={ag['assume_thinking']} transfer_tested={ag['transfer_tested_rate']:.2%}",
        f"Base (A100 T-018):  n={bs['num_problems']} (from t018_per_problem.csv)",
        "",
        "### 表 A：复现判定（A 组第 100 步 vs 论文 Eduardo-4B）",
        f"{'指标':<20} | {'A 组 (Pro 5000)':<28} | {'论文 Eduardo-4B':<24} | {'差值 / CI 重叠'}",
        "-" * 100,
        f"{'Δ_transfer (主口径)':<20} | {_fmt_stat_ci(ag_m['delta_transfer']):<28} | "
        f"+0.2500 [+0.2100, +0.2900] | diff={va['details']['diff_transfer']:.4f} overlap={va['details']['overlap_transfer']}",
        f"{'Δ_same (主口径)':<20} | {_fmt_stat_ci(ag_m['delta_same']):<28} | "
        f"+0.3700 [+0.3300, +0.4200] | diff={va['details']['diff_same']:.4f} overlap={va['details']['overlap_same']}",
        f"判定 A 结论: {va['verdict']}",
        "",
        "### 表 B：训练效果（A 组 Pro 5000 减 base A100 T-018，逐题配对 bootstrap）",
        f"{'题集':<20} | {'n_paired':>8} | {'base (A100) 增益':<24} | {'A 组 (Pro5000) 增益':<24} | {'配对差 (A组-base) [95% CI]'} | {'P(diff>0)'} | {'论文参考差'}",
        "-" * 118,
    ]
    for grp in ("author494_transfer", "author494_same"):
        st = vb[grp]
        lines.append(
            f"{grp:<20} | {st['n_paired']:>8} | {_p_ci(st, 'base'):<24} | {_p_ci(st, 'target'):<24} | "
            f"{_p_ci(st, 'diff'):<27} | {st['prop_diff_gt_zero']:.4f}    | {st['paper_ref_diff']:+.2f}"
        )
    lines.append(f"判定 B 结论 (Δ_transfer 配对差): {vb['verdict']}")
    lines.append("")

    cs = comp.get("compare_stack")
    lines.append("### 换机器检查（Pro 5000 vs A100 T-018）")
    if cs is None:
        lines.append("  未传入 --compare-stack-json（请参照阶段 1 换机器检查结果）")
    else:
        st_s = cs["student_side"]
        tt_s = cs["tutor_side"]
        lines.extend([
            f"  总结论: {cs['verdict']}",
            f"  学生侧 ({'通过' if st_s['passed'] else '不通过'}): n={st_s['n_problems']} failed={st_s['calls_failed']} | "
            f"orig_mean Pro5000={st_s['pro5000_ctl_same_mean']:.4f} vs A100={st_s['a100_ctl1_same_mean']:.4f} (|diff|={st_s['abs_diff_same']:.4f}, r={st_s['pearson_r_same']:.4f}) | "
            f"trans_mean Pro5000={st_s['pro5000_ctl_transfer_mean']:.4f} vs A100={st_s['a100_ctl1_transfer_mean']:.4f} (|diff|={st_s['abs_diff_transfer']:.4f}, r={st_s['pearson_r_transfer']:.4f})",
            f"  Tutor 侧 ({'通过' if tt_s['passed'] else '不通过'}): n={tt_s['n_problems']} assume_thinking={tt_s['assume_thinking']} | "
            f"Empty%={tt_s['blanked_teacher_turn_rate']*100:.2f}% vs T-018={tt_s['t018_smoke_empty_rate']*100:.2f}% (|diff|={tt_s['abs_diff_empty_rate']*100:.2f}pp) | failed={tt_s['student_calls_failed']}",
        ])
    lines.append("")

    ag_diag = ag["agnes_diagnostics"]
    ag_ok = ag_diag["judge_available"]
    ag_leak_str = _fmt_stat_ci(ag_m["leak_rate_author"], pct=True) if ag_ok else "不可用"
    bs_leak_str = _fmt_stat_ci(bs["leak_rate_author"], pct=True)
    ag_leak_sens = _fmt_stat_ci(ag_m["leak_rate_excl_tech_fail"], pct=True) if ag_ok else "不可用"
    ag_teach_str = f"{ag_m['teaching_reject_rate'] * 100:.2f}%" if ag_ok else "不可用"
    bs_teach_str = f"{bs['teaching_reject_rate'] * 100:.2f}%"
    ag_pf_str = f"{ag_m['eval_judge_pedagogy']:.2f} / {ag_m['eval_judge_correctness']:.2f}" if ag_ok else "不可用"
    bs_pf_str = f"{bs['eval_judge_pedagogy']:.2f} / {bs['eval_judge_correctness']:.2f}"

    lines.extend([
        "### 次要指标与诊断并排表",
        f"{'指标':<28} | {'A 组 (Pro 5000)':<24} | {'base (A100 T-018)':<24} | {'配对差 / 论文参考'}",
        "-" * 108,
        f"{'pre / post (transfer)':<28} | {ag_m['pre_solve_rate_transfer']:.3f} / {ag_m['post_solve_transfer']:.3f}{'':<11} | "
        f"{bs['pre_solve_rate_transfer']:.3f} / {bs['post_solve_transfer']:.3f}{'':<11} | raw_diff={_p_ci(sp['raw_post_minus_pre_transfer'], 'diff')}",
        f"{'pre / post (same)':<28} | {ag_m['pre_solve_rate']:.3f} / {ag_m['post_solve_same']:.3f}{'':<11} | "
        f"{bs['pre_solve_rate']:.3f} / {bs['post_solve_same']:.3f}{'':<11} | raw_diff={_p_ci(sp['raw_post_minus_pre_same'], 'diff')}",
        f"{'R1 (ctl1 -> post) transfer':<28} | {_p_ci(sp['ctl1_rebaselined_r1_transfer'], 'target'):<24} | "
        f"{_p_ci(sp['ctl1_rebaselined_r1_transfer'], 'base'):<24} | diff={_p_ci(sp['ctl1_rebaselined_r1_transfer'], 'diff')}",
        f"{'R1 (ctl1 -> post) same':<28} | {_p_ci(sp['ctl1_rebaselined_r1_same'], 'target'):<24} | "
        f"{_p_ci(sp['ctl1_rebaselined_r1_same'], 'base'):<24} | diff={_p_ci(sp['ctl1_rebaselined_r1_same'], 'diff')}",
        f"{'raw (post - ctl1) transfer':<28} | {sp['raw_post_minus_ctl1_transfer']['target_mean']:+.4f}{'':<17} | "
        f"{sp['raw_post_minus_ctl1_transfer']['base_mean']:+.4f}{'':<17} | diff={_p_ci(sp['raw_post_minus_ctl1_transfer'], 'diff')}",
        f"{'raw (post - ctl1) same':<28} | {sp['raw_post_minus_ctl1_same']['target_mean']:+.4f}{'':<17} | "
        f"{sp['raw_post_minus_ctl1_same']['base_mean']:+.4f}{'':<17} | diff={_p_ci(sp['raw_post_minus_ctl1_same'], 'diff')}",
    ])
    if "own_ctl_r1_transfer" in sp:
        lines.extend([
            f"{'R1 (own ctl->post) transfer':<28} | {_p_ci(sp['own_ctl_r1_transfer'], 'target'):<24} | "
            f"{_p_ci(sp['own_ctl_r1_transfer'], 'base'):<24} | diff={_p_ci(sp['own_ctl_r1_transfer'], 'diff')}",
            f"{'R1 (own ctl->post) same':<28} | {_p_ci(sp['own_ctl_r1_same'], 'target'):<24} | "
            f"{_p_ci(sp['own_ctl_r1_same'], 'base'):<24} | diff={_p_ci(sp['own_ctl_r1_same'], 'diff')}",
        ])
    lines.extend([
        f"{'Δ_transfer (gated)':<28} | {_fmt_stat_ci(ag_m['delta_transfer_gated']):<24} | "
        f"{_fmt_stat_ci(bs['delta_transfer_gated']):<24} | —",
        f"{'Δ_same (gated)':<28} | {_fmt_stat_ci(ag_m['delta_same_gated']):<24} | "
        f"{_fmt_stat_ci(bs['delta_same_gated']):<24} | —",
        f"{'R2 Hake g (transfer)':<28} | {_fmt_stat_ci(ag_m['r2_hake_g_transfer']):<24} | "
        f"{_fmt_stat_ci(bs['r2_hake_g_transfer']):<24} | —",
        f"{'R2 Hake g (same)':<28} | {_fmt_stat_ci(ag_m['r2_hake_g_same']):<24} | "
        f"{_fmt_stat_ci(bs['r2_hake_g_same']):<24} | —",
        f"{'R3 Cohen d (transfer)':<28} | {_fmt_stat_ci(ag_m['r3_cohens_d_transfer']):<24} | "
        f"{_fmt_stat_ci(bs['r3_cohens_d_transfer']):<24} | —",
        f"{'R3 Cohen d (same)':<28} | {_fmt_stat_ci(ag_m['r3_cohens_d_same']):<24} | "
        f"{_fmt_stat_ci(bs['r3_cohens_d_same']):<24} | —",
        f"{'Transfer gap (same-trans)':<28} | {_fmt_stat_ci(ag_m['transfer_gap']):<24} | "
        f"{_fmt_stat_ci(bs['transfer_gap']):<24} | —",
        f"{'Leak rate (Agnes author)':<28} | {ag_leak_str:<24} | {bs_leak_str:<24} | 论文 30.6% / 16.8% (仅内部比)",
        f"{'Leak rate (excl tech)':<28} | {ag_leak_sens:<24} | 6.68% [4.66%, 8.91%]     | —",
        f"{'Teaching reject (Agnes)':<28} | {ag_teach_str:<24} | {bs_teach_str:<24} | 仅内部比",
        f"{'Ped. / Fact. (Agnes 1-5)':<28} | {ag_pf_str:<24} | {bs_pf_str:<24} | 论文 3.66/4.19 vs 2.26/2.88 (仅内部比)",
        f"{'Empty% (blanked turns)':<28} | {ag_m['blanked_teacher_turn_rate']*100:>6.2f}%{'':<17} | "
        f"{bs['blanked_teacher_turn_rate']*100:>6.2f}%{'':<17} | 论文 3.7% / 70.0%",
        f"{'Seq. trunc.%':<28} | {ag_m['token_budget_truncated_rate']*100:>6.2f}%{'':<17} | "
        f"{bs['token_budget_truncated_rate']*100:>6.2f}%{'':<17} | 论文 10.7% / 31.2%",
        f"{'EoC rate':<28} | {ag_m['eoc_rate']*100:>6.2f}%{'':<17} | "
        f"{bs['eoc_rate']*100:>6.2f}%{'':<17} | 论文 69.6% / 68.8%",
        f"{'Thinking tok/turn':<28} | {ag_m['thinking_tokens_avg_per_turn']:>6.1f}{'':<18} | "
        f"{bs['thinking_tokens_avg_per_turn']:>6.1f}{'':<18} | 论文 544 / 596",
        "",
        "Agnes 诊断 (A 组 Pro 5000):",
        f"  votes={ag_diag['total_votes']} retry_fail={ag_diag['retry_failed_votes']} "
        f"parse_fail={ag_diag['parse_failed_votes']} regex_rec={ag_diag['regex_recovered_votes']} "
        f"eval_ok={ag_m['eval_judge_ok']:.2%} status={'可用' if ag_ok else '不可用'}",
        "=" * 100,
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Pro 5000 author-494 eval JSON(s) safely.")
    ap.add_argument("result_json", nargs="?", type=Path, default=None, help="Single transfer_eval_*.json")
    ap.add_argument("--paper-baseline", choices=("base", "eduardo4b"), default="base")
    ap.add_argument("--a-group-json", type=Path, default=None, help="Full A-group transfer_eval_*.json")
    ap.add_argument("--t018-table", type=Path, default=None, help="A100 T-018 t018_per_problem.csv")
    ap.add_argument("--paired-csv", type=Path, default=None, help="Paired CSV generated by make_paired_csv.py")
    ap.add_argument("--compare-stack-json", type=Path, default=None, help="Optional compare_stack.py JSON output")
    ap.add_argument("--pro5000-ctl-json", type=Path, default=None, help="Optional Pro 5000 ctl JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    try:
        if args.a_group_json or args.t018_table or args.paired_csv:
            if not (args.a_group_json and args.t018_table and args.paired_csv):
                raise ValueError("--a-group-json, --t018-table, and --paired-csv must all be provided together")
            ag_doc = json.loads(args.a_group_json.read_text(encoding="utf-8"))
            t018_rows = _load_t018_csv_rows(args.t018_table)
            cs_res = (
                json.loads(args.compare_stack_json.read_text(encoding="utf-8"))
                if args.compare_stack_json is not None
                else None
            )
            ctl_doc = (
                json.loads(args.pro5000_ctl_json.read_text(encoding="utf-8"))
                if args.pro5000_ctl_json is not None
                else None
            )
            out = build_full_comparison(
                a_group_doc=ag_doc,
                t018_by_row=t018_rows,
                paired_csv_path=args.paired_csv,
                seed=args.seed,
                draws=args.draws,
                compare_stack_res=cs_res,
                pro5000_ctl_doc=ctl_doc,
            )
            print(format_full_comparison(out))
        elif args.result_json is not None:
            doc = json.loads(args.result_json.read_text(encoding="utf-8"))
            ref = PAPER_EDUARDO_4B if args.paper_baseline == "eduardo4b" else PAPER_BASE_4B
            label = "Paper (Eduardo-4B)" if args.paper_baseline == "eduardo4b" else "Paper (base 4B)"
            out = summarize_single_run(doc, seed=args.seed, draws=args.draws, paper_ref=ref)
            print(format_single_report(out, paper_label=label))
        else:
            ap.error("Provide either <result_json> OR (--a-group-json, --t018-table, --paired-csv)")
            return 2

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
