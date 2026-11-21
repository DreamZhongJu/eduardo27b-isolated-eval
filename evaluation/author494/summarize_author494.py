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
) -> tuple[str, dict[str, Any]]:
    """Apply the pre-registered 3-tier decision rule on Δ_transfer and Δ_same."""
    diff_t = abs(delta_transfer - PAPER_BASE["delta_transfer"])
    diff_s = abs(delta_same - PAPER_BASE["delta_same"])
    overlap_t = intervals_overlap(ci_transfer, PAPER_BASE["delta_transfer_ci"])
    overlap_s = intervals_overlap(ci_same, PAPER_BASE["delta_same_ci"])

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


def build_summary(doc: dict[str, Any], seed: int = 42, draws: int = 10000) -> dict[str, Any]:
    """Validate, recompute, bootstrap, and assemble a completely sanitized summary dict."""
    recorded_metrics, records, judge_votes = validate_results_doc(doc)
    recomputed = recompute_author_summary(records)
    verify_author_summary_match(recomputed, recorded_metrics)

    judge_diag = analyze_judge_failures(records, expected_votes=judge_votes)

    dt_mean, dt_lo, dt_hi, n_prob = bootstrap_ci(problem_values(records, "delta_transfer"), seed=seed, draws=draws)
    ds_mean, ds_lo, ds_hi, _ = bootstrap_ci(problem_values(records, "delta_same"), seed=seed, draws=draws)
    dt_g_mean, dt_g_lo, dt_g_hi, _ = bootstrap_ci(problem_values(records, "delta_transfer_gated"), seed=seed, draws=draws)
    ds_g_mean, ds_g_lo, ds_g_hi, _ = bootstrap_ci(problem_values(records, "delta_same_gated"), seed=seed, draws=draws)
    leak_mean, leak_lo, leak_hi, _ = bootstrap_ci(problem_values(records, "leaked"), seed=seed, draws=draws)

    clean_retry_records = judge_diag["clean_retry_records"]
    if clean_retry_records:
        leak_sens_mean, leak_sens_lo, leak_sens_hi, leak_sens_n = bootstrap_ci(
            problem_values(clean_retry_records, "leaked"), seed=seed, draws=draws
        )
    else:
        leak_sens_mean, leak_sens_lo, leak_sens_hi, leak_sens_n = None, None, None, 0

    verdict, verdict_details = judge_alignment_verdict(
        dt_mean, (dt_lo, dt_hi), ds_mean, (ds_lo, ds_hi)
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
        },
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


def format_report(summary: dict[str, Any]) -> str:
    """Format a human-readable aggregate summary table (no sensitive fields)."""
    m = summary["metrics"]
    diag = summary["agnes_diagnostics"]
    vd = summary["verdict_details"]
    judge_ok = diag["judge_available"]

    def fmt_ci(val: dict[str, Any], pct: bool = False) -> str:
        if val["mean"] is None or val["ci"] is None:
            return "N/A"
        if pct:
            return f"{val['mean']*100:.2f}% [{val['ci'][0]*100:.2f}%, {val['ci'][1]*100:.2f}%]"
        return f"{val['mean']:+.4f} [{val['ci'][0]:+.4f}, {val['ci'][1]:+.4f}]"

    leak_author_str = fmt_ci(m["leak_rate_author"], pct=True) if judge_ok else "不可用 (>1% 技术失败)"
    if judge_ok:
        leak_sens_str = fmt_ci(m["leak_rate_excl_tech_fail"], pct=True)
    else:
        leak_sens_str = "不可用 (>1% 技术失败)"
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
        "-" * 78,
        f"{'Metric':<30} | {'Ours (A100 + Agnes)':<30} | {'Paper (base 4B)':<22}",
        "-" * 78,
        f"{'Δ_transfer (primary)':<30} | {fmt_ci(m['delta_transfer']):<30} | +0.1000 [+0.0600, +0.1300]",
        f"{'Δ_same (primary)':<30} | {fmt_ci(m['delta_same']):<30} | +0.1100 [+0.0800, +0.1500]",
        f"{'Δ_transfer (gated)':<30} | {fmt_ci(m['delta_transfer_gated']):<30} | —",
        f"{'Δ_same (gated)':<30} | {fmt_ci(m['delta_same_gated']):<30} | —",
        f"{'Leak rate (author)':<30} | {leak_author_str:<30} | 16.80% [13.60%, 20.20%]",
        f"{'Leak rate (excl tech-fail)':<30} | {leak_sens_str:<30} | —",
        f"{'Teaching reject rate':<30} | {teach_str:<30} | —",
        f"{'Ped. (1-5)':<30} | {ped_str:<30} | 2.26",
        f"{'Fact. (1-5)':<30} | {fact_str:<30} | 2.88",
        f"{'Empty% (blanked turns)':<30} | {m['blanked_teacher_turn_rate']*100:>6.2f}%{'':<23} | 70.00%",
        f"{'Seq. trunc.%':<30} | {m['token_budget_truncated_rate']*100:>6.2f}%{'':<23} | 31.20%",
        f"{'EoC rate':<30} | {m['eoc_rate']*100:>6.2f}%{'':<23} | 68.80%",
        f"{'Thinking tok/turn':<30} | {m['thinking_tokens_avg_per_turn']:>6.1f}{'':<24} | 596.0",
        "-" * 78,
        f"判定结论 (仅看主口径 Δ_transfer & Δ_same): {summary['verdict']}",
        f"  |Δ_transfer - 0.10| = {vd['diff_transfer']:.4f}, CI重叠={vd['overlap_transfer']}",
        f"  |Δ_same - 0.11|     = {vd['diff_same']:.4f}, CI重叠={vd['overlap_same']}",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Eduardo author-494 eval JSON safely.")
    ap.add_argument("result_json", type=Path, help="Path to transfer_eval_*.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path to write aggregate summary JSON")
    args = ap.parse_args()

    try:
        doc = json.loads(args.result_json.read_text(encoding="utf-8"))
        summary = build_summary(doc, seed=args.seed, draws=args.draws)
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
