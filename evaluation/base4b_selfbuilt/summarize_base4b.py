#!/usr/bin/env python3
"""Sanitized aggregate summary, diagnostics, and paired bootstrap for base-4B self-built eval.

Never emits source IDs, problem text, reference answers, student responses, raw
WARNING/ERROR lines, or unquoted exception messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ref27b_manifest import (
    DIFFICULTY_ORDER,
    GROUP_BY_DIFFICULTY,
    _source_key,
    build_manifest,
)

STAGE_BY_DIFF = {"easy": "EASY", "medium": "MEDIUM", "hard": "HARD"}

_TRACEBACK_FRAME_RE = re.compile(r'^\s+File ".*", line \d+, in ')
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|SystemExit)\b")
_SAFE_STATUS_RE = re.compile(r"^\[base4b(?:_eval)?\]\s+")


def extract_safe_log_lines(text: str) -> list[str]:
    """Return only allowlisted [base4b] status lines; drop WARNING/ERROR and raw payloads."""
    out: list[str] = []
    for line in text.splitlines():
        if "[WARNING]" in line or "[ERROR]" in line:
            continue
        if _SAFE_STATUS_RE.match(line):
            out.append(line)
    return out


def extract_safe_failure_summary(text: str, *, max_frames: int = 20) -> str:
    """Extract only traceback File/line/function frames and exception class names.

    Never emits WARNING lines, code lines, or exception messages (even unquoted).
    """
    frames: list[str] = []
    exc_classes: list[str] = []
    for line in text.splitlines():
        if "[WARNING]" in line or "[ERROR]" in line:
            continue
        if _TRACEBACK_FRAME_RE.match(line):
            frames.append(line.strip())
        else:
            m = _EXC_LINE_RE.match(line)
            if m:
                exc_classes.append(m.group(0))
    return "\n".join(frames[-max_frames:] + exc_classes[-5:])


def normalized_gain(pre: float, post: float) -> float:
    if not (0.0 <= pre <= 1.0 and 0.0 <= post <= 1.0):
        raise ValueError("Probabilities must be in [0, 1]")
    return (post - pre) / (1.0 - pre) if post > pre else post - pre


def p_value(counts: dict[str, int], unknown_as_wrong: bool) -> float | None:
    correct = int(counts.get("correct", 0))
    incorrect = int(counts.get("incorrect", 0))
    unknown = int(counts.get("unknown", 0))
    if correct < 0 or incorrect < 0 or unknown < 0 or (correct + incorrect + unknown) != 16:
        raise ValueError("Sample counts must sum to 16")
    den = 16 if unknown_as_wrong else (correct + incorrect)
    return (correct / den) if den > 0 else None


def _percentile(sorted_xs: list[float], q: float) -> float:
    rank = q * (len(sorted_xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (rank - lo)


def _derived_seed(base_seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_mean_ci(
    values: list[float], *, seed: int, draws: int = 10000
) -> dict[str, Any] | None:
    if not values:
        return None
    if draws <= 0:
        raise ValueError("draws must be positive")
    n = len(values)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    return {
        "mean": sum(values) / n,
        "ci": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "n": n,
    }


def _hake_g(pre_vals: list[float], post_vals: list[float]) -> float | None:
    if not pre_vals or len(pre_vals) != len(post_vals):
        return None
    m_pre = sum(pre_vals) / len(pre_vals)
    m_post = sum(post_vals) / len(post_vals)
    if m_pre >= 1.0:
        return 0.0
    return (m_post - m_pre) / (1.0 - m_pre) if m_post > m_pre else (m_post - m_pre)


def _cohens_d(pre_vals: list[float], post_vals: list[float]) -> float | None:
    if len(pre_vals) < 2 or len(pre_vals) != len(post_vals):
        return None
    m_pre = statistics.fmean(pre_vals)
    m_post = statistics.fmean(post_vals)
    s_pre = statistics.stdev(pre_vals)
    s_post = statistics.stdev(post_vals)
    s_pooled = math.sqrt((s_pre * s_pre + s_post * s_post) / 2.0)
    return ((m_post - m_pre) / s_pooled) if s_pooled > 0.0 else 0.0


def _classify_agnes_attempt(response_doc: dict[str, Any] | None) -> str:
    if not isinstance(response_doc, dict):
        return "technical_unknown"
    code = response_doc.get("http_status")
    if code in (401, 403):
        return "auth_failure"
    if code != 200 or response_doc.get("retryable") is True:
        return "technical_unknown"
    raw = response_doc.get("raw_response")
    try:
        if not isinstance(raw, dict) or raw.get("model") != "agnes-3.0-flash":
            return "technical_unknown"
        choice = raw["choices"][0]
        if choice["finish_reason"] != "stop":
            return "technical_unknown"
        content = choice["message"]["content"].strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3 and lines[0].strip().lower() in ("```", "```json") and lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1]).strip()
        parsed = json.loads(content)
        if parsed is None or (isinstance(parsed, dict) and set(parsed) == {"correct"} and parsed["correct"] is None):
            return "semantic_unknown"
    except Exception:
        return "technical_unknown"
    return "technical_unknown"


def _load_post_counts(
    out_root: Path,
    grade_subdir: str,
    stage: str,
    key: str,
    source_id: str,
    transfer_target: str,
    *,
    inspect_receipts: bool = True,
) -> tuple[dict[str, dict[str, int]] | None, dict[str, int]]:
    diag = Counter()
    results_path = out_root / grade_subdir / stage / "sources" / key / "RESULTS.json"
    if not results_path.is_file():
        return None, dict(diag)
    doc = json.loads(results_path.read_bytes())
    results = doc.get("results")
    if not isinstance(results, list) or len(results) != 32:
        raise ValueError(f"Expected 32 post results in {grade_subdir}/{stage}")
    by_target: dict[str, Counter] = {"原题": Counter(), "迁移题": Counter()}
    for item in results:
        ident = item.get("identity") or item
        if ident.get("source_id") != source_id or ident.get("phase") != "post":
            raise ValueError("Post result identity mismatch")
        tgt = ident.get("target")
        if tgt == "source":
            tgt_zh = "原题"
        elif tgt == transfer_target:
            tgt_zh = "迁移题"
        else:
            raise ValueError("Unexpected target in post results")
        status = item.get("status")
        if status not in ("correct", "incorrect", "unknown"):
            raise ValueError("Invalid status in post results")
        by_target[tgt_zh][status] += 1
        diag[f"grade_{status}"] += 1
        if status == "unknown" and inspect_receipts and grade_subdir == "AGNES":
            req_id = hashlib.sha256(
                json.dumps({"identity": ident}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            rec_dir = out_root / "AGNES" / "receipts" / stage / key / req_id
            attempts = sorted(rec_dir.glob("ATTEMPT-????-RESPONSE.json"))
            resp = json.loads(attempts[-1].read_bytes()) if attempts else None
            diag[_classify_agnes_attempt(resp)] += 1
    for tgt_zh in ("原题", "迁移题"):
        if sum(by_target[tgt_zh].values()) != 16:
            raise ValueError("Post results target count != 16")
    return {
        t: {k: int(by_target[t].get(k, 0)) for k in ("correct", "incorrect", "unknown")}
        for t in ("原题", "迁移题")
    }, dict(diag)


def collect_base4b_records(
    manifest: dict[str, Any],
    out_root: Path,
    repo_root: Path,
    *,
    grade_subdir: str = "AGNES",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair 4B-base post outcomes with verified 27B pre/control outcomes per (tier, source_id)."""
    eval_app = str(repo_root / "evaluation" / "app")
    eval_off = str(repo_root / "evaluation" / "official")
    for p in (eval_off, eval_app):
        if p not in sys.path:
            sys.path.insert(0, p)
    import protocol  # noqa: E402

    paired_records: list[dict[str, Any]] = []
    tier_diag: dict[str, Counter] = {d: Counter() for d in DIFFICULTY_ORDER}
    assume_values: set[bool] = set()
    end_reasons: Counter = Counter()

    for entry in manifest["eligible"]:
        diff = entry["difficulty"]
        stage = STAGE_BY_DIFF[diff]
        key = entry["source_key"]
        sid = entry["source_id"]
        src_dir = out_root / stage / "sources" / key
        if not src_dir.is_dir():
            continue
        dc = tier_diag[diff]
        dc["started"] += 1
        if (src_dir / "SOURCE_HELD.json").exists():
            dc["held"] += 1
            continue
        if not (src_dir / "RAW_COMPLETE.json").is_file():
            dc["partial"] += 1
            continue
        dc["gen_complete"] += 1

        # Dialogue diagnostics (never print content).
        roll_res_p = src_dir / "dialogue" / "ROLLOUT_RESULT.json"
        if roll_res_p.is_file():
            roll = json.loads(roll_res_p.read_bytes())
            assume = bool(roll.get("assume_thinking", True))
            assume_values.add(assume)
            end_reasons[str(roll.get("end_reason") or "unknown")] += 1
            if roll.get("eoc_used"):
                dc["eoc_used"] += 1
            for t in roll.get("thinking_tokens") or []:
                dc["thinking_tokens_sum"] += int(t)
                dc["thinking_tokens_count"] += 1
            for msg in roll.get("raw_messages") or []:
                if msg.get("role") == "teacher":
                    raw_c = msg.get("content") or ""
                    dc["teacher_turns"] += 1
                    if not protocol.strip_teacher_controls(raw_c, assume_thinking=assume).strip():
                        dc["blanked_turns"] += 1
                    if "</think>" not in raw_c:
                        dc["unterminated_think_turns"] += 1

        dc["student_calls_failed"] += len(list((src_dir / "samples").glob("*/*/*/ERROR.json")))

        post_counts, gdiag = _load_post_counts(
            out_root,
            grade_subdir,
            stage,
            key,
            sid,
            entry["row"]["transfer_target"],
            inspect_receipts=(grade_subdir == "AGNES"),
        )
        for k, v in gdiag.items():
            dc[k] += v
        if post_counts is None:
            dc["grading_pending"] += 1
            continue
        dc["graded_sources"] += 1

        counts_by_target = {}
        for tgt_zh in ("原题", "迁移题"):
            counts_by_target[tgt_zh] = {
                "pre": dict(entry["ref_counts"][tgt_zh]["pre"]),
                "control": dict(entry["ref_counts"][tgt_zh]["control"]),
                "post": post_counts[tgt_zh],
            }

        paired_records.append(
            {
                "difficulty": diff,
                "tier_zh": GROUP_BY_DIFFICULTY[diff],
                "source_id": sid,
                "source_key": key,
                "counts": counts_by_target,
                "ref_counts_27b": entry["ref_counts"],
            }
        )

    diagnostics = {
        "per_tier": {d: dict(tier_diag[d]) for d in DIFFICULTY_ORDER},
        "assume_thinking_values": sorted(assume_values),
        "end_reasons": dict(end_reasons),
    }
    return paired_records, diagnostics


def _estimate_arm(
    records: list[dict[str, Any]],
    *,
    tier_filter: str,
    target_zh: str,
    baseline: str,
    outcome_phase: str,
    unknown_as_wrong: bool,
    easy_filtered: bool,
    counts_key: str = "counts",
) -> tuple[list[tuple[str, float]], list[float], list[float], int, int]:
    per_id_delta: dict[str, list[float]] = defaultdict(list)
    per_id_base: dict[str, list[float]] = defaultdict(list)
    per_id_out: dict[str, list[float]] = defaultdict(list)
    filtered_sids: set[str] = set()
    excluded_sids: set[str] = set()

    for rec in records:
        if tier_filter != "汇总" and rec["tier_zh"] != tier_filter:
            continue
        if tier_filter == "汇总" and target_zh != "原题":
            raise ValueError("Combined tier is only defined for 原题")
        c = rec[counts_key][target_zh]
        if easy_filtered and c["pre"].get("correct", 0) >= 15:
            filtered_sids.add(rec["source_id"])
            continue
        p_base = p_value(c[baseline], unknown_as_wrong)
        p_out = p_value(c[outcome_phase], unknown_as_wrong)
        if p_base is None or p_out is None:
            excluded_sids.add(rec["source_id"])
            continue
        sid = rec["source_id"]
        per_id_delta[sid].append(normalized_gain(p_base, p_out))
        per_id_base[sid].append(p_base)
        per_id_out[sid].append(p_out)

    pairs = [(sid, sum(v) / len(v)) for sid, v in sorted(per_id_delta.items())]
    base_means = [sum(per_id_base[sid]) / len(per_id_base[sid]) for sid, _ in pairs]
    out_means = [sum(per_id_out[sid]) / len(per_id_out[sid]) for sid, _ in pairs]
    return pairs, base_means, out_means, len(filtered_sids), len(excluded_sids - set(per_id_delta))


def paired_bootstrap_diff(
    pairs_4b: list[tuple[str, float]],
    pairs_27b: list[tuple[str, float]],
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute paired difference (base4b - eduardo27b) on sources present on both sides."""
    map_4b = dict(pairs_4b)
    map_27b = dict(pairs_27b)
    common = sorted(set(map_4b) & set(map_27b))
    only_4b = len(set(map_4b) - set(map_27b))
    only_27b = len(set(map_27b) - set(map_4b))
    if not common:
        return {
            "n_paired": 0,
            "n_only_4b": only_4b,
            "n_only_27b": only_27b,
            "mean_diff": None,
            "ci": None,
            "p_gt_0": None,
        }
    diffs = [map_4b[sid] - map_27b[sid] for sid in common]
    n = len(diffs)
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        boot.append(sum(rng.choices(diffs, k=n)) / n)
    boot.sort()
    p_gt_0 = sum(1 for x in boot if x > 0.0) / draws
    return {
        "n_paired": n,
        "n_only_4b": only_4b,
        "n_only_27b": only_27b,
        "mean_diff": sum(diffs) / n,
        "ci": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "p_gt_0": p_gt_0,
    }


def compute_summary(
    records: list[dict[str, Any]],
    *,
    all_27b_records: list[dict[str, Any]] | None = None,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute R0-R4, sensitivity analyses, and paired difference vs 27B."""
    rows_spec = [
        ("原题合并", "汇总", "原题", "pre", False),
        ("原题-简单对话", "简单", "原题", "pre", False),
        ("原题-中等对话", "中等", "原题", "pre", False),
        ("原题-困难对话", "困难", "原题", "pre", False),
        ("简单迁移题", "简单", "迁移题", "pre", False),
        ("中等迁移题", "中等", "迁移题", "pre", False),
        ("困难迁移题", "困难", "迁移题", "pre", False),
        ("简单迁移题-剔除高分(对照基线)", "简单", "迁移题", "control", True),
        ("简单迁移题-剔除高分(前测基线)", "简单", "迁移题", "pre", True),
    ]

    ref_27b = all_27b_records if all_27b_records is not None else records
    analyses: dict[str, Any] = {}
    for unknown_as_wrong, policy in ((True, "未决按错"), (False, "排除未决")):
        table_rows = []
        for label, tier_zh, tgt_zh, base_phase, filtered in rows_spec:
            p4, b4, o4, n_filt, n_excl = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline=base_phase,
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            r1 = bootstrap_mean_ci(
                [v for _, v in p4],
                seed=_derived_seed(seed, f"{policy}|{label}|4b"),
                draws=draws,
            )
            r4_pairs, _, _, _, _ = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="pre",
                outcome_phase="control",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            r4 = bootstrap_mean_ci(
                [v for _, v in r4_pairs],
                seed=_derived_seed(seed, f"{policy}|{label}|r4"),
                draws=draws,
            )
            p27, _, _, _, _ = _estimate_arm(
                ref_27b,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline=base_phase,
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="ref_counts_27b" if all_27b_records is None else "ref_counts",
            )
            paired = paired_bootstrap_diff(
                p4, p27, seed=_derived_seed(seed, f"{policy}|{label}|paired"), draws=draws
            )
            table_rows.append(
                {
                    "label": label,
                    "n_4b": len(p4),
                    "n_filtered": n_filt,
                    "n_excluded_unknown": n_excl,
                    "r0_base": (sum(b4) / len(b4)) if b4 else None,
                    "r0_post": (sum(o4) / len(o4)) if o4 else None,
                    "r1": r1,
                    "r2_hake_g": _hake_g(b4, o4),
                    "r3_cohens_d": _cohens_d(b4, o4),
                    "r4_control": r4,
                    "paired_4b_minus_27b": paired,
                }
            )
        analyses[policy] = table_rows

    return {"seed": seed, "draws": draws, "analyses": analyses}


def format_report(
    manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    summary: dict[str, Any],
    *,
    elapsed_seconds: float = 0.0,
) -> str:
    lines: list[str] = []
    lines.append("=" * 84)
    lines.append("Qwen3.5-4B Base Self-Built Three-Tier Evaluation Summary (Sanitized)")
    lines.append(
        f"27B Manifest: eligible={manifest['eligible_count']} skipped_incomplete={manifest['skipped_count']} "
        f"(seed={summary['seed']}, draws={summary['draws']})"
    )
    lines.append("-" * 84)
    lines.append("Diagnostics by Tier:")
    tot_turns = 0
    tot_blanked = 0
    tot_unterm = 0
    tot_failed_stu = 0
    tot_graded_src = 0
    tot_agnes = Counter()
    for d in DIFFICULTY_ORDER:
        dc = diagnostics["per_tier"][d]
        turns = dc.get("teacher_turns", 0)
        blanked = dc.get("blanked_turns", 0)
        unterm = dc.get("unterminated_think_turns", 0)
        tot_turns += turns
        tot_blanked += blanked
        tot_unterm += unterm
        tot_failed_stu += dc.get("student_calls_failed", 0)
        tot_graded_src += dc.get("graded_sources", 0)
        for k in ("grade_correct", "grade_incorrect", "grade_unknown", "semantic_unknown", "technical_unknown", "auth_failure"):
            tot_agnes[k] += dc.get(k, 0)
        b_pct = (100.0 * blanked / turns) if turns else 0.0
        u_pct = (100.0 * unterm / turns) if turns else 0.0
        lines.append(
            f"  {d:<6}: started={dc.get('started', 0)} gen_complete={dc.get('gen_complete', 0)} "
            f"held={dc.get('held', 0)} partial={dc.get('partial', 0)} graded_src={dc.get('graded_sources', 0)} "
            f"blanked={blanked}/{turns} ({b_pct:.2f}%) unterminated_think={unterm}/{turns} ({u_pct:.2f}%) "
            f"stu_fail={dc.get('student_calls_failed', 0)}"
        )

    tot_calls = tot_agnes["grade_correct"] + tot_agnes["grade_incorrect"] + tot_agnes["grade_unknown"]
    tech_rate = (100.0 * tot_agnes["technical_unknown"] / tot_calls) if tot_calls else 0.0
    unk_rate = (100.0 * tot_agnes["grade_unknown"] / tot_calls) if tot_calls else 0.0
    lines.append(
        f"Agnes total_calls={tot_calls} correct={tot_agnes['grade_correct']} incorrect={tot_agnes['grade_incorrect']} "
        f"unknown={tot_agnes['grade_unknown']} ({unk_rate:.2f}%) [semantic_null={tot_agnes['semantic_unknown']}, "
        f"tech_fail={tot_agnes['technical_unknown']} ({tech_rate:.2f}%), auth_fail={tot_agnes['auth_failure']}]"
    )
    lines.append(
        f"Tutor assume_thinking={diagnostics['assume_thinking_values']} end_reasons={diagnostics['end_reasons']} "
        f"blanked_total={tot_blanked}/{tot_turns} unterminated_total={tot_unterm}/{tot_turns} stu_fail_total={tot_failed_stu}"
    )
    if elapsed_seconds > 0 and tot_graded_src > 0:
        sec_per = elapsed_seconds / tot_graded_src
        est_hours = (sec_per * 1489.0) / 3600.0
        lines.append(
            f"Timing: elapsed={elapsed_seconds:.1f}s sources={tot_graded_src} sec_per_source={sec_per:.1f}s "
            f"est_1489_hours={est_hours:.1f}h ({est_hours/24.0:.2f}d)"
        )
    lines.append("-" * 84)
    for policy, rows in summary["analyses"].items():
        lines.append(f"[{policy}]")
        lines.append(f"{'项目':<28} | {'n':>4} | {'R0_base':>7} | {'R0_post':>7} | {'R1 Δ [95% CI]':<23} | {'R2 g':>6} | {'R3 d':>6} | {'4B-27B Paired [95% CI]':<25}")
        for r in rows:
            r1 = r["r1"]
            r1_s = f"{r1['mean']:+.4f} [{r1['ci'][0]:+.4f},{r1['ci'][1]:+.4f}]" if r1 else "N/A"
            pd = r["paired_4b_minus_27b"]
            pd_s = (
                f"{pd['mean_diff']:+.4f} [{pd['ci'][0]:+.4f},{pd['ci'][1]:+.4f}] (n={pd['n_paired']})"
                if pd["mean_diff"] is not None
                else "N/A"
            )
            rb = f"{r['r0_base']*100:6.2f}%" if r["r0_base"] is not None else "    N/A"
            ro = f"{r['r0_post']*100:6.2f}%" if r["r0_post"] is not None else "    N/A"
            r2 = f"{r['r2_hake_g']:+6.3f}" if r["r2_hake_g"] is not None else "   N/A"
            r3 = f"{r['r3_cohens_d']:+6.3f}" if r["r3_cohens_d"] is not None else "   N/A"
            lines.append(f"{r['label']:<28} | {r['n_4b']:>4} | {rb} | {ro} | {r1_s:<23} | {r2} | {r3} | {pd_s}")
    lines.append("=" * 84)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanitized summary for Qwen3.5-4B self-built 3-tier eval.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref-root", type=Path, default=Path("/workspace/results/own-test-499-v1"))
    ap.add_argument("--overlay", type=Path, default=Path("/workspace/results/agnes-topbool-recovery-v1"))
    ap.add_argument("--decisions", type=Path, default=Path("/workspace/results/agnes-rejudge-v1"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--grade-subdir", default="AGNES", choices=("AGNES", "AUTHOR_RULE"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--elapsed-seconds", type=float, default=0.0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    manifest = build_manifest(
        args.data,
        args.ref_root,
        args.overlay,
        args.decisions,
        repo_root=args.repo,
        require_exact_totals=True,
    )
    records, diagnostics = collect_base4b_records(
        manifest, args.out_root, args.repo, grade_subdir=args.grade_subdir
    )
    all_27b = [
        {"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]}
        for e in manifest["eligible"]
    ]
    summary = compute_summary(records, all_27b_records=all_27b, seed=args.seed, draws=args.draws)
    print(format_report(manifest, diagnostics, summary, elapsed_seconds=args.elapsed_seconds))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "per_tier_manifest": manifest["per_tier_counts"],
            "eligible_count": manifest["eligible_count"],
            "skipped_count": manifest["skipped_count"],
            "diagnostics": diagnostics,
            "summary": summary,
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
