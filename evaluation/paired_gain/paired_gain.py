#!/usr/bin/env python3
"""Paired bootstrap evaluation of normalized learning gains between two models.

Reads a conversation-level summary CSV and computes source-weighted normalized
gains (Formula 4 from arXiv:2610.06446) and paired bootstrap 95% confidence
intervals for (target - base) over matched unit_ids.

No unit_id values are ever emitted in JSON or Markdown outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HIGH_PRE_THRESHOLD = 15.0 / 16.0
PRE_CONSISTENCY_TOL = 1e-9
REQUIRED_COLUMNS = (
    "unit_id",
    "condition",
    "group",
    "pre",
    "post_correct",
    "post_total",
    "post_unknown",
)


def normalized_gain(pre: float, post: float) -> float:
    """Compute asymmetric Hake normalized learning gain (Formula 4).

    Matches source_gain.normalized_gain:
    - If post > pre: (post - pre) / (1 - pre)
    - Otherwise: post - pre
    """
    if not (0.0 <= pre <= 1.0 and 0.0 <= post <= 1.0):
        raise ValueError(f"Probabilities must be in [0, 1], got pre={pre}, post={post}")
    return (post - pre) / (1.0 - pre) if post > pre else post - pre


def compute_post_rate(
    post_correct: int,
    post_total: int,
    post_unknown: int,
    unknown_as_wrong: bool,
) -> Optional[float]:
    """Compute post-test pass rate under the selected unknown-handling policy.

    Returns None when unknown_as_wrong is False and all answers are unknown
    (post_total - post_unknown == 0).
    """
    if post_total <= 0 or post_correct < 0 or post_unknown < 0:
        raise ValueError(
            f"Invalid counts: correct={post_correct}, total={post_total}, unknown={post_unknown}"
        )
    if post_correct + post_unknown > post_total:
        raise ValueError(
            f"post_correct + post_unknown exceeds post_total: "
            f"{post_correct} + {post_unknown} > {post_total}"
        )
    den = post_total if unknown_as_wrong else (post_total - post_unknown)
    if den == 0:
        return None
    return post_correct / den


def load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load and validate conversation-level summary rows from CSV."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or missing header")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        for line_no, raw in enumerate(reader, start=2):
            unit_id = (raw["unit_id"] or "").strip()
            condition = (raw["condition"] or "").strip()
            group = (raw["group"] or "").strip()
            if not unit_id or not condition or not group:
                raise ValueError(f"Empty identifier on CSV line {line_no}")

            pre = float(raw["pre"])
            if not (0.0 <= pre <= 1.0):
                raise ValueError(f"Invalid pre={pre} on CSV line {line_no}")

            post_correct = int(raw["post_correct"])
            post_total = int(raw["post_total"])
            post_unknown = int(raw["post_unknown"])
            # Validate counts eagerly
            compute_post_rate(post_correct, post_total, post_unknown, True)

            rows.append(
                {
                    "unit_id": unit_id,
                    "condition": condition,
                    "group": group,
                    "pre": pre,
                    "post_correct": post_correct,
                    "post_total": post_total,
                    "post_unknown": post_unknown,
                    "line_no": line_no,
                }
            )
    validate_pre_consistency(rows)
    return rows


def validate_pre_consistency(
    rows: Sequence[Dict[str, Any]],
    group_filter: Optional[str] = None,
    tol: float = PRE_CONSISTENCY_TOL,
    max_lines: int = 5,
) -> None:
    """Verify that all rows with the same (group, unit_id) share the same pre.

    Error messages report only group name, count of inconsistent sources, and
    the first few conflicting CSV line numbers (never unit_id).
    """
    seen: Dict[str, Dict[str, Tuple[float, Any]]] = defaultdict(dict)
    bad_units: Dict[str, set] = defaultdict(set)
    bad_lines: Dict[str, List[Any]] = defaultdict(list)

    for idx, row in enumerate(rows, start=1):
        group = row["group"]
        if group_filter is not None and group != group_filter:
            continue
        uid = row["unit_id"]
        pre = float(row["pre"])
        line_no = row.get("line_no", idx)

        group_seen = seen[group]
        if uid not in group_seen:
            group_seen[uid] = (pre, line_no)
        else:
            ref_pre, _ = group_seen[uid]
            if abs(pre - ref_pre) > tol:
                bad_units[group].add(uid)
                if len(bad_lines[group]) < max_lines:
                    bad_lines[group].append(line_no)

    if bad_units:
        parts = []
        for group in sorted(bad_units.keys()):
            n_bad = len(bad_units[group])
            lines_str = ", ".join(str(ln) for ln in bad_lines[group])
            parts.append(
                f"group '{group}': {n_bad} source(s) have inconsistent pre "
                f"(diff > {tol:g}); first conflicting lines: [{lines_str}]"
            )
        raise ValueError("Pre-test consistency check failed: " + "; ".join(parts))


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 64-bit integer seed independent of PYTHONHASHSEED."""
    payload = "|".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def aggregate_condition(
    rows: Sequence[Dict[str, Any]],
    group: str,
    condition: str,
    unknown_as_wrong: bool,
) -> Tuple[Dict[str, float], Dict[str, List[float]], int, int]:
    """Aggregate normalized gains per unit_id for one (group, condition).

    Returns:
        unit_gains: {unit_id: mean_gain}
        unit_pres: {unit_id: [pre, ...]}
        n_skipped_rows: number of rows skipped due to 0 valid denominator
        n_excluded_units: number of unit_ids with no valid rows at all
    """
    per_unit: Dict[str, List[float]] = defaultdict(list)
    per_unit_pre: Dict[str, List[float]] = defaultdict(list)
    excluded_zero = set()
    n_skipped_rows = 0

    for row in rows:
        if row["group"] != group or row["condition"] != condition:
            continue
        uid = row["unit_id"]
        per_unit_pre[uid].append(row["pre"])
        post = compute_post_rate(
            row["post_correct"],
            row["post_total"],
            row["post_unknown"],
            unknown_as_wrong=unknown_as_wrong,
        )
        if post is None:
            n_skipped_rows += 1
            excluded_zero.add(uid)
            continue
        per_unit[uid].append(normalized_gain(row["pre"], post))

    unit_gains = {uid: sum(vals) / len(vals) for uid, vals in per_unit.items()}
    n_excluded_units = len(excluded_zero - set(per_unit.keys()))
    return unit_gains, per_unit_pre, n_skipped_rows, n_excluded_units


def _bootstrap_stats_from_arrays(
    base_arr: np.ndarray,
    target_arr: np.ndarray,
    seed: int,
    draws: int,
) -> Dict[str, float]:
    """Run vectorized paired bootstrap over aligned 1D arrays."""
    n = len(base_arr)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array")
    if draws <= 0:
        raise ValueError(f"draws must be positive, got {draws}")

    diff_arr = target_arr - base_arr
    rng = np.random.default_rng(seed)

    # Chunk draws to keep peak temporary memory bounded for very large inputs.
    max_elements = 5_000_000
    chunk_size = max(1, min(draws, max_elements // max(1, n)))

    boot_base = np.empty(draws, dtype=np.float64)
    boot_target = np.empty(draws, dtype=np.float64)
    boot_diff = np.empty(draws, dtype=np.float64)

    offset = 0
    while offset < draws:
        cur = min(chunk_size, draws - offset)
        idx = rng.integers(0, n, size=(cur, n))
        boot_base[offset : offset + cur] = base_arr[idx].mean(axis=1)
        boot_target[offset : offset + cur] = target_arr[idx].mean(axis=1)
        boot_diff[offset : offset + cur] = diff_arr[idx].mean(axis=1)
        offset += cur

    base_ci_low, base_ci_high = np.percentile(boot_base, [2.5, 97.5])
    target_ci_low, target_ci_high = np.percentile(boot_target, [2.5, 97.5])
    diff_ci_low, diff_ci_high = np.percentile(boot_diff, [2.5, 97.5])

    return {
        "base_mean": float(base_arr.mean()),
        "base_ci_low": float(base_ci_low),
        "base_ci_high": float(base_ci_high),
        "target_mean": float(target_arr.mean()),
        "target_ci_low": float(target_ci_low),
        "target_ci_high": float(target_ci_high),
        "diff_mean": float(diff_arr.mean()),
        "diff_ci_low": float(diff_ci_low),
        "diff_ci_high": float(diff_ci_high),
        "prop_diff_gt_zero": float(np.mean(boot_diff > 0.0)),
    }


def evaluate_group(
    rows: Sequence[Dict[str, Any]],
    group: str,
    base_condition: str,
    target_condition: str,
    unknown_as_wrong: bool,
    filter_high_pre: bool,
    draws: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate paired gain difference for one (group, policy, filter) slice."""
    validate_pre_consistency(rows, group_filter=group)
    base_gains, base_pres, base_skipped, base_excl_zero = aggregate_condition(
        rows, group, base_condition, unknown_as_wrong
    )
    target_gains, target_pres, target_skipped, target_excl_zero = aggregate_condition(
        rows, group, target_condition, unknown_as_wrong
    )

    all_base_ids = set(base_gains.keys())
    all_target_ids = set(target_gains.keys())
    paired_ids = sorted(all_base_ids & all_target_ids)

    n_unpaired_base = len(all_base_ids - all_target_ids)
    n_unpaired_target = len(all_target_ids - all_base_ids)

    n_filtered_high_pre = 0
    if filter_high_pre:
        kept_ids: List[str] = []
        for uid in paired_ids:
            pre_val = base_pres[uid][0]
            if pre_val >= HIGH_PRE_THRESHOLD:
                n_filtered_high_pre += 1
            else:
                kept_ids.append(uid)
        paired_ids = kept_ids

    n_paired = len(paired_ids)
    result: Dict[str, Any] = {
        "group": group,
        "unknown_policy": "unknown_as_wrong" if unknown_as_wrong else "exclude_unknown",
        "unknown_policy_label": "未决按错" if unknown_as_wrong else "排除未决",
        "source_filter": "exclude_pre_ge_15_16" if filter_high_pre else "all",
        "source_filter_label": "剔除前测>=15/16" if filter_high_pre else "全部来源",
        "n_paired": n_paired,
        "n_unpaired_base": n_unpaired_base,
        "n_unpaired_target": n_unpaired_target,
        "n_unpaired_total": n_unpaired_base + n_unpaired_target,
        "n_skipped_rows_base": base_skipped,
        "n_skipped_rows_target": target_skipped,
        "n_excluded_zero_base": base_excl_zero,
        "n_excluded_zero_target": target_excl_zero,
        "n_filtered_high_pre": n_filtered_high_pre,
        "stats": None,
    }

    if n_paired > 0:
        base_vals = np.array([base_gains[uid] for uid in paired_ids], dtype=np.float64)
        target_vals = np.array([target_gains[uid] for uid in paired_ids], dtype=np.float64)
        sub_seed = derive_seed(seed, group, unknown_as_wrong, filter_high_pre)
        result["stats"] = _bootstrap_stats_from_arrays(
            base_vals, target_vals, seed=sub_seed, draws=draws
        )

    return result


def evaluate_dataset(
    rows: Sequence[Dict[str, Any]],
    base_condition: str = "base",
    target_condition: str = "a_group",
    draws: int = 10000,
    seed: int = 42,
    include_high_pre_filtered: bool = True,
) -> Dict[str, Any]:
    """Run paired bootstrap evaluation across all groups and policies."""
    groups = sorted({row["group"] for row in rows})
    if not groups:
        raise ValueError("No data rows found")
    validate_pre_consistency(rows)

    filter_modes = [False, True] if include_high_pre_filtered else [False]
    results: List[Dict[str, Any]] = []

    for filter_high_pre in filter_modes:
        for unknown_as_wrong in (True, False):
            for group in groups:
                results.append(
                    evaluate_group(
                        rows=rows,
                        group=group,
                        base_condition=base_condition,
                        target_condition=target_condition,
                        unknown_as_wrong=unknown_as_wrong,
                        filter_high_pre=filter_high_pre,
                        draws=draws,
                        seed=seed,
                    )
                )

    return {
        "meta": {
            "base_condition": base_condition,
            "target_condition": target_condition,
            "draws": draws,
            "seed": seed,
            "ci_level": 0.95,
            "high_pre_threshold": HIGH_PRE_THRESHOLD,
            "groups": groups,
        },
        "results": results,
    }


def _fmt_pct_ci(mean: float, low: float, high: float) -> str:
    return f"{mean * 100:+.2f}% [{low * 100:+.2f}%, {high * 100:+.2f}%]"


def render_markdown(report: Dict[str, Any]) -> str:
    """Render evaluation report as a Markdown summary table (no unit_ids)."""
    meta = report["meta"]
    base_name = meta["base_condition"]
    target_name = meta["target_condition"]

    lines = [
        f"# 配对 Bootstrap 归一化增益评测报告（`{target_name}` vs `{base_name}`）",
        "",
        (
            f"- 随机种子：`{meta['seed']}`；重采样次数：`{meta['draws']}`；"
            f"置信区间：95% 百分位区间（按来源 `unit_id` 有放回重采样）"
        ),
        f"- 差值定义：`{target_name} - {base_name}`；`P(diff > 0)` 为重采样差值大于 0 的比例。",
    ]

    sections = [
        ("all", "主口径：全部来源"),
        ("exclude_pre_ge_15_16", "次要口径：剔除前测 >= 15/16 的来源"),
    ]

    for filter_key, section_title in sections:
        section_rows = [r for r in report["results"] if r["source_filter"] == filter_key]
        if not section_rows:
            continue

        lines.extend(["", f"## {section_title}"])

        for policy_key, policy_title in (
            ("unknown_as_wrong", "未决按错（主处理）"),
            ("exclude_unknown", "排除未决（敏感性）"),
        ):
            policy_rows = [r for r in section_rows if r["unknown_policy"] == policy_key]
            if not policy_rows:
                continue

            lines.extend(
                [
                    "",
                    f"### {policy_title}",
                    "",
                    (
                        f"| 题集 | 配对来源数 | 未配对({base_name}/{target_name}) | "
                        f"跳过行/全未决源 | 高前测剔除 | `{base_name}` 增益 [95% CI] | "
                        f"`{target_name}` 增益 [95% CI] | 差值 ({target_name}-{base_name}) [95% CI] | P(diff>0) |"
                    ),
                    "|---|---:|---:|---:|---:|---|---|---|---:|",
                ]
            )

            for r in policy_rows:
                unpaired_str = f"{r['n_unpaired_base']}/{r['n_unpaired_target']}"
                skip_rows = r["n_skipped_rows_base"] + r["n_skipped_rows_target"]
                excl_src = r["n_excluded_zero_base"] + r["n_excluded_zero_target"]
                skip_str = f"{skip_rows}/{excl_src}"

                if r["stats"] is None:
                    base_str = "-"
                    target_str = "-"
                    diff_str = "-"
                    p_str = "-"
                else:
                    s = r["stats"]
                    base_str = _fmt_pct_ci(s["base_mean"], s["base_ci_low"], s["base_ci_high"])
                    target_str = _fmt_pct_ci(s["target_mean"], s["target_ci_low"], s["target_ci_high"])
                    diff_str = _fmt_pct_ci(s["diff_mean"], s["diff_ci_low"], s["diff_ci_high"])
                    p_str = f"{s['prop_diff_gt_zero']:.4f}"

                lines.append(
                    f"| {r['group']} | {r['n_paired']} | {unpaired_str} | {skip_str} | "
                    f"{r['n_filtered_high_pre']} | {base_str} | {target_str} | {diff_str} | {p_str} |"
                )

    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap evaluation of normalized learning gains."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input summary CSV path")
    parser.add_argument("--base", default="base", help="Baseline condition name (default: base)")
    parser.add_argument(
        "--target", default="a_group", help="Target condition name (default: a_group)"
    )
    parser.add_argument("--draws", type=int, default=10000, help="Bootstrap draws (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--json-out", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--md-out", type=Path, default=None, help="Output Markdown path")
    parser.add_argument(
        "--no-high-pre-filter",
        action="store_true",
        help="Skip secondary report that filters out sources with pre >= 15/16",
    )
    args = parser.parse_args(argv)

    rows = load_csv(args.input)
    report = evaluate_dataset(
        rows=rows,
        base_condition=args.base,
        target_condition=args.target,
        draws=args.draws,
        seed=args.seed,
        include_high_pre_filtered=not args.no_high_pre_filter,
    )
    md_text = render_markdown(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md_text, encoding="utf-8")
    else:
        print(md_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
