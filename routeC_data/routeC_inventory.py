#!/usr/bin/env python3
"""Route C difficulty-routing inventory (T-031 Stage 1 preliminary & Stage 2 final).

PRIVACY GUARANTEE:
- Reads `routeA_stage2_measured.jsonl` (and optional `source_pretest_measured.jsonl`)
  in-memory only for numerical computations.
- Never prints or writes problem statements, reference answers, or student responses.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from routeA_data.hybrid_grader_caller import (
    CANONICAL_HYBRID_GRADER_SHA256,
    assert_canonical_hybrid_grader,
)
from routeA_data.rasch_ladder import compute_smoothed_p_and_b

EXPECTED_T023_MEASURED_SHA256 = (
    "9ff8449b781906290f9b671bc5be1982ca31e5401ebfba485474d475a60752ce"
)
EXPECTED_OLD_SOURCE_PRETEST_TABLE_SHA256 = (
    "33b039c420375c79ec1024ef4393918f8f60fced878ebe764a7c44d28f6c1478"
)
DEFAULT_TAUS: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
DEFAULT_MARGINS: Tuple[float, ...] = (0.5, 1.0)
TARGET_ROUTE_C_SOURCES = 1600


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_source_pretest_table_sha256(rows: Sequence[Tuple[str, int, int]]) -> str:
    lines = [
        f"{sid}\t{int(nc)}\t{int(ns)}\n"
        for sid, nc, ns in sorted(rows, key=lambda x: x[0])
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _median(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def _load_source_pretest_map(
    path: Path,
    expected_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    actual_sha = sha256_file(path)
    if expected_sha256 and actual_sha != expected_sha256.strip().lower():
        raise ValueError(
            f"source_pretest_jsonl SHA-256 mismatch: got={actual_sha} expected={expected_sha256}"
        )
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r["source_id"])
            if sid in out:
                raise ValueError(f"Duplicate source_id in source_pretest_jsonl: {sid}")
            n_scored = int(r["n_scored"])
            n_correct = int(r["n_correct_hybrid"])
            p_orig, b_orig = compute_smoothed_p_and_b(n_correct, n_scored)
            out[sid] = {
                "source_id": sid,
                "n_correct": n_correct,
                "n_scored": n_scored,
                "p_orig": p_orig,
                "b_orig": b_orig,
            }
    return out, actual_sha


def _evaluate_tau_margin(
    items: Sequence[Dict[str, Any]],
    tau: float,
    margin: float,
    target_total: int = TARGET_ROUTE_C_SOURCES,
) -> Dict[str, Any]:
    easy_total = 0
    easy_usable = 0
    easy_usable_all_qual_ceil = 0
    easy_all_strict_harder_ceil = 0

    hard_total = 0
    hard_usable = 0
    hard_usable_orig_floor = 0
    hard_all_orig_floor = 0

    for it in items:
        p_orig = float(it["p_orig"])
        n_orig = int(it["n_orig"])
        n_orig_scored = int(it["n_orig_scored"])
        is_orig_floor = (n_orig / n_orig_scored) <= (1.0 / 16.0 + 1e-9)

        variants = it["variants"]
        if p_orig < tau:
            hard_total += 1
            if is_orig_floor:
                hard_all_orig_floor += 1
            qualifying = [v for v in variants if v["delta_b"] <= -margin]
            if qualifying:
                hard_usable += 1
                if is_orig_floor:
                    hard_usable_orig_floor += 1
        else:
            easy_total += 1
            strictly_harder = [v for v in variants if v["delta_b"] > 0.0]
            if strictly_harder and all(v["is_ceil_ge_15_16"] for v in strictly_harder):
                easy_all_strict_harder_ceil += 1
            qualifying = [v for v in variants if v["delta_b"] >= margin]
            if qualifying:
                easy_usable += 1
                if all(v["is_ceil_ge_15_16"] for v in qualifying):
                    easy_usable_all_qual_ceil += 1

    total_sources = easy_total + hard_total
    assert total_sources == len(items)
    total_usable = easy_usable + hard_usable
    usable_excl_floor_and_ceil = (
        (easy_usable - easy_usable_all_qual_ceil)
        + (hard_usable - hard_usable_orig_floor)
    )
    return {
        "tau": tau,
        "margin": margin,
        "n_sources": total_sources,
        "easy_total": easy_total,
        "easy_usable": easy_usable,
        "easy_usable_all_qualifying_harder_ceil_ge_15_16": easy_usable_all_qual_ceil,
        "easy_all_strictly_harder_ceil_ge_15_16": easy_all_strict_harder_ceil,
        "hard_total": hard_total,
        "hard_usable": hard_usable,
        "hard_usable_orig_floor_le_1_16": hard_usable_orig_floor,
        "hard_all_orig_floor_le_1_16": hard_all_orig_floor,
        "total_usable": total_usable,
        "gap_vs_1600": total_usable - target_total,
        "usable_excl_floor_and_ceil": usable_excl_floor_and_ceil,
        "gap_excl_floor_and_ceil_vs_1600": usable_excl_floor_and_ceil - target_total,
    }


def _summarize_direction_group(
    variants: Sequence[Dict[str, Any]],
    expected_sign: Optional[str],
) -> Dict[str, Any]:
    deltas = [float(v["delta_b"]) for v in variants]
    n = len(deltas)
    n_le_m1 = sum(1 for d in deltas if d <= -1.0)
    n_le_m05 = sum(1 for d in deltas if d <= -0.5)
    n_lt_0 = sum(1 for d in deltas if d < -1e-12)
    n_eq_0 = sum(1 for d in deltas if abs(d) <= 1e-12)
    n_gt_0 = sum(1 for d in deltas if d > 1e-12)
    n_ge_p05 = sum(1 for d in deltas if d >= 0.5)
    n_ge_p1 = sum(1 for d in deltas if d >= 1.0)

    matched_strict = None
    matched_ge_05 = None
    matched_ge_10 = None
    if expected_sign == "negative":
        matched_strict = n_lt_0
        matched_ge_05 = n_le_m05
        matched_ge_10 = n_le_m1
    elif expected_sign == "positive":
        matched_strict = n_gt_0
        matched_ge_05 = n_ge_p05
        matched_ge_10 = n_ge_p1

    return {
        "n_variants": n,
        "delta_b_le_neg1_0": n_le_m1,
        "delta_b_le_neg0_5": n_le_m05,
        "delta_b_lt_0": n_lt_0,
        "delta_b_eq_0": n_eq_0,
        "delta_b_gt_0": n_gt_0,
        "delta_b_ge_pos0_5": n_ge_p05,
        "delta_b_ge_pos1_0": n_ge_p1,
        "mean_delta_b": round(sum(deltas) / n, 6) if n else None,
        "median_delta_b": round(_median(deltas), 6) if n else None,
        "expected_direction": expected_sign or "mixed_or_neutral",
        "matched_strict_count": matched_strict,
        "matched_strict_rate": round(matched_strict / n, 6)
        if (matched_strict is not None and n)
        else None,
        "matched_ge_0_5_count": matched_ge_05,
        "matched_ge_0_5_rate": round(matched_ge_05 / n, 6)
        if (matched_ge_05 is not None and n)
        else None,
        "matched_ge_1_0_count": matched_ge_10,
        "matched_ge_1_0_rate": round(matched_ge_10 / n, 6)
        if (matched_ge_10 is not None and n)
        else None,
    }


def compute_inventory(
    measured_rows: Sequence[Dict[str, Any]],
    *,
    source_pretest_map: Optional[Dict[str, Dict[str, Any]]] = None,
    taus: Sequence[float] = DEFAULT_TAUS,
    margins: Sequence[float] = DEFAULT_MARGINS,
) -> Tuple[Dict[str, Any], str]:
    """Compute Route C routing inventory and direction cross-tabs (pure numbers)."""
    items: List[Dict[str, Any]] = []
    old_table_tuples: List[Tuple[str, int, int]] = []
    all_variants_flat: List[Dict[str, Any]] = []
    seen_sids = set()

    for row in measured_rows:
        sid = str(row["source_id"])
        if sid in seen_sids:
            raise ValueError(f"Duplicate source_id in measured_jsonl: {sid}")
        seen_sids.add(sid)
        split = str(row["split"])
        direction = str(row["direction"])

        if source_pretest_map is not None:
            if sid not in source_pretest_map:
                raise ValueError(f"source_id {sid} missing from source_pretest_jsonl")
            sp = source_pretest_map[sid]
            n_orig = int(sp["n_correct"])
            n_orig_scored = int(sp["n_scored"])
            p_orig = float(sp["p_orig"])
            b_orig = float(sp["b_orig"])
        else:
            old_rate = row.get("source_pre_solve_rate_old")
            if old_rate is None:
                raise ValueError(f"Missing source_pre_solve_rate_old for {sid}")
            old_rate_f = float(old_rate)
            n_orig = int(round(old_rate_f * 16))
            if abs(old_rate_f - n_orig / 16.0) > 1e-9:
                raise ValueError(f"Non-k/16 source_pre_solve_rate_old for {sid}: {old_rate_f}")
            n_orig_scored = 16
            p_orig, b_orig = compute_smoothed_p_and_b(n_orig, n_orig_scored)
            old_table_tuples.append((sid, n_orig, n_orig_scored))

        var_entries: List[Dict[str, Any]] = []
        for var in row["variants"]:
            n_hyb = int(var["n_correct_hybrid"])
            n_scored = int(var.get("n_scored", 16))
            p_var, b_var = compute_smoothed_p_and_b(n_hyb, n_scored)
            if var.get("p_new") is not None and abs(float(var["p_new"]) - p_var) > 1e-5:
                raise ValueError(
                    f"Variant p_new mismatch for {sid}/{var.get('variant_id')}: {var.get('p_new')} vs {p_var}"
                )
            delta_b = b_var - b_orig
            is_ceil = (n_hyb / n_scored) >= (15.0 / 16.0 - 1e-9)
            ve = {
                "source_id": sid,
                "split": split,
                "source_direction": direction,
                "origin": str(var.get("origin", "")),
                "generation_tier": str(var.get("generation_tier", "")),
                "generation_tag": str(var.get("generation_tag", "")),
                "n_correct_hybrid": n_hyb,
                "n_scored": n_scored,
                "p_var": p_var,
                "b_var": b_var,
                "delta_b": delta_b,
                "is_ceil_ge_15_16": is_ceil,
            }
            var_entries.append(ve)
            all_variants_flat.append(ve)

        items.append(
            {
                "source_id": sid,
                "split": split,
                "direction": direction,
                "n_orig": n_orig,
                "n_orig_scored": n_orig_scored,
                "p_orig": p_orig,
                "b_orig": b_orig,
                "variants": var_entries,
            }
        )

    if source_pretest_map is not None and len(source_pretest_map) != len(items):
        raise ValueError(
            f"source_pretest_jsonl count ({len(source_pretest_map)}) != measured_jsonl count ({len(items)})"
        )

    old_table_sha = (
        compute_source_pretest_table_sha256(old_table_tuples)
        if old_table_tuples
        else ""
    )

    by_split: Dict[str, List[Dict[str, Any]]] = {
        "all": items,
        "train": [it for it in items if it["split"] == "train"],
        "dev": [it for it in items if it["split"] == "dev"],
    }

    routing_tables: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, sub_items in by_split.items():
        rows_out: List[Dict[str, Any]] = []
        for tau in taus:
            for margin in margins:
                rows_out.append(_evaluate_tau_margin(sub_items, float(tau), float(margin)))
        routing_tables[split_name] = rows_out

    # Cross-tab 1: by source direction label (easier / mixed / harder)
    def _expected_sign_for_label(lbl: str) -> Optional[str]:
        if lbl == "easier":
            return "negative"
        if lbl == "harder":
            return "positive"
        return None

    cross_by_source_dir_all: Dict[str, Any] = {}
    cross_by_source_dir_self: Dict[str, Any] = {}
    cross_by_source_dir_paper: Dict[str, Any] = {}
    for lbl in ("easier", "mixed", "harder"):
        exp_sign = _expected_sign_for_label(lbl)
        sub_all = [v for v in all_variants_flat if v["source_direction"] == lbl]
        sub_self = [v for v in sub_all if v["origin"] == "self_generated"]
        sub_paper = [v for v in sub_all if v["origin"] == "paper_transfer"]
        cross_by_source_dir_all[lbl] = _summarize_direction_group(sub_all, exp_sign)
        cross_by_source_dir_self[lbl] = _summarize_direction_group(sub_self, exp_sign)
        cross_by_source_dir_paper[lbl] = _summarize_direction_group(sub_paper, exp_sign)

    # Cross-tab 2: by variant generation_tier (easier / nearby / harder / paper_transfer)
    cross_by_gen_tier: Dict[str, Any] = {}
    for tier in ("easier", "nearby", "harder", "paper_transfer"):
        exp_sign = _expected_sign_for_label(tier)
        sub_t = [v for v in all_variants_flat if v["generation_tier"] == tier]
        cross_by_gen_tier[tier] = _summarize_direction_group(sub_t, exp_sign)

    summary = {
        "mode": "stage2_final" if source_pretest_map is not None else "stage1_preliminary_old_pretest",
        "n_sources": len(items),
        "n_train": len(by_split["train"]),
        "n_dev": len(by_split["dev"]),
        "n_variants": len(all_variants_flat),
        "orig_floor_le_1_16_total": sum(
            1 for it in items if (it["n_orig"] / it["n_orig_scored"]) <= (1.0 / 16.0 + 1e-9)
        ),
        "orig_ceil_ge_15_16_total": sum(
            1 for it in items if (it["n_orig"] / it["n_orig_scored"]) >= (15.0 / 16.0 - 1e-9)
        ),
        "routing_tables": routing_tables,
        "direction_crosstab": {
            "by_source_direction_all_variants": cross_by_source_dir_all,
            "by_source_direction_self_generated_only": cross_by_source_dir_self,
            "by_source_direction_paper_transfer_only": cross_by_source_dir_paper,
            "by_variant_generation_tier": cross_by_gen_tier,
        },
    }
    return summary, old_table_sha


def format_inventory_report(
    summary: Dict[str, Any],
    *,
    measured_sha256: str,
    old_table_sha256: str = "",
    source_pretest_sha256: str = "",
    hybrid_grader_sha256: str = CANONICAL_HYBRID_GRADER_SHA256,
) -> str:
    lines: List[str] = []
    mode_label = (
        "阶段 2 最终清点（bf16 新原题前测）"
        if summary["mode"] == "stage2_final"
        else "阶段 1 初步清点（旧原题前测 pre_solve_rate + T-023 bf16 重测变体）"
    )
    lines.append(f"=== T-031 Route C Inventory: {mode_label} ===")
    lines.append(
        f"n_sources={summary['n_sources']} (train={summary['n_train']}, dev={summary['n_dev']}), "
        f"n_variants={summary['n_variants']}, orig_floor_le_1_16={summary['orig_floor_le_1_16_total']}, "
        f"orig_ceil_ge_15_16={summary['orig_ceil_ge_15_16_total']}"
    )
    lines.append(f"measured_jsonl_sha256={measured_sha256}")
    if old_table_sha256:
        lines.append(f"old_pretest_table_sha256={old_table_sha256}")
    if source_pretest_sha256:
        lines.append(f"source_pretest_sha256={source_pretest_sha256}")
    lines.append(f"hybrid_grader_sha256={hybrid_grader_sha256}")

    for split_name in ("all", "train", "dev"):
        lines.append("")
        lines.append(f"--- Routing Table ({split_name}) ---")
        lines.append(
            "| tau | m | easy_total | easy_usable | easy_ceil(>=15/16) | hard_total | hard_usable | hard_floor(<=1/16) | total_usable | gap_vs_1600 | net_excl_floor_ceil |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in summary["routing_tables"][split_name]:
            lines.append(
                f"| {r['tau']:.1f} | {r['margin']:.1f} | {r['easy_total']} | {r['easy_usable']} | "
                f"{r['easy_usable_all_qualifying_harder_ceil_ge_15_16']} (strict:{r['easy_all_strictly_harder_ceil_ge_15_16']}) | "
                f"{r['hard_total']} | {r['hard_usable']} | "
                f"{r['hard_usable_orig_floor_le_1_16']} (all:{r['hard_all_orig_floor_le_1_16']}) | "
                f"{r['total_usable']} | {r['gap_vs_1600']:+d} | {r['usable_excl_floor_and_ceil']} |"
            )

    lines.append("")
    lines.append("--- Direction Tag Cross-Tab (by source direction: all / self_generated / paper_transfer) ---")
    for sub_key in (
        "by_source_direction_all_variants",
        "by_source_direction_self_generated_only",
        "by_source_direction_paper_transfer_only",
        "by_variant_generation_tier",
    ):
        lines.append(f"[{sub_key}]")
        lines.append("| tag | n | <=-1.0 | <=-0.5 | <0 | ==0 | >0 | >=+0.5 | >=+1.0 | mean_db | med_db | match(<0 or >0) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for lbl, rec in summary["direction_crosstab"][sub_key].items():
            mc = rec["matched_strict_count"]
            mr = rec["matched_strict_rate"]
            match_str = f"{mc}/{rec['n_variants']} ({100.0 * mr:.2f}%)" if mc is not None and mr is not None else "—"
            lines.append(
                f"| {lbl} | {rec['n_variants']} | {rec['delta_b_le_neg1_0']} | {rec['delta_b_le_neg0_5']} | "
                f"{rec['delta_b_lt_0']} | {rec['delta_b_eq_0']} | {rec['delta_b_gt_0']} | "
                f"{rec['delta_b_ge_pos0_5']} | {rec['delta_b_ge_pos1_0']} | "
                f"{rec['mean_delta_b']} | {rec['median_delta_b']} | {match_str} |"
            )

    # Pick tau=0.4, m=0.5 (all) as compact machine-readable headline indicator
    headline = next(
        (r for r in summary["routing_tables"]["all"] if abs(r["tau"] - 0.4) < 1e-9 and abs(r["margin"] - 0.5) < 1e-9),
        summary["routing_tables"]["all"][0],
    )
    lines.append("")
    lines.append("status: STAGE1_DONE" if summary["mode"] != "stage2_final" else "status: STAGE2_INVENTORY_DONE")
    lines.append("rc: 0")
    lines.append("calls: 0")
    lines.append("failures: 0")
    lines.append(f"n_sources: {summary['n_sources']}")
    lines.append(f"n_variants: {summary['n_variants']}")
    lines.append(f"usable_tau04_m05: {headline['total_usable']}")
    lines.append(f"measured_jsonl_sha256: {measured_sha256}")
    if old_table_sha256:
        lines.append(f"old_pretest_table_sha256: {old_table_sha256}")
    if source_pretest_sha256:
        lines.append(f"source_pretest_sha256: {source_pretest_sha256}")
    return "\n".join(lines) + "\n"


def run_inventory(
    *,
    measured_jsonl: Path,
    out_dir: Path,
    source_pretest_jsonl: Optional[Path] = None,
    expected_measured_sha256: Optional[str] = EXPECTED_T023_MEASURED_SHA256,
    expected_old_table_sha256: Optional[str] = EXPECTED_OLD_SOURCE_PRETEST_TABLE_SHA256,
    expected_source_pretest_sha256: Optional[str] = None,
    allow_custom_sha256: bool = False,
) -> Dict[str, Any]:
    hybrid_sha = assert_canonical_hybrid_grader()
    actual_measured_sha = sha256_file(measured_jsonl)
    if (not allow_custom_sha256) and expected_measured_sha256:
        if actual_measured_sha != expected_measured_sha256.strip().lower():
            raise ValueError(
                f"measured_jsonl SHA-256 mismatch: got={actual_measured_sha} expected={expected_measured_sha256}"
            )

    measured_rows: List[Dict[str, Any]] = []
    with measured_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                measured_rows.append(json.loads(line))

    sp_map = None
    sp_sha = ""
    if source_pretest_jsonl is not None:
        sp_map, sp_sha = _load_source_pretest_map(
            source_pretest_jsonl,
            expected_sha256=None if allow_custom_sha256 else expected_source_pretest_sha256,
        )

    summary, old_table_sha = compute_inventory(
        measured_rows,
        source_pretest_map=sp_map,
    )
    if (
        sp_map is None
        and (not allow_custom_sha256)
        and expected_old_table_sha256
        and old_table_sha != expected_old_table_sha256.strip().lower()
    ):
        raise ValueError(
            f"Old source pretest table SHA-256 mismatch: got={old_table_sha} expected={expected_old_table_sha256}"
        )

    summary["measured_jsonl_sha256"] = actual_measured_sha
    summary["old_pretest_table_sha256"] = old_table_sha
    summary["source_pretest_sha256"] = sp_sha
    summary["hybrid_grader_sha256"] = hybrid_sha

    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    json_path = out_dir / "inventory_summary.json"
    txt_path = out_dir / "inventory_summary.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    txt = format_inventory_report(
        summary,
        measured_sha256=actual_measured_sha,
        old_table_sha256=old_table_sha,
        source_pretest_sha256=sp_sha,
        hybrid_grader_sha256=hybrid_sha,
    )
    txt_path.write_text(txt, encoding="utf-8")
    os.chmod(json_path, 0o600)
    os.chmod(txt_path, 0o600)
    return summary


def main() -> int:
    default_base = os.environ.get(
        "PRO5000_BASE", str(Path.home() / "bzhang" / "a_group_pro5000")
    )
    default_measured = (
        Path(default_base)
        / "runs"
        / "t023_stage2_20261125_163319"
        / "routeA_stage2_measured.jsonl"
    )
    ap = argparse.ArgumentParser(
        description="T-031 Route C difficulty-routing inventory (CPU-only, zero text)."
    )
    ap.add_argument("--measured-jsonl", type=Path, default=default_measured)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-pretest-jsonl", type=Path, default=None)
    ap.add_argument("--expected-measured-sha256", type=str, default=EXPECTED_T023_MEASURED_SHA256)
    ap.add_argument(
        "--expected-old-table-sha256",
        type=str,
        default=EXPECTED_OLD_SOURCE_PRETEST_TABLE_SHA256,
    )
    ap.add_argument("--expected-source-pretest-sha256", type=str, default=None)
    ap.add_argument("--allow-custom-sha256", action="store_true")
    args = ap.parse_args()

    run_inventory(
        measured_jsonl=args.measured_jsonl,
        out_dir=args.out_dir,
        source_pretest_jsonl=args.source_pretest_jsonl,
        expected_measured_sha256=args.expected_measured_sha256,
        expected_old_table_sha256=args.expected_old_table_sha256,
        expected_source_pretest_sha256=args.expected_source_pretest_sha256,
        allow_custom_sha256=args.allow_custom_sha256,
    )
    print((args.out_dir / "inventory_summary.txt").read_text(encoding="utf-8"), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
