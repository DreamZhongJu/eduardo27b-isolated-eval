#!/usr/bin/env python3
"""Route C difficulty-routing inventory (T-031 Stage 1 & Stage 2, Card v1 / Inbox #2).

Inbox #2 rules implemented here:
1. Floor problems are NOT excluded — only reported separately.
2. Hard-branch floor looks at the ENTIRE post-test ladder:
   `orig` + all remaining variants other than `v_taught` are ALL <= 1/16.
3. Unroutable sources fall back to Route A (teach orig, post-test all variants).
   Main pool matches T-026: `train` after removing ceiling groups (`1,895` in T-026)
   plus `dev` (`100`) = `1,995`, alongside `all_2100` for card-level verification.
4. Hard branch teaches the variant with `delta_b <= -m` that is CLOSEST to the
   original problem (i.e. maximum `b_var`).
5. When `source_pretest_jsonl` is provided (Stage 2), also reports old-vs-new
   branch switching counts and 3-state route transitions.

PRIVACY GUARANTEE:
- Never prints or writes problem statements, reference answers, or student responses.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
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
CEILING_MIN_CORRECT = 15


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


def is_t026_ceiling_group(variants: Sequence[Dict[str, Any]]) -> bool:
    """Return True iff every variant in the source group has n_correct_hybrid >= 15."""
    if not variants:
        return False
    return all(int(v["n_correct_hybrid"]) >= CEILING_MIN_CORRECT for v in variants)


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


def _classify_route_state(
    it: Dict[str, Any],
    tau: float,
    margin: float,
    *,
    use_old: bool = False,
) -> str:
    p_orig = float(it["old_p_orig"] if use_old else it["p_orig"])
    b_orig = float(it["old_b_orig"] if use_old else it["b_orig"])
    variants = it["variants"]
    if p_orig >= tau:
        if any((float(v["b_var"]) - b_orig) >= margin for v in variants):
            return "easy_routed"
        return "fallback_routeA"
    if any((float(v["b_var"]) - b_orig) <= -margin for v in variants):
        return "hard_routed"
    return "fallback_routeA"


def _evaluate_tau_margin(
    items: Sequence[Dict[str, Any]],
    tau: float,
    margin: float,
) -> Dict[str, Any]:
    easy_routed = 0
    easy_routed_ceil = 0
    easy_fallback_routeA = 0

    hard_routed = 0
    hard_routed_ladder_floor = 0
    hard_routed_orig_only_floor = 0
    hard_fallback_routeA = 0

    level_1 = 0
    level_2 = 0
    level_3plus = 0
    taught_origin_counts: collections.Counter[str] = collections.Counter()
    taught_delta_bs: List[float] = []

    for it in items:
        p_orig = float(it["p_orig"])
        is_orig_floor = (it["n_orig"] / it["n_orig_scored"]) <= (1.0 / 16.0 + 1e-9)
        variants = it["variants"]

        if p_orig >= tau:
            qualifying = [v for v in variants if v["delta_b"] >= margin]
            if qualifying:
                easy_routed += 1
                if all(v["is_ceil_ge_15_16"] for v in qualifying):
                    easy_routed_ceil += 1
            else:
                easy_fallback_routeA += 1
        else:
            qualifying_pairs = [
                (idx, v) for idx, v in enumerate(variants) if v["delta_b"] <= -margin
            ]
            if qualifying_pairs:
                hard_routed += 1
                # Decision 4: pick the variant in delta_b <= -m closest to original (max b_var).
                taught_idx, v_taught = min(
                    qualifying_pairs,
                    key=lambda pair: (-float(pair[1]["b_var"]), str(pair[1]["variant_id"]), pair[0]),
                )
                taught_origin_counts[str(v_taught["origin"])] += 1
                taught_delta_bs.append(float(v_taught["delta_b"]))

                rem_vars = [v for idx, v in enumerate(variants) if idx != taught_idx]
                n_levels = 1 + len(rem_vars)
                if n_levels == 1:
                    level_1 += 1
                elif n_levels == 2:
                    level_2 += 1
                else:
                    level_3plus += 1

                # Decision 2: entire post-test ladder (orig + all untaught variants) <= 1/16
                if is_orig_floor:
                    hard_routed_orig_only_floor += 1
                    if all(v["is_floor_le_1_16"] for v in rem_vars):
                        hard_routed_ladder_floor += 1
            else:
                hard_fallback_routeA += 1

    n_sources = len(items)
    fallback_routeA = easy_fallback_routeA + hard_fallback_routeA
    assert easy_routed + hard_routed + fallback_routeA == n_sources
    assert level_1 + level_2 + level_3plus == hard_routed

    return {
        "tau": tau,
        "margin": margin,
        "n_sources": n_sources,
        "easy_routed": easy_routed,
        "easy_routed_all_qualifying_harder_ceil_ge_15_16": easy_routed_ceil,
        "hard_routed": hard_routed,
        "hard_routed_ladder_floor_le_1_16": hard_routed_ladder_floor,
        "hard_routed_orig_only_floor_le_1_16": hard_routed_orig_only_floor,
        "fallback_routeA": fallback_routeA,
        "easy_fallback_routeA": easy_fallback_routeA,
        "hard_fallback_routeA": hard_fallback_routeA,
        "hard_routed_posttest_levels": {
            "1": level_1,
            "2": level_2,
            "3+": level_3plus,
        },
        "hard_routed_taught_origin": dict(sorted(taught_origin_counts.items())),
        "hard_routed_taught_mean_delta_b": round(sum(taught_delta_bs) / len(taught_delta_bs), 6)
        if taught_delta_bs
        else None,
    }


def _compute_switching_stats(
    items: Sequence[Dict[str, Any]],
    taus: Sequence[float],
    margins: Sequence[float],
) -> Optional[Dict[str, Any]]:
    if not items or any(it.get("old_p_orig") is None for it in items):
        return None

    by_tau: List[Dict[str, Any]] = []
    for tau in taus:
        old_hard_new_easy = 0
        old_easy_new_hard = 0
        stayed_hard = 0
        stayed_easy = 0
        for it in items:
            old_hard = float(it["old_p_orig"]) < tau
            new_hard = float(it["p_orig"]) < tau
            if old_hard and not new_hard:
                old_hard_new_easy += 1
            elif (not old_hard) and new_hard:
                old_easy_new_hard += 1
            elif old_hard and new_hard:
                stayed_hard += 1
            else:
                stayed_easy += 1
        assert old_hard_new_easy + old_easy_new_hard + stayed_hard + stayed_easy == len(items)
        by_tau.append(
            {
                "tau": float(tau),
                "n_sources": len(items),
                "old_hard_to_new_easy": old_hard_new_easy,
                "old_easy_to_new_hard": old_easy_new_hard,
                "total_branch_switched": old_hard_new_easy + old_easy_new_hard,
                "stayed_hard": stayed_hard,
                "stayed_easy": stayed_easy,
            }
        )

    by_tau_margin: List[Dict[str, Any]] = []
    for tau in taus:
        for margin in margins:
            trans: collections.Counter[str] = collections.Counter()
            switched = 0
            for it in items:
                s_old = _classify_route_state(it, float(tau), float(margin), use_old=True)
                s_new = _classify_route_state(it, float(tau), float(margin), use_old=False)
                trans[f"{s_old}->{s_new}"] += 1
                if s_old != s_new:
                    switched += 1
            by_tau_margin.append(
                {
                    "tau": float(tau),
                    "margin": float(margin),
                    "n_sources": len(items),
                    "total_route_switched": switched,
                    "transitions": dict(sorted(trans.items())),
                }
            )

    return {"by_tau_branch_switch": by_tau, "by_tau_margin_route_switch": by_tau_margin}


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


def _compute_crosstab(variants_flat: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
        sub_all = [v for v in variants_flat if v["source_direction"] == lbl]
        sub_self = [v for v in sub_all if v["origin"] == "self_generated"]
        sub_paper = [v for v in sub_all if v["origin"] == "paper_transfer"]
        cross_by_source_dir_all[lbl] = _summarize_direction_group(sub_all, exp_sign)
        cross_by_source_dir_self[lbl] = _summarize_direction_group(sub_self, exp_sign)
        cross_by_source_dir_paper[lbl] = _summarize_direction_group(sub_paper, exp_sign)

    cross_by_gen_tier: Dict[str, Any] = {}
    for tier in ("easier", "nearby", "harder", "paper_transfer"):
        exp_sign = _expected_sign_for_label(tier)
        sub_t = [v for v in variants_flat if v["generation_tier"] == tier]
        cross_by_gen_tier[tier] = _summarize_direction_group(sub_t, exp_sign)

    return {
        "by_source_direction_all_variants": cross_by_source_dir_all,
        "by_source_direction_self_generated_only": cross_by_source_dir_self,
        "by_source_direction_paper_transfer_only": cross_by_source_dir_paper,
        "by_variant_generation_tier": cross_by_gen_tier,
    }


def compute_inventory(
    measured_rows: Sequence[Dict[str, Any]],
    *,
    source_pretest_map: Optional[Dict[str, Dict[str, Any]]] = None,
    taus: Sequence[float] = DEFAULT_TAUS,
    margins: Sequence[float] = DEFAULT_MARGINS,
    enforce_t026_counts: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Compute Route C routing inventory (Card v1 / Inbox #2, pure numbers)."""
    items: List[Dict[str, Any]] = []
    old_table_tuples: List[Tuple[str, int, int]] = []
    seen_sids = set()

    for row in measured_rows:
        sid = str(row["source_id"])
        if sid in seen_sids:
            raise ValueError(f"Duplicate source_id in measured_jsonl: {sid}")
        seen_sids.add(sid)
        split = str(row["split"])
        direction = str(row["direction"])

        old_rate = row.get("source_pre_solve_rate_old")
        old_p_orig: Optional[float] = None
        old_b_orig: Optional[float] = None
        old_k16: Optional[int] = None
        if old_rate is not None:
            old_rate_f = float(old_rate)
            old_k16 = int(round(old_rate_f * 16))
            if abs(old_rate_f - old_k16 / 16.0) > 1e-9:
                raise ValueError(f"Non-k/16 source_pre_solve_rate_old for {sid}: {old_rate_f}")
            old_p_orig, old_b_orig = compute_smoothed_p_and_b(old_k16, 16)
            old_table_tuples.append((sid, old_k16, 16))

        if source_pretest_map is not None:
            if sid not in source_pretest_map:
                raise ValueError(f"source_id {sid} missing from source_pretest_jsonl")
            sp = source_pretest_map[sid]
            n_orig = int(sp["n_correct"])
            n_orig_scored = int(sp["n_scored"])
            p_orig = float(sp["p_orig"])
            b_orig = float(sp["b_orig"])
        else:
            if old_k16 is None or old_p_orig is None or old_b_orig is None:
                raise ValueError(f"Missing source_pre_solve_rate_old for {sid}")
            n_orig = old_k16
            n_orig_scored = 16
            p_orig = old_p_orig
            b_orig = old_b_orig

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
            is_floor = (n_hyb / n_scored) <= (1.0 / 16.0 + 1e-9)
            var_entries.append(
                {
                    "source_id": sid,
                    "variant_id": str(var["variant_id"]),
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
                    "is_floor_le_1_16": is_floor,
                }
            )

        is_train_ceil = (split == "train") and is_t026_ceiling_group(row["variants"])
        items.append(
            {
                "source_id": sid,
                "split": split,
                "direction": direction,
                "n_orig": n_orig,
                "n_orig_scored": n_orig_scored,
                "p_orig": p_orig,
                "b_orig": b_orig,
                "old_p_orig": old_p_orig,
                "old_b_orig": old_b_orig,
                "is_t026_train_ceiling": is_train_ceil,
                "variants": var_entries,
            }
        )

    if source_pretest_map is not None and len(source_pretest_map) != len(items):
        raise ValueError(
            f"source_pretest_jsonl count ({len(source_pretest_map)}) != measured_jsonl count ({len(items)})"
        )

    old_table_sha = (
        compute_source_pretest_table_sha256(old_table_tuples)
        if len(old_table_tuples) == len(items)
        else ""
    )

    by_scope: Dict[str, List[Dict[str, Any]]] = {
        "train_t026": [it for it in items if it["split"] == "train" and not it["is_t026_train_ceiling"]],
        "dev": [it for it in items if it["split"] == "dev"],
        "t026_pool": [
            it
            for it in items
            if (it["split"] == "train" and not it["is_t026_train_ceiling"]) or it["split"] == "dev"
        ],
        "train_all": [it for it in items if it["split"] == "train"],
        "all_2100": items,
    }
    removed_train_ceiling = len(by_scope["train_all"]) - len(by_scope["train_t026"])

    if enforce_t026_counts:
        if (
            len(by_scope["train_t026"]) != 1895
            or len(by_scope["dev"]) != 100
            or len(by_scope["t026_pool"]) != 1995
            or len(by_scope["all_2100"]) != 2100
            or removed_train_ceiling != 105
        ):
            raise ValueError(
                f"Unexpected T-026 pool sizes: train_t026={len(by_scope['train_t026'])}, "
                f"dev={len(by_scope['dev'])}, removed={removed_train_ceiling}, all={len(by_scope['all_2100'])}"
            )

    routing_tables: Dict[str, List[Dict[str, Any]]] = {}
    switching_stats: Dict[str, Any] = {}
    for scope_name, sub_items in by_scope.items():
        rows_out: List[Dict[str, Any]] = []
        for tau in taus:
            for margin in margins:
                rows_out.append(_evaluate_tau_margin(sub_items, float(tau), float(margin)))
        routing_tables[scope_name] = rows_out
        if source_pretest_map is not None:
            sw = _compute_switching_stats(sub_items, taus, margins)
            if sw is not None:
                switching_stats[scope_name] = sw

    t026_vars_flat = [v for it in by_scope["t026_pool"] for v in it["variants"]]
    all_vars_flat = [v for it in by_scope["all_2100"] for v in it["variants"]]

    summary: Dict[str, Any] = {
        "mode": "stage2_final" if source_pretest_map is not None else "stage1_preliminary_old_pretest",
        "pool_counts": {
            "train_t026_sources": len(by_scope["train_t026"]),
            "train_t026_variants": sum(len(it["variants"]) for it in by_scope["train_t026"]),
            "removed_train_ceiling_sources": removed_train_ceiling,
            "dev_sources": len(by_scope["dev"]),
            "dev_variants": sum(len(it["variants"]) for it in by_scope["dev"]),
            "t026_pool_sources": len(by_scope["t026_pool"]),
            "t026_pool_variants": len(t026_vars_flat),
            "all_2100_sources": len(by_scope["all_2100"]),
            "all_2100_variants": len(all_vars_flat),
        },
        "orig_floor_le_1_16_t026_pool": sum(
            1 for it in by_scope["t026_pool"] if (it["n_orig"] / it["n_orig_scored"]) <= (1.0 / 16.0 + 1e-9)
        ),
        "orig_ceil_ge_15_16_t026_pool": sum(
            1 for it in by_scope["t026_pool"] if (it["n_orig"] / it["n_orig_scored"]) >= (15.0 / 16.0 - 1e-9)
        ),
        "routing_tables": routing_tables,
        "switching_stats": switching_stats,
        "direction_crosstab_t026_pool": _compute_crosstab(t026_vars_flat),
        "direction_crosstab_all_2100": _compute_crosstab(all_vars_flat),
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
    pc = summary["pool_counts"]
    lines: List[str] = []
    mode_label = (
        "阶段 2 最终清点（bf16 新原题前测，收件箱第 2 条新口径）"
        if summary["mode"] == "stage2_final"
        else "阶段 1 初步清点（旧原题前测 + 收件箱第 2 条新口径）"
    )
    lines.append(f"=== T-031 Route C Inventory: {mode_label} ===")
    lines.append(
        f"t026_pool={pc['t026_pool_sources']} (train_t026={pc['train_t026_sources']}, "
        f"removed_train_ceiling={pc['removed_train_ceiling_sources']}, dev={pc['dev_sources']}), "
        f"t026_variants={pc['t026_pool_variants']}, all_2100={pc['all_2100_sources']} ({pc['all_2100_variants']} variants)"
    )
    lines.append(f"measured_jsonl_sha256={measured_sha256}")
    if old_table_sha256:
        lines.append(f"old_pretest_table_sha256={old_table_sha256}")
    if source_pretest_sha256:
        lines.append(f"source_pretest_sha256={source_pretest_sha256}")
    lines.append(f"hybrid_grader_sha256={hybrid_grader_sha256}")

    for scope_name in ("train_t026", "dev", "t026_pool", "all_2100"):
        lines.append("")
        lines.append(f"--- Routing Table ({scope_name}, n={summary['routing_tables'][scope_name][0]['n_sources']}) ---")
        lines.append(
            "| tau | m | easy_routed (ceil>=15/16) | hard_routed | hard_ladder_floor(<=1/16) [orig_only] | fallback_routeA (easy+hard) | posttest_levels (1/2/3+) | total_check |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|---:|")
        for r in summary["routing_tables"][scope_name]:
            lv = r["hard_routed_posttest_levels"]
            chk = r["easy_routed"] + r["hard_routed"] + r["fallback_routeA"]
            lines.append(
                f"| {r['tau']:.1f} | {r['margin']:.1f} | "
                f"{r['easy_routed']} ({r['easy_routed_all_qualifying_harder_ceil_ge_15_16']}) | "
                f"{r['hard_routed']} | "
                f"{r['hard_routed_ladder_floor_le_1_16']} [{r['hard_routed_orig_only_floor_le_1_16']}] | "
                f"{r['fallback_routeA']} ({r['easy_fallback_routeA']}+{r['hard_fallback_routeA']}) | "
                f"{lv['1']}/{lv['2']}/{lv['3+']} | {chk} |"
            )

    if summary.get("switching_stats"):
        for scope_name in ("train_t026", "dev", "t026_pool", "all_2100"):
            sw = summary["switching_stats"].get(scope_name)
            if not sw:
                continue
            lines.append("")
            lines.append(f"--- Branch Switch Old->New ({scope_name}) ---")
            lines.append("| tau | n | old_hard->new_easy | old_easy->new_hard | total_switched | stayed_hard | stayed_easy |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            for b in sw["by_tau_branch_switch"]:
                lines.append(
                    f"| {b['tau']:.1f} | {b['n_sources']} | {b['old_hard_to_new_easy']} | "
                    f"{b['old_easy_to_new_hard']} | {b['total_branch_switched']} | "
                    f"{b['stayed_hard']} | {b['stayed_easy']} |"
                )

    lines.append("")
    lines.append("--- Direction Tag Cross-Tab (t026_pool) ---")
    for sub_key in (
        "by_source_direction_all_variants",
        "by_source_direction_self_generated_only",
        "by_source_direction_paper_transfer_only",
        "by_variant_generation_tier",
    ):
        lines.append(f"[{sub_key}]")
        lines.append("| tag | n | <=-1.0 | <=-0.5 | <0 | ==0 | >0 | >=+0.5 | >=+1.0 | mean_db | med_db | match(<0 or >0) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for lbl, rec in summary["direction_crosstab_t026_pool"][sub_key].items():
            mc = rec["matched_strict_count"]
            mr = rec["matched_strict_rate"]
            match_str = f"{mc}/{rec['n_variants']} ({100.0 * mr:.2f}%)" if mc is not None and mr is not None else "—"
            lines.append(
                f"| {lbl} | {rec['n_variants']} | {rec['delta_b_le_neg1_0']} | {rec['delta_b_le_neg0_5']} | "
                f"{rec['delta_b_lt_0']} | {rec['delta_b_eq_0']} | {rec['delta_b_gt_0']} | "
                f"{rec['delta_b_ge_pos0_5']} | {rec['delta_b_ge_pos1_0']} | "
                f"{rec['mean_delta_b']} | {rec['median_delta_b']} | {match_str} |"
            )

    hl = next(
        (r for r in summary["routing_tables"]["train_t026"] if abs(r["tau"] - 0.4) < 1e-9 and abs(r["margin"] - 0.5) < 1e-9),
        summary["routing_tables"]["train_t026"][0],
    )
    lines.append("")
    lines.append("status: STAGE1_DONE" if summary["mode"] != "stage2_final" else "status: STAGE2_INVENTORY_DONE")
    lines.append("rc: 0")
    lines.append("calls: 0")
    lines.append("failures: 0")
    lines.append(f"t026_train_sources: {pc['train_t026_sources']}")
    lines.append(f"dev_sources: {pc['dev_sources']}")
    lines.append(f"t026_train_easy_tau04_m05: {hl['easy_routed']}")
    lines.append(f"t026_train_hard_tau04_m05: {hl['hard_routed']}")
    lines.append(f"t026_train_fallback_tau04_m05: {hl['fallback_routeA']}")
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
        enforce_t026_counts=(not allow_custom_sha256),
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
