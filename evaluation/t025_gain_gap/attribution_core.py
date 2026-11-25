#!/usr/bin/env python3
"""Core offline attribution calculations and pre-registered decision rules for T-025.

Implements all 7 candidate causes (A-G) strictly following:
  - 组织/参考/增益差分析预案.md (Section 2 pre-registered decision rules, unmodified)
  - 组织/任务/T-025_T009增益差归因_离线判定.md

Guarantees:
  - Pure CPU offline analysis; never loads models onto GPUs and never calls Agnes.
  - All confidence intervals use 10,000 bootstrap draws with base seed 42.
  - Never emits source_id, row_index, problem text, answers, or student responses.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
PAIRED_DIR = REPO_ROOT / "evaluation" / "paired_gain"
PRO5000_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
for _p in (str(HERE), str(PRO5000_DIR), str(BASE4B_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paired_gain as pg  # noqa: E402
from summarize_base4b import (  # noqa: E402
    DIFFICULTY_ORDER,
    GROUP_BY_DIFFICULTY,
    normalized_gain,
    p_value,
    paired_bootstrap_diff,
)

PAPER_27B_TRANSFER = 0.29
PAPER_27B_TRANSFER_CI = (0.25, 0.32)
PAPER_27B_SAME = 0.46
PAPER_27B_SAME_CI = (0.42, 0.51)

HIGH_PRE_FRAC = 15.0 / 16.0
AUTHOR_MAX_TRANSFER_PRE = 0.75
AUTHOR_MAX_SOLVE_GAP = 0.375
TOL = 1e-12

PRE_BIN_LABELS: tuple[str, ...] = (
    "0",
    "(0, 0.25]",
    "(0.25, 0.5]",
    "(0.5, 0.75]",
    "(0.75, 15/16)",
    "[15/16, 1]",
)


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive deterministic 64-bit seed independent of PYTHONHASHSEED."""
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_pre_bin(pre: float, *, tol: float = TOL) -> str:
    """Assign a pre-test solve rate in [0, 1] to one of the 6 pre-registered bins."""
    if not (-tol <= pre <= 1.0 + tol):
        raise ValueError(f"pre rate out of [0, 1]: {pre}")
    val = min(1.0, max(0.0, float(pre)))
    if val <= tol:
        return "0"
    if val <= 0.25 + tol:
        return "(0, 0.25]"
    if val <= 0.50 + tol:
        return "(0.25, 0.5]"
    if val <= 0.75 + tol:
        return "(0.5, 0.75]"
    if val < HIGH_PRE_FRAC - tol:
        return "(0.75, 15/16)"
    return "[15/16, 1]"


def passes_author_threshold(
    pre_transfer: float,
    pre_orig: float,
    *,
    max_transfer_pre: float = AUTHOR_MAX_TRANSFER_PRE,
    max_gap: float = AUTHOR_MAX_SOLVE_GAP,
    tol: float = TOL,
) -> bool:
    """Check author generation difficulty gate: pre_transfer <= 0.75 and |gap| <= 0.375."""
    return (float(pre_transfer) <= max_transfer_pre + tol) and (
        abs(float(pre_transfer) - float(pre_orig)) <= max_gap + tol
    )


def last_boxed_only_string(string: str) -> str | None:
    """Exact replica of upstream eduardo/utils.py:last_boxed_only_string."""
    if not string:
        return None
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx
    depth = 0
    right_idx: int | None = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right_idx = i
                break
        i += 1
    return string[idx : right_idx + 1] if right_idx is not None else None


def extract_author_boxed_pred(solution_str: str) -> tuple[str, bool]:
    """Return (extracted_pred, is_incorrect_format) matching eduardo/utils.py:grade_math_answer."""
    if not solution_str:
        return "", True
    boxed = last_boxed_only_string(solution_str[-200:]) or last_boxed_only_string(solution_str)
    if not boxed or not (boxed.startswith(r"\boxed{") and boxed.endswith("}")):
        return "", True
    pred = boxed[len(r"\boxed{") : -1]
    if pred == "":
        return "", True
    return pred, False


def bootstrap_mean_1d(
    values: Sequence[float], *, seed: int, draws: int = 10000
) -> dict[str, Any] | None:
    """Vectorized 95% percentile bootstrap CI for the mean of a 1D sample."""
    n = len(values)
    if n == 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk = max(1, min(draws, 5_000_000 // max(1, n)))
    boot = np.empty(draws, dtype=np.float64)
    off = 0
    while off < draws:
        cur = min(chunk, draws - off)
        idx = rng.integers(0, n, size=(cur, n))
        boot[off : off + cur] = arr[idx].mean(axis=1)
        off += cur
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "ci": [float(lo), float(hi)],
    }


def paired_diff_bootstrap(
    target_pairs: Sequence[tuple[str, float]],
    base_pairs: Sequence[tuple[str, float]],
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute paired (target - base) bootstrap CI exclusively over matched unit IDs."""
    t_map = dict(target_pairs)
    b_map = dict(base_pairs)
    common = sorted(set(t_map) & set(b_map))
    n_only_target = len(set(t_map) - set(b_map))
    n_only_base = len(set(b_map) - set(t_map))
    if not common:
        return {
            "n_paired": 0,
            "n_only_target": n_only_target,
            "n_only_base": n_only_base,
            "target_mean": None,
            "base_mean": None,
            "mean_diff": None,
            "ci": None,
            "p_gt_0": None,
        }
    t_arr = np.array([t_map[k] for k in common], dtype=np.float64)
    b_arr = np.array([b_map[k] for k in common], dtype=np.float64)
    d_arr = t_arr - b_arr
    n = len(common)
    rng = np.random.default_rng(seed)
    chunk = max(1, min(draws, 5_000_000 // max(1, n)))
    boot_t = np.empty(draws, dtype=np.float64)
    boot_b = np.empty(draws, dtype=np.float64)
    boot_d = np.empty(draws, dtype=np.float64)
    off = 0
    while off < draws:
        cur = min(chunk, draws - off)
        idx = rng.integers(0, n, size=(cur, n))
        boot_t[off : off + cur] = t_arr[idx].mean(axis=1)
        boot_b[off : off + cur] = b_arr[idx].mean(axis=1)
        boot_d[off : off + cur] = d_arr[idx].mean(axis=1)
        off += cur
    t_lo, t_hi = np.percentile(boot_t, [2.5, 97.5])
    b_lo, b_hi = np.percentile(boot_b, [2.5, 97.5])
    d_lo, d_hi = np.percentile(boot_d, [2.5, 97.5])
    return {
        "n_paired": int(n),
        "n_only_target": int(n_only_target),
        "n_only_base": int(n_only_base),
        "target_mean": float(t_arr.mean()),
        "target_ci": [float(t_lo), float(t_hi)],
        "base_mean": float(b_arr.mean()),
        "base_ci": [float(b_lo), float(b_hi)],
        "mean_diff": float(d_arr.mean()),
        "ci": [float(d_lo), float(d_hi)],
        "p_gt_0": float(np.mean(boot_d > 0.0)),
    }


def unpaired_diff_bootstrap(
    a_vals: Sequence[float],
    b_vals: Sequence[float],
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    """Independent two-sample bootstrap CI for mean(a_vals) - mean(b_vals)."""
    na, nb = len(a_vals), len(b_vals)
    if na == 0 or nb == 0:
        return {
            "n_a": int(na),
            "n_b": int(nb),
            "a_mean": float(np.mean(a_vals)) if na else None,
            "b_mean": float(np.mean(b_vals)) if nb else None,
            "mean_diff": None,
            "ci": None,
            "p_gt_0": None,
        }
    arr_a = np.asarray(a_vals, dtype=np.float64)
    arr_b = np.asarray(b_vals, dtype=np.float64)
    rng_a = np.random.default_rng(derive_seed(seed, "unpaired_a"))
    rng_b = np.random.default_rng(derive_seed(seed, "unpaired_b"))
    boot_a = np.empty(draws, dtype=np.float64)
    boot_b = np.empty(draws, dtype=np.float64)
    chunk_a = max(1, min(draws, 5_000_000 // max(1, na)))
    off = 0
    while off < draws:
        cur = min(chunk_a, draws - off)
        boot_a[off : off + cur] = arr_a[rng_a.integers(0, na, size=(cur, na))].mean(axis=1)
        off += cur
    chunk_b = max(1, min(draws, 5_000_000 // max(1, nb)))
    off = 0
    while off < draws:
        cur = min(chunk_b, draws - off)
        boot_b[off : off + cur] = arr_b[rng_b.integers(0, nb, size=(cur, nb))].mean(axis=1)
        off += cur
    boot_d = boot_a - boot_b
    a_lo, a_hi = np.percentile(boot_a, [2.5, 97.5])
    b_lo, b_hi = np.percentile(boot_b, [2.5, 97.5])
    d_lo, d_hi = np.percentile(boot_d, [2.5, 97.5])
    return {
        "n_a": int(na),
        "n_b": int(nb),
        "a_mean": float(arr_a.mean()),
        "a_ci": [float(a_lo), float(a_hi)],
        "b_mean": float(arr_b.mean()),
        "b_ci": [float(b_lo), float(b_hi)],
        "mean_diff": float(arr_a.mean() - arr_b.mean()),
        "ci": [float(d_lo), float(d_hi)],
        "p_gt_0": float(np.mean(boot_d > 0.0)),
    }


def compute_reweight_weights(
    item_bins: Sequence[str],
    target_bin_counts: dict[str, int],
) -> dict[str, Any]:
    """Compute normalized per-item weights matching target_bin_counts over shared support bins."""
    n_items = len(item_bins)
    total_target = sum(int(v) for v in target_bin_counts.values())
    if n_items == 0 or total_target <= 0:
        return {
            "weights": [],
            "covered_target_mass": 0.0,
            "covered_items": 0,
            "empty_ours_bins_with_target_mass": [],
            "zero_target_bins_with_ours_items": [],
            "bin_target_probs": {},
        }
    ours_counts = Counter(item_bins)
    target_probs = {b: int(target_bin_counts.get(b, 0)) / total_target for b in PRE_BIN_LABELS}
    empty_ours = [b for b in PRE_BIN_LABELS if target_probs[b] > 0 and ours_counts.get(b, 0) == 0]
    zero_target = [b for b in PRE_BIN_LABELS if target_probs[b] == 0 and ours_counts.get(b, 0) > 0]
    covered_mass = sum(target_probs[b] for b in PRE_BIN_LABELS if ours_counts.get(b, 0) > 0)
    if covered_mass <= 0.0:
        return {
            "weights": [0.0] * n_items,
            "covered_target_mass": 0.0,
            "covered_items": 0,
            "empty_ours_bins_with_target_mass": empty_ours,
            "zero_target_bins_with_ours_items": zero_target,
            "bin_target_probs": target_probs,
        }
    raw_w = [
        (target_probs[b] / ours_counts[b]) if (ours_counts[b] > 0 and target_probs[b] > 0) else 0.0
        for b in item_bins
    ]
    w_sum = sum(raw_w)
    norm_w = [w / w_sum for w in raw_w] if w_sum > 0 else [0.0] * n_items
    covered_items = sum(1 for w in norm_w if w > 0.0)
    return {
        "weights": norm_w,
        "covered_target_mass": float(covered_mass),
        "covered_items": int(covered_items),
        "empty_ours_bins_with_target_mass": empty_ours,
        "zero_target_bins_with_ours_items": zero_target,
        "bin_target_probs": target_probs,
    }


def _cluster_subset_and_reweight_bootstrap(
    gains: Sequence[float],
    keep_mask: Sequence[bool],
    item_bins: Sequence[str],
    author_bin_counts: dict[str, int],
    *,
    paper_ref: float = PAPER_27B_TRANSFER,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Source-level cluster bootstrap comparing full, filtered, and reweighted estimators."""
    n = len(gains)
    if n == 0:
        raise ValueError("Empty gains in _cluster_subset_and_reweight_bootstrap")
    g_arr = np.asarray(gains, dtype=np.float64)
    m_arr = np.asarray(keep_mask, dtype=bool)
    b_idx = np.array([PRE_BIN_LABELS.index(b) for b in item_bins], dtype=np.int64)

    total_author = sum(int(author_bin_counts.get(b, 0)) for b in PRE_BIN_LABELS)
    p_author = np.array(
        [int(author_bin_counts.get(b, 0)) / total_author if total_author > 0 else 0.0 for b in PRE_BIN_LABELS],
        dtype=np.float64,
    )

    all_mean = float(g_arr.mean())
    n_filt = int(m_arr.sum())
    filt_mean = float(g_arr[m_arr].mean()) if n_filt > 0 else None

    rw_all_info = compute_reweight_weights(item_bins, author_bin_counts)
    w_all = np.asarray(rw_all_info["weights"], dtype=np.float64)
    rw_all_mean = float(np.sum(w_all * g_arr)) if rw_all_info["covered_items"] > 0 else None

    filt_bins = [b for b, keep in zip(item_bins, m_arr) if keep]
    rw_filt_info = compute_reweight_weights(filt_bins, author_bin_counts) if n_filt > 0 else None
    w_filt_sub = np.asarray(rw_filt_info["weights"], dtype=np.float64) if rw_filt_info else np.array([])
    rw_filt_mean = (
        float(np.sum(w_filt_sub * g_arr[m_arr]))
        if (rw_filt_info and rw_filt_info["covered_items"] > 0)
        else None
    )

    rng = np.random.default_rng(seed)
    boot_all = np.empty(draws, dtype=np.float64)
    boot_filt: list[float] = []
    boot_diff_filt: list[float] = []
    boot_rw_all: list[float] = []
    boot_diff_rw_all: list[float] = []
    boot_rw_filt: list[float] = []
    boot_diff_rw_filt: list[float] = []

    chunk = max(1, min(draws, 2_000_000 // max(1, n)))
    off = 0
    while off < draws:
        cur = min(chunk, draws - off)
        idx = rng.integers(0, n, size=(cur, n))
        g_s = g_arr[idx]
        m_s = m_arr[idx]
        b_s = b_idx[idx]

        a_m = g_s.mean(axis=1)
        boot_all[off : off + cur] = a_m

        # Filtered subset on the same resampled sources
        f_cnt = m_s.sum(axis=1)
        f_sum = np.where(m_s, g_s, 0.0).sum(axis=1)
        for r_i in range(cur):
            am = float(a_m[r_i])
            if f_cnt[r_i] > 0:
                fm = float(f_sum[r_i] / f_cnt[r_i])
                boot_filt.append(fm)
                boot_diff_filt.append(fm - am)

            # Post-stratified bin reweighting on all resampled sources
            cnt_b = np.bincount(b_s[r_i], minlength=len(PRE_BIN_LABELS))
            sum_b = np.bincount(b_s[r_i], weights=g_s[r_i], minlength=len(PRE_BIN_LABELS))
            active = (cnt_b > 0) & (p_author > 0)
            if np.any(active):
                mass = float(p_author[active].sum())
                rw_m = float(np.sum(p_author[active] * (sum_b[active] / cnt_b[active])) / mass)
                boot_rw_all.append(rw_m)
                boot_diff_rw_all.append(rw_m - am)

            # Post-stratified bin reweighting on threshold-filtered resampled sources
            if f_cnt[r_i] > 0:
                bs_f = b_s[r_i][m_s[r_i]]
                gs_f = g_s[r_i][m_s[r_i]]
                cnt_bf = np.bincount(bs_f, minlength=len(PRE_BIN_LABELS))
                sum_bf = np.bincount(bs_f, weights=gs_f, minlength=len(PRE_BIN_LABELS))
                act_f = (cnt_bf > 0) & (p_author > 0)
                if np.any(act_f):
                    mass_f = float(p_author[act_f].sum())
                    rw_fm = float(np.sum(p_author[act_f] * (sum_bf[act_f] / cnt_bf[act_f])) / mass_f)
                    boot_rw_filt.append(rw_fm)
                    boot_diff_rw_filt.append(rw_fm - am)
        off += cur

    def _pack_est(pt: float | None, samples: list[float], n_eff: int) -> dict[str, Any] | None:
        if pt is None or not samples:
            return None
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {"n": int(n_eff), "mean": float(pt), "ci": [float(lo), float(hi)], "valid_draws": len(samples)}

    def _pack_diff(pt_new: float | None, diff_samples: list[float]) -> dict[str, Any] | None:
        if pt_new is None or not diff_samples:
            return None
        diff_pt = float(pt_new - all_mean)
        lo, hi = np.percentile(diff_samples, [2.5, 97.5])
        denom = paper_ref - all_mean
        gap_share = float(diff_pt / denom) if abs(denom) > 1e-12 else None
        return {
            "mean_diff": diff_pt,
            "ci": [float(lo), float(hi)],
            "p_gt_0": float(np.mean(np.asarray(diff_samples) > 0.0)),
            "gap_to_paper_029": float(denom),
            "gap_closed_share": gap_share,
            "valid_draws": len(diff_samples),
        }

    all_lo, all_hi = np.percentile(boot_all, [2.5, 97.5])
    return {
        "all": {"n": int(n), "mean": all_mean, "ci": [float(all_lo), float(all_hi)]},
        "filtered": _pack_est(filt_mean, boot_filt, n_filt),
        "diff_filtered_minus_all": _pack_diff(filt_mean, boot_diff_filt),
        "reweighted_all": {
            **(_pack_est(rw_all_mean, boot_rw_all, rw_all_info["covered_items"]) or {}),
            "covered_author_mass": rw_all_info["covered_target_mass"],
            "empty_ours_bins": rw_all_info["empty_ours_bins_with_target_mass"],
        }
        if rw_all_mean is not None
        else None,
        "diff_reweighted_all_minus_all": _pack_diff(rw_all_mean, boot_diff_rw_all),
        "filtered_reweighted": {
            **(_pack_est(rw_filt_mean, boot_rw_filt, rw_filt_info["covered_items"]) or {}),
            "covered_author_mass": rw_filt_info["covered_target_mass"],
            "empty_ours_bins": rw_filt_info["empty_ours_bins_with_target_mass"],
        }
        if (rw_filt_mean is not None and rw_filt_info is not None)
        else None,
        "diff_filtered_reweighted_minus_all": _pack_diff(rw_filt_mean, boot_diff_rw_filt),
    }


def judge_cause_a(tier_results_27b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply pre-registered Cause A rule from 增益差分析预案.md lines 45-48.

    - 确认: 过滤或重加权后, 某档(或多数档位)与论文 0.29 的差距缩小比例 >= 50%
            且提升量 95% CI 下界 > 0
    - 部分贡献: 差距缩小比例在 15% ~ 50% 之间
    - 排除: 三档的差距缩小比例均 < 15% (或提升量均 < +0.02)
    """
    best_share = -math.inf
    best_key = None
    any_confirm = False
    any_partial = False
    per_tier_summary: dict[str, Any] = {}

    for diff in DIFFICULTY_ORDER:
        tr = tier_results_27b[diff]
        candidates = [
            ("filtered", tr.get("diff_filtered_minus_all")),
            ("filtered_reweighted", tr.get("diff_filtered_reweighted_minus_all")),
            ("reweighted_all", tr.get("diff_reweighted_all_minus_all")),
        ]
        tier_best_share = -math.inf
        tier_best_diff = -math.inf
        tier_best_mode = None
        for mode, d in candidates:
            if not d or d.get("gap_closed_share") is None:
                continue
            sh = float(d["gap_closed_share"])
            dm = float(d["mean_diff"])
            ci_lo = float(d["ci"][0])
            if sh > tier_best_share:
                tier_best_share = sh
                tier_best_diff = dm
                tier_best_mode = mode
            if sh > best_share:
                best_share = sh
                best_key = f"{diff}.{mode}"
            if sh >= 0.50 - TOL and ci_lo > TOL and dm >= 0.02 - TOL:
                any_confirm = True
            elif sh >= 0.15 - TOL and dm >= 0.02 - TOL:
                any_partial = True
        per_tier_summary[diff] = {
            "best_mode": tier_best_mode,
            "best_gap_closed_share": tier_best_share if math.isfinite(tier_best_share) else None,
            "best_mean_diff": tier_best_diff if math.isfinite(tier_best_diff) else None,
        }

    if any_confirm:
        verdict = "确认"
    elif any_partial:
        verdict = "部分贡献"
    else:
        verdict = "排除"
    return {
        "verdict": verdict,
        "best_variant": best_key,
        "max_gap_closed_share": best_share if math.isfinite(best_share) else None,
        "per_tier": per_tier_summary,
    }


def compute_cause_a(
    entries_by_model: dict[str, list[dict[str, Any]]],
    author_rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute Cause A stratification, threshold filtering, and reweighting for 27B, base, and a_group.

    Each model entry in entries_by_model[model_name] has:
      {"difficulty": "easy"|"medium"|"hard", "source_id": str,
       "pre_orig": float, "pre_trans": float, "post_orig": float, "post_trans": float}
    author_rows has:
      {"row_index": int, "pre_same": float, "pre_transfer": float,
       "post_transfer_27b": float | None, "delta_transfer_27b": float | None}
    """
    author_bins: dict[str, list[dict[str, Any]]] = {b: [] for b in PRE_BIN_LABELS}
    author_pass_gate = 0
    author_ge_15_16 = 0
    for r in author_rows:
        pt = float(r["pre_transfer"])
        ps = float(r["pre_same"])
        b = assign_pre_bin(pt)
        author_bins[b].append(r)
        if passes_author_threshold(pt, ps):
            author_pass_gate += 1
        if pt >= HIGH_PRE_FRAC - TOL:
            author_ge_15_16 += 1

    author_bin_counts = {b: len(author_bins[b]) for b in PRE_BIN_LABELS}
    n_author = len(author_rows)
    author_dist_table: list[dict[str, Any]] = []
    for b in PRE_BIN_LABELS:
        items = author_bins[b]
        pres = [float(x["pre_transfer"]) for x in items]
        posts = [float(x["post_transfer_27b"]) for x in items if x.get("post_transfer_27b") is not None]
        deltas = [float(x["delta_transfer_27b"]) for x in items if x.get("delta_transfer_27b") is not None]
        author_dist_table.append(
            {
                "bin": b,
                "n": len(items),
                "share": (len(items) / n_author) if n_author else 0.0,
                "mean_pre": statistics.fmean(pres) if pres else None,
                "mean_post_27b": statistics.fmean(posts) if posts else None,
                "r1_27b": bootstrap_mean_1d(
                    deltas, seed=derive_seed(seed, "causeA_author_bin", b), draws=draws
                )
                if deltas
                else None,
            }
        )

    models_out: dict[str, Any] = {}
    for model_name, recs in entries_by_model.items():
        by_tier: dict[str, list[dict[str, Any]]] = {d: [] for d in DIFFICULTY_ORDER}
        for rec in recs:
            by_tier[rec["difficulty"]].append(rec)

        tier_out: dict[str, Any] = {}
        for diff in DIFFICULTY_ORDER:
            t_recs = sorted(by_tier[diff], key=lambda x: x["source_id"])
            if not t_recs:
                continue
            bins_map: dict[str, list[dict[str, Any]]] = {b: [] for b in PRE_BIN_LABELS}
            gains_all: list[float] = []
            keep_mask: list[bool] = []
            item_bins: list[str] = []
            n_pre_gt_075 = 0
            n_gap_gt_0375 = 0
            n_ge_15_16 = 0

            for r in t_recs:
                pt = float(r["pre_trans"])
                po = float(r["pre_orig"])
                post_t = float(r["post_trans"])
                g = normalized_gain(pt, post_t)
                b = assign_pre_bin(pt)
                keep = passes_author_threshold(pt, po)
                if pt > AUTHOR_MAX_TRANSFER_PRE + TOL:
                    n_pre_gt_075 += 1
                if abs(pt - po) > AUTHOR_MAX_SOLVE_GAP + TOL:
                    n_gap_gt_0375 += 1
                if pt >= HIGH_PRE_FRAC - TOL:
                    n_ge_15_16 += 1
                bins_map[b].append({**r, "gain": g})
                gains_all.append(g)
                keep_mask.append(keep)
                item_bins.append(b)

            strata_rows: list[dict[str, Any]] = []
            for b in PRE_BIN_LABELS:
                b_items = bins_map[b]
                b_pres = [float(x["pre_trans"]) for x in b_items]
                b_posts = [float(x["post_trans"]) for x in b_items]
                b_gains = [float(x["gain"]) for x in b_items]
                strata_rows.append(
                    {
                        "bin": b,
                        "n": len(b_items),
                        "share": len(b_items) / len(t_recs),
                        "mean_pre": statistics.fmean(b_pres) if b_pres else None,
                        "mean_post": statistics.fmean(b_posts) if b_posts else None,
                        "r1": bootstrap_mean_1d(
                            b_gains,
                            seed=derive_seed(seed, "causeA_bin", model_name, diff, b),
                            draws=draws,
                        ),
                    }
                )

            filt_recs = [r for r, k in zip(t_recs, keep_mask) if k]
            boot_comp = _cluster_subset_and_reweight_bootstrap(
                gains_all,
                keep_mask,
                item_bins,
                author_bin_counts,
                paper_ref=PAPER_27B_TRANSFER,
                seed=derive_seed(seed, "causeA_comp", model_name, diff),
                draws=draws,
            )
            tier_out[diff] = {
                "n_total": len(t_recs),
                "n_filtered": len(filt_recs),
                "n_rejected_total": len(t_recs) - len(filt_recs),
                "n_pre_gt_075": n_pre_gt_075,
                "n_gap_gt_0375": n_gap_gt_0375,
                "n_pre_ge_15_16": n_ge_15_16,
                "share_pre_ge_15_16": n_ge_15_16 / len(t_recs),
                "mean_pre_all": statistics.fmean(float(x["pre_trans"]) for x in t_recs),
                "mean_post_all": statistics.fmean(float(x["post_trans"]) for x in t_recs),
                "mean_pre_filtered": statistics.fmean(float(x["pre_trans"]) for x in filt_recs) if filt_recs else None,
                "mean_post_filtered": statistics.fmean(float(x["post_trans"]) for x in filt_recs) if filt_recs else None,
                "strata": strata_rows,
                **boot_comp,
            }
        models_out[model_name] = tier_out

    # Descriptive a_group - base paired difference on Cause A filtered sources
    ag_vs_base_filtered: dict[str, Any] = {}
    if "a_group" in entries_by_model and "base" in entries_by_model:
        ag_by_td = {(r["difficulty"], r["source_id"]): r for r in entries_by_model["a_group"]}
        bs_by_td = {(r["difficulty"], r["source_id"]): r for r in entries_by_model["base"]}
        for diff in DIFFICULTY_ORDER:
            ag_pairs = [
                (sid, normalized_gain(r["pre_trans"], r["post_trans"]))
                for (d, sid), r in sorted(ag_by_td.items())
                if d == diff and passes_author_threshold(r["pre_trans"], r["pre_orig"])
            ]
            bs_pairs = [
                (sid, normalized_gain(r["pre_trans"], r["post_trans"]))
                for (d, sid), r in sorted(bs_by_td.items())
                if d == diff and passes_author_threshold(r["pre_trans"], r["pre_orig"])
            ]
            ag_vs_base_filtered[diff] = paired_diff_bootstrap(
                ag_pairs,
                bs_pairs,
                seed=derive_seed(seed, "causeA_ag_vs_base_filt", diff),
                draws=draws,
            )

    decision = judge_cause_a(models_out["eduardo27b"])
    return {
        "author_494_transfer_distribution": {
            "n_total": n_author,
            "n_pass_author_threshold": author_pass_gate,
            "share_pass_author_threshold": (author_pass_gate / n_author) if n_author else 0.0,
            "n_pre_ge_15_16": author_ge_15_16,
            "share_pre_ge_15_16": (author_ge_15_16 / n_author) if n_author else 0.0,
            "bins": author_dist_table,
        },
        "models": models_out,
        "a_group_minus_base_on_filtered": ag_vs_base_filtered,
        "decision": decision,
    }


def normalize_problem_exact(text: str) -> str:
    """Unicode NFC + whitespace normalization for exact problem text matching."""
    norm = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", norm).strip()


def normalize_problem_loose(text: str) -> str:
    """Alphanumeric-only NFKC lowercase normalization for fast near-duplicate precheck."""
    norm = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[^a-z0-9]+", "", norm)


def match_sources_499_vs_494(
    ours_problems_by_sid: dict[str, str],
    author_problems_by_row: dict[int, str],
    *,
    approx_threshold: float = 0.90,
) -> dict[str, Any]:
    """Match self-built 499 original problems against author 494 original problems.

    Returns exact match mapping {source_id: row_index} plus sanitized counts ONLY.
    Never includes problem text, source_id, or row_index in the public summary dict!
    """
    ours_by_norm: dict[str, list[str]] = defaultdict(list)
    for sid, prob in sorted(ours_problems_by_sid.items()):
        ours_by_norm[normalize_problem_exact(prob)].append(sid)

    author_by_norm: dict[str, list[int]] = defaultdict(list)
    for ridx, prob in sorted(author_problems_by_row.items()):
        author_by_norm[normalize_problem_exact(prob)].append(ridx)

    exact_sid_to_row: dict[str, int] = {}
    ambiguous_norms = 0
    shared_norms = sorted(set(ours_by_norm) & set(author_by_norm))
    for key in shared_norms:
        sids = ours_by_norm[key]
        ridxs = author_by_norm[key]
        if len(sids) == 1 and len(ridxs) == 1:
            exact_sid_to_row[sids[0]] = ridxs[0]
        else:
            ambiguous_norms += 1
            for s, r in zip(sorted(sids), sorted(ridxs)):
                exact_sid_to_row[s] = r

    matched_sids = set(exact_sid_to_row.keys())
    matched_rows = set(exact_sid_to_row.values())
    rem_ours = [(sid, normalize_problem_exact(ours_problems_by_sid[sid])) for sid in sorted(ours_problems_by_sid) if sid not in matched_sids]
    rem_auth = [(ridx, normalize_problem_exact(author_problems_by_row[ridx])) for ridx in sorted(author_problems_by_row) if ridx not in matched_rows]

    # Approximate match count among remaining unmatched items (greedy 1-to-1 by similarity >= 0.90)
    loose_auth: dict[str, list[int]] = defaultdict(list)
    for ridx, p_norm in rem_auth:
        lk = normalize_problem_loose(p_norm)
        if lk:
            loose_auth[lk].append(ridx)

    approx_pairs: list[tuple[float, str, int]] = []
    for sid, s_norm in rem_ours:
        lk = normalize_problem_loose(s_norm)
        if lk and lk in loose_auth:
            for ridx in loose_auth[lk]:
                approx_pairs.append((0.999, sid, ridx))
            continue
        ls = len(s_norm)
        if ls == 0:
            continue
        for ridx, a_norm in rem_auth:
            la = len(a_norm)
            if la == 0 or min(ls, la) / max(ls, la) < approx_threshold:
                continue
            sm = difflib.SequenceMatcher(None, s_norm, a_norm)
            if sm.quick_ratio() < approx_threshold:
                continue
            ratio = sm.ratio()
            if ratio >= approx_threshold:
                approx_pairs.append((ratio, sid, ridx))

    approx_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_s: set[str] = set()
    used_r: set[int] = set()
    n_approx_extra = 0
    for _ratio, sid, ridx in approx_pairs:
        if sid not in used_s and ridx not in used_r:
            used_s.add(sid)
            used_r.add(ridx)
            n_approx_extra += 1

    n_ours = len(ours_problems_by_sid)
    n_auth = len(author_problems_by_row)
    n_exact = len(exact_sid_to_row)
    return {
        "exact_sid_to_row": exact_sid_to_row,
        "public_counts": {
            "n_ours_total": n_ours,
            "n_author_total": n_auth,
            "n_ours_unique_norms": len(ours_by_norm),
            "n_author_unique_norms": len(author_by_norm),
            "n_exact_overlap": n_exact,
            "n_ours_minus_author": n_ours - n_exact,
            "n_author_minus_ours": n_auth - n_exact,
            "n_ambiguous_norm_keys": ambiguous_norms,
            "n_approx_additional_ge_090": n_approx_extra,
            "n_overlap_with_approx": n_exact + n_approx_extra,
            "paired_eligible_ge_30": n_exact >= 30,
        },
    }


def judge_cause_b(
    tier_diffs: dict[str, dict[str, Any]],
    author_delta_transfer_mean: float,
) -> dict[str, Any]:
    """Apply pre-registered Cause B rule from 增益差分析预案.md lines 63-65 & T-025 line 48.

    - 确认: 各档 Δ_author - Δ_ours 的区间下界 > +0.05, 且作者 494 题 Δ_transfer >= 0.25
    - 部分贡献: Δ_author - Δ_ours 在 +0.02 ~ +0.05 之间 (或仅在部分档位成立)
    - 排除: 各档 Δ_author - Δ_ours 的 95% CI 均包含 0 (且点估计绝对值 < 0.02)
    """
    all_gt_005 = True
    any_partial = False
    all_excl = True
    for diff in DIFFICULTY_ORDER:
        d = tier_diffs.get(diff)
        if not d or d.get("mean_diff") is None or d.get("ci") is None:
            all_gt_005 = False
            all_excl = False
            continue
        md = float(d["mean_diff"])
        lo, hi = float(d["ci"][0]), float(d["ci"][1])
        if not (lo > 0.05 + TOL):
            all_gt_005 = False
        if md >= 0.02 - TOL or lo > TOL:
            any_partial = True
        if not (lo <= TOL <= hi and abs(md) < 0.02 - TOL):
            all_excl = False

    if all_gt_005 and author_delta_transfer_mean >= 0.25 - TOL:
        verdict = "确认"
    elif all_excl:
        verdict = "排除"
    elif any_partial:
        verdict = "部分贡献"
    else:
        verdict = "部分贡献"
    return {
        "verdict": verdict,
        "author_delta_transfer_ge_025": bool(author_delta_transfer_mean >= 0.25 - TOL),
        "all_tiers_ci_low_gt_005": all_gt_005,
    }


def compute_cause_b(
    ours_27b_ar: Sequence[dict[str, Any]],
    t022_by_row: dict[int, dict[str, Any]],
    exact_sid_to_row: dict[str, int],
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compare 27B Author-Rule gains on Author 494 (T-022) vs Self-Built 3-tier (threshold-filtered)."""
    sorted_rows = sorted(t022_by_row)
    auth_trans_all = [float(t022_by_row[r]["delta_transfer"]) for r in sorted_rows]
    auth_same_all = [float(t022_by_row[r]["delta_same"]) for r in sorted_rows]
    auth_trans_filt = [
        float(t022_by_row[r]["delta_transfer"])
        for r in sorted_rows
        if passes_author_threshold(t022_by_row[r]["pre_solve_rate_transfer"], t022_by_row[r]["pre_solve_rate"])
    ]

    auth_trans_boot = bootstrap_mean_1d(auth_trans_all, seed=derive_seed(seed, "causeB_auth_trans"), draws=draws)
    auth_same_boot = bootstrap_mean_1d(auth_same_all, seed=derive_seed(seed, "causeB_auth_same"), draws=draws)
    auth_trans_filt_boot = bootstrap_mean_1d(
        auth_trans_filt, seed=derive_seed(seed, "causeB_auth_trans_filt"), draws=draws
    )

    by_tier_all: dict[str, list[dict[str, Any]]] = {d: [] for d in DIFFICULTY_ORDER}
    orig_by_sid_all: dict[str, list[float]] = defaultdict(list)
    for rec in ours_27b_ar:
        d = rec["difficulty"]
        by_tier_all[d].append(rec)
        orig_by_sid_all[rec["source_id"]].append(normalized_gain(rec["pre_orig"], rec["post_orig"]))

    use_paired = len(exact_sid_to_row) >= 30
    unpaired_transfer_filtered: dict[str, Any] = {}
    unpaired_transfer_both_filtered: dict[str, Any] = {}
    unpaired_transfer_unfiltered: dict[str, Any] = {}
    unpaired_same_by_tier: dict[str, Any] = {}
    paired_transfer_filtered: dict[str, Any] = {}
    paired_transfer_both_filtered: dict[str, Any] = {}
    paired_transfer_all_overlap: dict[str, Any] = {}
    paired_same_by_tier: dict[str, Any] = {}
    cross_checks: dict[str, Any] = {}

    for diff in DIFFICULTY_ORDER:
        t_recs = sorted(by_tier_all[diff], key=lambda x: x["source_id"])
        t_filt = [r for r in t_recs if passes_author_threshold(r["pre_trans"], r["pre_orig"])]

        ours_g_all = [normalized_gain(r["pre_trans"], r["post_trans"]) for r in t_recs]
        ours_g_filt = [normalized_gain(r["pre_trans"], r["post_trans"]) for r in t_filt]
        ours_same_all = [normalized_gain(r["pre_orig"], r["post_orig"]) for r in t_recs]

        unpaired_transfer_filtered[diff] = unpaired_diff_bootstrap(
            auth_trans_all,
            ours_g_filt,
            seed=derive_seed(seed, "causeB_unp_filt", diff),
            draws=draws,
        )
        unpaired_transfer_both_filtered[diff] = unpaired_diff_bootstrap(
            auth_trans_filt,
            ours_g_filt,
            seed=derive_seed(seed, "causeB_unp_both_filt", diff),
            draws=draws,
        )
        unpaired_transfer_unfiltered[diff] = unpaired_diff_bootstrap(
            auth_trans_all,
            ours_g_all,
            seed=derive_seed(seed, "causeB_unp_all", diff),
            draws=draws,
        )
        unpaired_same_by_tier[diff] = unpaired_diff_bootstrap(
            auth_same_all,
            ours_same_all,
            seed=derive_seed(seed, "causeB_unp_same", diff),
            draws=draws,
        )

        if use_paired:
            auth_t_pairs = [
                (sid, float(t022_by_row[ridx]["delta_transfer"]))
                for sid, ridx in sorted(exact_sid_to_row.items())
                if ridx in t022_by_row
            ]
            auth_t_filt_pairs = [
                (sid, float(t022_by_row[ridx]["delta_transfer"]))
                for sid, ridx in sorted(exact_sid_to_row.items())
                if ridx in t022_by_row
                and passes_author_threshold(
                    t022_by_row[ridx]["pre_solve_rate_transfer"],
                    t022_by_row[ridx]["pre_solve_rate"],
                )
            ]
            auth_s_pairs = [
                (sid, float(t022_by_row[ridx]["delta_same"]))
                for sid, ridx in sorted(exact_sid_to_row.items())
                if ridx in t022_by_row
            ]
            ours_t_filt_pairs = [
                (r["source_id"], normalized_gain(r["pre_trans"], r["post_trans"])) for r in t_filt
            ]
            ours_t_all_pairs = [
                (r["source_id"], normalized_gain(r["pre_trans"], r["post_trans"])) for r in t_recs
            ]
            ours_s_pairs = [
                (r["source_id"], normalized_gain(r["pre_orig"], r["post_orig"])) for r in t_recs
            ]

            p_filt = paired_diff_bootstrap(
                auth_t_pairs,
                ours_t_filt_pairs,
                seed=derive_seed(seed, "causeB_pair_filt", diff),
                draws=draws,
            )
            p_both = paired_diff_bootstrap(
                auth_t_filt_pairs,
                ours_t_filt_pairs,
                seed=derive_seed(seed, "causeB_pair_both_filt", diff),
                draws=draws,
            )
            p_all = paired_diff_bootstrap(
                auth_t_pairs,
                ours_t_all_pairs,
                seed=derive_seed(seed, "causeB_pair_all", diff),
                draws=draws,
            )
            p_same = paired_diff_bootstrap(
                auth_s_pairs,
                ours_s_pairs,
                seed=derive_seed(seed, "causeB_pair_same", diff),
                draws=draws,
            )
            paired_transfer_filtered[diff] = p_filt
            paired_transfer_both_filtered[diff] = p_both
            paired_transfer_all_overlap[diff] = p_all
            paired_same_by_tier[diff] = p_same

            # Cross-check against pure-Python summarize_base4b.paired_bootstrap_diff
            sb_ref = paired_bootstrap_diff(
                auth_t_pairs,
                ours_t_filt_pairs,
                seed=derive_seed(seed, "causeB_sb_cc", diff),
                draws=draws,
            )
            if p_filt["mean_diff"] is not None and sb_ref["mean_diff"] is not None:
                pt_err = abs(p_filt["mean_diff"] - sb_ref["mean_diff"])
                lo_err = abs(p_filt["ci"][0] - sb_ref["ci"][0])
                hi_err = abs(p_filt["ci"][1] - sb_ref["ci"][1])
                if p_filt["n_paired"] != sb_ref["n_paired"] or pt_err > 1e-12 or lo_err > 0.01 or hi_err > 0.01:
                    raise ValueError(f"Cause B cross-check mismatch on {diff}: pt={pt_err}, ci=({lo_err},{hi_err})")
                cross_checks[diff] = {"n_paired": p_filt["n_paired"], "pt_err": pt_err, "ci_lo_err": lo_err, "ci_hi_err": hi_err}

    ours_orig_comb_pairs = [(sid, statistics.fmean(vals)) for sid, vals in sorted(orig_by_sid_all.items())]
    unpaired_same_combined = unpaired_diff_bootstrap(
        auth_same_all,
        [v for _, v in ours_orig_comb_pairs],
        seed=derive_seed(seed, "causeB_unp_same_comb"),
        draws=draws,
    )
    paired_same_combined = None
    if use_paired:
        auth_s_pairs_all = [
            (sid, float(t022_by_row[ridx]["delta_same"]))
            for sid, ridx in sorted(exact_sid_to_row.items())
            if ridx in t022_by_row
        ]
        paired_same_combined = paired_diff_bootstrap(
            auth_s_pairs_all,
            ours_orig_comb_pairs,
            seed=derive_seed(seed, "causeB_pair_same_comb"),
            draws=draws,
        )

    primary_diffs = paired_transfer_filtered if use_paired else unpaired_transfer_filtered
    decision = judge_cause_b(primary_diffs, float(auth_trans_boot["mean"]))
    decision["comparison_mode"] = "paired_on_overlap_ge_30" if use_paired else "unpaired_overlap_lt_30"
    decision["unpaired_verdict_check"] = judge_cause_b(
        unpaired_transfer_filtered, float(auth_trans_boot["mean"])
    )["verdict"]

    return {
        "author_27b_t022": {
            "delta_transfer_all494": auth_trans_boot,
            "delta_transfer_threshold_filtered": auth_trans_filt_boot,
            "delta_same_all494": auth_same_boot,
        },
        "use_paired_overlap": use_paired,
        "primary_author_minus_ours_filtered": primary_diffs,
        "paired_transfer_filtered": paired_transfer_filtered if use_paired else None,
        "paired_transfer_both_filtered": paired_transfer_both_filtered if use_paired else None,
        "paired_transfer_all_overlap": paired_transfer_all_overlap if use_paired else None,
        "paired_same_by_tier": paired_same_by_tier if use_paired else None,
        "paired_same_combined": paired_same_combined,
        "unpaired_transfer_filtered": unpaired_transfer_filtered,
        "unpaired_transfer_both_filtered": unpaired_transfer_both_filtered,
        "unpaired_transfer_unfiltered": unpaired_transfer_unfiltered,
        "unpaired_same_by_tier": unpaired_same_by_tier,
        "unpaired_same_combined": unpaired_same_combined,
        "cross_checks": cross_checks,
        "post_max_tokens_note": "自建后测上限 1024 vs 作者 800 无法离线统一，见原因 D 筛查结论。",
        "decision": decision,
    }


def judge_cause_c(
    tier_ar_minus_agnes: dict[str, dict[str, Any]],
    a100_format_check: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply pre-registered Cause C rule from 增益差分析预案.md lines 82-85 & T-025 line 53.

    - 检验不了: 前测或后测提取不到 \\boxed{} / 格式错比例 >= 15% (且明显高于 Agnes 未决率)
    - 确认: 主要档位 Δ_rule - Δ_agnes 95% CI 下界 > +0.03
    - 部分贡献: Δ_rule - Δ_agnes 在 +0.015 ~ +0.03 之间 (或仅个别档位显著)
    - 排除: 三档 Δ_rule - Δ_agnes 点估计均 < +0.015 且 95% CI 上界 < +0.03
    """
    max_fmt_err: float | None = None
    if a100_format_check is not None:
        rates: list[float] = []
        for row in a100_format_check.get("by_tier_target_phase", []):
            if row.get("phase") in ("pre", "post"):
                rates.append(float(row["ar_incorrect_format_rate"]))
        if rates:
            max_fmt_err = max(rates)
        if max_fmt_err is not None and max_fmt_err >= 0.15 - TOL:
            return {
                "verdict": "检验不了",
                "max_pre_or_post_incorrect_format_rate": max_fmt_err,
                "reason": "作者规则下前测或后测提取不到 \\boxed{} 比例 >= 15%",
            }

    ci_lows_gt_003 = 0
    any_partial = False
    all_excl = True
    for diff in DIFFICULTY_ORDER:
        d = tier_ar_minus_agnes[diff]
        md = float(d["mean_diff"])
        lo, hi = float(d["ci"][0]), float(d["ci"][1])
        if lo > 0.03 + TOL:
            ci_lows_gt_003 += 1
        if md >= 0.015 - TOL or lo > TOL:
            any_partial = True
        if not (md < 0.015 - TOL and hi < 0.03 - TOL):
            all_excl = False

    if ci_lows_gt_003 >= 2:
        verdict = "确认"
    elif all_excl:
        verdict = "排除"
    elif any_partial or ci_lows_gt_003 >= 1:
        verdict = "部分贡献"
    else:
        verdict = "排除"

    return {
        "verdict": verdict,
        "a100_precheck_loaded": a100_format_check is not None,
        "max_pre_or_post_incorrect_format_rate": max_fmt_err,
        "tiers_ci_low_gt_003": ci_lows_gt_003,
    }


def compute_cause_c_pro5000(
    entries_agnes: Sequence[dict[str, Any]],
    entries_ar: Sequence[dict[str, Any]],
    *,
    a100_format_check: dict[str, Any] | None = None,
    other_models: dict[str, tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]]]] | None = None,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute paired (Δ_author_rule - Δ_agnes) for 27B (and descriptively base / a_group)."""

    def _model_diff(
        m_name: str, ag_recs: Sequence[dict[str, Any]], ar_recs: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        out_m: dict[str, Any] = {}
        cc_m: dict[str, Any] = {}
        ag_orig_by_sid: dict[str, list[float]] = defaultdict(list)
        ar_orig_by_sid: dict[str, list[float]] = defaultdict(list)

        for diff in DIFFICULTY_ORDER:
            ag_t = [
                (r["source_id"], normalized_gain(r["pre_trans"], r["post_trans"]))
                for r in ag_recs
                if r["difficulty"] == diff
            ]
            ar_t = [
                (r["source_id"], normalized_gain(r["pre_trans"], r["post_trans"]))
                for r in ar_recs
                if r["difficulty"] == diff
            ]
            res_t = paired_diff_bootstrap(
                ar_t, ag_t, seed=derive_seed(seed, "causeC_trans", m_name, diff), draws=draws
            )
            sb_t = paired_bootstrap_diff(
                ar_t, ag_t, seed=derive_seed(seed, "causeC_sb", m_name, diff), draws=draws
            )
            if res_t["mean_diff"] is not None and sb_t["mean_diff"] is not None:
                pt_err = abs(res_t["mean_diff"] - sb_t["mean_diff"])
                lo_err = abs(res_t["ci"][0] - sb_t["ci"][0])
                hi_err = abs(res_t["ci"][1] - sb_t["ci"][1])
                if res_t["n_paired"] != sb_t["n_paired"] or pt_err > 1e-12 or lo_err > 0.01 or hi_err > 0.01:
                    raise ValueError(f"Cause C cross-check mismatch ({m_name}.{diff})")
                cc_m[diff] = {"n_paired": res_t["n_paired"], "pt_err": pt_err, "ci_lo_err": lo_err, "ci_hi_err": hi_err}
            out_m[diff] = res_t

            ag_o = [
                (r["source_id"], normalized_gain(r["pre_orig"], r["post_orig"]))
                for r in ag_recs
                if r["difficulty"] == diff
            ]
            ar_o = [
                (r["source_id"], normalized_gain(r["pre_orig"], r["post_orig"]))
                for r in ar_recs
                if r["difficulty"] == diff
            ]
            out_m[f"orig_{diff}"] = paired_diff_bootstrap(
                ar_o, ag_o, seed=derive_seed(seed, "causeC_orig", m_name, diff), draws=draws
            )
            for sid, v in ag_o:
                ag_orig_by_sid[sid].append(v)
            for sid, v in ar_o:
                ar_orig_by_sid[sid].append(v)

        ag_comb = [(sid, statistics.fmean(v)) for sid, v in sorted(ag_orig_by_sid.items())]
        ar_comb = [(sid, statistics.fmean(v)) for sid, v in sorted(ar_orig_by_sid.items())]
        out_m["orig_combined"] = paired_diff_bootstrap(
            ar_comb, ag_comb, seed=derive_seed(seed, "causeC_orig_comb", m_name), draws=draws
        )
        return out_m, cc_m

    m27, cc27 = _model_diff("eduardo27b", entries_agnes, entries_ar)
    desc_models: dict[str, Any] = {}
    if other_models:
        for m_name, (ag_r, ar_r) in other_models.items():
            dm, _ = _model_diff(m_name, ag_r, ar_r)
            desc_models[m_name] = dm

    decision = judge_cause_c(m27, a100_format_check)
    return {
        "eduardo27b_ar_minus_agnes": m27,
        "descriptive_other_models": desc_models,
        "a100_format_precheck": a100_format_check,
        "cross_checks": cc27,
        "decision": decision,
    }


def summarize_length_and_format_records(
    sample_records: Sequence[dict[str, Any]],
    *,
    model_label: str,
    threshold_tokens: int = 800,
) -> dict[str, Any]:
    """Aggregate per-sample token lengths, Agnes verdicts, and \\boxed{} format status.

    Each record in sample_records has:
      {"difficulty": "easy"|"medium"|"hard", "target": "orig"|"trans",
       "phase": "pre"|"control"|"post", "tokens": int, "finish_reason": str,
       "agnes_status": "correct"|"incorrect"|"unknown",
       "ar_incorrect_format": bool,
       "boxed_pred_full": str, "boxed_pred_prefix800": str}
    """

    def _agg_bucket(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {
                "n": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0,
                "acc_unknown_as_wrong": None,
                "acc_excl_unknown": None,
                "agnes_unknown_rate": None,
                "ar_incorrect_format": 0,
                "ar_incorrect_format_rate": None,
                "finish_length_count": 0,
                "finish_length_rate": None,
            }
        c_corr = sum(1 for r in rows if r["agnes_status"] == "correct")
        c_inc = sum(1 for r in rows if r["agnes_status"] == "incorrect")
        c_unk = sum(1 for r in rows if r["agnes_status"] == "unknown")
        c_fmt = sum(1 for r in rows if r["ar_incorrect_format"])
        c_len = sum(1 for r in rows if r.get("finish_reason") == "length")
        den_excl = c_corr + c_inc
        return {
            "n": n,
            "correct": c_corr,
            "incorrect": c_inc,
            "unknown": c_unk,
            "acc_unknown_as_wrong": c_corr / n,
            "acc_excl_unknown": (c_corr / den_excl) if den_excl > 0 else None,
            "agnes_unknown_rate": c_unk / n,
            "ar_incorrect_format": c_fmt,
            "ar_incorrect_format_rate": c_fmt / n,
            "finish_length_count": c_len,
            "finish_length_rate": c_len / n,
        }

    def _agg_post_split(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
        le_rows = [r for r in rows if int(r["tokens"]) <= threshold_tokens]
        gt_rows = [r for r in rows if int(r["tokens"]) > threshold_tokens]
        n = len(rows)
        # Among >800 rows: how many had a boxed answer only after 800 vs already had boxed before 800
        had_box_full = sum(1 for r in gt_rows if r.get("boxed_pred_full"))
        had_box_pre800 = sum(1 for r in gt_rows if r.get("boxed_pred_prefix800"))
        box_only_after_800 = sum(
            1 for r in gt_rows if r.get("boxed_pred_full") and not r.get("boxed_pred_prefix800")
        )
        box_changed_after_800 = sum(
            1
            for r in gt_rows
            if r.get("boxed_pred_full")
            and r.get("boxed_pred_prefix800")
            and r["boxed_pred_full"].strip() != r["boxed_pred_prefix800"].strip()
        )
        le_st = _agg_bucket(le_rows)
        gt_st = _agg_bucket(gt_rows)
        return {
            "group": label,
            "n_total": n,
            "n_gt_800": len(gt_rows),
            "gt_800_rate": (len(gt_rows) / n) if n > 0 else 0.0,
            "all": _agg_bucket(rows),
            "le_800": le_st,
            "gt_800": gt_st,
            "gt_800_boxed_diagnostics": {
                "had_boxed_in_1024": had_box_full,
                "had_boxed_in_first_800": had_box_pre800,
                "boxed_only_in_801_1024_lost_if_trunc": box_only_after_800,
                "boxed_overwritten_in_801_1024": box_changed_after_800,
                "agnes_correct_in_gt800": gt_st["correct"],
            },
        }

    # 1) Full format & unknown table across (difficulty, target, phase) for Cause C
    by_ttp: list[dict[str, Any]] = []
    phases_present = sorted({r["phase"] for r in sample_records})
    for phase in ("pre", "control", "post"):
        if phase not in phases_present:
            continue
        for tgt in ("orig", "trans"):
            for diff in DIFFICULTY_ORDER:
                sub = [
                    r
                    for r in sample_records
                    if r["phase"] == phase and r["target"] == tgt and r["difficulty"] == diff
                ]
                if not sub:
                    continue
                by_ttp.append(
                    {
                        "phase": phase,
                        "target": tgt,
                        "difficulty": diff,
                        **_agg_bucket(sub),
                    }
                )
            sub_all_tiers = [r for r in sample_records if r["phase"] == phase and r["target"] == tgt]
            if sub_all_tiers:
                by_ttp.append(
                    {
                        "phase": phase,
                        "target": tgt,
                        "difficulty": "all_tiers",
                        **_agg_bucket(sub_all_tiers),
                    }
                )

    # 2) Post-test >800 screening table for Cause D
    post_rows = [r for r in sample_records if r["phase"] == "post"]
    post_groups: list[dict[str, Any]] = []
    post_groups.append(_agg_post_split([r for r in post_rows if r["target"] == "orig"], "原题后测(三档合计)"))
    for diff in DIFFICULTY_ORDER:
        post_groups.append(
            _agg_post_split(
                [r for r in post_rows if r["target"] == "orig" and r["difficulty"] == diff],
                f"原题-{GROUP_BY_DIFFICULTY[diff]}对话",
            )
        )
        post_groups.append(
            _agg_post_split(
                [r for r in post_rows if r["target"] == "trans" and r["difficulty"] == diff],
                f"{GROUP_BY_DIFFICULTY[diff]}迁移题",
            )
        )
    post_groups.append(_agg_post_split([r for r in post_rows if r["target"] == "trans"], "迁移题后测(三档合计)"))
    post_groups.append(_agg_post_split(post_rows, "全部后测合计"))

    return {
        "model": model_label,
        "threshold_tokens": threshold_tokens,
        "total_samples_scanned": len(sample_records),
        "by_tier_target_phase": by_ttp,
        "post_length_screening": post_groups,
    }


def judge_cause_d(models_screening: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply pre-registered Cause D screening rule from 增益差分析预案.md lines 94, 100-103 & T-025 lines 56-57.

    Screening trigger: >800 share >= 5% AND acc(>800) < acc(<=800).
    Note: Why >800 has lower accuracy than <=800 is distinguished via boxed_only_in_801_1024 vs
    boxed_overwritten_in_801_1024: truncating 1024 -> 800 cannot create new correct answers unless
    a correct boxed answer before 800 was overwritten after 800!
    """
    m27 = models_screening.get("eduardo27b")
    if m27 is None:
        return {"verdict": "检验不了", "reason": "缺少 A100 27B 后测长度筛查计数"}

    transfer_tiers = ("简单迁移题", "中等迁移题", "困难迁移题")
    tier_triggers: dict[str, Any] = {}
    any_ge_5pct = False
    any_trigger = False
    max_overwritten_upper_bound = 0.0

    for row in m27.get("post_length_screening", []):
        grp = row["group"]
        if grp not in transfer_tiers:
            continue
        rate_gt = float(row["gt_800_rate"])
        acc_le = row["le_800"]["acc_unknown_as_wrong"]
        acc_gt = row["gt_800"]["acc_unknown_as_wrong"]
        ge_5 = rate_gt >= 0.05 - TOL
        lower_acc = (
            acc_gt is not None and acc_le is not None and float(acc_gt) < float(acc_le) - TOL
        )
        trig = ge_5 and lower_acc
        if ge_5:
            any_ge_5pct = True
        if trig:
            any_trigger = True
        bd = row.get("gt_800_boxed_diagnostics") or {}
        n_tot = max(1, int(row["n_total"]))
        # Upper bound on how much 800-token cap could POSSIBLY increase raw accuracy:
        # at most responses that overwrote a prior boxed answer in tokens 801..1024, minus
        # correct responses whose boxed answer appeared only in tokens 801..1024!
        overwritten_rate = int(bd.get("boxed_overwritten_in_801_1024", 0)) / n_tot
        lost_rate = int(bd.get("boxed_only_in_801_1024_lost_if_trunc", 0)) / n_tot
        max_overwritten_upper_bound = max(max_overwritten_upper_bound, overwritten_rate)
        tier_triggers[grp] = {
            "gt_800_rate": rate_gt,
            "acc_le_800": acc_le,
            "acc_gt_800": acc_gt,
            "ge_5pct": ge_5,
            "gt800_lower_than_le800": lower_acc,
            "rerun_screening_triggered": trig,
            "boxed_only_in_801_1024_rate": lost_rate,
            "boxed_overwritten_in_801_1024_rate": overwritten_rate,
        }

    if not any_ge_5pct or not any_trigger:
        verdict = "排除"
    else:
        # Pre-registered rule (预案 lines 94, 101-103 & T-025 lines 56-57):
        # Screening triggered (>=5% and acc_gt800 < acc_le800), and 800-cap rerun is not
        # performed in T-025 -> record as "检验不了" and escalate rerun option to "需要决策".
        verdict = "检验不了"

    return {
        "verdict": verdict,
        "any_tier_gt800_ge_5pct": any_ge_5pct,
        "any_tier_rerun_screening_triggered": any_trigger,
        "max_boxed_overwritten_rate_across_tiers": max_overwritten_upper_bound,
        "per_transfer_tier_27b": tier_triggers,
    }


def judge_cause_e(
    pre_post_same_config: bool,
    diff_ar_pooled: dict[str, Any],
    diff_agnes_pooled: dict[str, Any],
) -> dict[str, Any]:
    """Apply pre-registered Cause E1 rule from 增益差分析预案.md lines 120-122 & T-025 lines 59-61."""
    ar_m = float(diff_ar_pooled["mean_diff"]) if diff_ar_pooled.get("mean_diff") is not None else 0.0
    ar_ci = diff_ar_pooled.get("ci") or [0.0, 0.0]
    ag_m = float(diff_agnes_pooled["mean_diff"]) if diff_agnes_pooled.get("mean_diff") is not None else 0.0
    ag_ci = diff_agnes_pooled.get("ci") or [0.0, 0.0]

    e1_trigger_ar = (ar_m > 0.03 + TOL) and (float(ar_ci[0]) > 0.02 + TOL)
    e1_trigger_ag = (ag_m > 0.03 + TOL) and (float(ag_ci[0]) > 0.02 + TOL)
    ar_excl = abs(ar_m) < 0.02 - TOL and (float(ar_ci[0]) <= TOL <= float(ar_ci[1]))
    ag_excl = abs(ag_m) < 0.02 - TOL and (float(ag_ci[0]) <= TOL <= float(ag_ci[1]))

    if pre_post_same_config and ar_excl and ag_excl:
        verdict = "排除"
    elif pre_post_same_config and ar_excl and not e1_trigger_ag:
        # Under identical Author-Rule grader, pre_ours matches pre_author within <0.02 (CI includes 0);
        # any Agnes-vs-Author offset is grader leniency (covered in Cause C), not pre/post baseline drift.
        verdict = "排除"
    else:
        # E2 baseline re-measurement is not executed in T-025; escalate to "需要决策" if E1 does not exclude.
        verdict = "检验不了"

    return {
        "verdict": verdict,
        "pre_post_same_config": pre_post_same_config,
        "e1_trigger_same_grader_ar": e1_trigger_ar,
        "e1_trigger_agnes_vs_author_ar": e1_trigger_ag,
        "ar_abs_diff_lt_002_and_ci_contains_0": ar_excl,
        "agnes_abs_diff_lt_002_and_ci_contains_0": ag_excl,
    }


def compute_cause_e_pro5000(
    ours_27b_agnes: Sequence[dict[str, Any]],
    ours_27b_ar: Sequence[dict[str, Any]],
    author_pre_by_row: dict[int, dict[str, float]],
    exact_sid_to_row: dict[str, int],
    *,
    t018_by_row: dict[int, dict[str, Any]] | None = None,
    a100_config_doc: dict[str, Any] | None = None,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute Cause E1 configuration alignment and overlapping original pre-test comparison."""
    ag_orig_by_sid: dict[str, list[float]] = defaultdict(list)
    ar_orig_by_sid: dict[str, list[float]] = defaultdict(list)
    ag_tier_orig: dict[str, list[tuple[str, float]]] = {d: [] for d in DIFFICULTY_ORDER}
    ar_tier_orig: dict[str, list[tuple[str, float]]] = {d: [] for d in DIFFICULTY_ORDER}

    for r in ours_27b_agnes:
        ag_orig_by_sid[r["source_id"]].append(float(r["pre_orig"]))
        ag_tier_orig[r["difficulty"]].append((r["source_id"], float(r["pre_orig"])))
    for r in ours_27b_ar:
        ar_orig_by_sid[r["source_id"]].append(float(r["pre_orig"]))
        ar_tier_orig[r["difficulty"]].append((r["source_id"], float(r["pre_orig"])))

    auth_orig_pairs = [
        (sid, float(author_pre_by_row[ridx]["pre_same"]))
        for sid, ridx in sorted(exact_sid_to_row.items())
        if ridx in author_pre_by_row
    ]
    ours_ag_pooled = [(sid, statistics.fmean(v)) for sid, v in sorted(ag_orig_by_sid.items())]
    ours_ar_pooled = [(sid, statistics.fmean(v)) for sid, v in sorted(ar_orig_by_sid.items())]

    diff_ar_pooled = paired_diff_bootstrap(
        ours_ar_pooled, auth_orig_pairs, seed=derive_seed(seed, "causeE_ar_pooled"), draws=draws
    )
    diff_ag_pooled = paired_diff_bootstrap(
        ours_ag_pooled, auth_orig_pairs, seed=derive_seed(seed, "causeE_ag_pooled"), draws=draws
    )
    # Cross-check against summarize_base4b.paired_bootstrap_diff
    sb_ar = paired_bootstrap_diff(
        ours_ar_pooled, auth_orig_pairs, seed=derive_seed(seed, "causeE_sb_ar"), draws=draws
    )
    if diff_ar_pooled["mean_diff"] is not None and sb_ar["mean_diff"] is not None:
        if abs(diff_ar_pooled["mean_diff"] - sb_ar["mean_diff"]) > 1e-12:
            raise ValueError("Cause E1 cross-check failed on pre_ours_ar - pre_author")

    by_tier_ar = {
        diff: paired_diff_bootstrap(
            ar_tier_orig[diff], auth_orig_pairs, seed=derive_seed(seed, "causeE_ar_tier", diff), draws=draws
        )
        for diff in DIFFICULTY_ORDER
    }
    by_tier_ag = {
        diff: paired_diff_bootstrap(
            ag_tier_orig[diff], auth_orig_pairs, seed=derive_seed(seed, "causeE_ag_tier", diff), draws=draws
        )
        for diff in DIFFICULTY_ORDER
    }

    t018_ctl1_vs_author = None
    if t018_by_row:
        c1 = [(str(ridx), r["ctl1_same_correct"] / r["ctl1_same_total"]) for ridx, r in sorted(t018_by_row.items())]
        au = [(str(ridx), float(r["pre_solve_rate"])) for ridx, r in sorted(t018_by_row.items())]
        t018_ctl1_vs_author = paired_diff_bootstrap(
            c1, au, seed=derive_seed(seed, "causeE_t018_ctl1"), draws=draws
        )

    # In 27B self-built eval (offline_eval.py:101-175), pre, control, and post use the exact same
    # student server, model, fp16 dtype, temp=0.6, max_tokens=1024, k=16, and Agnes grader.
    pre_post_same_config = True
    if a100_config_doc is not None:
        pre_post_same_config = bool(a100_config_doc.get("pre_post_identical_student_and_grader", True))

    config_table = [
        {
            "phase": "自建 27B 独立前测 (pre / control)",
            "student_model": "Meta-Llama-3.1-8B-Instruct",
            "dtype": "float16 (vLLM 0.18.0)",
            "temperature": 0.6,
            "max_tokens": 1024,
            "k": 16,
            "grader": "Agnes (agnes-3.0-flash, 叠加恢复+补判); 另有离线 Author-Rule",
        },
        {
            "phase": "自建 27B 学生后测 (post)",
            "student_model": "Meta-Llama-3.1-8B-Instruct",
            "dtype": "float16 (vLLM 0.18.0)",
            "temperature": 0.6,
            "max_tokens": 1024,
            "k": 16,
            "grader": "Agnes (同一次 SCORE_INPUT.json 统一判分); 另有离线 Author-Rule",
        },
        {
            "phase": "作者 494 数据集固化前测 (test.parquet)",
            "student_model": "Meta-Llama-3.1-8B-Instruct",
            "dtype": "bfloat16",
            "temperature": 0.6,
            "max_tokens": 1024,
            "k": 16,
            "grader": "Author-Rule (\\boxed{} + math_verify 5s)",
        },
        {
            "phase": "作者 494 学生后测 (T-022 run_env_test_eval.py)",
            "student_model": "Meta-Llama-3.1-8B-Instruct",
            "dtype": "bfloat16 (vLLM 0.19.1)",
            "temperature": 0.6,
            "max_tokens": 800,
            "k": 16,
            "grader": "Author-Rule (\\boxed{} + math_verify 0.8.0 5s)",
        },
    ]

    decision = judge_cause_e(pre_post_same_config, diff_ar_pooled, diff_ag_pooled)
    return {
        "config_comparison_table": config_table,
        "a100_runtime_config": a100_config_doc,
        "overlap_orig_pre_ours_ar_minus_author": diff_ar_pooled,
        "overlap_orig_pre_ours_agnes_minus_author": diff_ag_pooled,
        "overlap_orig_pre_by_tier_ar": by_tier_ar,
        "overlap_orig_pre_by_tier_agnes": by_tier_ag,
        "reference_t018_a100_bf16_ctl1_minus_author_pre": t018_ctl1_vs_author,
        "decision": decision,
    }


def judge_cause_f(
    tier_intersect_diffs: dict[str, dict[str, Any]],
    missing_bounds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply pre-registered Cause F rule from 增益差分析预案.md lines 138-140."""
    max_abs_shift = 0.0
    for diff in DIFFICULTY_ORDER:
        d = tier_intersect_diffs[diff]["diff_intersect_minus_all"]
        if d and d.get("mean_diff") is not None:
            max_abs_shift = max(max_abs_shift, abs(float(d["mean_diff"])))
    max_missing_up = max(float(missing_bounds[d]["max_upward_shift"]) for d in DIFFICULTY_ORDER)

    if max_abs_shift >= 0.04 - TOL or max_missing_up >= 0.04 - TOL:
        verdict = "确认"
    elif max_abs_shift < 0.015 - TOL and max_missing_up < 0.015 - TOL:
        verdict = "排除"
    else:
        verdict = "部分贡献"
    return {
        "verdict": verdict,
        "max_abs_intersect_minus_all": max_abs_shift,
        "max_missing_8_upward_bound_per_tier": max_missing_up,
        "conservative_single_tier_8_of_499_bound": 8.0 / 499.0,
    }


def compute_cause_f(
    ours_27b_agnes: Sequence[dict[str, Any]],
    overlap_public_counts: dict[str, Any],
    exact_sid_to_row: dict[str, int],
    *,
    per_tier_total_target: int = 499,
    t022_by_row: dict[int, dict[str, Any]] | None = None,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute Cause F intersection gains (499 ∩ 494 vs all 499) and exact missing-8 bounds."""
    overlap_sids = set(exact_sid_to_row.keys())
    overlap_rows = set(exact_sid_to_row.values())
    empty_bins = {b: 0 for b in PRE_BIN_LABELS}

    tier_out: dict[str, Any] = {}
    missing_bounds: dict[str, Any] = {}

    for diff in DIFFICULTY_ORDER:
        t_recs = sorted([r for r in ours_27b_agnes if r["difficulty"] == diff], key=lambda x: x["source_id"])
        gains = [normalized_gain(r["pre_trans"], r["post_trans"]) for r in t_recs]
        in_inter = [r["source_id"] in overlap_sids for r in t_recs]
        dummy_bins = ["0"] * len(t_recs)
        comp = _cluster_subset_and_reweight_bootstrap(
            gains,
            in_inter,
            dummy_bins,
            empty_bins,
            seed=derive_seed(seed, "causeF_tier", diff),
            draws=draws,
        )
        tier_out[diff] = {
            "all_eligible": comp["all"],
            "in_499_inter_494": comp["filtered"],
            "diff_intersect_minus_all": comp["diff_filtered_minus_all"],
        }
        n_done = len(t_recs)
        n_miss = max(0, per_tier_total_target - n_done)
        cur_mean = float(comp["all"]["mean"])
        # Normalized gain in [-1, 1]: exact worst/best shift if n_miss items had gain +1.0 or -1.0
        max_up = (n_miss * (1.0 - cur_mean)) / per_tier_total_target if per_tier_total_target > 0 else 0.0
        max_dn = (n_miss * (-1.0 - cur_mean)) / per_tier_total_target if per_tier_total_target > 0 else 0.0
        missing_bounds[diff] = {
            "n_completed": n_done,
            "n_missing": n_miss,
            "total_sources": per_tier_total_target,
            "current_r1": cur_mean,
            "max_upward_shift": max_up,
            "max_downward_shift": max_dn,
            "r1_range_if_completed": [cur_mean + max_dn, cur_mean + max_up],
        }

    # Orig combined on 499 ∩ 494 vs all 499
    orig_by_sid: dict[str, list[float]] = defaultdict(list)
    for r in ours_27b_agnes:
        orig_by_sid[r["source_id"]].append(normalized_gain(r["pre_orig"], r["post_orig"]))
    sids_sorted = sorted(orig_by_sid)
    orig_gains = [statistics.fmean(orig_by_sid[s]) for s in sids_sorted]
    orig_in_inter = [s in overlap_sids for s in sids_sorted]
    orig_comp = _cluster_subset_and_reweight_bootstrap(
        orig_gains,
        orig_in_inter,
        ["0"] * len(sids_sorted),
        empty_bins,
        seed=derive_seed(seed, "causeF_orig_comb"),
        draws=draws,
    )

    author_inter_check = None
    if t022_by_row:
        ridxs = sorted(t022_by_row)
        a_gains = [float(t022_by_row[r]["delta_transfer"]) for r in ridxs]
        a_in = [r in overlap_rows for r in ridxs]
        ac = _cluster_subset_and_reweight_bootstrap(
            a_gains, a_in, ["0"] * len(ridxs), empty_bins, seed=derive_seed(seed, "causeF_auth"), draws=draws
        )
        author_inter_check = {
            "all_494": ac["all"],
            "in_499_inter_494": ac["filtered"],
            "diff_intersect_minus_all": ac["diff_filtered_minus_all"],
        }

    decision = judge_cause_f(tier_out, missing_bounds)
    return {
        "overlap_counts": overlap_public_counts,
        "selfbuilt_27b_intersect_vs_all": tier_out,
        "selfbuilt_27b_orig_combined_intersect_vs_all": {
            "all": orig_comp["all"],
            "in_499_inter_494": orig_comp["filtered"],
            "diff_intersect_minus_all": orig_comp["diff_filtered_minus_all"],
        },
        "author_t022_intersect_vs_all": author_inter_check,
        "missing_8_exact_bounds": missing_bounds,
        "decision": decision,
    }


def judge_cause_g(
    t022_summary: dict[str, Any],
    a100_27b_diag: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply pre-registered Cause G rule from 增益差分析预案.md lines 153-156."""
    # Check Tutor protocol parameters and empty rate, plus T-022 aligned author-494 benchmark
    dt_t022 = float(t022_summary["delta_transfer"]["mean"])
    ds_t022 = float(t022_summary["delta_same"]["mean"])
    diff_vs_paper_transfer = abs(dt_t022 - PAPER_27B_TRANSFER)
    diff_vs_paper_same = abs(ds_t022 - PAPER_27B_SAME)

    tutor_aligned = True
    empty_lt_2pct = None
    student_fp16_deviation = True
    if a100_27b_diag is not None:
        cfg = a100_27b_diag.get("run_configuration") or {}
        roll = a100_27b_diag.get("rollout_diagnostics") or {}
        tutor_aligned = (
            cfg.get("teacher_dtype") == "bfloat16"
            and int(cfg.get("teacher_max_tokens", 0)) == 4192
            and bool(a100_27b_diag.get("omit_teacher_turns_in_post", True))
        )
        emp = roll.get("blanked_turn_rate")
        if emp is not None:
            empty_lt_2pct = float(emp) < 0.02 - TOL
        student_fp16_deviation = cfg.get("student_dtype") == "float16"

    # Under 预案 line 156: Tutor BF16, temp 0.6, 4192/16384, (hidden) masking, empty < 2%,
    # and T-022 aligned run reproduces paper (+0.2775 vs +0.29, |diff|=0.0125 < 0.02);
    # the only remaining protocol difference vs paper is Student fp16 (and post 1024/Agnes in C, D).
    if tutor_aligned and (empty_lt_2pct is not False) and diff_vs_paper_transfer < 0.02:
        verdict = "排除"
    else:
        verdict = "部分贡献"

    return {
        "verdict": verdict,
        "tutor_protocol_aligned_bf16_4192_16384_hidden": tutor_aligned,
        "selfbuilt_27b_empty_rate_lt_2pct": empty_lt_2pct,
        "student_dtype_was_fp16": student_fp16_deviation,
        "t022_author494_abs_diff_vs_paper_transfer": diff_vs_paper_transfer,
        "t022_author494_abs_diff_vs_paper_same": diff_vs_paper_same,
    }


def assert_output_sanitized(
    output_text: str,
    *,
    forbidden_exact_ids: Sequence[str] = (),
    forbidden_text_snippets: Sequence[str] = (),
) -> None:
    """Raise ValueError if any source_id or problem/answer/response text snippet appears in output."""
    for sid in forbidden_exact_ids:
        s = (sid or "").strip()
        # Check non-trivial identifiers (e.g. "test:123", "train_456", sha24 keys)
        if len(s) >= 4 and not s.isdigit() and s in output_text:
            raise ValueError("Sanitization check failed: source identifier leaked into output")
    for raw in forbidden_text_snippets:
        norm = (raw or "").strip()
        if len(norm) >= 18:
            probe = norm[:40]
            if probe in output_text:
                raise ValueError("Sanitization check failed: problem/answer/response text leaked into output")
