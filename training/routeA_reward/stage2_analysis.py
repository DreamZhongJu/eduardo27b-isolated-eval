"""Stage 2 dev-set simulation statistical analysis (checks (a) through (g)).

Implements every pre-registered check in T-024 v4 (inbox #4):
  (a) Paired bootstrap (10,000 draws, seed=42) of theta_hat(a_group) - theta_hat(base)
  (b) Split-half reliability (8 vs 8) & Spearman-Brown prophecy for k in {8, 12, 16}
  (c) Within-source dialogue variance vs measurement noise variance
  (d) Reward scale distribution, prior-pulled rate, and comparison vs -0.2 gate penalty
  (e) Alternative scoring rules on identical ladder responses:
      (1) 1 - p sum per climb, (2) max passed Rasch b, (3) discrete ladder score
  (f) Compute & 27B grader load projection for 16 problems x 16 rollouts = 256 / step
  (g) Four-stratum breakdown of low-discrimination / flat-difficulty source problems

Privacy: operates strictly on numeric/metadata records; never outputs problem,
answer, or student response text.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from .rasch import rasch_map_estimate

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42
RELIABILITY_THRESHOLD = 0.7
GATE_PENALTY_ABS = 0.2
TRAIN_ROLLOUTS_PER_STEP = 16 * 16  # 16 prompts x 16 trajectories
AUTHOR_GATE_JUDGE_CALLS_PER_STEP = TRAIN_ROLLOUTS_PER_STEP * 2 * 3  # 2 criteria x 3 votes

STRATUM_ORDER = (
    "all_le_1_16",
    "all_ge_15_16",
    "narrow_span_lt_0_5",
    "other",
)


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: Sequence[float], ddof: int = 1) -> float:
    n = len(vals)
    if n <= ddof:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (n - ddof))


def _var(vals: Sequence[float], ddof: int = 1) -> float:
    n = len(vals)
    if n <= ddof:
        return 0.0
    mu = _mean(vals)
    return sum((x - mu) ** 2 for x in vals) / (n - ddof)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def paired_bootstrap(
    diffs: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Non-parametric paired bootstrap over source-level differences."""
    n = len(diffs)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_gt_0": None,
            "cohen_dz": None,
        }
    vals = [float(x) for x in diffs]
    obs_mean = _mean(vals)
    obs_std = _std(vals, ddof=1)
    dz = (obs_mean / obs_std) if obs_std > 0.0 else None

    if n == 1:
        return {
            "n": 1,
            "mean": obs_mean,
            "std": 0.0,
            "ci95_low": obs_mean,
            "ci95_high": obs_mean,
            "p_gt_0": 1.0 if obs_mean > 0.0 else 0.0,
            "cohen_dz": None,
        }

    rng = random.Random(seed)
    boot_means = [
        sum(vals[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    ]
    boot_means.sort()
    p_gt_0 = sum(1 for m in boot_means if m > 0.0) / draws
    return {
        "n": n,
        "mean": obs_mean,
        "std": obs_std,
        "ci95_low": _quantile(boot_means, 0.025),
        "ci95_high": _quantile(boot_means, 0.975),
        "p_gt_0": p_gt_0,
        "cohen_dz": dz,
    }


def spearman_brown(r_base: float, multiplier: float) -> float:
    """Spearman-Brown prophecy reliability for ``multiplier * k_base`` climbs."""
    if r_base <= 0.0:
        return float(r_base)
    denom = 1.0 + (multiplier - 1.0) * r_base
    if denom <= 0.0:
        return 0.0
    return float((multiplier * r_base) / denom)


def choose_k_from_split_half(
    r_half8: float,
    *,
    threshold: float = RELIABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Compute Spearman-Brown reliabilities for k=8,12,16 and pick k."""
    rel_8 = spearman_brown(r_half8, 1.0)
    rel_12 = spearman_brown(r_half8, 1.5)
    rel_16 = spearman_brown(r_half8, 2.0)
    if rel_8 >= threshold:
        chosen_k = 8
    elif rel_12 >= threshold:
        chosen_k = 12
    else:
        chosen_k = 16
    return {
        "r_split_half_8v8": float(r_half8),
        "reliability_k8": rel_8,
        "reliability_k12": rel_12,
        "reliability_k16": rel_16,
        "threshold": threshold,
        "chosen_k": chosen_k,
    }


def classify_source_stratum(variants: Sequence[dict[str, Any]]) -> str:
    """Classify a source problem into T-024 (g)'s four mutually exclusive strata.

    1. ``all_le_1_16``: every variant has ``n_correct_hybrid <= 1``
    2. ``all_ge_15_16``: every variant has ``n_correct_hybrid >= 15``
    3. ``narrow_span_lt_0_5``: not in (1)/(2), and ``max(b) - min(b) < 0.5``
    4. ``other``: all remaining source problems (``max(b) - min(b) >= 0.5``)
    """
    if not variants:
        raise ValueError("variants must be non-empty")
    counts = [int(v["n_correct_hybrid"]) for v in variants]
    if all(c <= 1 for c in counts):
        return "all_le_1_16"
    if all(c >= 15 for c in counts):
        return "all_ge_15_16"
    bs = [float(v["b"]) for v in variants]
    if (max(bs) - min(bs)) < 0.5:
        return "narrow_span_lt_0_5"
    return "other"


def score_dialogue_from_climbs(
    climbs: Sequence[dict[str, Any]],
    variants: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Rasch MAP (full 16, half A 0..7, half B 8..15) and (e) alternatives.

    Each climb dict has ``steps``: list of dicts with keys
    ``b`` (float), ``p_new`` (float), ``correct`` (bool), ``tech_failed`` (bool).
    """
    n_climbs = len(climbs)
    if n_climbs == 0:
        raise ValueError("climbs must be non-empty")
    num_variants = len(variants)
    b_min = min(float(v["b"]) for v in variants)

    def _collect_obs(subset: Sequence[dict[str, Any]]) -> list[tuple[float, int]]:
        obs: list[tuple[float, int]] = []
        for cl in subset:
            for st in cl.get("steps", []):
                if st.get("tech_failed"):
                    continue
                obs.append((float(st["b"]), 1 if st.get("correct") else 0))
        return obs

    obs_full = _collect_obs(climbs)
    est_full = rasch_map_estimate(obs_full)

    half = n_climbs // 2
    obs_a = _collect_obs(climbs[:half])
    obs_b = _collect_obs(climbs[half : 2 * half])
    est_a = rasch_map_estimate(obs_a)
    est_b = rasch_map_estimate(obs_b)

    # Alternative scorers (e) computed on the exact same climbs (excluding climbs
    # that failed technically on rung 0 with zero graded steps).
    one_minus_p_vals: list[float] = []
    max_b_vals: list[float] = []
    discrete_frac_vals: list[float] = []
    discrete_raw_vals: list[float] = []

    for cl in climbs:
        graded_steps = [st for st in cl.get("steps", []) if not st.get("tech_failed")]
        if not graded_steps:
            continue
        passed_steps = [st for st in graded_steps if st.get("correct")]
        one_minus_p_vals.append(sum(1.0 - float(st["p_new"]) for st in passed_steps))
        if passed_steps:
            max_b_vals.append(max(float(st["b"]) for st in passed_steps))
        else:
            # Failed rung 0 -> strictly below easiest rung b_min (constant offset per source cancels in paired diff).
            max_b_vals.append(b_min - 1.0)
        discrete_raw_vals.append(float(len(passed_steps)))
        discrete_frac_vals.append(len(passed_steps) / num_variants)

    return {
        "rasch_valid": bool(est_full.valid),
        "theta_full": est_full.theta,
        "prior_pulled": bool(est_full.prior_pulled),
        "all_correct": bool(est_full.all_correct),
        "all_wrong": bool(est_full.all_wrong),
        "n_obs_full": est_full.n_obs,
        "theta_half_a": est_a.theta if est_a.valid else None,
        "theta_half_b": est_b.theta if est_b.valid else None,
        "alt_one_minus_p": _mean(one_minus_p_vals) if one_minus_p_vals else None,
        "alt_max_b": _mean(max_b_vals) if max_b_vals else None,
        "alt_discrete_frac": _mean(discrete_frac_vals) if discrete_frac_vals else None,
        "alt_discrete_raw": _mean(discrete_raw_vals) if discrete_raw_vals else None,
    }


def _describe_distribution(vals: Sequence[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    s = sorted(float(x) for x in vals)
    q25 = _quantile(s, 0.25)
    q75 = _quantile(s, 0.75)
    return {
        "n": len(s),
        "mean": _mean(s),
        "std": _std(s, ddof=1),
        "min": s[0],
        "p05": _quantile(s, 0.05),
        "p10": _quantile(s, 0.10),
        "p25": q25,
        "p50": _quantile(s, 0.50),
        "p75": q75,
        "p90": _quantile(s, 0.90),
        "p95": _quantile(s, 0.95),
        "max": s[-1],
        "iqr": q75 - q25,
        "frac_lt_minus_0_2": sum(1 for x in s if x < -GATE_PENALTY_ABS) / len(s),
        "frac_gt_0": sum(1 for x in s if x > 0.0) / len(s),
    }


def _variance_decomposition(
    records: Sequence[dict[str, Any]],
    *,
    chosen_k: int,
) -> dict[str, Any]:
    """Compute within-source dialogue variance vs measurement noise variance."""
    by_tutor_source: dict[tuple[str, str], list[float]] = {}
    all_thetas: list[float] = []
    half_sq_diffs: list[float] = []

    for r in records:
        th = r.get("theta_full")
        if th is not None:
            all_thetas.append(float(th))
            key = (str(r["tutor"]), str(r["source_id"]))
            by_tutor_source.setdefault(key, []).append(float(th))
        ha, hb = r.get("theta_half_a"), r.get("theta_half_b")
        if ha is not None and hb is not None:
            half_sq_diffs.append((float(ha) - float(hb)) ** 2)

    within_vars: list[float] = []
    for vals in by_tutor_source.values():
        if len(vals) >= 2:
            within_vars.append(_var(vals, ddof=1))

    within_obs_var = _mean(within_vars) if within_vars else 0.0
    total_var = _var(all_thetas, ddof=1) if len(all_thetas) >= 2 else 0.0
    noise_var_k8 = 0.5 * _mean(half_sq_diffs) if half_sq_diffs else 0.0
    noise_var_k16 = 0.25 * _mean(half_sq_diffs) if half_sq_diffs else 0.0
    noise_var_chosen = (8.0 / chosen_k) * noise_var_k8 if chosen_k > 0 else noise_var_k16

    true_signal_var_k16 = max(0.0, within_obs_var - noise_var_k16)
    snr_k16 = (true_signal_var_k16 / noise_var_k16) if noise_var_k16 > 0.0 else None
    snr_chosen = (
        (true_signal_var_k16 / noise_var_chosen) if noise_var_chosen > 0.0 else None
    )

    return {
        "n_dialogues": len(all_thetas),
        "n_tutor_source_pairs": len(within_vars),
        "total_var": total_var,
        "within_source_obs_var": within_obs_var,
        "within_var_share_of_total": (within_obs_var / total_var) if total_var > 0.0 else None,
        "measurement_noise_var_k8": noise_var_k8,
        "measurement_noise_var_k16": noise_var_k16,
        "measurement_noise_var_chosen_k": noise_var_chosen,
        "true_dialogue_signal_var": true_signal_var_k16,
        "snr_at_k16": snr_k16,
        "snr_at_chosen_k": snr_chosen,
    }


def analyze_stage2(
    dialogue_records: Sequence[dict[str, Any]],
    source_meta: dict[str, dict[str, Any]],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run all Stage 2 checks (a)-(g) on completed dev-set simulation records.

    ``dialogue_records``: list of per-dialogue dicts with keys:
      ``tutor`` ("base" | "a_group"), ``source_id``, ``dialogue_idx``,
      ``climbs`` (list of 16 climb dicts), ``student_calls``, ``student_failed``,
      ``grader_llm_calls``, ``grader_rule_correct``, ``grader_llm_unknown``,
      ``grader_tech_failed``, ``grader_llm_time_s`` (total 27B wall seconds for
      this dialogue's grading calls), ``grader_llm_latencies_ms`` (optional list).
    ``source_meta``: ``{source_id: {"variants": [...]}}``.
    """
    enriched: list[dict[str, Any]] = []
    for rec in dialogue_records:
        sid = str(rec["source_id"])
        variants = source_meta[sid]["variants"]
        scored = score_dialogue_from_climbs(rec["climbs"], variants)
        enriched.append({**rec, **scored})

    # Group by source_id and tutor for paired analyses
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in enriched:
        sid = str(r["source_id"])
        tutor = str(r["tutor"])
        by_source.setdefault(sid, {"base": [], "a_group": []})
        by_source[sid][tutor].append(r)

    def _paired_for_metric(
        metric_key: str,
        source_ids: Sequence[str],
    ) -> dict[str, Any]:
        base_means: list[float] = []
        ag_means: list[float] = []
        diffs: list[float] = []
        for sid in source_ids:
            grp = by_source.get(sid)
            if not grp:
                continue
            b_vals = [float(r[metric_key]) for r in grp["base"] if r.get(metric_key) is not None]
            a_vals = [float(r[metric_key]) for r in grp["a_group"] if r.get(metric_key) is not None]
            if not b_vals or not a_vals:
                continue
            bm = _mean(b_vals)
            am = _mean(a_vals)
            base_means.append(bm)
            ag_means.append(am)
            diffs.append(am - bm)
        boot = paired_bootstrap(diffs, draws=bootstrap_draws, seed=bootstrap_seed)
        return {
            "n_paired": len(diffs),
            "base_mean": _mean(base_means) if base_means else None,
            "a_group_mean": _mean(ag_means) if ag_means else None,
            **boot,
        }

    all_sids = sorted(source_meta.keys())

    # (a) Main discrimination check
    check_a = _paired_for_metric("theta_full", all_sids)
    check_a["passed_ci_low_gt_0"] = (
        check_a["ci95_low"] is not None and check_a["ci95_low"] > 0.0
    )

    # (b) Split-half reliability & Spearman-Brown k selection
    valid_halves = [
        (float(r["theta_half_a"]), float(r["theta_half_b"]))
        for r in enriched
        if r.get("theta_half_a") is not None and r.get("theta_half_b") is not None
    ]
    r_all = _pearson([x[0] for x in valid_halves], [x[1] for x in valid_halves])
    check_b = choose_k_from_split_half(r_all)
    check_b["n_dialogues"] = len(valid_halves)
    for tutor in ("base", "a_group"):
        sub = [
            (float(r["theta_half_a"]), float(r["theta_half_b"]))
            for r in enriched
            if r["tutor"] == tutor
            and r.get("theta_half_a") is not None
            and r.get("theta_half_b") is not None
        ]
        check_b[f"r_split_half_{tutor}"] = _pearson([x[0] for x in sub], [x[1] for x in sub])
    chosen_k = int(check_b["chosen_k"])

    # (c) Signal vs noise variance decomposition
    check_c = {
        "pooled": _variance_decomposition(enriched, chosen_k=chosen_k),
        "base": _variance_decomposition(
            [r for r in enriched if r["tutor"] == "base"], chosen_k=chosen_k
        ),
        "a_group": _variance_decomposition(
            [r for r in enriched if r["tutor"] == "a_group"], chosen_k=chosen_k
        ),
    }

    # (d) Reward scale & prior-pulled rate vs -0.2 gate penalty
    valid_recs = [r for r in enriched if r.get("rasch_valid") and r.get("theta_full") is not None]
    thetas_all = [float(r["theta_full"]) for r in valid_recs]
    thetas_base = [float(r["theta_full"]) for r in valid_recs if r["tutor"] == "base"]
    thetas_ag = [float(r["theta_full"]) for r in valid_recs if r["tutor"] == "a_group"]

    def _prior_rates(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        n = len(rows)
        if n == 0:
            return {"prior_pulled": 0.0, "all_correct": 0.0, "all_wrong": 0.0}
        return {
            "prior_pulled": sum(1 for r in rows if r.get("prior_pulled")) / n,
            "all_correct": sum(1 for r in rows if r.get("all_correct")) / n,
            "all_wrong": sum(1 for r in rows if r.get("all_wrong")) / n,
        }

    dist_all = _describe_distribution(thetas_all)
    check_d = {
        "all": dist_all,
        "base": _describe_distribution(thetas_base),
        "a_group": _describe_distribution(thetas_ag),
        "prior_pulled_all": _prior_rates(valid_recs),
        "prior_pulled_base": _prior_rates([r for r in valid_recs if r["tutor"] == "base"]),
        "prior_pulled_a_group": _prior_rates([r for r in valid_recs if r["tutor"] == "a_group"]),
        "vs_gate_penalty_0_2": {
            "std_to_penalty_ratio": (
                dist_all["std"] / GATE_PENALTY_ABS if dist_all.get("std") is not None else None
            ),
            "iqr_to_penalty_ratio": (
                dist_all["iqr"] / GATE_PENALTY_ABS if dist_all.get("iqr") is not None else None
            ),
            "paired_diff_to_penalty_ratio": (
                abs(check_a["mean"]) / GATE_PENALTY_ABS if check_a.get("mean") is not None else None
            ),
        },
    }

    # (e) Alternative scoring rules on identical ladder responses
    check_e = {
        "rasch_map_main": {
            "mean_diff": check_a["mean"],
            "ci95_low": check_a["ci95_low"],
            "ci95_high": check_a["ci95_high"],
            "cohen_dz": check_a["cohen_dz"],
        },
        "alt1_one_minus_p": _paired_for_metric("alt_one_minus_p", all_sids),
        "alt2_max_passed_b": _paired_for_metric("alt_max_b", all_sids),
        "alt3_discrete_ladder_frac": _paired_for_metric("alt_discrete_frac", all_sids),
        "alt3_discrete_ladder_raw": _paired_for_metric("alt_discrete_raw", all_sids),
    }

    # (f) Cost & 27B judge load projection
    n_diag = len(enriched)
    total_student_calls = sum(int(r.get("student_calls", 0)) for r in enriched)
    total_student_failed = sum(int(r.get("student_failed", 0)) for r in enriched)
    total_llm_calls = sum(int(r.get("grader_llm_calls", 0)) for r in enriched)
    total_rule_correct = sum(int(r.get("grader_rule_correct", 0)) for r in enriched)
    total_llm_unknown = sum(int(r.get("grader_llm_unknown", 0)) for r in enriched)
    total_grader_tech_failed = sum(int(r.get("grader_tech_failed", 0)) for r in enriched)
    total_llm_time_s = sum(float(r.get("grader_llm_time_s", 0.0)) for r in enriched)
    all_lat_ms = [
        float(x)
        for r in enriched
        for x in r.get("grader_llm_latencies_ms", [])
    ]
    all_lat_ms.sort()

    mean_student_calls_k16 = total_student_calls / n_diag if n_diag else 0.0
    mean_llm_calls_k16 = total_llm_calls / n_diag if n_diag else 0.0
    mean_llm_time_k16 = total_llm_time_s / n_diag if n_diag else 0.0

    def _project_step(k: int) -> dict[str, float]:
        scale = k / 16.0
        llm_per_diag = mean_llm_calls_k16 * scale
        llm_per_step = llm_per_diag * TRAIN_ROLLOUTS_PER_STEP
        return {
            "k": k,
            "student_calls_per_dialogue": mean_student_calls_k16 * scale,
            "student_calls_per_step_256": mean_student_calls_k16 * scale * TRAIN_ROLLOUTS_PER_STEP,
            "llm_27b_calls_per_dialogue": llm_per_diag,
            "llm_27b_calls_per_step_256": llm_per_step,
            "extra_judge_load_ratio_vs_1536": llm_per_step / AUTHOR_GATE_JUDGE_CALLS_PER_STEP,
            "est_llm_27b_call_seconds_per_step": mean_llm_time_k16 * scale * TRAIN_ROLLOUTS_PER_STEP,
        }

    check_f = {
        "n_dialogues": n_diag,
        "total_student_calls_k16": total_student_calls,
        "total_student_failed": total_student_failed,
        "total_grader_rule_correct": total_rule_correct,
        "total_grader_llm_calls_k16": total_llm_calls,
        "total_grader_llm_unknown": total_llm_unknown,
        "total_grader_tech_failed": total_grader_tech_failed,
        "llm_call_rate_of_graded": (
            total_llm_calls / (total_llm_calls + total_rule_correct)
            if (total_llm_calls + total_rule_correct) > 0
            else 0.0
        ),
        "llm_unknown_rate": (total_llm_unknown / total_llm_calls) if total_llm_calls else 0.0,
        "llm_tech_failure_rate": (
            total_grader_tech_failed / total_llm_calls if total_llm_calls else 0.0
        ),
        "mean_student_calls_per_dialogue_k16": mean_student_calls_k16,
        "mean_llm_27b_calls_per_dialogue_k16": mean_llm_calls_k16,
        "mean_llm_27b_latency_ms": _mean(all_lat_ms) if all_lat_ms else None,
        "p95_llm_27b_latency_ms": _quantile(all_lat_ms, 0.95) if all_lat_ms else None,
        "total_llm_27b_wall_s": total_llm_time_s,
        "author_gate_judge_calls_per_step": AUTHOR_GATE_JUDGE_CALLS_PER_STEP,
        "step_projection_k8": _project_step(8),
        "step_projection_k12": _project_step(12),
        "step_projection_k16": _project_step(16),
        "step_projection_chosen_k": _project_step(chosen_k),
    }

    # (g) Four-stratum breakdown
    strata_sids: dict[str, list[str]] = {name: [] for name in STRATUM_ORDER}
    narrow_single = 0
    narrow_multi = 0
    for sid in all_sids:
        variants = source_meta[sid]["variants"]
        st = classify_source_stratum(variants)
        strata_sids[st].append(sid)
        if st == "narrow_span_lt_0_5":
            if len(variants) == 1:
                narrow_single += 1
            else:
                narrow_multi += 1

    check_g: dict[str, Any] = {}
    for name in STRATUM_ORDER:
        sids = strata_sids[name]
        sid_set = set(sids)
        sub_recs = [r for r in enriched if str(r["source_id"]) in sid_set]
        vdec = _variance_decomposition(sub_recs, chosen_k=chosen_k)
        paired = _paired_for_metric("theta_full", sids)
        entry: dict[str, Any] = {
            "n_sources": len(sids),
            "n_dialogues": vdec["n_dialogues"],
            "within_source_obs_var": vdec["within_source_obs_var"],
            "total_var": vdec["total_var"],
            "within_var_share_of_total": vdec["within_var_share_of_total"],
            "paired_diff": paired,
        }
        if name == "narrow_span_lt_0_5":
            entry["n_single_variant"] = narrow_single
            entry["n_multi_variant"] = narrow_multi
        check_g[name] = entry

    return {
        "n_sources": len(all_sids),
        "n_dialogues_total": n_diag,
        "check_a_discrimination": check_a,
        "check_b_reliability_k": check_b,
        "check_c_variance_signal": check_c,
        "check_d_reward_scale": check_d,
        "check_e_alternatives": check_e,
        "check_f_cost_and_load": check_f,
        "check_g_low_disc_strata": check_g,
    }
