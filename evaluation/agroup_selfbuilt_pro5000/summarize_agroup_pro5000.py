#!/usr/bin/env python3
"""Sanitized Pro 5000 summary + paired_gain.py vs summarize_base4b cross-check for T-021."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = HERE.parents[1]
BASE4B_DIR = REPO_ROOT_DEFAULT / "evaluation" / "base4b_selfbuilt"
PAIRED_DIR = REPO_ROOT_DEFAULT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(BASE4B_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paired_gain as pg  # noqa: E402
from import_bundle import load_and_verify_bundle  # noqa: E402
from make_selfbuilt_paired_csv import (  # noqa: E402
    _load_base4b_csv_counts,
    build_selfbuilt_paired_csv,
)
from summarize_base4b import (  # noqa: E402
    _cohens_d,
    _derived_seed,
    _estimate_arm,
    _hake_g,
    bootstrap_mean_ci,
    collect_base4b_records,
    compute_summary,
    extract_safe_failure_summary,
    format_report,
    paired_bootstrap_diff,
)

GROUP_TO_SUMMARY_LABEL = {
    "orig": "原题合并",
    "orig_easy": "原题-简单对话",
    "orig_medium": "原题-中等对话",
    "orig_hard": "原题-困难对话",
    "easy": "简单迁移题",
    "medium": "中等迁移题",
    "hard": "困难迁移题",
}


def _with_scalar_pre(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold pre unknown into pre incorrect to match paired_gain CSV scalar pre column."""
    out: list[dict[str, Any]] = []
    for rec in records:
        new_rec = {**rec}
        for ck in ("counts", "ref_counts", "ref_counts_27b", "ref_counts_ar"):
            c_map = rec.get(ck)
            if not isinstance(c_map, dict):
                continue
            new_c = {}
            for tgt_k, phases in c_map.items():
                if not isinstance(phases, dict):
                    new_c[tgt_k] = phases
                    continue
                new_phases = {**phases}
                if isinstance(phases.get("pre"), dict):
                    pre_c = dict(phases["pre"])
                    unk = int(pre_c.get("unknown", 0))
                    if unk:
                        pre_c["incorrect"] = int(pre_c.get("incorrect", 0)) + unk
                        pre_c["unknown"] = 0
                    new_phases["pre"] = pre_c
                new_c[tgt_k] = new_phases
            new_rec[ck] = new_c
        out.append(new_rec)
    return out


def cross_check_paired_diff(
    paired_csv_rows: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
    ref_records: list[dict[str, Any]],
    *,
    base_name: str,
    target_name: str = "a_group",
    draws: int = 10000,
    seed: int = 42,
    ci_tol: float = 0.01,
) -> dict[str, Any]:
    """Compare paired_gain.py output against summarize_base4b bootstrap on matched sources."""
    pg_res = pg.evaluate_dataset(
        paired_csv_rows,
        base_condition=base_name,
        target_condition=target_name,
        draws=draws,
        seed=seed,
        include_high_pre_filtered=False,
    )
    sb_sum = compute_summary(
        summary_records, all_27b_records=ref_records, seed=seed, draws=draws
    )
    sb_sum_aligned = compute_summary(
        _with_scalar_pre(summary_records),
        all_27b_records=_with_scalar_pre(ref_records),
        seed=seed,
        draws=draws,
    )
    pg_index = {
        (r["unknown_policy_label"], r["group"]): r
        for r in pg_res.get("results", [])
        if r.get("source_filter") == "all"
    }
    mismatches: list[str] = []
    comparisons: dict[str, Any] = {}
    for policy in ("未决按错", "排除未决"):
        sb_rows_raw = {r["label"]: r for r in sb_sum["analyses"][policy]}
        sb_rows_aln = {r["label"]: r for r in sb_sum_aligned["analyses"][policy]}
        for grp in GROUP_TO_SUMMARY_LABEL:
            pg_entry = pg_index.get((policy, grp))
            lbl = GROUP_TO_SUMMARY_LABEL[grp]
            sb_entry = sb_rows_aln.get(lbl, {}).get("paired_4b_minus_27b")
            sb_raw_entry = sb_rows_raw.get(lbl, {}).get("paired_4b_minus_27b")
            if not pg_entry or not pg_entry.get("stats") or not sb_entry or sb_entry.get("mean_diff") is None:
                continue
            pg_st = pg_entry["stats"]
            diff_pt = abs(float(pg_st["diff_mean"]) - float(sb_entry["mean_diff"]))
            diff_lo = abs(float(pg_st["diff_ci_low"]) - float(sb_entry["ci"][0]))
            diff_hi = abs(float(pg_st["diff_ci_high"]) - float(sb_entry["ci"][1]))
            raw_diff = (
                float(sb_raw_entry["mean_diff"])
                if sb_raw_entry and sb_raw_entry.get("mean_diff") is not None
                else None
            )
            comparisons[f"{policy}.{grp}"] = {
                "n_paired_pg": pg_entry["n_paired"],
                "n_paired_sb": sb_entry["n_paired"],
                "pg_diff": pg_st["diff_mean"],
                "sb_diff": sb_entry["mean_diff"],
                "sb_diff_pre_excl_unk": raw_diff,
                "pt_diff": diff_pt,
                "ci_lo_diff": diff_lo,
                "ci_hi_diff": diff_hi,
            }
            if pg_entry["n_paired"] != sb_entry["n_paired"] or diff_pt > 1e-12 or diff_lo > ci_tol or diff_hi > ci_tol:
                mismatches.append(f"{policy}.{grp}")

    if mismatches:
        raise ValueError(f"paired_gain vs summarize_base4b cross-check mismatch: {mismatches}")
    return {"pg_report": pg_res, "sb_summary": sb_sum, "comparisons": comparisons}


def _count_inconsistent_orig_pre(loaded: dict[str, Any]) -> dict[str, int]:
    """Count sources whose 27B orig pre differs across tiers under Agnes vs Author-Rule."""
    out: dict[str, int] = {}
    for label, ref_key in (("agnes", "ref_counts"), ("ar", "ref_counts_ar")):
        by_sid: dict[str, list[float]] = {}
        for e in loaded["eligible"]:
            rc = e.get(ref_key)
            if not isinstance(rc, dict):
                continue
            by_sid.setdefault(e["source_id"], []).append(rc["原题"]["pre"]["correct"] / 16.0)
        out[label] = sum(1 for vals in by_sid.values() if max(vals) - min(vals) > pg.PRE_CONSISTENCY_TOL)
    return out


def build_base_records_from_csv(
    loaded_view: dict[str, Any],
    base4b_counts_csv: Path,
    *,
    grade_subdir: str = "AGNES",
) -> list[dict[str, Any]]:
    """Build summarize_base4b-compatible records for T-019 base from base4b_counts.csv."""
    prefix = "agnes" if grade_subdir == "AGNES" else "ar"
    counts_map = _load_base4b_csv_counts(base4b_counts_csv, prefix)
    out: list[dict[str, Any]] = []
    for e in loaded_view["eligible"]:
        k = (e["difficulty"], e["source_id"])
        if k not in counts_map:
            continue
        bc = counts_map[k]
        counts_by_target = {
            "原题": {
                "pre": dict(e["ref_counts"]["原题"]["pre"]),
                "control": dict(e["ref_counts"]["原题"]["control"]),
                "post": dict(bc["orig"]),
            },
            "迁移题": {
                "pre": dict(e["ref_counts"]["迁移题"]["pre"]),
                "control": dict(e["ref_counts"]["迁移题"]["control"]),
                "post": dict(bc["trans"]),
            },
        }
        out.append(
            {
                "difficulty": e["difficulty"],
                "tier_zh": e["tier_zh"],
                "source_id": e["source_id"],
                "source_key": e["source_key"],
                "counts": counts_by_target,
                "ref_counts_27b": e["ref_counts"],
            }
        )
    return out


def compute_extended_agroup_table(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute R0(pre/ctl/post), raw diffs, R1(pre/ctl), R2, R3, R4 for a_group."""
    rows_spec = [
        ("原题合并", "汇总", "原题", False),
        ("原题-简单对话", "简单", "原题", False),
        ("原题-中等对话", "中等", "原题", False),
        ("原题-困难对话", "困难", "原题", False),
        ("简单迁移题", "简单", "迁移题", False),
        ("中等迁移题", "中等", "迁移题", False),
        ("困难迁移题", "困难", "迁移题", False),
        ("简单迁移题-剔除高分", "简单", "迁移题", True),
    ]
    out: dict[str, Any] = {}
    for unknown_as_wrong, policy in ((True, "未决按错"), (False, "排除未决")):
        table: list[dict[str, Any]] = []
        for label, tier_zh, tgt_zh, filtered in rows_spec:
            p_pre, b_pre, o_post, n_filt, n_excl = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="pre",
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            p_ctl, b_ctl, o_post_ctl, _, _ = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="control",
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            p_r4, b_r4, o_r4, _, _ = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="pre",
                outcome_phase="control",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            m_pre = (sum(b_pre) / len(b_pre)) if b_pre else None
            m_ctl = (sum(b_ctl) / len(b_ctl)) if b_ctl else None
            m_post = (sum(o_post) / len(o_post)) if o_post else None
            m_post_ctl = (sum(o_post_ctl) / len(o_post_ctl)) if o_post_ctl else None
            table.append(
                {
                    "label": label,
                    "n": len(p_pre),
                    "n_filtered": n_filt,
                    "n_excluded_unknown": n_excl,
                    "r0_pre": m_pre,
                    "r0_ctl": m_ctl,
                    "r0_post": m_post,
                    "raw_diff_pre": (m_post - m_pre) if (m_post is not None and m_pre is not None) else None,
                    "raw_diff_ctl": (m_post_ctl - m_ctl) if (m_post_ctl is not None and m_ctl is not None) else None,
                    "r1_pre": bootstrap_mean_ci([v for _, v in p_pre], seed=_derived_seed(seed, f"{policy}|{label}|ag_pre"), draws=draws),
                    "r1_ctl": bootstrap_mean_ci([v for _, v in p_ctl], seed=_derived_seed(seed, f"{policy}|{label}|ag_ctl"), draws=draws),
                    "r2_g": _hake_g(b_pre, o_post),
                    "r3_d": _cohens_d(b_pre, o_post),
                    "r4_pre_to_ctl": bootstrap_mean_ci([v for _, v in p_r4], seed=_derived_seed(seed, f"{policy}|{label}|ag_r4"), draws=draws),
                    "r4_r2_g": _hake_g(b_r4, o_r4),
                    "r4_r3_d": _cohens_d(b_r4, o_r4),
                }
            )
        out[policy] = table
    return out


def cross_check_agroup_vs_base(
    paired_csv_rows: list[dict[str, Any]],
    a_records: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
    *,
    draws: int = 10000,
    seed: int = 42,
    ci_tol: float = 0.01,
) -> dict[str, Any]:
    """Compute paired (a_group - base) incl. 原题合并 via summarize_base4b and cross-check paired_gain.py."""
    pg_base = pg.evaluate_dataset(
        paired_csv_rows,
        base_condition="base",
        target_condition="a_group",
        draws=draws,
        seed=seed,
        include_high_pre_filtered=True,
    )
    pg_index = {
        (r["unknown_policy_label"], r["group"]): r
        for r in pg_base.get("results", [])
        if r.get("source_filter") == "all"
    }
    a_aln = _with_scalar_pre(a_records)
    b_aln = _with_scalar_pre(base_records)

    orig_combined: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    mismatches: list[str] = []
    specs = [
        ("orig", "原题合并", "汇总", "原题"),
        ("orig_easy", "原题-简单对话", "简单", "原题"),
        ("orig_medium", "原题-中等对话", "中等", "原题"),
        ("orig_hard", "原题-困难对话", "困难", "原题"),
        ("easy", "简单迁移题", "简单", "迁移题"),
        ("medium", "中等迁移题", "中等", "迁移题"),
        ("hard", "困难迁移题", "困难", "迁移题"),
    ]
    for unknown_as_wrong, policy in ((True, "未决按错"), (False, "排除未决")):
        for grp, lbl, tier_zh, tgt_zh in specs:
            p_ag_raw, _, _, _, _ = _estimate_arm(
                a_records, tier_filter=tier_zh, target_zh=tgt_zh, baseline="pre",
                outcome_phase="post", unknown_as_wrong=unknown_as_wrong, easy_filtered=False,
            )
            p_bs_raw, _, _, _, _ = _estimate_arm(
                base_records, tier_filter=tier_zh, target_zh=tgt_zh, baseline="pre",
                outcome_phase="post", unknown_as_wrong=unknown_as_wrong, easy_filtered=False,
            )
            sb_raw = paired_bootstrap_diff(
                p_ag_raw, p_bs_raw, seed=_derived_seed(seed, f"{policy}|{lbl}|ag_vs_base_raw"), draws=draws
            )
            # Note: paired_bootstrap_diff computes (first - second) = (a_group - base)
            if grp == "orig":
                orig_combined[policy] = sb_raw
                continue

            p_ag_aln, _, _, _, _ = _estimate_arm(
                a_aln, tier_filter=tier_zh, target_zh=tgt_zh, baseline="pre",
                outcome_phase="post", unknown_as_wrong=unknown_as_wrong, easy_filtered=False,
            )
            p_bs_aln, _, _, _, _ = _estimate_arm(
                b_aln, tier_filter=tier_zh, target_zh=tgt_zh, baseline="pre",
                outcome_phase="post", unknown_as_wrong=unknown_as_wrong, easy_filtered=False,
            )
            sb_aln = paired_bootstrap_diff(
                p_ag_aln, p_bs_aln, seed=_derived_seed(seed, f"{policy}|{lbl}|ag_vs_base_aln"), draws=draws
            )
            pg_entry = pg_index.get((policy, grp))
            if not pg_entry or not pg_entry.get("stats") or sb_aln.get("mean_diff") is None:
                continue
            pg_st = pg_entry["stats"]
            pt_diff = abs(float(pg_st["diff_mean"]) - float(sb_aln["mean_diff"]))
            lo_diff = abs(float(pg_st["diff_ci_low"]) - float(sb_aln["ci"][0]))
            hi_diff = abs(float(pg_st["diff_ci_high"]) - float(sb_aln["ci"][1]))
            comparisons[f"{policy}.{grp}"] = {
                "n_paired_pg": pg_entry["n_paired"],
                "n_paired_sb": sb_aln["n_paired"],
                "pg_diff": pg_st["diff_mean"],
                "sb_diff_scalar_pre": sb_aln["mean_diff"],
                "sb_diff_source_gain": sb_raw["mean_diff"],
                "pt_diff": pt_diff,
                "ci_lo_diff": lo_diff,
                "ci_hi_diff": hi_diff,
            }
            if pg_entry["n_paired"] != sb_aln["n_paired"] or pt_diff > 1e-12 or lo_diff > ci_tol or hi_diff > ci_tol:
                mismatches.append(f"{policy}.{grp}")

    if mismatches:
        raise ValueError(f"a_group vs base paired_gain vs summarize_base4b mismatch: {mismatches}")
    return {"pg_report": pg_base, "orig_combined": orig_combined, "comparisons": comparisons}


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanitized summary + paired comparison for T-021.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--base4b-counts-csv", type=Path, default=None)
    ap.add_argument("--grade-subdir", choices=("AGNES", "AUTHOR_RULE"), default="AGNES")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--elapsed-seconds", type=float, default=0.0)
    ap.add_argument("--gen-elapsed-seconds", type=float, default=0.0)
    ap.add_argument("--grade-elapsed-seconds", type=float, default=0.0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    loaded = load_and_verify_bundle(args.bundle_dir, require_exact_totals=True)
    if args.grade_subdir == "AUTHOR_RULE":
        loaded_view = {
            **loaded,
            "eligible": [{**e, "ref_counts": e["ref_counts_ar"]} for e in loaded["eligible"]],
        }
    else:
        loaded_view = loaded
    records, diagnostics = collect_base4b_records(
        loaded_view, args.out_root, args.repo, grade_subdir=args.grade_subdir
    )
    all_27b = [
        {"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]}
        for e in loaded_view["eligible"]
    ]
    summary = compute_summary(records, all_27b_records=all_27b, seed=args.seed, draws=args.draws)
    print(
        format_report(
            loaded_view,
            diagnostics,
            summary,
            elapsed_seconds=args.elapsed_seconds,
            gen_elapsed_seconds=args.gen_elapsed_seconds,
            grade_elapsed_seconds=args.grade_elapsed_seconds,
        )
    )

    paired_csv = args.out_root / f"paired_{args.grade_subdir.lower()}.csv"
    paired_info = build_selfbuilt_paired_csv(
        loaded_bundle=loaded,
        a_group_root=args.out_root,
        base4b_counts_csv=args.base4b_counts_csv,
        out_csv=paired_csv,
        grade_subdir=args.grade_subdir,
    )
    csv_rows = pg.load_csv(paired_csv)
    incons = _count_inconsistent_orig_pre(loaded)
    if not paired_info.get("emit_combined_orig", True):
        print(
            "[base4b_summary] note: group=orig omitted from paired_gain CSV "
            f"(27B orig_pre differs across tiers for {paired_info['orig_inconsistent_sources']} sources "
            f"[agnes={incons['agnes']}, author_rule={incons['ar']}]; "
            "combined 原题合并 uses exact per-tier 27B aggregation in summary above, "
            "and orig_easy/orig_medium/orig_hard are included in paired_gain CSV)"
        )

    ext_agroup = compute_extended_agroup_table(records, seed=args.seed, draws=args.draws)
    print(f"[base4b_summary] === T-021 {args.grade_subdir} Extended Secondary (R0_pre/ctl/post, raw_diff, R1_ctl, R4) ===")
    for policy in ("未决按错", "排除未决"):
        for r in ext_agroup[policy]:
            r1p = r["r1_pre"]
            r1c = r["r1_ctl"]
            r4c = r["r4_pre_to_ctl"]
            r1p_s = f"{r1p['mean']:+.4f}[{r1p['ci'][0]:+.4f},{r1p['ci'][1]:+.4f}]" if r1p else "N/A"
            r1c_s = f"{r1c['mean']:+.4f}[{r1c['ci'][0]:+.4f},{r1c['ci'][1]:+.4f}]" if r1c else "N/A"
            r4c_s = f"{r4c['mean']:+.4f}[{r4c['ci'][0]:+.4f},{r4c['ci'][1]:+.4f}]" if r4c else "N/A"
            print(
                f"[base4b_summary] [{policy}] {r['label']} n={r['n']} pre={r['r0_pre']*100:.2f}% "
                f"ctl={r['r0_ctl']*100:.2f}% post={r['r0_post']*100:.2f}% "
                f"diff_pre={r['raw_diff_pre']:+.4f} diff_ctl={r['raw_diff_ctl']:+.4f} "
                f"R1={r1p_s} R1_ctl={r1c_s} R2={r['r2_g']:+.3f} R3={r['r3_d']:+.3f} R4={r4c_s}"
            )

    payload: dict[str, Any] = {
        "diagnostics": diagnostics,
        "summary_vs_27b": summary,
        "extended_agroup": ext_agroup,
        "paired_csv_info": {
            "emit_combined_orig": paired_info.get("emit_combined_orig"),
            "orig_inconsistent_sources": paired_info.get("orig_inconsistent_sources"),
            "orig_inconsistent_by_grader": incons,
        },
    }
    agnes_rep_p = args.out_root / "AGNES" / "AGNES_GRADING_REPORT.json"
    if args.grade_subdir == "AGNES" and agnes_rep_p.is_file():
        ag_rep = json.loads(agnes_rep_p.read_text(encoding="utf-8"))
        payload["agnes_grading_report"] = ag_rep
        w = ag_rep.get("window") or {}
        print(
            f"[base4b_summary] agnes_window: max_limit={w.get('max_limit_reached')} "
            f"drops={w.get('window_drops')} http429={w.get('http_429_events')} "
            f"tech_fail={ag_rep.get('technical_failures')} sem_null={ag_rep.get('semantic_unknown')} "
            f"primary_status={ag_rep.get('primary_status')}"
        )
    if records:
        cc_27b = cross_check_paired_diff(
            csv_rows,
            records,
            all_27b,
            base_name="eduardo27b",
            target_name="a_group",
            draws=args.draws,
            seed=args.seed,
        )
        md_27b = pg.render_markdown(cc_27b["pg_report"])
        for ln in md_27b.splitlines():
            print(f"[base4b_summary] {ln}")
        payload["paired_vs_27b"] = cc_27b["pg_report"]
        payload["cross_check_27b"] = cc_27b["comparisons"]

    if args.base4b_counts_csv is not None and records:
        base_records = build_base_records_from_csv(
            loaded_view, args.base4b_counts_csv, grade_subdir=args.grade_subdir
        )
        cc_base = cross_check_agroup_vs_base(
            csv_rows,
            records,
            base_records,
            draws=args.draws,
            seed=args.seed,
        )
        payload["paired_vs_base"] = cc_base["pg_report"]
        payload["orig_combined_vs_base"] = cc_base["orig_combined"]
        payload["cross_check_base"] = cc_base["comparisons"]
        for pol, oc in cc_base["orig_combined"].items():
            if oc and oc.get("mean_diff") is not None:
                print(
                    f"[base4b_summary] orig_combined_vs_base[{pol}] n={oc['n_paired']} "
                    f"diff={oc['mean_diff']:+.4f} [{oc['ci'][0]:+.4f},{oc['ci'][1]:+.4f}] p_gt_0={oc['p_gt_0']:.4f}"
                )
        for ln in pg.render_markdown(cc_base["pg_report"]).splitlines():
            print(f"[base4b_summary] {ln}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
