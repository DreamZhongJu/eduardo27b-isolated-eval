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
from make_selfbuilt_paired_csv import build_selfbuilt_paired_csv  # noqa: E402
from summarize_base4b import (  # noqa: E402
    collect_base4b_records,
    compute_summary,
    extract_safe_failure_summary,
    format_report,
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
    if not paired_info.get("emit_combined_orig", True):
        print(
            "[base4b_summary] note: group=orig omitted from paired_gain CSV "
            f"(27B orig_pre differs across tiers for {paired_info['orig_inconsistent_sources']} sources; "
            "combined 原题合并 uses exact per-tier 27B aggregation in summary above, "
            "and orig_easy/orig_medium/orig_hard are included in paired_gain CSV)"
        )

    payload: dict[str, Any] = {
        "diagnostics": diagnostics,
        "summary_vs_27b": summary,
        "paired_csv_info": {
            "emit_combined_orig": paired_info.get("emit_combined_orig"),
            "orig_inconsistent_sources": paired_info.get("orig_inconsistent_sources"),
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
        pg_base = pg.evaluate_dataset(
            csv_rows,
            base_condition="base",
            target_condition="a_group",
            draws=args.draws,
            seed=args.seed,
            include_high_pre_filtered=True,
        )
        payload["paired_vs_base"] = pg_base
        for ln in pg.render_markdown(pg_base).splitlines():
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
