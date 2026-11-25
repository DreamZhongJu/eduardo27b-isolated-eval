#!/usr/bin/env python3
"""Read-only Pro 5000 offline attribution runner for T-025 (Causes A-G).

Reads existing verified artifacts on Pro 5000:
  - t021_bundle (selfbuilt_eval_inputs_1489.jsonl + ref27b_counts.csv + manifest_export.json)
  - t019_counts/base4b_counts.csv
  - runs/t021_stage2_full_* (A-group AGNES & AUTHOR_RULE RESULTS.json + SCORE_INPUT.json)
  - t018_bundle/t018_per_problem.csv
  - runs/t022_27b_pro5000_*/full494/transfer_eval_*.json
  - data/author494_eval/test.parquet
  - Meta-Llama-3.1-8B-Instruct/tokenizer.json

Never loads GPU models, never calls Agnes, never modifies existing directories,
and strictly verifies zero problem/answer/source_id leakage in outputs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
import traceback
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PRO5000_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
PAIRED_DIR = REPO_ROOT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(PRO5000_DIR), str(BASE4B_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paired_gain as pg  # noqa: E402
from attribution_core import (  # noqa: E402
    DIFFICULTY_ORDER,
    assert_output_sanitized,
    bootstrap_mean_1d,
    compute_cause_a,
    compute_cause_b,
    compute_cause_c_pro5000,
    compute_cause_e_pro5000,
    compute_cause_f,
    derive_seed,
    extract_author_boxed_pred,
    judge_cause_d,
    judge_cause_g,
    match_sources_499_vs_494,
    passes_author_threshold,
    summarize_length_and_format_records,
)
from import_bundle import load_and_verify_bundle  # noqa: E402
from make_selfbuilt_paired_csv import _load_base4b_csv_counts, _load_results_dir_counts  # noqa: E402
from run_t025_a100 import _load_fast_tokenizer  # noqa: E402
from summarize_base4b import STAGE_BY_DIFF, extract_safe_failure_summary, p_value  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_model_rate_entries(
    loaded_bundle: dict[str, Any],
    *,
    grader: str,
    post_override: dict[tuple[str, str], dict[str, dict[str, int]]] | None = None,
) -> list[dict[str, Any]]:
    """Build normalized [0, 1] pre/post rate records (未决按错) for one model & grader."""
    ref_key = "ref_counts" if grader == "agnes" else "ref_counts_ar"
    out: list[dict[str, Any]] = []
    for e in loaded_bundle["eligible"]:
        diff = e["difficulty"]
        sid = e["source_id"]
        rc = e[ref_key]
        if post_override is not None:
            if (diff, sid) not in post_override:
                continue
            post_orig_c = post_override[(diff, sid)]["orig"]
            post_trans_c = post_override[(diff, sid)]["trans"]
        else:
            post_orig_c = rc["原题"]["post"]
            post_trans_c = rc["迁移题"]["post"]
        out.append(
            {
                "difficulty": diff,
                "source_id": sid,
                "pre_orig": float(p_value(rc["原题"]["pre"], unknown_as_wrong=True)),
                "pre_trans": float(p_value(rc["迁移题"]["pre"], unknown_as_wrong=True)),
                "post_orig": float(p_value(post_orig_c, unknown_as_wrong=True)),
                "post_trans": float(p_value(post_trans_c, unknown_as_wrong=True)),
            }
        )
    return out


def _cross_check_same_pre_with_paired_gain(
    loaded_bundle: dict[str, Any],
    ag_post: dict[tuple[str, str], dict[str, dict[str, int]]],
    base_post: dict[tuple[str, str], dict[str, dict[str, int]]],
    *,
    cause_a_res: dict[str, Any],
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Cross-check Cause A filtered (a_group - base) against paired_gain.py on matched sources."""
    rows: list[dict[str, Any]] = []
    for e in loaded_bundle["eligible"]:
        diff = e["difficulty"]
        sid = e["source_id"]
        rc = e["ref_counts"]
        pre_o = rc["原题"]["pre"]["correct"] / 16.0
        pre_t = rc["迁移题"]["pre"]["correct"] / 16.0
        if not passes_author_threshold(pre_t, pre_o):
            continue
        if (diff, sid) in base_post:
            bc = base_post[(diff, sid)]["trans"]
            rows.append(
                {
                    "unit_id": sid,
                    "condition": "base",
                    "group": diff,
                    "pre": pre_t,
                    "post_correct": int(bc["correct"]),
                    "post_total": 16,
                    "post_unknown": int(bc["unknown"]),
                    "line_no": len(rows) + 2,
                }
            )
        if (diff, sid) in ag_post:
            ac = ag_post[(diff, sid)]["trans"]
            rows.append(
                {
                    "unit_id": sid,
                    "condition": "a_group",
                    "group": diff,
                    "pre": pre_t,
                    "post_correct": int(ac["correct"]),
                    "post_total": 16,
                    "post_unknown": int(ac["unknown"]),
                    "line_no": len(rows) + 2,
                }
            )
    if not rows:
        return {}
    pg_rep = pg.evaluate_dataset(
        rows,
        base_condition="base",
        target_condition="a_group",
        draws=draws,
        seed=seed,
        include_high_pre_filtered=False,
    )
    pg_idx = {
        r["group"]: r
        for r in pg_rep["results"]
        if r["unknown_policy"] == "unknown_as_wrong" and r["source_filter"] == "all"
    }
    checks: dict[str, Any] = {}
    for diff in DIFFICULTY_ORDER:
        if diff not in pg_idx or pg_idx[diff]["stats"] is None:
            continue
        pg_st = pg_idx[diff]["stats"]
        ca_st = cause_a_res["a_group_minus_base_on_filtered"][diff]
        pt_err = abs(float(pg_st["diff_mean"]) - float(ca_st["mean_diff"]))
        lo_err = abs(float(pg_st["diff_ci_low"]) - float(ca_st["ci"][0]))
        hi_err = abs(float(pg_st["diff_ci_high"]) - float(ca_st["ci"][1]))
        if pg_idx[diff]["n_paired"] != ca_st["n_paired"] or pt_err > 1e-12 or lo_err > 0.01 or hi_err > 0.01:
            raise ValueError(f"paired_gain cross-check mismatch on filtered {diff}: pt={pt_err}")
        checks[diff] = {
            "n_paired": ca_st["n_paired"],
            "pt_err": pt_err,
            "ci_lo_err": lo_err,
            "ci_hi_err": hi_err,
        }
    return checks


def _load_t022_json(path: Path, *, seed: int = 42, draws: int = 10000) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    convs = doc.get("conversations") or doc.get("records") or []
    if not convs:
        raise ValueError(f"Empty conversations in T-022 JSON: {path}")
    by_row: dict[int, dict[str, Any]] = {}
    for rec in convs:
        ridx = int(rec["row_index"])
        if ridx in by_row:
            raise ValueError("Expected 1 conversation per row_index in T-022 JSON")
        by_row[ridx] = rec
    metrics = doc.get("metrics") or {}
    dt_boot = bootstrap_mean_1d(
        [float(by_row[r]["delta_transfer"]) for r in sorted(by_row)],
        seed=derive_seed(seed, "t022_dt"),
        draws=draws,
    )
    ds_boot = bootstrap_mean_1d(
        [float(by_row[r]["delta_same"]) for r in sorted(by_row)],
        seed=derive_seed(seed, "t022_ds"),
        draws=draws,
    )
    summary = {
        "json_sha256": _sha256_file(path),
        "num_problems": len(by_row),
        "delta_transfer": dt_boot,
        "delta_same": ds_boot,
        "blanked_teacher_turn_rate": metrics.get("blanked_teacher_turn_rate"),
        "token_budget_truncated_rate": metrics.get("token_budget_truncated_rate"),
        "eoc_rate": metrics.get("eoc_rate"),
        "thinking_tokens_avg_per_turn": metrics.get("thinking_tokens_avg_per_turn"),
        "n_student_calls_failed": metrics.get("n_student_calls_failed"),
        "assume_thinking": metrics.get("assume_thinking"),
    }
    return by_row, summary


def _load_t018_csv(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ridx = int(row["row_index"])
            out[ridx] = {
                "row_index": ridx,
                "pre_solve_rate": float(row["pre_solve_rate"]),
                "pre_solve_rate_transfer": float(row["pre_solve_rate_transfer"]),
                "post_same_correct": int(row["post_same_correct"]),
                "post_same_total": int(row["post_same_total"]),
                "post_transfer_correct": int(row["post_transfer_correct"]),
                "post_transfer_total": int(row["post_transfer_total"]),
                "ctl1_same_correct": int(row["ctl1_same_correct"]),
                "ctl1_same_total": int(row["ctl1_same_total"]),
                "ctl1_transfer_correct": int(row["ctl1_transfer_correct"]),
                "ctl1_transfer_total": int(row["ctl1_transfer_total"]),
            }
    return out


def _collect_agroup_post_lengths(
    loaded_bundle: dict[str, Any],
    agroup_root: Path,
    tok_batch_fn: Any,
) -> dict[str, Any]:
    sample_records: list[dict[str, Any]] = []
    sources_scanned = 0
    for e in loaded_bundle["eligible"]:
        diff = e["difficulty"]
        stage = STAGE_BY_DIFF[diff]
        key = e["source_key"]
        exp_tgt = e["row"]["transfer_target"]
        src_dir = agroup_root / stage / "sources" / key
        score_p = src_dir / "SCORE_INPUT.json"
        agnes_p = agroup_root / "AGNES" / stage / "sources" / key / "RESULTS.json"
        if not score_p.is_file() or not agnes_p.is_file():
            continue
        rows = json.loads(score_p.read_bytes())
        ag_doc = json.loads(agnes_p.read_bytes())
        st_by_coord: dict[tuple[str, str, int], str] = {}
        for item in ag_doc.get("results") or []:
            ident = item.get("identity") or item
            st_by_coord[(ident["target"], ident["phase"], int(ident["index"]))] = item["status"]
        texts = [r.get("student_response") or "" for r in rows]
        tok_info = tok_batch_fn(texts)
        for r, (n_tok, pref800) in zip(rows, tok_info):
            t_raw = r["target"]
            if t_raw != "source" and t_raw != exp_tgt:
                raise ValueError("Unexpected target in A-group SCORE_INPUT.json")
            tgt = "orig" if t_raw == "source" else "trans"
            st = st_by_coord[(t_raw, r["phase"], int(r["index"]))]
            pred_full, fmt_err = extract_author_boxed_pred(r.get("student_response") or "")
            pred_800, _ = extract_author_boxed_pred(pref800) if n_tok > 800 else (pred_full, fmt_err)
            sample_records.append(
                {
                    "difficulty": diff,
                    "target": tgt,
                    "phase": r["phase"],
                    "tokens": int(n_tok),
                    "finish_reason": str(r.get("finish_reason") or ""),
                    "agnes_status": st,
                    "ar_incorrect_format": bool(fmt_err),
                    "boxed_pred_full": pred_full,
                    "boxed_pred_prefix800": pred_800,
                }
            )
        sources_scanned += 1
    return {
        "sources_scanned": sources_scanned,
        "length_and_format": summarize_length_and_format_records(sample_records, model_label="a_group"),
    }


def _fmt_est(d: dict[str, Any] | None) -> str:
    if not d or d.get("mean") is None:
        return "N/A"
    ci = d.get("ci")
    if ci:
        return f"{d['mean']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
    return f"{d['mean']:+.4f}"


def _fmt_diff(d: dict[str, Any] | None, *, with_closed: bool = False) -> str:
    if not d or d.get("mean_diff") is None:
        return "N/A"
    ci = d.get("ci")
    s = f"{d['mean_diff']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else f"{d['mean_diff']:+.4f}"
    if with_closed and d.get("gap_closed_share") is not None:
        s += f" closed={d['gap_closed_share']*100:5.1f}%"
    return s


def format_pro5000_report(payload: dict[str, Any]) -> str:
    """Render concise human-readable Pro 5000 attribution report (zero problem text, zero IDs)."""
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append("T-025 Pro 5000 Read-Only Attribution Summary (Causes A, B, C, D[A-group], E1, F, G)")
    ov = payload["cause_f"]["overlap_counts"]
    lines.append(
        f"Source Overlap (499 vs 494): exact={ov['n_exact_overlap']} (499\\494={ov['n_ours_minus_author']}, "
        f"494\\499={ov['n_author_minus_ours']}), approx_extra(>=0.90)={ov['n_approx_additional_ge_090']}, "
        f"with_approx={ov['n_overlap_with_approx']}, paired_ge_30={ov['paired_eligible_ge_30']}"
    )
    lines.append("-" * 92)

    # Cause A
    ca = payload["cause_a"]
    mgc = ca["decision"].get("max_gap_closed_share")
    mgc_s = f"{mgc*100:.1f}%" if mgc is not None else "N/A"
    lines.append(f"[Cause A] Verdict (27B): {ca['decision']['verdict']} (max_gap_closed={mgc_s})")
    ad = ca["author_494_transfer_distribution"]
    lines.append(
        f"  Author 494 pre_transfer: pass_gate={ad['n_pass_author_threshold']}/{ad['n_total']} ({ad['share_pass_author_threshold']*100:.1f}%), "
        f"pre>=15/16={ad['n_pre_ge_15_16']}/{ad['n_total']} ({ad['share_pre_ge_15_16']*100:.2f}%)"
    )
    for b in ad["bins"]:
        r1s = f"{b['r1_27b']['mean']:+.4f}" if b["r1_27b"] else "N/A"
        mp = f"{b['mean_pre']*100:5.1f}%" if b["mean_pre"] is not None else "  N/A"
        lines.append(f"    Author bin {b['bin']:<14}: n={b['n']:>3} ({b['share']*100:5.1f}%) pre={mp} R1_27b={r1s}")

    for m_name in ("eduardo27b", "base", "a_group"):
        if m_name not in ca["models"]:
            continue
        lines.append(f"  Model={m_name}:")
        for diff in DIFFICULTY_ORDER:
            tr = ca["models"][m_name][diff]
            df = tr["diff_filtered_minus_all"]
            drw = tr["diff_filtered_reweighted_minus_all"]
            drwa = tr["diff_reweighted_all_minus_all"]
            lines.append(
                f"    {diff:<6} all(n={tr['n_total']}): R1={_fmt_est(tr['all'])} pre>=15/16={tr['n_pre_ge_15_16']} | "
                f"filt(n={tr['n_filtered']}): R1={_fmt_est(tr['filtered'])} diff={_fmt_diff(df, with_closed=True)} | "
                f"rewt_all: R1={_fmt_est(tr['reweighted_all'])} diff={_fmt_diff(drwa, with_closed=True)} | "
                f"filt+rewt: R1={_fmt_est(tr['filtered_reweighted'])} diff={_fmt_diff(drw, with_closed=True)}"
            )
            if m_name == "eduardo27b":
                for st in tr["strata"]:
                    r1_b = _fmt_est(st["r1"])
                    mp = f"{st['mean_pre']*100:5.1f}%" if st["mean_pre"] is not None else "  N/A"
                    mo = f"{st['mean_post']*100:5.1f}%" if st["mean_post"] is not None else "  N/A"
                    lines.append(
                        f"      bin {st['bin']:<14}: n={st['n']:>3} pre={mp} post={mo} R1={r1_b}"
                    )

    lines.append("-" * 92)
    # Cause B
    cb = payload["cause_b"]
    lines.append(
        f"[Cause B] Verdict: {cb['decision']['verdict']} (mode={cb['decision']['comparison_mode']}, "
        f"T-022 Δ_transfer={cb['author_27b_t022']['delta_transfer_all494']['mean']:+.4f}, "
        f"Δ_same={cb['author_27b_t022']['delta_same_all494']['mean']:+.4f})"
    )
    for diff in DIFFICULTY_ORDER:
        pr = cb["primary_author_minus_ours_filtered"][diff]
        up = cb["unpaired_transfer_filtered"][diff]
        lines.append(
            f"  {diff:<6} primary(author-ours_filt): diff={_fmt_diff(pr)} "
            f"(n={pr.get('n_paired', up['n_b'])}) | unpaired(494 vs {up['n_b']}): {_fmt_diff(up)}"
        )
    sc = cb["paired_same_combined"] or cb["unpaired_same_combined"]
    lines.append(f"  Δ_same (author - ours_orig_combined): diff={_fmt_diff(sc)}")

    lines.append("-" * 92)
    # Cause C
    cc = payload["cause_c"]
    lines.append(f"[Cause C] Pro5000 Verdict (pending/with A100 precheck): {cc['decision']['verdict']}")
    for k in ("easy", "medium", "hard", "orig_combined"):
        d = cc["eduardo27b_ar_minus_agnes"][k]
        lines.append(
            f"  27B {k:<14} (AR - Agnes, n={d['n_paired']}): AR={d['target_mean']:+.4f} Agnes={d['base_mean']:+.4f} "
            f"diff={_fmt_diff(d)}"
        )

    lines.append("-" * 92)
    # Cause D (A-group on Pro 5000)
    cd_ag = payload["cause_d_pro5000_agroup"]["length_and_format"]["post_length_screening"]
    lines.append("[Cause D] Pro 5000 A-group Post-test Length (>800 vs <=800 tokens):")
    for r in cd_ag:
        le, gt = r["le_800"], r["gt_800"]
        le_a = f"{le['acc_unknown_as_wrong']*100:6.2f}%" if le["acc_unknown_as_wrong"] is not None else "   N/A"
        gt_a = f"{gt['acc_unknown_as_wrong']*100:6.2f}%" if gt["acc_unknown_as_wrong"] is not None else "   N/A"
        lines.append(
            f"  {r['group']:<18} N={r['n_total']:>5} >800={r['n_gt_800']:>4} ({r['gt_800_rate']*100:5.2f}%) | "
            f"<=800 acc={le_a} | >800 acc={gt_a}"
        )

    lines.append("-" * 92)
    # Cause E1
    ce = payload["cause_e"]
    ar_p = ce["overlap_orig_pre_ours_ar_minus_author"]
    ag_p = ce["overlap_orig_pre_ours_agnes_minus_author"]
    tm_s = f"{ar_p['target_mean']:.4f}" if ar_p.get("target_mean") is not None else "N/A"
    bm_s = f"{ar_p['base_mean']:.4f}" if ar_p.get("base_mean") is not None else "N/A"
    lines.append(
        f"[Cause E1] Verdict: {ce['decision']['verdict']} | Overlap orig pre (n={ar_p['n_paired']}): "
        f"ours_AR - author = {_fmt_diff(ar_p)} (ours={tm_s}, author={bm_s}); "
        f"ours_Agnes - author = {_fmt_diff(ag_p)}"
    )

    lines.append("-" * 92)
    # Cause F
    cf = payload["cause_f"]
    lines.append(f"[Cause F] Verdict: {cf['decision']['verdict']}")
    for diff in DIFFICULTY_ORDER:
        it = cf["selfbuilt_27b_intersect_vs_all"][diff]
        dd = it["diff_intersect_minus_all"]
        mb = cf["missing_8_exact_bounds"][diff]
        inter_est = it["in_499_inter_494"]
        in_n = inter_est["n"] if inter_est else 0
        in_m = f"{inter_est['mean']:+.4f}" if inter_est else "N/A"
        lines.append(
            f"  {diff:<6}: all(n={it['all_eligible']['n']})={it['all_eligible']['mean']:+.4f}, "
            f"499∩494(n={in_n})={in_m}, diff={_fmt_diff(dd)} | "
            f"missing={mb['n_missing']} bound=[{mb['max_downward_shift']:+.4f}, {mb['max_upward_shift']:+.4f}]"
        )

    lines.append("-" * 92)
    # Cause G
    cg = payload["cause_g"]
    lines.append(
        f"[Cause G] Verdict: {cg['verdict']} | T-022 |Δ_transfer - 0.29|={cg['t022_author494_abs_diff_vs_paper_transfer']:.4f}, "
        f"|Δ_same - 0.46|={cg['t022_author494_abs_diff_vs_paper_same']:.4f}"
    )
    lines.append("=" * 92)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only Pro 5000 attribution runner for T-025.")
    ap.add_argument("--bundle-dir", type=Path, required=True, help="T-021 imported bundle dir")
    ap.add_argument("--agroup-dir", type=Path, required=True, help="T-021 full run dir")
    ap.add_argument("--base4b-counts-csv", type=Path, required=True, help="T-019 base4b_counts.csv")
    ap.add_argument("--t018-table", type=Path, default=None, help="T-018 t018_per_problem.csv")
    ap.add_argument("--t022-json", type=Path, required=True, help="T-022 full494 transfer_eval_*.json")
    ap.add_argument("--test-parquet", type=Path, required=True, help="Author 494 test.parquet")
    ap.add_argument("--student-tokenizer-dir", type=Path, required=True)
    ap.add_argument("--a100-json", type=Path, default=None, help="Optional A100 counts JSON")
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-small-test", action="store_true")
    args = ap.parse_args()

    loaded_bundle = load_and_verify_bundle(
        args.bundle_dir, require_exact_totals=not args.allow_small_test
    )
    entries = loaded_bundle["eligible"]

    ours_problems_by_sid: dict[str, str] = {}
    for e in entries:
        ours_problems_by_sid[e["source_id"]] = e["row"]["problem"]
    if not args.allow_small_test and len(ours_problems_by_sid) != 499:
        raise ValueError(f"Expected 499 unique sources in bundle, got {len(ours_problems_by_sid)}")

    pq_sha = _sha256_file(args.test_parquet)
    df_pq = pd.read_parquet(args.test_parquet)
    if not args.allow_small_test and len(df_pq) != 494:
        raise ValueError(f"Expected 494 rows in test.parquet, got {len(df_pq)}")

    author_problems_by_row: dict[int, str] = {}
    author_pre_by_row: dict[int, dict[str, float]] = {}
    for ridx, extra in enumerate(df_pq["extra_info"].tolist()):
        author_problems_by_row[ridx] = str(extra["problem"])
        author_pre_by_row[ridx] = {
            "pre_same": float(extra.get("pre_solve_rate") or 0.0),
            "pre_transfer": float(
                extra.get("pre_solve_rate_transfer_problem")
                if extra.get("pre_solve_rate_transfer_problem") is not None
                else extra.get("pre_solve_rate") or 0.0
            ),
        }

    t022_by_row, t022_summary = _load_t022_json(args.t022_json, seed=args.seed, draws=args.draws)
    if set(t022_by_row) != set(author_problems_by_row):
        raise ValueError("row_index mismatch between T-022 JSON and test.parquet")

    t018_by_row = _load_t018_csv(args.t018_table) if (args.t018_table and args.t018_table.is_file()) else None
    a100_doc = json.loads(args.a100_json.read_text(encoding="utf-8")) if (args.a100_json and args.a100_json.is_file()) else None

    # Build Agnes & Author-Rule rate entries for 27B, base, and a_group
    ag_post_agnes = _load_results_dir_counts(entries, args.agroup_dir, "AGNES")
    ag_post_ar = _load_results_dir_counts(entries, args.agroup_dir, "AUTHOR_RULE")
    bs_post_agnes = _load_base4b_csv_counts(args.base4b_counts_csv, "agnes")
    bs_post_ar = _load_base4b_csv_counts(args.base4b_counts_csv, "ar")

    e27_agnes = _build_model_rate_entries(loaded_bundle, grader="agnes")
    e27_ar = _build_model_rate_entries(loaded_bundle, grader="ar")
    ebs_agnes = _build_model_rate_entries(loaded_bundle, grader="agnes", post_override=bs_post_agnes)
    ebs_ar = _build_model_rate_entries(loaded_bundle, grader="ar", post_override=bs_post_ar)
    eag_agnes = _build_model_rate_entries(loaded_bundle, grader="agnes", post_override=ag_post_agnes)
    eag_ar = _build_model_rate_entries(loaded_bundle, grader="ar", post_override=ag_post_ar)

    # Cause A
    author_rows_for_a = [
        {
            "row_index": ridx,
            "pre_same": author_pre_by_row[ridx]["pre_same"],
            "pre_transfer": author_pre_by_row[ridx]["pre_transfer"],
            "post_transfer_27b": float(t022_by_row[ridx]["post_solve_transfer"]),
            "delta_transfer_27b": float(t022_by_row[ridx]["delta_transfer"]),
        }
        for ridx in sorted(author_pre_by_row)
    ]
    cause_a = compute_cause_a(
        {"eduardo27b": e27_agnes, "base": ebs_agnes, "a_group": eag_agnes},
        author_rows_for_a,
        seed=args.seed,
        draws=args.draws,
    )
    pg_cc_a = _cross_check_same_pre_with_paired_gain(
        loaded_bundle,
        ag_post_agnes,
        bs_post_agnes,
        cause_a_res=cause_a,
        seed=args.seed,
        draws=args.draws,
    )
    cause_a["paired_gain_cross_check_filtered_ag_vs_base"] = pg_cc_a

    # Overlap matching (499 vs 494) for Causes B, E1, F
    overlap_res = match_sources_499_vs_494(ours_problems_by_sid, author_problems_by_row)
    exact_sid_to_row = overlap_res["exact_sid_to_row"]
    overlap_public = overlap_res["public_counts"]

    # Cause B
    cause_b = compute_cause_b(
        e27_ar,
        t022_by_row,
        exact_sid_to_row,
        seed=args.seed,
        draws=args.draws,
    )

    # Cause C
    a100_fmt = a100_doc["eduardo27b"]["length_and_format"] if a100_doc else None
    other_c: dict[str, Any] = {"base": (ebs_agnes, ebs_ar)}
    if eag_ar:
        other_c["a_group"] = (eag_agnes, eag_ar)
    cause_c = compute_cause_c_pro5000(
        e27_agnes,
        e27_ar,
        a100_format_check=a100_fmt,
        other_models=other_c,
        seed=args.seed,
        draws=args.draws,
    )

    # Cause D (Pro 5000 A-group + optional A100 27B/base merge)
    tok_fn, tok_backend = _load_fast_tokenizer(args.student_tokenizer_dir)
    tok_sha = _sha256_file(args.student_tokenizer_dir / "tokenizer.json")
    ag_len_doc = _collect_agroup_post_lengths(loaded_bundle, args.agroup_dir, tok_fn)
    models_d = {"a_group": ag_len_doc["length_and_format"]}
    if a100_doc:
        models_d["eduardo27b"] = a100_doc["eduardo27b"]["length_and_format"]
        if a100_doc.get("base4b"):
            models_d["base"] = a100_doc["base4b"]["length_and_format"]
    cause_d_decision = judge_cause_d(models_d)

    # Cause E1
    cause_e = compute_cause_e_pro5000(
        e27_agnes,
        e27_ar,
        author_pre_by_row,
        exact_sid_to_row,
        t018_by_row=t018_by_row,
        a100_config_doc=a100_doc["eduardo27b"] if a100_doc else None,
        seed=args.seed,
        draws=args.draws,
    )

    # Cause F
    target_total = 499 if not args.allow_small_test else len(ours_problems_by_sid)
    cause_f = compute_cause_f(
        e27_agnes,
        overlap_public,
        exact_sid_to_row,
        per_tier_total_target=target_total,
        t022_by_row=t022_by_row,
        seed=args.seed,
        draws=args.draws,
    )

    # Cause G
    cause_g = judge_cause_g(t022_summary, a100_doc["eduardo27b"] if a100_doc else None)

    payload: dict[str, Any] = {
        "schema": "t025_pro5000_summary_v1",
        "seed": args.seed,
        "draws": args.draws,
        "inputs_verified": {
            "test_parquet_sha256": pq_sha,
            "t022_json_sha256": t022_summary["json_sha256"],
            "base4b_counts_csv_sha256": _sha256_file(args.base4b_counts_csv),
            "t018_table_sha256": _sha256_file(args.t018_table) if (args.t018_table and args.t018_table.is_file()) else None,
            "tokenizer_backend": tok_backend,
            "tokenizer_json_sha256": tok_sha,
            "bundle_math_verify_version": loaded_bundle.get("math_verify_version"),
            "a100_json_merged": a100_doc is not None,
        },
        "t022_summary": t022_summary,
        "cause_a": cause_a,
        "cause_b": cause_b,
        "cause_c": cause_c,
        "cause_d_pro5000_agroup": ag_len_doc,
        "cause_d_decision": cause_d_decision,
        "cause_e": cause_e,
        "cause_f": cause_f,
        "cause_g": cause_g,
    }

    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    report_text = format_pro5000_report(payload)

    forbidden_ids = list(ours_problems_by_sid.keys())
    forbidden_snippets = (
        list(ours_problems_by_sid.values())[:50]
        + list(author_problems_by_row.values())[:50]
        + [e["row"]["transfer_problem"] for e in entries[:50]]
    )
    assert_output_sanitized(json_text, forbidden_exact_ids=forbidden_ids, forbidden_text_snippets=forbidden_snippets)
    assert_output_sanitized(report_text, forbidden_exact_ids=forbidden_ids, forbidden_text_snippets=forbidden_snippets)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text, encoding="utf-8")
    json_sha = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    compressed_b64 = base64.b64encode(zlib.compress(json_text.encode("utf-8"), level=9)).decode("ascii")

    print(report_text)
    print(f"T025_PRO5000_JSON_SHA256={json_sha}")
    print("T025_PRO5000_B64_BEGIN")
    print(compressed_b64)
    print("T025_PRO5000_B64_END")
    print(f"T025_PRO5000_DONE json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
