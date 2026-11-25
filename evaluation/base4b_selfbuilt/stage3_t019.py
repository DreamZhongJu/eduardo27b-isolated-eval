#!/usr/bin/env python3
"""Stage 3 sanitized summary, paired_gain cross-check, Author-Rule sensitivity, and base4b_counts export for T-019."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import tarfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = HERE.parents[1]
PAIRED_DIR = REPO_ROOT_DEFAULT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paired_gain as pg  # noqa: E402
from ref27b_manifest import DIFFICULTY_ORDER, build_manifest  # noqa: E402
from summarize_base4b import (  # noqa: E402
    STAGE_BY_DIFF,
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

BASE4B_COUNT_COLUMNS = (
    "difficulty",
    "tier_zh",
    "source_key",
    "source_id",
    "row_index",
    *(
        f"{grader}_{tgt}_post_{st}"
        for grader in ("agnes", "ar")
        for tgt in ("orig", "trans")
        for st in ("correct", "incorrect", "unknown")
    ),
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _with_scalar_pre(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold pre unknown into pre incorrect to align with paired_gain CSV scalar pre column."""
    out: list[dict[str, Any]] = []
    for rec in records:
        new_rec = {**rec}
        for ck in ("counts", "ref_counts", "ref_counts_27b"):
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


def export_t019_base_counts(
    *,
    manifest: dict[str, Any],
    base4b_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Export sanitized per-source T-019 base post counts (no raw text) matching T-021 BASE4B_COUNT_COLUMNS."""
    if not base4b_dir.is_dir():
        raise ValueError(f"base4b_dir does not exist: {base4b_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    held_count = 0
    for e in manifest["eligible"]:
        diff = e["difficulty"]
        stage = STAGE_BY_DIFF[diff]
        key = e["source_key"]
        sid = e["source_id"]
        exp_tgt = e["row"]["transfer_target"]
        src_dir = base4b_dir / stage / "sources" / key
        if (src_dir / "SOURCE_HELD.json").exists():
            held_count += 1
            continue
        if not (src_dir / "RAW_COMPLETE.json").is_file():
            continue
        agnes_p = base4b_dir / "AGNES" / stage / "sources" / key / "RESULTS.json"
        ar_p = base4b_dir / "AUTHOR_RULE" / stage / "sources" / key / "RESULTS.json"
        if not agnes_p.is_file():
            continue

        row_rec: dict[str, Any] = {
            "difficulty": diff,
            "tier_zh": e["tier_zh"],
            "source_key": key,
            "source_id": sid,
            "row_index": e["row_index"],
        }
        for grader, p in (("agnes", agnes_p), ("ar", ar_p)):
            if not p.is_file():
                for tgt in ("orig", "trans"):
                    for st in ("correct", "incorrect", "unknown"):
                        row_rec[f"{grader}_{tgt}_post_{st}"] = ""
                continue
            doc = json.loads(p.read_bytes())
            if doc.get("source_id") != sid:
                raise ValueError("base4b RESULTS.json source_id mismatch")
            by_t: dict[str, Counter] = {"orig": Counter(), "trans": Counter()}
            for item in doc.get("results") or []:
                ident = item.get("identity") or item
                t_raw = ident.get("target")
                tgt = "orig" if t_raw == "source" else "trans"
                if t_raw != "source" and t_raw != exp_tgt:
                    raise ValueError("base4b RESULTS.json target mismatch")
                st = item.get("status")
                if st not in ("correct", "incorrect", "unknown"):
                    raise ValueError("Invalid status in RESULTS.json")
                by_t[tgt][st] += 1
            for tgt in ("orig", "trans"):
                if sum(by_t[tgt].values()) != 16:
                    raise ValueError("base4b target count != 16")
                for st in ("correct", "incorrect", "unknown"):
                    row_rec[f"{grader}_{tgt}_post_{st}"] = int(by_t[tgt].get(st, 0))
        rows_out.append(row_rec)

    csv_path = out_dir / "base4b_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BASE4B_COUNT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows_out)

    man = {
        "schema": "t021_t019_base_counts_v1",
        "exported_sources": len(rows_out),
        "eligible_total": manifest["eligible_count"],
        "held_sources": held_count,
        "files_sha256": {csv_path.name: _sha256_file(csv_path)},
    }
    man_path = out_dir / "base4b_manifest.json"
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle = out_dir / "t019_base_counts_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(csv_path, arcname=csv_path.name)
        tf.add(man_path, arcname=man_path.name)
    bsha = _sha256_file(bundle)
    (out_dir / "t019_base_counts_bundle.tar.gz.sha256").write_text(
        f"{bsha}  {bundle.name}\n", encoding="utf-8"
    )
    man["bundle_sha256"] = bsha
    man["bundle_bytes"] = bundle.stat().st_size
    man["csv_path"] = str(csv_path)
    man["bundle_path"] = str(bundle)
    return man


def load_ref27b_ar_overlay(
    *,
    ref27b_csv: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, dict[str, dict[str, int]]]]]:
    """Load pre-exported 27B offline Author-Rule counts from T-021's ref27b_counts.csv."""
    by_key: dict[tuple[str, str], dict[str, dict[str, dict[str, int]]]] = {}
    with ref27b_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            diff = row["difficulty"]
            sid = row["source_id"]
            rc: dict[str, dict[str, dict[str, int]]] = {}
            for tgt_zh, tgt_csv in (("原题", "orig"), ("迁移题", "trans")):
                rc[tgt_zh] = {}
                for phase in ("pre", "control", "post"):
                    c = {
                        st: int(row[f"ar_{tgt_csv}_{phase}_{st}"])
                        for st in ("correct", "incorrect", "unknown")
                    }
                    if sum(c.values()) != 16:
                        raise ValueError("ref27b_counts.csv AR count sum != 16")
                    rc[tgt_zh][phase] = c
            by_key[(diff, sid)] = rc

    all_27b_ar: list[dict[str, Any]] = []
    for e in manifest["eligible"]:
        k = (e["difficulty"], e["source_id"])
        if k not in by_key:
            raise ValueError("Missing source in ref27b_counts.csv")
        all_27b_ar.append(
            {
                "tier_zh": e["tier_zh"],
                "source_id": e["source_id"],
                "ref_counts": by_key[k],
            }
        )
    return all_27b_ar, by_key


def apply_27b_ar_ref_to_records(
    records: list[dict[str, Any]],
    ref_ar_by_key: dict[tuple[str, str], dict[str, dict[str, dict[str, int]]]],
) -> list[dict[str, Any]]:
    """Replace pre/control/ref_counts_27b in base4b records with 27B offline Author-Rule counts."""
    out: list[dict[str, Any]] = []
    for rec in records:
        k = (rec["difficulty"], rec["source_id"])
        rc_ar = ref_ar_by_key[k]
        new_counts = {}
        for tgt_zh in ("原题", "迁移题"):
            new_counts[tgt_zh] = {
                "pre": dict(rc_ar[tgt_zh]["pre"]),
                "control": dict(rc_ar[tgt_zh]["control"]),
                "post": dict(rec["counts"][tgt_zh]["post"]),
            }
        out.append({**rec, "counts": new_counts, "ref_counts_27b": rc_ar})
    return out


def compute_extended_stage3_table(
    records: list[dict[str, Any]],
    all_27b_records: list[dict[str, Any]],
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Compute R0(pre/ctl/post), raw diffs, R1(pre/ctl), R2, R3, R4, and base-27B paired diff."""
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
            p4_pre, b_pre, o_post, n_filt, n_excl = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="pre",
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            r1_pre = bootstrap_mean_ci(
                [v for _, v in p4_pre], seed=_derived_seed(seed, f"{policy}|{label}|4b_pre"), draws=draws
            )
            p4_ctl, b_ctl, o_post_ctl, _, _ = _estimate_arm(
                records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="control",
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="counts",
            )
            r1_ctl = bootstrap_mean_ci(
                [v for _, v in p4_ctl], seed=_derived_seed(seed, f"{policy}|{label}|4b_ctl"), draws=draws
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
            r4_ci = bootstrap_mean_ci(
                [v for _, v in p_r4], seed=_derived_seed(seed, f"{policy}|{label}|r4"), draws=draws
            )
            p27, _, _, _, _ = _estimate_arm(
                all_27b_records,
                tier_filter=tier_zh,
                target_zh=tgt_zh,
                baseline="pre",
                outcome_phase="post",
                unknown_as_wrong=unknown_as_wrong,
                easy_filtered=filtered,
                counts_key="ref_counts",
            )
            paired = paired_bootstrap_diff(
                p4_pre, p27, seed=_derived_seed(seed, f"{policy}|{label}|paired"), draws=draws
            )
            m_pre = (sum(b_pre) / len(b_pre)) if b_pre else None
            m_ctl = (sum(b_ctl) / len(b_ctl)) if b_ctl else None
            m_post = (sum(o_post) / len(o_post)) if o_post else None
            m_post_ctl = (sum(o_post_ctl) / len(o_post_ctl)) if o_post_ctl else None
            m_r4_pre = (sum(b_r4) / len(b_r4)) if b_r4 else None
            m_r4_ctl = (sum(o_r4) / len(o_r4)) if o_r4 else None
            table.append(
                {
                    "label": label,
                    "n": len(p4_pre),
                    "n_filtered": n_filt,
                    "n_excluded_unknown": n_excl,
                    "r0_pre": m_pre,
                    "r0_ctl": m_ctl,
                    "r0_post": m_post,
                    "raw_diff_pre": (m_post - m_pre) if (m_post is not None and m_pre is not None) else None,
                    "raw_diff_ctl": (m_post_ctl - m_ctl) if (m_post_ctl is not None and m_ctl is not None) else None,
                    "r1_pre": r1_pre,
                    "r1_ctl": r1_ctl,
                    "r2_g": _hake_g(b_pre, o_post),
                    "r3_d": _cohens_d(b_pre, o_post),
                    "r4_pre_to_ctl": r4_ci,
                    "r4_raw_diff": (m_r4_ctl - m_r4_pre) if (m_r4_ctl is not None and m_r4_pre is not None) else None,
                    "r4_r2_g": _hake_g(b_r4, o_r4),
                    "r4_r3_d": _cohens_d(b_r4, o_r4),
                    "paired_base_minus_27b": paired,
                }
            )
        out[policy] = table
    return out


def build_base_vs_27b_paired_csv(
    *,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    ref_counts_by_key: dict[tuple[str, str], dict[str, Any]] | None = None,
    out_csv: Path,
) -> dict[str, Any]:
    """Write paired_gain.py CSV for eduardo27b vs base across orig_easy/medium/hard and easy/medium/hard."""
    rec_by_key = {(r["difficulty"], r["source_id"]): r for r in records}
    orig_pre_by_sid: dict[str, list[float]] = {}
    for e in manifest["eligible"]:
        rc = (
            ref_counts_by_key[(e["difficulty"], e["source_id"])]
            if ref_counts_by_key is not None
            else e["ref_counts"]
        )
        orig_pre_by_sid.setdefault(e["source_id"], []).append(rc["原题"]["pre"]["correct"] / 16.0)
    orig_inconsistent = sum(
        1 for vals in orig_pre_by_sid.values() if max(vals) - min(vals) > pg.PRE_CONSISTENCY_TOL
    )
    emit_combined_orig = orig_inconsistent == 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_name(f".{out_csv.name}.{os.getpid()}.tmp")
    written = Counter()
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pg.REQUIRED_COLUMNS))
            writer.writeheader()
            for e in manifest["eligible"]:
                diff = e["difficulty"]
                sid = e["source_id"]
                rc = (
                    ref_counts_by_key[(diff, sid)]
                    if ref_counts_by_key is not None
                    else e["ref_counts"]
                )
                pre_orig = rc["原题"]["pre"]["correct"] / 16.0
                pre_trans = rc["迁移题"]["pre"]["correct"] / 16.0

                def _emit(cond: str, grp: str, pre_v: float, post_c: dict[str, int]) -> None:
                    writer.writerow(
                        {
                            "unit_id": sid,
                            "condition": cond,
                            "group": grp,
                            "pre": f"{pre_v:.17g}",
                            "post_correct": int(post_c["correct"]),
                            "post_total": 16,
                            "post_unknown": int(post_c["unknown"]),
                        }
                    )
                    written[f"{cond}.{grp}"] += 1

                if emit_combined_orig:
                    _emit("eduardo27b", "orig", pre_orig, rc["原题"]["post"])
                _emit("eduardo27b", f"orig_{diff}", pre_orig, rc["原题"]["post"])
                _emit("eduardo27b", diff, pre_trans, rc["迁移题"]["post"])

                if (diff, sid) in rec_by_key:
                    bc = rec_by_key[(diff, sid)]["counts"]
                    if emit_combined_orig:
                        _emit("base", "orig", pre_orig, bc["原题"]["post"])
                    _emit("base", f"orig_{diff}", pre_orig, bc["原题"]["post"])
                    _emit("base", diff, pre_trans, bc["迁移题"]["post"])

        pg.load_csv(tmp_csv)
        os.replace(tmp_csv, out_csv)
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise
    return {
        "csv_path": str(out_csv),
        "emit_combined_orig": emit_combined_orig,
        "orig_inconsistent_sources": orig_inconsistent,
        "row_counts": dict(sorted(written.items())),
    }


def cross_check_base_vs_27b(
    *,
    csv_path: Path,
    records: list[dict[str, Any]],
    all_27b_records: list[dict[str, Any]],
    draws: int = 10000,
    seed: int = 42,
    ci_tol: float = 0.01,
) -> dict[str, Any]:
    """Run paired_gain.py (eduardo27b -> base) and verify alignment against summarize_base4b."""
    csv_rows = pg.load_csv(csv_path)
    pg_res = pg.evaluate_dataset(
        csv_rows,
        base_condition="eduardo27b",
        target_condition="base",
        draws=draws,
        seed=seed,
        include_high_pre_filtered=True,
    )
    sb_raw = compute_summary(records, all_27b_records=all_27b_records, seed=seed, draws=draws)
    sb_aln = compute_summary(
        _with_scalar_pre(records),
        all_27b_records=_with_scalar_pre(all_27b_records),
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
        sb_rows_raw = {r["label"]: r for r in sb_raw["analyses"][policy]}
        sb_rows_aln = {r["label"]: r for r in sb_aln["analyses"][policy]}
        for grp, lbl in GROUP_TO_SUMMARY_LABEL.items():
            pg_e = pg_index.get((policy, grp))
            sb_e = sb_rows_aln.get(lbl, {}).get("paired_4b_minus_27b")
            sb_r = sb_rows_raw.get(lbl, {}).get("paired_4b_minus_27b")
            if not pg_e or not pg_e.get("stats") or not sb_e or sb_e.get("mean_diff") is None:
                continue
            pg_st = pg_e["stats"]
            pt_diff = abs(float(pg_st["diff_mean"]) - float(sb_e["mean_diff"]))
            lo_diff = abs(float(pg_st["diff_ci_low"]) - float(sb_e["ci"][0]))
            hi_diff = abs(float(pg_st["diff_ci_high"]) - float(sb_e["ci"][1]))
            comparisons[f"{policy}.{grp}"] = {
                "n_paired_pg": pg_e["n_paired"],
                "n_paired_sb": sb_e["n_paired"],
                "pg_diff": pg_st["diff_mean"],
                "sb_diff_scalar_pre": sb_e["mean_diff"],
                "sb_diff_source_gain": float(sb_r["mean_diff"]) if sb_r and sb_r.get("mean_diff") is not None else None,
                "pt_diff": pt_diff,
                "ci_lo_diff": lo_diff,
                "ci_hi_diff": hi_diff,
            }
            if pg_e["n_paired"] != sb_e["n_paired"] or pt_diff > 1e-12 or lo_diff > ci_tol or hi_diff > ci_tol:
                mismatches.append(f"{policy}.{grp}")
    if mismatches:
        raise ValueError(f"T-019 paired_gain vs summarize_base4b mismatch: {mismatches}")
    return {"pg_report": pg_res, "sb_summary": sb_raw, "comparisons": comparisons}


def _fmt_ci(stat: dict[str, Any] | None) -> str:
    if not stat or stat.get("mean") is None:
        return "N/A"
    return f"{stat['mean']:+.4f} [{stat['ci'][0]:+.4f},{stat['ci'][1]:+.4f}]"


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:6.2f}%" if v is not None else "    N/A"


def _fmt_signed(v: float | None, fmt: str = "+.4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


def print_extended_table(title: str, ext: dict[str, Any]) -> None:
    print(f"[base4b_summary] === {title} (exclude_unknown_policy=source_gain: pre & post both drop unknown) ===")
    for policy in ("未决按错", "排除未决"):
        print(f"[base4b_summary] [{policy}]")
        print(
            "[base4b_summary] "
            + f"{'项目':<20} | {'n':>4} | {'R0_pre':>7} | {'R0_ctl':>7} | {'R0_post':>7} | "
            + f"{'差(pre)':>8} | {'差(ctl)':>8} | {'R1(pre->post) [CI]':<23} | "
            + f"{'R1(ctl->post) [CI]':<23} | {'R2 g':>6} | {'R3 d':>6} | {'R4(pre->ctl) [CI]':<23}"
        )
        for r in ext[policy]:
            print(
                "[base4b_summary] "
                + f"{r['label']:<20} | {r['n']:>4} | {_fmt_pct(r['r0_pre'])} | {_fmt_pct(r['r0_ctl'])} | "
                + f"{_fmt_pct(r['r0_post'])} | {_fmt_signed(r['raw_diff_pre']):>8} | "
                + f"{_fmt_signed(r['raw_diff_ctl']):>8} | {_fmt_ci(r['r1_pre']):<23} | "
                + f"{_fmt_ci(r['r1_ctl']):<23} | {_fmt_signed(r['r2_g'], '+6.3f'):>6} | "
                + f"{_fmt_signed(r['r3_d'], '+6.3f'):>6} | {_fmt_ci(r['r4_pre_to_ctl']):<23}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run T-019 Stage 3 sanitized summaries and base4b_counts export.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref-root", type=Path, default=Path("/workspace/results/own-test-499-v1"))
    ap.add_argument("--overlay", type=Path, default=Path("/workspace/results/agnes-topbool-recovery-v1"))
    ap.add_argument("--decisions", type=Path, default=Path("/workspace/results/agnes-rejudge-v1"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--stage3-dir", type=Path, required=True)
    ap.add_argument("--ref27b-csv", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        mv_ver = importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        mv_ver = "NOT_INSTALLED"

    manifest = build_manifest(
        args.data,
        args.ref_root,
        args.overlay,
        args.decisions,
        repo_root=args.repo,
        require_exact_totals=True,
    )
    args.stage3_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export base4b_counts.csv & bundle for T-021 Stage 3
    exp_dir = args.stage3_dir / "base4b_counts_export"
    exp_info = export_t019_base_counts(manifest=manifest, base4b_dir=args.out_root, out_dir=exp_dir)
    print(
        f"[export_t019] OK exported={exp_info['exported_sources']}/{exp_info['eligible_total']} "
        f"held={exp_info['held_sources']} math_verify={mv_ver} "
        f"bundle_bytes={exp_info['bundle_bytes']} bundle_sha256={exp_info['bundle_sha256']}",
        flush=True,
    )

    # 2. AGNES Stage 3 tables + paired_gain.py (eduardo27b vs base)
    ag_records, ag_diag = collect_base4b_records(
        manifest, args.out_root, args.repo, grade_subdir="AGNES"
    )
    all_27b_ag = [
        {"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]}
        for e in manifest["eligible"]
    ]
    ext_ag = compute_extended_stage3_table(ag_records, all_27b_ag, seed=args.seed, draws=args.draws)
    print(format_report(manifest, ag_diag, compute_summary(ag_records, all_27b_records=all_27b_ag, seed=args.seed, draws=args.draws)))
    print_extended_table("T-019 AGNES Extended Stage 3", ext_ag)

    ag_csv = args.stage3_dir / "paired_t019_agnes.csv"
    ag_csv_info = build_base_vs_27b_paired_csv(manifest=manifest, records=ag_records, out_csv=ag_csv)
    ag_cc = cross_check_base_vs_27b(
        csv_path=ag_csv, records=ag_records, all_27b_records=all_27b_ag, draws=args.draws, seed=args.seed
    )
    print(
        f"[base4b_summary] agnes_paired_crosscheck_ok orig_inconsistent_sources={ag_csv_info['orig_inconsistent_sources']}"
    )
    for ln in pg.render_markdown(ag_cc["pg_report"]).splitlines():
        print(f"[base4b_summary] {ln}")

    payload: dict[str, Any] = {
        "math_verify_version": mv_ver,
        "export_info": exp_info,
        "diagnostics_agnes": ag_diag,
        "extended_agnes": ext_ag,
        "agnes_csv_info": ag_csv_info,
        "agnes_paired_report": ag_cc["pg_report"],
        "agnes_cross_check": ag_cc["comparisons"],
    }

    # 3. Optional AUTHOR_RULE sensitivity using T-021 exported 27B ref27b_counts.csv
    if args.ref27b_csv is not None:
        all_27b_ar, ref_ar_by_key = load_ref27b_ar_overlay(ref27b_csv=args.ref27b_csv, manifest=manifest)
        ar_records_raw, ar_diag = collect_base4b_records(
            manifest, args.out_root, args.repo, grade_subdir="AUTHOR_RULE"
        )
        ar_records = apply_27b_ar_ref_to_records(ar_records_raw, ref_ar_by_key)
        ext_ar = compute_extended_stage3_table(ar_records, all_27b_ar, seed=args.seed, draws=args.draws)
        print_extended_table("T-019 AUTHOR_RULE Sensitivity (base 0.8.0 vs 27B exported AR)", ext_ar)
        ar_csv = args.stage3_dir / "paired_t019_author_rule.csv"
        ar_csv_info = build_base_vs_27b_paired_csv(
            manifest=manifest, records=ar_records, ref_counts_by_key=ref_ar_by_key, out_csv=ar_csv
        )
        ar_cc = cross_check_base_vs_27b(
            csv_path=ar_csv, records=ar_records, all_27b_records=all_27b_ar, draws=args.draws, seed=args.seed
        )
        print(
            f"[base4b_summary] ar_paired_crosscheck_ok orig_inconsistent_sources={ar_csv_info['orig_inconsistent_sources']}"
        )
        for ln in pg.render_markdown(ar_cc["pg_report"]).splitlines():
            print(f"[base4b_summary] {ln}")
        payload["diagnostics_ar"] = ar_diag
        payload["extended_ar"] = ext_ar
        payload["ar_csv_info"] = ar_csv_info
        payload["ar_paired_report"] = ar_cc["pg_report"]
        payload["ar_cross_check"] = ar_cc["comparisons"]

    out_json = args.stage3_dir / "t019_stage3_summary.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[base4b_summary] T019_STAGE3_DONE summary_json={out_json}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
