#!/usr/bin/env python3
"""Check zero overlap between T-026 train/val sets and Author 494 / Self-built 3-tier test sets.

Privacy guarantee:
    NEVER prints or writes any problem text, answers, or source_id strings.
    Outputs ONLY integer counts, booleans, and SHA-256 digests of input files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AGROUP_SB_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(AGROUP_SB_DIR), str(BASE4B_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DEFAULT_BASE,
    format_machine_summary,
    sha256_file,
    write_json_atomic,
)
from import_bundle import load_and_verify_bundle  # noqa: E402


def _norm_hash(text: str, *, casefold: bool = False) -> str:
    norm = " ".join(str(text).strip().split())
    if casefold:
        norm = norm.casefold()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _parse_extra_info(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {}


def _extract_parquet_items(parquet_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    source_ids: set[str] = set()
    prob_exact: set[str] = set()
    prob_fold: set[str] = set()
    all_prob_exact: set[str] = set()
    all_prob_fold: set[str] = set()
    n_variants = 0

    for raw_ei in df["extra_info"].tolist():
        ei = _parse_extra_info(raw_ei)
        sid = ei.get("source_id")
        if sid is not None and str(sid).strip():
            source_ids.add(str(sid).strip())
        prob = ei.get("problem")
        if prob:
            h_ex = _norm_hash(str(prob), casefold=False)
            h_cf = _norm_hash(str(prob), casefold=True)
            prob_exact.add(h_ex)
            prob_fold.add(h_cf)
            all_prob_exact.add(h_ex)
            all_prob_fold.add(h_cf)
        tp = ei.get("transfer_problem")
        if tp:
            all_prob_exact.add(_norm_hash(str(tp), casefold=False))
            all_prob_fold.add(_norm_hash(str(tp), casefold=True))
        raw_vars = ei.get("variants")
        if isinstance(raw_vars, str) and raw_vars.strip():
            raw_vars = json.loads(raw_vars)
        if isinstance(raw_vars, list):
            for v in raw_vars:
                if isinstance(v, dict) and v.get("problem"):
                    n_variants += 1
                    all_prob_exact.add(_norm_hash(str(v["problem"]), casefold=False))
                    all_prob_fold.add(_norm_hash(str(v["problem"]), casefold=True))

    return {
        "rows": len(df),
        "n_variants": n_variants,
        "has_source_id": bool(source_ids),
        "source_ids": source_ids,
        "prob_exact": prob_exact,
        "prob_fold": prob_fold,
        "all_prob_exact": all_prob_exact,
        "all_prob_fold": all_prob_fold,
    }


def _extract_selfbuilt_items(bundle_dir: Path) -> dict[str, Any]:
    loaded = load_and_verify_bundle(bundle_dir, require_exact_totals=True)
    source_ids: set[str] = set()
    prob_exact: set[str] = set()
    prob_fold: set[str] = set()
    all_prob_exact: set[str] = set()
    all_prob_fold: set[str] = set()

    for entry in loaded["eligible"]:
        sid = str(entry["source_id"]).strip()
        source_ids.add(sid)
        row = entry["row"]
        p_orig = str(row["problem"])
        p_trans = str(row["transfer_problem"])
        h_ex = _norm_hash(p_orig, casefold=False)
        h_cf = _norm_hash(p_orig, casefold=True)
        prob_exact.add(h_ex)
        prob_fold.add(h_cf)
        all_prob_exact.add(h_ex)
        all_prob_fold.add(h_cf)
        all_prob_exact.add(_norm_hash(p_trans, casefold=False))
        all_prob_fold.add(_norm_hash(p_trans, casefold=True))

    return {
        "rows": len(loaded["eligible"]),
        "unique_sources": len(source_ids),
        "source_ids": source_ids,
        "prob_exact": prob_exact,
        "prob_fold": prob_fold,
        "all_prob_exact": all_prob_exact,
        "all_prob_fold": all_prob_fold,
    }


def compute_overlap_report(
    *,
    train_parquet: Path,
    val_parquet: Path,
    author494_parquet: Path,
    selfbuilt_bundle_dir: Path,
    require_exact_counts: bool = True,
) -> dict[str, Any]:
    tr = _extract_parquet_items(train_parquet)
    va = _extract_parquet_items(val_parquet)
    a494 = _extract_parquet_items(author494_parquet)
    sb = _extract_selfbuilt_items(selfbuilt_bundle_dir)

    if require_exact_counts:
        if tr["rows"] != 1895:
            raise RuntimeError(f"Expected 1895 train rows, got {tr['rows']}")
        if va["rows"] != 100:
            raise RuntimeError(f"Expected 100 val rows, got {va['rows']}")
        if a494["rows"] != 494:
            raise RuntimeError(f"Expected 494 author494 rows, got {a494['rows']}")
        if sb["rows"] != 1489:
            raise RuntimeError(f"Expected 1489 selfbuilt entries, got {sb['rows']}")

    pairs = {
        "train_vs_author494": {
            "source_id_overlap": len(tr["source_ids"] & a494["source_ids"]) if a494["has_source_id"] else 0,
            "source_problem_exact_overlap": len(tr["prob_exact"] & a494["prob_exact"]),
            "all_problem_exact_overlap": len(tr["all_prob_exact"] & a494["all_prob_exact"]),
            "all_problem_casefold_overlap": len(tr["all_prob_fold"] & a494["all_prob_fold"]),
        },
        "val_vs_author494": {
            "source_id_overlap": len(va["source_ids"] & a494["source_ids"]) if a494["has_source_id"] else 0,
            "source_problem_exact_overlap": len(va["prob_exact"] & a494["prob_exact"]),
            "all_problem_exact_overlap": len(va["all_prob_exact"] & a494["all_prob_exact"]),
            "all_problem_casefold_overlap": len(va["all_prob_fold"] & a494["all_prob_fold"]),
        },
        "train_vs_selfbuilt": {
            "source_id_overlap": len(tr["source_ids"] & sb["source_ids"]),
            "source_problem_exact_overlap": len(tr["prob_exact"] & sb["prob_exact"]),
            "all_problem_exact_overlap": len(tr["all_prob_exact"] & sb["all_prob_exact"]),
            "all_problem_casefold_overlap": len(tr["all_prob_fold"] & sb["all_prob_fold"]),
        },
        "val_vs_selfbuilt": {
            "source_id_overlap": len(va["source_ids"] & sb["source_ids"]),
            "source_problem_exact_overlap": len(va["prob_exact"] & sb["prob_exact"]),
            "all_problem_exact_overlap": len(va["all_prob_exact"] & sb["all_prob_exact"]),
            "all_problem_casefold_overlap": len(va["all_prob_fold"] & sb["all_prob_fold"]),
        },
    }

    overlap_total = sum(v for p in pairs.values() for v in p.values())
    passed = overlap_total == 0
    return {
        "status": "OVERLAP_ZERO" if passed else "OVERLAP_FOUND",
        "rc": 0 if passed else 1,
        "passed": passed,
        "overlap_total": overlap_total,
        "counts": {
            "train_rows": tr["rows"],
            "train_variants": tr["n_variants"],
            "val_rows": va["rows"],
            "val_variants": va["n_variants"],
            "author494_rows": a494["rows"],
            "author494_has_source_id": a494["has_source_id"],
            "selfbuilt_entries": sb["rows"],
            "selfbuilt_unique_sources": sb["unique_sources"],
        },
        "input_sha256": {
            "train_parquet": sha256_file(train_parquet),
            "val_parquet": sha256_file(val_parquet),
            "author494_parquet": sha256_file(author494_parquet),
        },
        "overlaps": pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check train/val vs test overlap (counts only).")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--val-parquet", type=Path, required=True)
    ap.add_argument("--author494-parquet", type=Path, default=None)
    ap.add_argument("--selfbuilt-bundle-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    base = args.base.resolve()
    report = compute_overlap_report(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        author494_parquet=args.author494_parquet or (base / "data" / "author494_eval" / "test.parquet"),
        selfbuilt_bundle_dir=args.selfbuilt_bundle_dir or (base / "t021_bundle"),
        require_exact_counts=True,
    )
    if args.out_json is not None:
        write_json_atomic(args.out_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(
        format_machine_summary(
            {
                "status": report["status"],
                "rc": report["rc"],
                "calls": 4,
                "failures": 0 if report["passed"] else 1,
                "overlap_total": report["overlap_total"],
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
