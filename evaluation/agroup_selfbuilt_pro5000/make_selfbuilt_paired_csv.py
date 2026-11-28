#!/usr/bin/env python3
"""Build paired_gain.py CSV for T-021 self-built 3-tier eval (a_group, base, eduardo27b).

Reads:
  - Imported 27B bundle (ref27b_counts.csv + selfbuilt_eval_inputs_1489.jsonl)
  - Optional T-019 exported base4b_counts.csv (for Stage 3 a_group vs base)
  - Pro 5000 A-group RESULTS.json (AGNES or AUTHOR_RULE)
Outputs strictly sanitized CSV (no problem text, answers, or responses) ready for paired_gain.py
with groups: orig, easy, medium, hard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from collections import Counter
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
from summarize_base4b import STAGE_BY_DIFF, extract_safe_failure_summary  # noqa: E402


def _load_results_dir_counts(
    entries: list[dict[str, Any]],
    out_root: Path,
    grade_subdir: str,
) -> dict[tuple[str, str], dict[str, dict[str, int]]]:
    out: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    for e in entries:
        diff = e["difficulty"]
        stage = STAGE_BY_DIFF[diff]
        key = e["source_key"]
        sid = e["source_id"]
        exp_tgt = e["row"]["transfer_target"]
        src_dir = out_root / stage / "sources" / key
        if not (src_dir / "RAW_COMPLETE.json").is_file() or (src_dir / "SOURCE_HELD.json").exists():
            continue
        res_p = out_root / grade_subdir / stage / "sources" / key / "RESULTS.json"
        if not res_p.is_file():
            continue
        doc = json.loads(res_p.read_bytes())
        if doc.get("source_id") != sid:
            raise ValueError("RESULTS.json source_id mismatch")
        by_t: dict[str, Counter] = {"orig": Counter(), "trans": Counter()}
        for item in doc.get("results") or []:
            ident = item.get("identity") or item
            t_raw = ident.get("target")
            if t_raw == "source":
                tgt = "orig"
            elif t_raw == exp_tgt:
                tgt = "trans"
            else:
                raise ValueError("Unexpected target in RESULTS.json")
            st = item.get("status")
            if st not in ("correct", "incorrect", "unknown"):
                raise ValueError("Invalid status in RESULTS.json")
            by_t[tgt][st] += 1
        for tgt in ("orig", "trans"):
            if sum(by_t[tgt].values()) != 16:
                raise ValueError("Post result target count != 16")
        out[(diff, sid)] = {
            tgt: {st: int(by_t[tgt].get(st, 0)) for st in ("correct", "incorrect", "unknown")}
            for tgt in ("orig", "trans")
        }
    return out


def _load_base4b_csv_counts(
    csv_path: Path, grader_prefix: str
) -> dict[tuple[str, str], dict[str, dict[str, int]]]:
    out: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            diff = row["difficulty"]
            sid = row["source_id"]
            if row.get(f"{grader_prefix}_orig_post_correct", "") == "":
                continue
            by_t: dict[str, dict[str, int]] = {}
            for tgt in ("orig", "trans"):
                c = {
                    st: int(row[f"{grader_prefix}_{tgt}_post_{st}"])
                    for st in ("correct", "incorrect", "unknown")
                }
                if sum(c.values()) != 16:
                    raise ValueError("base4b_counts.csv post sum != 16")
                by_t[tgt] = c
            out[(diff, sid)] = by_t
    return out


def build_selfbuilt_paired_csv(
    *,
    loaded_bundle: dict[str, Any],
    a_group_root: Path | None,
    base4b_counts_csv: Path | None,
    out_csv: Path,
    grade_subdir: str = "AGNES",
) -> dict[str, Any]:
    """Write paired_gain.py CSV for orig_easy/orig_medium/orig_hard/easy/medium/hard (and orig when consistent)."""
    grader_prefix = "agnes" if grade_subdir == "AGNES" else "ar"
    ref_key = "ref_counts" if grade_subdir == "AGNES" else "ref_counts_ar"
    entries = loaded_bundle["eligible"]

    ag_counts = (
        _load_results_dir_counts(entries, a_group_root, grade_subdir)
        if a_group_root is not None
        else {}
    )
    base_counts = (
        _load_base4b_csv_counts(base4b_counts_csv, grader_prefix)
        if base4b_counts_csv is not None
        else {}
    )

    # paired_gain.py rejects duplicate (group="orig", unit_id) rows if 27B EASY/MEDIUM/HARD
    # measured slightly different independent pre-test draws for the same source_id.
    orig_pre_by_sid: dict[str, list[float]] = {}
    for e in entries:
        pre_orig = e[ref_key]["原题"]["pre"]["correct"] / 16.0
        orig_pre_by_sid.setdefault(e["source_id"], []).append(pre_orig)
    orig_inconsistent_sources = sum(
        1 for vals in orig_pre_by_sid.values() if max(vals) - min(vals) > pg.PRE_CONSISTENCY_TOL
    )
    emit_combined_orig = orig_inconsistent_sources == 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_name(f".{out_csv.name}.{os.getpid()}.tmp")
    written = Counter()
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pg.REQUIRED_COLUMNS))
            writer.writeheader()

            for e in entries:
                diff = e["difficulty"]
                sid = e["source_id"]
                rc = e[ref_key]
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

                # Eduardo-27B always available across all eligible entries.
                if emit_combined_orig:
                    _emit("eduardo27b", "orig", pre_orig, rc["原题"]["post"])
                _emit("eduardo27b", f"orig_{diff}", pre_orig, rc["原题"]["post"])
                _emit("eduardo27b", diff, pre_trans, rc["迁移题"]["post"])

                if (diff, sid) in base_counts:
                    bc = base_counts[(diff, sid)]
                    if emit_combined_orig:
                        _emit("base", "orig", pre_orig, bc["orig"])
                    _emit("base", f"orig_{diff}", pre_orig, bc["orig"])
                    _emit("base", diff, pre_trans, bc["trans"])

                if (diff, sid) in ag_counts:
                    ac = ag_counts[(diff, sid)]
                    if emit_combined_orig:
                        _emit("a_group", "orig", pre_orig, ac["orig"])
                    _emit("a_group", f"orig_{diff}", pre_orig, ac["orig"])
                    _emit("a_group", diff, pre_trans, ac["trans"])

        # Validate immediately with paired_gain loader.
        pg.load_csv(tmp_csv)
        os.replace(tmp_csv, out_csv)
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise

    return {
        "csv_path": str(out_csv),
        "row_counts": dict(sorted(written.items())),
        "emit_combined_orig": emit_combined_orig,
        "orig_inconsistent_sources": orig_inconsistent_sources,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build paired_gain CSV for T-021 self-built 3-tier.")
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--a-group-root", type=Path, default=None)
    ap.add_argument("--base4b-counts-csv", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--grade-subdir", choices=("AGNES", "AUTHOR_RULE"), default="AGNES")
    args = ap.parse_args()

    loaded = load_and_verify_bundle(args.bundle_dir, require_exact_totals=True)
    info = build_selfbuilt_paired_csv(
        loaded_bundle=loaded,
        a_group_root=args.a_group_root,
        base4b_counts_csv=args.base4b_counts_csv,
        out_csv=args.out_csv,
        grade_subdir=args.grade_subdir,
    )
    print(f"[make_selfbuilt_paired_csv] OK counts={json.dumps(info['row_counts'], sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
