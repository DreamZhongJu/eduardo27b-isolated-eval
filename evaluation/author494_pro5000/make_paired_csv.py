#!/usr/bin/env python3
"""Build paired_gain.py CSV from Pro 5000 A-group JSON and A100 T-018 per-problem table.

- Uses row_index as unit_id (kept strictly local; never printed).
- Requires 0 student failed calls in A-group JSON.
- Requires identical row_index sets and identical pre_solve_rate / pre_solve_rate_transfer
  for every row_index across A-group JSON and T-018 table (within 1e-9).
- Immediately verifies that paired_gain.py's mean normalized delta per (condition, group)
  matches A-group JSON metrics and T-018 table recomputed deltas within < 1e-9.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PAIRED_DIR = HERE / "paired_gain"
for p in (HERE, PAIRED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paired_gain as pg  # noqa: E402
from export_t018_table import load_t018_table, recompute_from_t018_table  # noqa: E402
from summarize_author494_pro5000 import (  # noqa: E402
    recompute_author_summary,
    validate_results_doc,
    verify_author_summary_match,
)

PRE_TOL = 1e-9
DELTA_TOL = 1e-9


def _extract_num_attempts(doc: dict[str, Any], label: str) -> int:
    args = doc.get("args")
    if isinstance(args, dict) and "num_student_attempts" in args:
        k = args["num_student_attempts"]
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"{label} JSON has invalid args['num_student_attempts']")
        return k
    return 16


def _counts_from_rate(rate: float, total: int, field: str) -> int:
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"Out-of-range {field}: {rate}")
    raw = rate * total
    rounded = int(round(raw))
    if not math.isclose(raw, rounded, abs_tol=1e-9):
        raise ValueError(
            f"{field}={rate} is not an integer multiple of 1/{total} (cannot reconstruct exact post_correct)"
        )
    return rounded


def build_and_verify_paired_csv(
    *,
    t018_by_row: dict[int, dict[str, Any]],
    a_group_doc: dict[str, Any],
    out_csv: Path,
    base_name: str = "base",
    target_name: str = "a_group",
    t018_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate inputs, write paired CSV atomically, and verify <1e-9 delta match."""
    ag_metrics, ag_records, _ = validate_results_doc(a_group_doc)
    ag_re = recompute_author_summary(ag_records)
    verify_author_summary_match(ag_re, ag_metrics)

    if ag_re["n_student_calls_failed"] != 0:
        raise ValueError(f"a_group JSON has student_calls_failed={ag_re['n_student_calls_failed']} != 0")

    k_ag = _extract_num_attempts(a_group_doc, "a_group")
    ag_by_row: dict[int, list[dict[str, Any]]] = {}
    for rec in ag_records:
        ag_by_row.setdefault(rec["row_index"], []).append(rec)

    if set(t018_by_row) != set(ag_by_row):
        only_base = len(set(t018_by_row) - set(ag_by_row))
        only_ag = len(set(ag_by_row) - set(t018_by_row))
        raise ValueError(
            f"row_index sets differ between T-018 table and a_group (only_base={only_base}, only_a_group={only_ag})"
        )

    sorted_rows = sorted(t018_by_row)
    mismatch_same = 0
    mismatch_transfer = 0
    for ridx in sorted_rows:
        t_rec = t018_by_row[ridx]
        for a_rec in ag_by_row[ridx]:
            if not math.isclose(
                float(t_rec["pre_solve_rate"]),
                float(a_rec["pre_solve_rate"]),
                abs_tol=PRE_TOL,
            ):
                mismatch_same += 1
            if not math.isclose(
                float(t_rec["pre_solve_rate_transfer"]),
                float(a_rec["pre_solve_rate_transfer"]),
                abs_tol=PRE_TOL,
            ):
                mismatch_transfer += 1

    if mismatch_same or mismatch_transfer:
        raise ValueError(
            "Pre-test consistency mismatch between T-018 table and a_group "
            f"(pre_same_mismatch={mismatch_same}, pre_transfer_mismatch={mismatch_transfer})"
        )

    t018_recomp = recompute_from_t018_table(t018_by_row)
    if t018_manifest is not None:
        mc = t018_manifest.get("recomputed_checks") or {}
        for k in ("delta_transfer", "delta_same"):
            if k in mc and abs(t018_recomp[k] - float(mc[k])) >= DELTA_TOL:
                raise ValueError(f"T-018 table {k} mismatch against t018_manifest.json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_name(f".{out_csv.name}.{os.getpid()}.tmp")
    n_rows_written = 0
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pg.REQUIRED_COLUMNS))
            writer.writeheader()
            # Write base rows from A100 T-018 per-problem table.
            for ridx in sorted_rows:
                t_rec = t018_by_row[ridx]
                writer.writerow({
                    "unit_id": str(ridx),
                    "condition": base_name,
                    "group": "author494_transfer",
                    "pre": f"{float(t_rec['pre_solve_rate_transfer']):.17g}",
                    "post_correct": int(t_rec["post_transfer_correct"]),
                    "post_total": int(t_rec["post_transfer_total"]),
                    "post_unknown": 0,
                })
                writer.writerow({
                    "unit_id": str(ridx),
                    "condition": base_name,
                    "group": "author494_same",
                    "pre": f"{float(t_rec['pre_solve_rate']):.17g}",
                    "post_correct": int(t_rec["post_same_correct"]),
                    "post_total": int(t_rec["post_same_total"]),
                    "post_unknown": 0,
                })
                n_rows_written += 2

            # Write a_group rows from Pro 5000 result JSON.
            for ridx in sorted_rows:
                for rec in ag_by_row[ridx]:
                    c_trans = _counts_from_rate(
                        float(rec["post_solve_transfer"]), k_ag, "post_solve_transfer"
                    )
                    c_same = _counts_from_rate(
                        float(rec["post_solve_same"]), k_ag, "post_solve_same"
                    )
                    writer.writerow({
                        "unit_id": str(ridx),
                        "condition": target_name,
                        "group": "author494_transfer",
                        "pre": f"{float(rec['pre_solve_rate_transfer']):.17g}",
                        "post_correct": c_trans,
                        "post_total": k_ag,
                        "post_unknown": 0,
                    })
                    writer.writerow({
                        "unit_id": str(ridx),
                        "condition": target_name,
                        "group": "author494_same",
                        "pre": f"{float(rec['pre_solve_rate']):.17g}",
                        "post_correct": c_same,
                        "post_total": k_ag,
                        "post_unknown": 0,
                    })
                    n_rows_written += 2

        loaded_rows = pg.load_csv(tmp_csv)
        checks: dict[str, dict[str, float]] = {}
        expected_by_cond = {
            base_name: {
                "delta_transfer": t018_recomp["delta_transfer"],
                "delta_same": t018_recomp["delta_same"],
            },
            target_name: {
                "delta_transfer": float(ag_metrics["delta_transfer"]),
                "delta_same": float(ag_metrics["delta_same"]),
            },
        }
        for cond_name, expected_metrics in expected_by_cond.items():
            for grp_name, metric_key in (
                ("author494_transfer", "delta_transfer"),
                ("author494_same", "delta_same"),
            ):
                unit_gains, _, n_skip, n_excl = pg.aggregate_condition(
                    loaded_rows, grp_name, cond_name, unknown_as_wrong=True
                )
                if n_skip or n_excl or len(unit_gains) != len(sorted_rows):
                    raise ValueError(
                        f"Unexpected CSV aggregation count for ({cond_name}, {grp_name}): "
                        f"units={len(unit_gains)} expected={len(sorted_rows)} skip={n_skip} excl={n_excl}"
                    )
                csv_mean = sum(unit_gains.values()) / len(unit_gains)
                ref_mean = float(expected_metrics[metric_key])
                diff = abs(csv_mean - ref_mean)
                if diff >= DELTA_TOL:
                    raise ValueError(
                        f"CSV delta check failed for ({cond_name}, {grp_name}): "
                        f"csv={csv_mean:.12f} ref={ref_mean:.12f} diff={diff:.3e} >= {DELTA_TOL:.1e}"
                    )
                checks[f"{cond_name}.{grp_name}"] = {
                    "csv_mean": csv_mean,
                    "ref_mean": ref_mean,
                    "abs_diff": diff,
                }

        os.replace(tmp_csv, out_csv)
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise

    return {
        "n_problems": len(sorted_rows),
        "n_csv_rows": n_rows_written,
        "post_total_a_group": k_ag,
        "delta_checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create and verify paired_gain CSV from Pro 5000 A-group JSON and A100 T-018 table."
    )
    ap.add_argument("--a-group-json", required=True, type=Path)
    ap.add_argument("--t018-table", required=True, type=Path)
    ap.add_argument("--t018-manifest", type=Path, default=None)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--target-name", default="a_group")
    args = ap.parse_args()

    try:
        t018_rows = load_t018_table(args.t018_table)
        ag_doc = json.loads(args.a_group_json.read_text(encoding="utf-8"))
        manifest = (
            json.loads(args.t018_manifest.read_text(encoding="utf-8"))
            if args.t018_manifest is not None
            else None
        )
        info = build_and_verify_paired_csv(
            t018_by_row=t018_rows,
            a_group_doc=ag_doc,
            out_csv=args.out_csv,
            base_name=args.base_name,
            target_name=args.target_name,
            t018_manifest=manifest,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"[make_paired_csv] OK problems={info['n_problems']} csv_rows={info['n_csv_rows']} "
        f"post_total_a_group={info['post_total_a_group']}"
    )
    for label, d in info["delta_checks"].items():
        print(
            f"[make_paired_csv] check {label}: csv={d['csv_mean']:+.6f} "
            f"ref={d['ref_mean']:+.6f} diff={d['abs_diff']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
