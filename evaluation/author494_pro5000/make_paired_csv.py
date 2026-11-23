#!/usr/bin/env python3
"""Build paired_gain.py CSV from Pro 5000 A-group and base author-494 result JSONs.

- Uses row_index as unit_id (kept strictly local; never printed).
- Requires 0 student failed calls in both JSONs.
- Requires identical row_index sets and identical pre_solve_rate / pre_solve_rate_transfer
  for every row_index across both runs (within 1e-9).
- Immediately verifies that paired_gain.py's mean normalized delta per (condition, group)
  matches the author JSON metrics within < 1e-9 before committing the output CSV.
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


def _group_by_row(records: list[dict[str, Any]], label: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["row_index"], []).append(rec)
    for ridx, recs in grouped.items():
        if len(recs) > 1:
            # Verify internal consistency if multiple samples exist for one row.
            p_s0 = float(recs[0]["pre_solve_rate"])
            p_t0 = float(recs[0]["pre_solve_rate_transfer"])
            for r in recs[1:]:
                if not math.isclose(float(r["pre_solve_rate"]), p_s0, abs_tol=PRE_TOL) or not math.isclose(
                    float(r["pre_solve_rate_transfer"]), p_t0, abs_tol=PRE_TOL
                ):
                    raise ValueError(f"{label} JSON has inconsistent pre_solve_rate within a problem")
    return grouped


def build_and_verify_paired_csv(
    *,
    base_doc: dict[str, Any],
    a_group_doc: dict[str, Any],
    out_csv: Path,
    base_name: str = "base",
    target_name: str = "a_group",
) -> dict[str, Any]:
    """Validate both docs, write paired CSV atomically, and verify exact delta match."""
    base_metrics, base_records, _ = validate_results_doc(base_doc)
    ag_metrics, ag_records, _ = validate_results_doc(a_group_doc)

    base_re = recompute_author_summary(base_records)
    ag_re = recompute_author_summary(ag_records)
    verify_author_summary_match(base_re, base_metrics)
    verify_author_summary_match(ag_re, ag_metrics)

    if base_re["n_student_calls_failed"] != 0:
        raise ValueError(f"base JSON has student_calls_failed={base_re['n_student_calls_failed']} != 0")
    if ag_re["n_student_calls_failed"] != 0:
        raise ValueError(f"a_group JSON has student_calls_failed={ag_re['n_student_calls_failed']} != 0")

    k_base = _extract_num_attempts(base_doc, "base")
    k_ag = _extract_num_attempts(a_group_doc, "a_group")
    if k_base != k_ag:
        raise ValueError(f"num_student_attempts mismatch: base={k_base} vs a_group={k_ag}")

    base_by_row = _group_by_row(base_records, "base")
    ag_by_row = _group_by_row(ag_records, "a_group")

    if set(base_by_row) != set(ag_by_row):
        only_base = len(set(base_by_row) - set(ag_by_row))
        only_ag = len(set(ag_by_row) - set(base_by_row))
        raise ValueError(
            f"row_index sets differ between base and a_group (only_base={only_base}, only_a_group={only_ag})"
        )

    sorted_rows = sorted(base_by_row)
    mismatch_same = 0
    mismatch_transfer = 0
    sample_mismatch = 0
    for ridx in sorted_rows:
        b_recs = base_by_row[ridx]
        a_recs = ag_by_row[ridx]
        if len(b_recs) != len(a_recs):
            sample_mismatch += 1
        if not math.isclose(
            float(b_recs[0]["pre_solve_rate"]),
            float(a_recs[0]["pre_solve_rate"]),
            abs_tol=PRE_TOL,
        ):
            mismatch_same += 1
        if not math.isclose(
            float(b_recs[0]["pre_solve_rate_transfer"]),
            float(a_recs[0]["pre_solve_rate_transfer"]),
            abs_tol=PRE_TOL,
        ):
            mismatch_transfer += 1

    if sample_mismatch or mismatch_same or mismatch_transfer:
        raise ValueError(
            "Pre-test or sample-count mismatch across runs "
            f"(sample_mismatch={sample_mismatch}, pre_same_mismatch={mismatch_same}, "
            f"pre_transfer_mismatch={mismatch_transfer})"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_name(f".{out_csv.name}.{os.getpid()}.tmp")
    n_rows_written = 0
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pg.REQUIRED_COLUMNS))
            writer.writeheader()
            for cond_name, by_row in ((base_name, base_by_row), (target_name, ag_by_row)):
                for ridx in sorted_rows:
                    for rec in by_row[ridx]:
                        c_trans = _counts_from_rate(
                            float(rec["post_solve_transfer"]), k_base, "post_solve_transfer"
                        )
                        c_same = _counts_from_rate(
                            float(rec["post_solve_same"]), k_base, "post_solve_same"
                        )
                        writer.writerow({
                            "unit_id": str(ridx),
                            "condition": cond_name,
                            "group": "author494_transfer",
                            "pre": f"{float(rec['pre_solve_rate_transfer']):.17g}",
                            "post_correct": c_trans,
                            "post_total": k_base,
                            "post_unknown": 0,
                        })
                        writer.writerow({
                            "unit_id": str(ridx),
                            "condition": cond_name,
                            "group": "author494_same",
                            "pre": f"{float(rec['pre_solve_rate']):.17g}",
                            "post_correct": c_same,
                            "post_total": k_base,
                            "post_unknown": 0,
                        })
                        n_rows_written += 2

        loaded_rows = pg.load_csv(tmp_csv)
        checks: dict[str, dict[str, float]] = {}
        for cond_name, expected_metrics in (
            (base_name, base_metrics),
            (target_name, ag_metrics),
        ):
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
                json_mean = float(expected_metrics[metric_key])
                diff = abs(csv_mean - json_mean)
                if diff >= DELTA_TOL:
                    raise ValueError(
                        f"CSV delta check failed for ({cond_name}, {grp_name}): "
                        f"csv={csv_mean:.12f} json={json_mean:.12f} diff={diff:.3e} >= {DELTA_TOL:.1e}"
                    )
                checks[f"{cond_name}.{grp_name}"] = {
                    "csv_mean": csv_mean,
                    "json_mean": json_mean,
                    "abs_diff": diff,
                }

        os.replace(tmp_csv, out_csv)
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise

    return {
        "n_problems": len(sorted_rows),
        "n_csv_rows": n_rows_written,
        "post_total": k_base,
        "delta_checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Create and verify paired_gain CSV from two author-494 JSONs.")
    ap.add_argument("--base-json", required=True, type=Path)
    ap.add_argument("--a-group-json", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--target-name", default="a_group")
    args = ap.parse_args()

    try:
        base_doc = json.loads(args.base_json.read_text(encoding="utf-8"))
        ag_doc = json.loads(args.a_group_json.read_text(encoding="utf-8"))
        info = build_and_verify_paired_csv(
            base_doc=base_doc,
            a_group_doc=ag_doc,
            out_csv=args.out_csv,
            base_name=args.base_name,
            target_name=args.target_name,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"[make_paired_csv] OK problems={info['n_problems']} csv_rows={info['n_csv_rows']} "
        f"post_total={info['post_total']}"
    )
    for label, d in info["delta_checks"].items():
        print(
            f"[make_paired_csv] check {label}: csv={d['csv_mean']:+.6f} "
            f"json={d['json_mean']:+.6f} diff={d['abs_diff']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
