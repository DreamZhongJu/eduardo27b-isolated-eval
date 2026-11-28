#!/usr/bin/env python3
"""Extract sanitized per-problem table from A100 T-018 Stage 2 & Stage 3 JSONs.

Writes:
  - t018_per_problem.csv (494 rows, NO problem text, answers, dialogues, or reasoning)
  - t018_manifest.json (SHA-256 of test.parquet, both source JSONs, CSV, and checks)
  - t018_table_bundle.tar.gz + t018_table_bundle.tar.gz.sha256

Immediately recomputes T-018 Δ_transfer, Δ_same (primary & gated), and ctl1 means
from the written CSV and verifies < 1e-9 agreement with source JSON metrics before
building the tar.gz bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize_author494_pro5000 import (  # noqa: E402
    recompute_author_summary,
    validate_results_doc,
    verify_author_summary_match,
)

DEFAULT_STAGE2_JSON = Path(
    "/workspace/A_group_repro_20261117/eval_author494_full494/transfer_eval_base_qwen35_4b_author494_20261122_144033.json"
)
DEFAULT_STAGE3_JSON = Path(
    "/workspace/A_group_repro_20261117/eval_author494_control_r2/control_eval_base_qwen35_4b_author494_control_20261123_003400.json"
)
DEFAULT_TEST_PARQUET = Path("/workspace/A_group_repro_20261117/data/author494_eval/test.parquet")

TABLE_COLUMNS = (
    "row_index",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "post_same_correct",
    "post_same_total",
    "post_transfer_correct",
    "post_transfer_total",
    "ctl1_same_correct",
    "ctl1_same_total",
    "ctl1_transfer_correct",
    "ctl1_transfer_total",
    "ctl2_same_correct",
    "ctl2_same_total",
    "ctl2_transfer_correct",
    "ctl2_transfer_total",
    "leaked",
    "teaching_rejected",
    "eval_judge_pedagogy",
    "eval_judge_correctness",
    "has_blanked_turn",
    "is_truncated",
)

TOL = 1e-9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_delta(post: float, pre: float) -> float:
    if post <= pre or pre >= 1.0:
        return post - pre
    return (post - pre) / (1.0 - pre)


def _int_correct_from_rate(rate: float, total: int, label: str) -> int:
    if total <= 0:
        raise ValueError(f"Invalid total={total} for {label}")
    raw = rate * total
    rounded = int(round(raw))
    if not math.isclose(raw, rounded, abs_tol=TOL):
        raise ValueError(f"{label} rate={rate} is not an integer count out of {total}")
    return rounded


def load_t018_table(csv_path: Path) -> dict[int, dict[str, Any]]:
    """Load and validate t018_per_problem.csv keyed by row_index."""
    by_row: dict[int, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("T-018 table CSV is empty")
        missing = [c for c in TABLE_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"T-018 table CSV missing columns: {missing}")
        for idx, raw in enumerate(reader, start=2):
            ridx = int(raw["row_index"])
            if ridx in by_row:
                raise ValueError(f"Duplicate row_index in T-018 table at line {idx}")
            ped_s = raw["eval_judge_pedagogy"].strip()
            fact_s = raw["eval_judge_correctness"].strip()
            rec = {
                "row_index": ridx,
                "pre_solve_rate": float(raw["pre_solve_rate"]),
                "pre_solve_rate_transfer": float(raw["pre_solve_rate_transfer"]),
                "post_same_correct": int(raw["post_same_correct"]),
                "post_same_total": int(raw["post_same_total"]),
                "post_transfer_correct": int(raw["post_transfer_correct"]),
                "post_transfer_total": int(raw["post_transfer_total"]),
                "ctl1_same_correct": int(raw["ctl1_same_correct"]),
                "ctl1_same_total": int(raw["ctl1_same_total"]),
                "ctl1_transfer_correct": int(raw["ctl1_transfer_correct"]),
                "ctl1_transfer_total": int(raw["ctl1_transfer_total"]),
                "ctl2_same_correct": int(raw["ctl2_same_correct"]),
                "ctl2_same_total": int(raw["ctl2_same_total"]),
                "ctl2_transfer_correct": int(raw["ctl2_transfer_correct"]),
                "ctl2_transfer_total": int(raw["ctl2_transfer_total"]),
                "leaked": int(raw["leaked"]),
                "teaching_rejected": int(raw["teaching_rejected"]),
                "eval_judge_pedagogy": float(ped_s) if ped_s != "" else None,
                "eval_judge_correctness": float(fact_s) if fact_s != "" else None,
                "has_blanked_turn": int(raw["has_blanked_turn"]),
                "is_truncated": int(raw["is_truncated"]),
            }
            if rec["leaked"] not in (0, 1) or rec["teaching_rejected"] not in (0, 1):
                raise ValueError(f"Invalid gate flag at CSV line {idx}")
            if rec["has_blanked_turn"] not in (0, 1) or rec["is_truncated"] not in (0, 1):
                raise ValueError(f"Invalid dialogue flag at CSV line {idx}")
            by_row[ridx] = rec
    if not by_row:
        raise ValueError("T-018 table CSV has 0 rows")
    return by_row


def recompute_from_t018_table(by_row: dict[int, dict[str, Any]]) -> dict[str, float]:
    """Recompute primary/gated deltas and ctl1/ctl2 means directly from table counts."""
    dt_vals: list[float] = []
    ds_vals: list[float] = []
    dt_g_vals: list[float] = []
    ds_g_vals: list[float] = []
    ctl1_s_vals: list[float] = []
    ctl1_t_vals: list[float] = []
    ctl2_s_vals: list[float] = []
    ctl2_t_vals: list[float] = []

    for ridx in sorted(by_row):
        r = by_row[ridx]
        post_t = r["post_transfer_correct"] / r["post_transfer_total"]
        post_s = r["post_same_correct"] / r["post_same_total"]
        dt = normalized_delta(post_t, r["pre_solve_rate_transfer"])
        ds = normalized_delta(post_s, r["pre_solve_rate"])
        dt_vals.append(dt)
        ds_vals.append(ds)
        dt_g_vals.append(0.0 if r["leaked"] else dt)
        ds_g_vals.append(0.0 if r["leaked"] else ds)
        ctl1_s_vals.append(r["ctl1_same_correct"] / r["ctl1_same_total"])
        ctl1_t_vals.append(r["ctl1_transfer_correct"] / r["ctl1_transfer_total"])
        ctl2_s_vals.append(r["ctl2_same_correct"] / r["ctl2_same_total"])
        ctl2_t_vals.append(r["ctl2_transfer_correct"] / r["ctl2_transfer_total"])

    return {
        "n_problems": float(len(by_row)),
        "delta_transfer": statistics.fmean(dt_vals),
        "delta_same": statistics.fmean(ds_vals),
        "delta_transfer_gated": statistics.fmean(dt_g_vals),
        "delta_same_gated": statistics.fmean(ds_g_vals),
        "ctl1_same_mean": statistics.fmean(ctl1_s_vals),
        "ctl1_transfer_mean": statistics.fmean(ctl1_t_vals),
        "ctl2_same_mean": statistics.fmean(ctl2_s_vals),
        "ctl2_transfer_mean": statistics.fmean(ctl2_t_vals),
    }


def export_t018_bundle(
    *,
    stage2_json: Path,
    stage3_json: Path,
    test_parquet: Path,
    out_dir: Path,
    expected_rows: int = 494,
) -> dict[str, Any]:
    """Extract, verify (<1e-9), and package t018_table_bundle.tar.gz."""
    s2_doc = json.loads(stage2_json.read_text(encoding="utf-8"))
    s3_doc = json.loads(stage3_json.read_text(encoding="utf-8"))

    s2_metrics, s2_records, _ = validate_results_doc(s2_doc)
    s2_re = recompute_author_summary(s2_records)
    verify_author_summary_match(s2_re, s2_metrics)
    if s2_re["n_student_calls_failed"] != 0:
        raise ValueError(f"Stage 2 has student_calls_failed={s2_re['n_student_calls_failed']}")

    s3_metrics = s3_doc.get("metrics") or {}
    s3_records = s3_doc.get("records") or []
    if not s3_records:
        raise ValueError("Stage 3 JSON missing non-empty 'records'")
    if int(s3_metrics.get("calls_failed", -1)) != 0:
        raise ValueError(f"Stage 3 has calls_failed={s3_metrics.get('calls_failed')}")

    k_post = int((s2_doc.get("args") or {}).get("num_student_attempts", 16))
    s2_by_row: dict[int, dict[str, Any]] = {}
    for rec in s2_records:
        ridx = rec["row_index"]
        if ridx in s2_by_row:
            raise ValueError("Stage 2 has multiple conversations per row_index; expected 1:1")
        s2_by_row[ridx] = rec

    s3_by_row: dict[int, dict[str, Any]] = {}
    for rec in s3_records:
        ridx = int(rec["row_index"])
        if ridx in s3_by_row:
            raise ValueError("Stage 3 has duplicate row_index")
        s3_by_row[ridx] = rec

    if set(s2_by_row) != set(s3_by_row):
        raise ValueError("row_index sets differ between Stage 2 and Stage 3 JSONs")
    if expected_rows > 0 and len(s2_by_row) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(s2_by_row)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "t018_per_problem.csv"
    tmp_csv = out_dir / f".t018_per_problem.{os.getpid()}.tmp"

    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(TABLE_COLUMNS))
            writer.writeheader()
            for ridx in sorted(s2_by_row):
                r2 = s2_by_row[ridx]
                r3 = s3_by_row[ridx]
                pre_s = float(r2["pre_solve_rate"])
                pre_t = float(r2["pre_solve_rate_transfer"])
                if not math.isclose(pre_s, float(r3["pre_solve_rate"]), abs_tol=TOL) or not math.isclose(
                    pre_t, float(r3["pre_solve_rate_transfer"]), abs_tol=TOL
                ):
                    raise ValueError("pre_solve_rate mismatch between Stage 2 and Stage 3")

                for rnd in ("ctl1", "ctl2"):
                    for kind in ("same", "transfer"):
                        if int(r3[f"{rnd}_{kind}_failed"]) != 0:
                            raise ValueError(f"Stage 3 has failed calls in {rnd}_{kind}")

                c1_s_tot = int(r3["ctl1_same_scored"])
                c1_t_tot = int(r3["ctl1_transfer_scored"])
                c2_s_tot = int(r3["ctl2_same_scored"])
                c2_t_tot = int(r3["ctl2_transfer_scored"])

                has_blanked = int(
                    float(r2["blanked_teacher_turn_rate"]) > 0.0
                    or int(r2.get("num_blanked_teacher_turns", 0)) > 0
                )
                is_trunc = int(r2["end_reason"] == "token_budget")

                writer.writerow({
                    "row_index": ridx,
                    "pre_solve_rate": f"{pre_s:.17g}",
                    "pre_solve_rate_transfer": f"{pre_t:.17g}",
                    "post_same_correct": _int_correct_from_rate(float(r2["post_solve_same"]), k_post, "post_same"),
                    "post_same_total": k_post,
                    "post_transfer_correct": _int_correct_from_rate(float(r2["post_solve_transfer"]), k_post, "post_transfer"),
                    "post_transfer_total": k_post,
                    "ctl1_same_correct": _int_correct_from_rate(float(r3["ctl1_same"]), c1_s_tot, "ctl1_same"),
                    "ctl1_same_total": c1_s_tot,
                    "ctl1_transfer_correct": _int_correct_from_rate(float(r3["ctl1_transfer"]), c1_t_tot, "ctl1_transfer"),
                    "ctl1_transfer_total": c1_t_tot,
                    "ctl2_same_correct": _int_correct_from_rate(float(r3["ctl2_same"]), c2_s_tot, "ctl2_same"),
                    "ctl2_same_total": c2_s_tot,
                    "ctl2_transfer_correct": _int_correct_from_rate(float(r3["ctl2_transfer"]), c2_t_tot, "ctl2_transfer"),
                    "ctl2_transfer_total": c2_t_tot,
                    "leaked": int(r2["leaked"]),
                    "teaching_rejected": int(r2["teaching_rejected"]),
                    "eval_judge_pedagogy": "" if r2["eval_judge_pedagogy"] is None else f"{float(r2['eval_judge_pedagogy']):.17g}",
                    "eval_judge_correctness": "" if r2["eval_judge_correctness"] is None else f"{float(r2['eval_judge_correctness']):.17g}",
                    "has_blanked_turn": has_blanked,
                    "is_truncated": is_trunc,
                })

        loaded = load_t018_table(tmp_csv)
        recomp = recompute_from_t018_table(loaded)
        checks = [
            ("delta_transfer", recomp["delta_transfer"], float(s2_metrics["delta_transfer"])),
            ("delta_same", recomp["delta_same"], float(s2_metrics["delta_same"])),
            ("delta_transfer_gated", recomp["delta_transfer_gated"], float(s2_metrics["delta_transfer_gated"])),
            ("delta_same_gated", recomp["delta_same_gated"], float(s2_metrics["delta_same_gated"])),
            ("ctl1_same_mean", recomp["ctl1_same_mean"], float(s3_metrics["ctl1_same"])),
            ("ctl1_transfer_mean", recomp["ctl1_transfer_mean"], float(s3_metrics["ctl1_transfer"])),
            ("ctl2_same_mean", recomp["ctl2_same_mean"], float(s3_metrics["ctl2_same"])),
            ("ctl2_transfer_mean", recomp["ctl2_transfer_mean"], float(s3_metrics["ctl2_transfer"])),
        ]
        for name, got, exp in checks:
            if abs(got - exp) >= TOL:
                raise ValueError(f"Recomputed {name}={got:.12f} != source JSON {exp:.12f}")

        os.replace(tmp_csv, csv_path)
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise

    manifest = {
        "source": "T-018 A100 Qwen3.5-4B base",
        "num_problems": len(loaded),
        "test_parquet_sha256": sha256_file(test_parquet),
        "stage2_json_sha256": sha256_file(stage2_json),
        "stage3_json_sha256": sha256_file(stage3_json),
        "table_filename": "t018_per_problem.csv",
        "table_sha256": sha256_file(csv_path),
        "recomputed_checks": recomp,
    }
    manifest_path = out_dir / "t018_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle_path = out_dir / "t018_table_bundle.tar.gz"
    tmp_tar = out_dir / f".t018_table_bundle.{os.getpid()}.tar.gz.tmp"
    try:
        with tarfile.open(tmp_tar, "w:gz") as tf:
            for fname in ("t018_per_problem.csv", "t018_manifest.json"):
                fp = out_dir / fname
                data = fp.read_bytes()
                ti = tarfile.TarInfo(name=fname)
                ti.size = len(data)
                ti.mode = 0o644
                ti.mtime = 0
                tf.addfile(ti, io.BytesIO(data))
        os.replace(tmp_tar, bundle_path)
    except Exception:
        tmp_tar.unlink(missing_ok=True)
        raise

    bundle_sha = sha256_file(bundle_path)
    (out_dir / "t018_table_bundle.tar.gz.sha256").write_text(
        f"{bundle_sha}  t018_table_bundle.tar.gz\n", encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": bundle_sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export sanitized per-problem table from T-018 JSONs on A100.")
    ap.add_argument("--stage2-json", type=Path, default=DEFAULT_STAGE2_JSON)
    ap.add_argument("--stage3-json", type=Path, default=DEFAULT_STAGE3_JSON)
    ap.add_argument("--test-parquet", type=Path, default=DEFAULT_TEST_PARQUET)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-rows", type=int, default=494)
    args = ap.parse_args()

    try:
        res = export_t018_bundle(
            stage2_json=args.stage2_json,
            stage3_json=args.stage3_json,
            test_parquet=args.test_parquet,
            out_dir=args.out_dir,
            expected_rows=args.expected_rows,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    m = res["manifest"]
    rc = m["recomputed_checks"]
    print(f"[export_t018_table] OK num_problems={m['num_problems']}")
    print(f"test_parquet_sha256={m['test_parquet_sha256']}")
    print(f"stage2_json_sha256={m['stage2_json_sha256']}")
    print(f"stage3_json_sha256={m['stage3_json_sha256']}")
    print(f"table_sha256={m['table_sha256']}")
    print(f"bundle_sha256={res['bundle_sha256']} bytes={res['bundle_size_bytes']}")
    print(
        f"recomputed: delta_transfer={rc['delta_transfer']:+.6f} delta_same={rc['delta_same']:+.6f} "
        f"delta_transfer_gated={rc['delta_transfer_gated']:+.6f} delta_same_gated={rc['delta_same_gated']:+.6f} "
        f"ctl1_same={rc['ctl1_same_mean']:.6f} ctl1_transfer={rc['ctl1_transfer_mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
