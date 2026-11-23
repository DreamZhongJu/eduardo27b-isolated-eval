#!/usr/bin/env python3
"""Load and strictly verify imported A100 bundle on Pro 5000 for T-021."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE4B_DIR = HERE.parents[1] / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(BASE4B_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ref27b_manifest import DIFFICULTY_ORDER  # noqa: E402
from summarize_base4b import extract_safe_failure_summary  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_key(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def _extract_ref_counts(csv_row: dict[str, str], grader: str) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for tgt_zh, tgt_csv in (("原题", "orig"), ("迁移题", "trans")):
        out[tgt_zh] = {}
        for phase in ("pre", "control", "post"):
            item = {
                st: int(csv_row[f"{grader}_{tgt_csv}_{phase}_{st}"])
                for st in ("correct", "incorrect", "unknown")
            }
            if sum(item.values()) != 16:
                raise ValueError(f"27B phase count sum != 16 ({grader}/{tgt_csv}/{phase})")
            out[tgt_zh][phase] = item
    return out


def load_and_verify_bundle(
    bundle_dir: Path,
    *,
    require_exact_totals: bool = True,
) -> dict[str, Any]:
    """Verify SHA-256 hashes and strict (difficulty, source_id) alignment in imported bundle."""
    if not bundle_dir.is_dir():
        raise ValueError(f"Imported bundle directory does not exist: {bundle_dir}")

    man_path = bundle_dir / "manifest_export.json"
    inputs_path = bundle_dir / "selfbuilt_eval_inputs_1489.jsonl"
    counts_path = bundle_dir / "ref27b_counts.csv"
    for p in (man_path, inputs_path, counts_path):
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"Missing or symlink bundle file: {p.name}")

    manifest_doc = json.loads(man_path.read_text(encoding="utf-8"))
    if manifest_doc.get("schema") != "t021_a100_export_v1":
        raise ValueError("Unexpected export manifest schema")
    files_sha = manifest_doc.get("files_sha256") or {}
    if _sha256_file(inputs_path) != files_sha.get(inputs_path.name):
        raise ValueError("selfbuilt_eval_inputs_1489.jsonl SHA-256 mismatch with manifest_export.json")
    if _sha256_file(counts_path) != files_sha.get(counts_path.name):
        raise ValueError("ref27b_counts.csv SHA-256 mismatch with manifest_export.json")

    entries: list[dict[str, Any]] = []
    seen_inputs: dict[tuple[str, str], dict[str, Any]] = {}
    tier_counts = {d: 0 for d in DIFFICULTY_ORDER}
    for line in inputs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        diff = item["difficulty"]
        sid = item["source_id"]
        row = item["row"]
        if diff not in DIFFICULTY_ORDER or row.get("source_id") != sid:
            raise ValueError("Invalid difficulty or source_id in eval inputs")
        if item.get("source_key") != _source_key(sid):
            raise ValueError("source_key hash mismatch in eval inputs")
        if (diff, sid) in seen_inputs:
            raise ValueError(f"Duplicate (difficulty, source_id) in eval inputs: {diff}")
        seen_inputs[(diff, sid)] = item
        tier_counts[diff] += 1
        entries.append(item)

    with counts_path.open("r", encoding="utf-8", newline="") as cf:
        csv_rows = list(csv.DictReader(cf))

    counts_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for crow in csv_rows:
        diff = crow["difficulty"]
        sid = crow["source_id"]
        if crow.get("source_key") != _source_key(sid):
            raise ValueError("source_key hash mismatch in ref27b_counts.csv")
        if (diff, sid) in counts_by_pair:
            raise ValueError(f"Duplicate (difficulty, source_id) in ref27b_counts.csv: {diff}")
        counts_by_pair[(diff, sid)] = crow

    if set(seen_inputs) != set(counts_by_pair):
        only_in = len(set(seen_inputs) - set(counts_by_pair))
        only_cnt = len(set(counts_by_pair) - set(seen_inputs))
        raise ValueError(
            f"(difficulty, source_id) mismatch between inputs and counts (only_inputs={only_in}, only_counts={only_cnt})"
        )

    for entry in entries:
        crow = counts_by_pair[(entry["difficulty"], entry["source_id"])]
        if int(crow["row_index"]) != int(entry["row_index"]):
            raise ValueError("row_index mismatch between inputs and ref27b_counts.csv")
        entry["ref_counts"] = _extract_ref_counts(crow, "agnes")
        entry["ref_counts_ar"] = _extract_ref_counts(crow, "ar")

    if require_exact_totals:
        if len(entries) != 1489 or manifest_doc.get("eligible_count") != 1489 or manifest_doc.get("skipped_count") != 8:
            raise ValueError(
                f"Expected 1489 eligible & 8 skipped, got entries={len(entries)} "
                f"manifest=({manifest_doc.get('eligible_count')}, {manifest_doc.get('skipped_count')})"
            )
        if tier_counts != {"easy": 497, "medium": 497, "hard": 495}:
            raise ValueError(f"Unexpected per-tier counts: {tier_counts}")

    return {
        "manifest_export": manifest_doc,
        "eligible_count": len(entries),
        "skipped_count": int(manifest_doc.get("skipped_count", 0)),
        "per_tier_counts": manifest_doc.get("per_tier_counts") or tier_counts,
        "math_verify_version": manifest_doc.get("math_verify_version"),
        "eligible": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify imported A100 self-built bundle.")
    ap.add_argument("--bundle-dir", type=Path, required=True)
    args = ap.parse_args()
    loaded = load_and_verify_bundle(args.bundle_dir, require_exact_totals=True)
    print(
        f"[import_bundle] OK eligible={loaded['eligible_count']} skipped={loaded['skipped_count']} "
        f"math_verify={loaded['math_verify_version']} tiers={loaded['per_tier_counts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
