#!/usr/bin/env python3
"""Read-only A100 exporter for T-021 (Pro 5000 A-group step_100 x self-built 3-tier).

Modes:
- --mode ref27b (default): Exports ONLY the 1,489 eligible evaluation inputs (no 27B
  dialogues or responses) + per-source Agnes and offline math_verify Author-Rule
  counts + manifest & SHA-256s, recomputes and checks 27B R1 before packing tar.gz.
- --mode t019-base: Stage 3 helper to export T-019 base4b post counts (no raw text).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import signal
import sys
import tarfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = HERE.parents[1]
BASE4B_DIR = REPO_ROOT_DEFAULT / "evaluation" / "base4b_selfbuilt"
PAIRED_DIR = REPO_ROOT_DEFAULT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(BASE4B_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ref27b_manifest import DIFFICULTY_ORDER, GROUP_BY_DIFFICULTY, build_manifest  # noqa: E402
from summarize_base4b import (  # noqa: E402
    STAGE_BY_DIFF,
    bootstrap_mean_ci,
    extract_safe_failure_summary,
    normalized_gain,
    p_value,
)

COUNT_COLUMNS = (
    "difficulty",
    "tier_zh",
    "ref_stage",
    "source_key",
    "source_id",
    "row_index",
    *(
        f"{grader}_{tgt}_{phase}_{st}"
        for grader in ("agnes", "ar")
        for tgt in ("orig", "trans")
        for phase in ("pre", "control", "post")
        for st in ("correct", "incorrect", "unknown")
    ),
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _last_boxed_only_string(string: str) -> str | None:
    if not string:
        return None
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx
    depth = 0
    right_idx: int | None = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right_idx = i
                break
        i += 1
    return string[idx : right_idx + 1] if right_idx is not None else None


def _remove_boxed(s: str) -> str:
    left = r"\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return ""


class _Timeout:
    def __init__(self, seconds: int = 5):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, *a):
        signal.alarm(0)

    @staticmethod
    def _handle(_signum, _frame):
        raise TimeoutError("math_verify timeout")


def grade_math_answer(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Exact replica of upstream e8c89b7 eduardo/utils.py:grade_math_answer."""
    if not solution_str:
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 1, "truncated": 0}
    boxed = _last_boxed_only_string(solution_str[-200:]) or _last_boxed_only_string(solution_str)
    pred = _remove_boxed(boxed) if boxed else ""
    if pred == "":
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 1, "truncated": 0}
    correct = pred.strip() == ground_truth.strip()
    if not correct:
        try:
            with _Timeout(seconds=5):
                from math_verify import parse as mv_parse, verify as mv_verify

                gold = mv_parse(ground_truth)
                guess = mv_parse(pred)
                correct = bool(mv_verify(gold, guess))
        except Exception:
            pass
    reward = 1.0 if correct else 0.0
    return {"score": reward, "acc": reward, "pred": pred, "incorrect_format": 0, "truncated": 0}


def _score_input_author_rule_worker(task: tuple[str, str, str, str]) -> tuple[str, str, dict[str, int]]:
    """Grade one 27B source SCORE_INPUT.json (96 rows) offline via grade_math_answer."""
    difficulty, source_id, score_input_path_s, expected_target = task
    rows = json.loads(Path(score_input_path_s).read_bytes())
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError(f"Expected 96 rows in {difficulty} SCORE_INPUT.json")
    counts: Counter = Counter()
    for row in rows:
        tgt = "orig" if row["target"] == "source" else "trans"
        if row["target"] != "source" and row["target"] != expected_target:
            raise ValueError("Unexpected target in SCORE_INPUT.json")
        phase = row["phase"]
        verdict = grade_math_answer(row["student_response"], row["reference_answer"])
        st = "correct" if float(verdict["acc"]) >= 0.5 else "incorrect"
        counts[f"ar_{tgt}_{phase}_{st}"] += 1
    out: dict[str, int] = {}
    for tgt in ("orig", "trans"):
        for phase in ("pre", "control", "post"):
            c = int(counts.get(f"ar_{tgt}_{phase}_correct", 0))
            inc = int(counts.get(f"ar_{tgt}_{phase}_incorrect", 0))
            if c + inc != 16:
                raise ValueError("Author-rule count sum != 16")
            out[f"ar_{tgt}_{phase}_correct"] = c
            out[f"ar_{tgt}_{phase}_incorrect"] = inc
            out[f"ar_{tgt}_{phase}_unknown"] = 0
    return difficulty, source_id, out


def recompute_27b_r1_checks(
    count_rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    draws: int = 10000,
) -> dict[str, Any]:
    """Recompute 27B orig combined + 3-tier R1 directly from exported count rows."""
    out: dict[str, Any] = {}
    for grader in ("agnes", "ar"):
        # Orig combined: average gain per source_id across tiers present, then bootstrap.
        orig_by_sid: dict[str, list[float]] = defaultdict(list)
        tier_orig: dict[str, list[float]] = defaultdict(list)
        tier_trans: dict[str, list[float]] = defaultdict(list)
        for rec in count_rows:
            diff = rec["difficulty"]
            for tgt, dest_tier in (("orig", tier_orig[diff]), ("trans", tier_trans[diff])):
                c_pre = {
                    st: int(rec[f"{grader}_{tgt}_pre_{st}"])
                    for st in ("correct", "incorrect", "unknown")
                }
                c_post = {
                    st: int(rec[f"{grader}_{tgt}_post_{st}"])
                    for st in ("correct", "incorrect", "unknown")
                }
                p_pre = p_value(c_pre, unknown_as_wrong=True)
                p_post = p_value(c_post, unknown_as_wrong=True)
                if p_pre is not None and p_post is not None:
                    g = normalized_gain(p_pre, p_post)
                    dest_tier.append(g)
                    if tgt == "orig":
                        orig_by_sid[rec["source_id"]].append(g)

        orig_vals = [sum(v) / len(v) for _, v in sorted(orig_by_sid.items())]
        out[f"{grader}_orig_combined"] = bootstrap_mean_ci(orig_vals, seed=seed, draws=draws)
        for diff in DIFFICULTY_ORDER:
            out[f"{grader}_{diff}_orig"] = bootstrap_mean_ci(tier_orig[diff], seed=seed, draws=draws)
            out[f"{grader}_{diff}_trans"] = bootstrap_mean_ci(tier_trans[diff], seed=seed, draws=draws)
    return out


def export_ref27b(
    *,
    repo_root: Path,
    data_path: Path,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    out_dir: Path,
    workers: int = 8,
    require_exact_totals: bool = True,
    verify_27b_r1: bool = True,
    grade_fn=None,
) -> dict[str, Any]:
    """Export 1,489 inputs + Agnes/Author-Rule 27B counts + verified manifest & tar.gz."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Export directory already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        data_path,
        ref_root,
        overlay_root,
        decisions_root,
        repo_root=repo_root,
        require_exact_totals=require_exact_totals,
        show_progress=True,
    )
    entries = manifest["eligible"]

    # Grade 27B SCORE_INPUT.json offline via Author-Rule (\boxed{} + math_verify).
    tasks = [
        (
            e["difficulty"],
            e["source_id"],
            str(ref_root / e["ref_stage"] / "sources" / e["source_key"] / "SCORE_INPUT.json"),
            e["row"]["transfer_target"],
        )
        for e in entries
    ]
    ar_by_key: dict[tuple[str, str], dict[str, int]] = {}
    if grade_fn is not None:
        for t in tasks:
            diff, sid, s_path, exp_tgt = t
            rows = json.loads(Path(s_path).read_bytes())
            counts: Counter = Counter()
            for r in rows:
                tgt = "orig" if r["target"] == "source" else "trans"
                v = grade_fn(r["student_response"], r["reference_answer"])
                st = "correct" if float(v.get("acc", 0.0)) >= 0.5 else "incorrect"
                counts[f"ar_{tgt}_{r['phase']}_{st}"] += 1
            ar_by_key[(diff, sid)] = {
                f"ar_{tgt}_{phase}_{st}": int(counts.get(f"ar_{tgt}_{phase}_{st}", 0))
                for tgt in ("orig", "trans")
                for phase in ("pre", "control", "post")
                for st in ("correct", "incorrect", "unknown")
            }
    else:
        ctx = multiprocessing.get_context("spawn")
        max_w = max(1, min(workers, len(tasks))) if tasks else 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as pool:
            for done, res in enumerate(pool.map(_score_input_author_rule_worker, tasks, chunksize=8), 1):
                diff, sid, ar_counts = res
                ar_by_key[(diff, sid)] = ar_counts
                if done % 200 == 0 or done == len(tasks):
                    print(f"[export_27b] ar_offline_progress={done}/{len(tasks)}", flush=True)

    inputs_path = out_dir / "selfbuilt_eval_inputs_1489.jsonl"
    counts_path = out_dir / "ref27b_counts.csv"

    count_rows: list[dict[str, Any]] = []
    with inputs_path.open("w", encoding="utf-8") as jf:
        for e in entries:
            input_rec = {
                "difficulty": e["difficulty"],
                "tier_zh": e["tier_zh"],
                "ref_stage": e["ref_stage"],
                "source_key": e["source_key"],
                "source_id": e["source_id"],
                "row_index": e["row_index"],
                "row": e["row"],
            }
            jf.write(json.dumps(input_rec, ensure_ascii=False) + "\n")

            row_out: dict[str, Any] = {
                "difficulty": e["difficulty"],
                "tier_zh": e["tier_zh"],
                "ref_stage": e["ref_stage"],
                "source_key": e["source_key"],
                "source_id": e["source_id"],
                "row_index": e["row_index"],
            }
            rc = e["ref_counts"]
            for tgt_csv, tgt_zh in (("orig", "原题"), ("trans", "迁移题")):
                for phase in ("pre", "control", "post"):
                    for st in ("correct", "incorrect", "unknown"):
                        row_out[f"agnes_{tgt_csv}_{phase}_{st}"] = int(rc[tgt_zh][phase][st])
            row_out.update(ar_by_key[(e["difficulty"], e["source_id"])])
            count_rows.append(row_out)

    with counts_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(COUNT_COLUMNS))
        writer.writeheader()
        writer.writerows(count_rows)

    checks = recompute_27b_r1_checks(count_rows, seed=42, draws=10000)
    if verify_27b_r1:
        oc = checks["agnes_orig_combined"]
        eo = checks["agnes_easy_trans"]
        mo = checks["agnes_medium_trans"]
        ho = checks["agnes_hard_trans"]
        if oc is None or abs(oc["mean"] - 0.5284) > 0.003:
            raise ValueError(f"27B orig_combined R1 check failed: got {oc}")
        if (
            abs(eo["mean"] - 0.105) > 0.005
            or abs(mo["mean"] - 0.152) > 0.005
            or abs(ho["mean"] - 0.117) > 0.005
        ):
            raise ValueError(f"27B 3-tier transfer R1 check failed: easy={eo} med={mo} hard={ho}")

    try:
        mv_ver = importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        mv_ver = "NOT_INSTALLED"

    manifest_doc = {
        "schema": "t021_a100_export_v1",
        "data_sha256": manifest["data_sha256"],
        "eligible_count": manifest["eligible_count"],
        "skipped_count": manifest["skipped_count"],
        "per_tier_counts": manifest["per_tier_counts"],
        "math_verify_version": mv_ver,
        "python_version": sys.version.split()[0],
        "files_sha256": {
            inputs_path.name: _sha256_file(inputs_path),
            counts_path.name: _sha256_file(counts_path),
        },
        "recomputed_27b_checks": checks,
    }
    man_path = out_dir / "manifest_export.json"
    man_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle_path = out_dir / "t021_a100_export_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        for p in (inputs_path, counts_path, man_path):
            tf.add(p, arcname=p.name)
    bundle_sha = _sha256_file(bundle_path)
    (out_dir / "t021_a100_export_bundle.tar.gz.sha256").write_text(
        f"{bundle_sha}  {bundle_path.name}\n", encoding="utf-8"
    )
    manifest_doc["bundle_sha256"] = bundle_sha
    manifest_doc["bundle_bytes"] = bundle_path.stat().st_size
    return manifest_doc


def export_t019_base(
    *,
    repo_root: Path,
    data_path: Path,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    base4b_dir: Path,
    out_dir: Path,
    require_exact_totals: bool = True,
) -> dict[str, Any]:
    """Export T-019 base4b per-source post counts (Agnes & Author-Rule, no raw text) for Stage 3."""
    if not base4b_dir.is_dir():
        raise ValueError(f"base4b_dir does not exist: {base4b_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        data_path,
        ref_root,
        overlay_root,
        decisions_root,
        repo_root=repo_root,
        require_exact_totals=require_exact_totals,
        show_progress=True,
    )
    rows_out: list[dict[str, Any]] = []
    for e in manifest["eligible"]:
        diff = e["difficulty"]
        stage = STAGE_BY_DIFF[diff]
        key = e["source_key"]
        sid = e["source_id"]
        exp_tgt = e["row"]["transfer_target"]
        src_dir = base4b_dir / stage / "sources" / key
        if not (src_dir / "RAW_COMPLETE.json").is_file() or (src_dir / "SOURCE_HELD.json").exists():
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
                by_t[tgt][item["status"]] += 1
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
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description="Export sanitized A100 self-built inputs/counts for T-021.")
    ap.add_argument("--mode", choices=("ref27b", "t019-base"), default="ref27b")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl"))
    ap.add_argument("--ref-root", type=Path, default=Path("/workspace/results/own-test-499-v1"))
    ap.add_argument("--overlay", type=Path, default=Path("/workspace/results/agnes-topbool-recovery-v1"))
    ap.add_argument("--decisions", type=Path, default=Path("/workspace/results/agnes-rejudge-v1"))
    ap.add_argument("--base4b-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.mode == "ref27b":
        man = export_ref27b(
            repo_root=args.repo,
            data_path=args.data,
            ref_root=args.ref_root,
            overlay_root=args.overlay,
            decisions_root=args.decisions,
            out_dir=args.out_dir,
            workers=args.workers,
            require_exact_totals=True,
            verify_27b_r1=True,
        )
        oc = man["recomputed_27b_checks"]["agnes_orig_combined"]
        eo = man["recomputed_27b_checks"]["agnes_easy_trans"]
        mo = man["recomputed_27b_checks"]["agnes_medium_trans"]
        ho = man["recomputed_27b_checks"]["agnes_hard_trans"]
        ar_oc = man["recomputed_27b_checks"]["ar_orig_combined"]
        print(
            f"[export_27b] OK eligible={man['eligible_count']} skipped={man['skipped_count']} "
            f"math_verify={man['math_verify_version']}",
            flush=True,
        )
        print(
            f"[export_27b] 27B_AGNES_R1 orig_combined={oc['mean']:+.4f} [{oc['ci'][0]:+.4f},{oc['ci'][1]:+.4f}] "
            f"easy={eo['mean']:+.4f} medium={mo['mean']:+.4f} hard={ho['mean']:+.4f}",
            flush=True,
        )
        print(
            f"[export_27b] 27B_AR_R1 orig_combined={ar_oc['mean']:+.4f} [{ar_oc['ci'][0]:+.4f},{ar_oc['ci'][1]:+.4f}]",
            flush=True,
        )
        print(f"[export_27b] bundle_sha256={man['bundle_sha256']} bytes={man['bundle_bytes']}", flush=True)
    else:
        if args.base4b_dir is None:
            raise ValueError("--base4b-dir is required for --mode t019-base")
        man = export_t019_base(
            repo_root=args.repo,
            data_path=args.data,
            ref_root=args.ref_root,
            overlay_root=args.overlay,
            decisions_root=args.decisions,
            base4b_dir=args.base4b_dir,
            out_dir=args.out_dir,
            require_exact_totals=True,
        )
        print(
            f"[export_t019] OK exported={man['exported_sources']}/{man['eligible_total']} "
            f"bundle_sha256={man['bundle_sha256']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
