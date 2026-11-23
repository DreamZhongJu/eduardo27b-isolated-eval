#!/usr/bin/env python3
"""Pro 5000 post-only dialogue + student sampling + Agnes (<=3 workers, after T-020) + Author-Rule grading."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = HERE.parents[1]
BASE4B_DIR = REPO_ROOT_DEFAULT / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(BASE4B_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_27b_selfbuilt import grade_math_answer  # noqa: E402
from import_bundle import load_and_verify_bundle  # noqa: E402
from ref27b_manifest import select_pilot_entries  # noqa: E402
from run_post_eval import (  # noqa: E402
    generate_entries,
    grade_agnes_entries,
    grade_author_rule_entries,
)
from summarize_base4b import extract_safe_failure_summary  # noqa: E402


def _pid_alive(pid: int, proc_dir: Path = Path("/proc")) -> bool:
    if pid <= 0:
        return False
    return (proc_dir / str(pid)).is_dir()


def check_t020_finished(
    *,
    t020_base: Path,
    t020_dir: Path | None = None,
    workers: int = 3,
    proc_dir: Path = Path("/proc"),
) -> dict[str, Any]:
    """Fail closed unless Pro 5000 Agnes workers <= 3 AND T-020 full 494-problem run is finished."""
    if not 1 <= workers <= 3:
        raise ValueError("Pro 5000 Agnes concurrency cannot exceed 3 (1-3 required)")

    if t020_dir is None:
        runs_dir = t020_base / "runs"
        if not runs_dir.is_dir():
            raise RuntimeError("T-020 runs directory missing; cannot verify T-020 completion")
        candidates = sorted(
            p for p in runs_dir.glob("eval_author494_full*") if p.is_dir()
        )
        if not candidates:
            raise RuntimeError("No T-020 full run directory (eval_author494_full*) found")
        t020_dir = candidates[-1]

    if not t020_dir.is_dir():
        raise RuntimeError(f"T-020 run directory does not exist: {t020_dir}")

    # Check runner/student PID files in T-020 directory.
    for pid_name in ("runner.pid", "student.pid"):
        pid_file = t020_dir / pid_name
        if pid_file.is_file():
            raw = pid_file.read_text(encoding="utf-8", errors="replace").strip()
            if raw.isdigit() and _pid_alive(int(raw), proc_dir=proc_dir):
                raise RuntimeError(f"T-020 process ({pid_name}={raw}) is still running")

    # Scan /proc cmdlines for active T-020 or Agnes callers on Pro 5000.
    if proc_dir.is_dir():
        for p in proc_dir.iterdir():
            if not p.name.isdigit():
                continue
            if int(p.name) == os.getpid():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if any(
                tok in cmd
                for tok in (
                    "run_author494_pro5000.sh",
                    "eval.run_env_test_eval",
                    "agnes_ping_author494.py",
                )
            ):
                raise RuntimeError("Active T-020/Agnes process detected on host; cannot call Agnes")

    # Check full 494-problem result JSON exists and has 494 records.
    json_files = sorted(t020_dir.glob("transfer_eval_*.json"))
    if not json_files:
        raise RuntimeError("T-020 full transfer_eval_*.json not written yet")
    doc = json.loads(json_files[-1].read_bytes())
    records = doc.get("records")
    metrics = doc.get("metrics")
    if not isinstance(records, list) or len(records) != 494 or not isinstance(metrics, dict):
        raise RuntimeError("T-020 full result JSON incomplete (expected 494 records + metrics)")

    return {
        "t020_dir": str(t020_dir),
        "t020_json": json_files[-1].name,
        "records": len(records),
        "workers": workers,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Pro 5000 A-group step_100 post-only self-built 3-tier eval.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--teacher-served-name", default="A-group-step100")
    ap.add_argument("--teacher-url", default="http://127.0.0.1:18701/v1")
    ap.add_argument("--student-url", default="http://127.0.0.1:18702/v1")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".secrets" / "agnes_api_key")
    ap.add_argument("--workers", type=int, default=3, help="Agnes workers (Pro 5000 max 3)")
    ap.add_argument("--concurrency", type=int, default=8, help="Concurrent generation sources")
    ap.add_argument("--gate-first-n", type=int, default=0)
    ap.add_argument("--max-est-days", type=float, default=3.0)
    ap.add_argument("--pilot-per-tier", type=int, default=3, help="0 means all 1,489 eligible sources")
    ap.add_argument("--t020-base", type=Path, default=Path.home() / "bzhang" / "a_group_pro5000")
    ap.add_argument("--t020-dir", type=Path, default=None)
    ap.add_argument("--check-bundle", action="store_true")
    ap.add_argument("--check-t020", action="store_true")
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--run-grading", action="store_true")
    ap.add_argument("--run-author-rule", action="store_true")
    args = ap.parse_args()

    loaded = load_and_verify_bundle(args.bundle_dir, require_exact_totals=True)
    print(
        f"[base4b] bundle_ok eligible={loaded['eligible_count']} skipped={loaded['skipped_count']} "
        f"math_verify={loaded['math_verify_version']} tiers={loaded['per_tier_counts']}",
        flush=True,
    )
    if args.check_bundle:
        return 0

    if args.check_t020:
        info = check_t020_finished(
            t020_base=args.t020_base, t020_dir=args.t020_dir, workers=args.workers
        )
        print(f"[base4b] t020_check_ok records={info['records']} workers={info['workers']}", flush=True)
        return 0

    if args.out_root is None:
        raise ValueError("--out-root is required")
    args.out_root.mkdir(parents=True, exist_ok=True)
    entries = select_pilot_entries(loaded, per_tier=args.pilot_per_tier)
    print(f"[base4b] selected_entries={len(entries)} pilot_per_tier={args.pilot_per_tier}", flush=True)

    if args.run_generation:
        if not args.teacher_model:
            raise ValueError("--teacher-model is required for generation")
        gen_stats = asyncio.run(
            generate_entries(
                entries,
                out_root=args.out_root,
                repo_root=args.repo,
                teacher_model_name=args.teacher_served_name,
                teacher_path=args.teacher_model,
                teacher_url=args.teacher_url,
                student_url=args.student_url,
                concurrency=args.concurrency,
                gate_first_n=args.gate_first_n,
                max_est_days=args.max_est_days,
            )
        )
        print(f"[base4b] generation_done {json.dumps(gen_stats, sort_keys=True)}", flush=True)

    if args.run_grading:
        check_t020_finished(
            t020_base=args.t020_base, t020_dir=args.t020_dir, workers=args.workers
        )
        grade_stats = grade_agnes_entries(
            entries,
            out_root=args.out_root,
            repo_root=args.repo,
            key_file=args.key_file,
            workers=args.workers,
        )
        print(f"[base4b] agnes_grading_done {json.dumps(grade_stats, sort_keys=True)}", flush=True)

    if args.run_author_rule:
        ar_stats = grade_author_rule_entries(
            entries,
            out_root=args.out_root,
            repo_root=args.repo,
            grade_math_fn=grade_math_answer,
        )
        print(f"[base4b] author_rule_done {json.dumps(ar_stats, sort_keys=True)}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
