#!/usr/bin/env python3
"""Pro 5000 post-only dialogue + student sampling + Agnes (AIMD 1..16 after T-020) + Author-Rule grading."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
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

from agnes_window import AgnesWindowCaller, SyncCongestionWindow  # noqa: E402
from export_27b_selfbuilt import grade_math_answer  # noqa: E402
from import_bundle import load_and_verify_bundle  # noqa: E402
from ref27b_manifest import select_pilot_entries  # noqa: E402
from run_post_eval import (  # noqa: E402
    _sha256_bytes,
    generate_entries,
    grade_author_rule_entries,
    load_verified_agnes_judge,
)
from summarize_base4b import (  # noqa: E402
    STAGE_BY_DIFF,
    _classify_agnes_attempt,
    extract_safe_failure_summary,
)


class AgnesTechnicalFailureLimitError(RuntimeError):
    """Raised when Agnes technical failure rate exceeds 1% of calls."""


def _pid_alive(pid: int, proc_dir: Path = Path("/proc")) -> bool:
    if pid <= 0:
        return False
    return (proc_dir / str(pid)).is_dir()


def check_t020_finished(
    *,
    t020_base: Path,
    t020_dir: Path | None = None,
    workers: int = 16,
    proc_dir: Path = Path("/proc"),
) -> dict[str, Any]:
    """Fail closed unless Pro 5000 Agnes workers in 1..16 AND T-020 full 494-problem run is finished."""
    if not 1 <= workers <= 16:
        raise ValueError("Pro 5000 Agnes concurrency must be in 1..16")

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


def _count_source_unknowns(
    *,
    out_root: Path,
    stage: str,
    skey: str,
    results: list[dict[str, Any]],
    judge_mod: Any,
) -> tuple[int, int]:
    """Return (semantic_unknown, technical_failure) across 32 sample verdicts of one source."""
    semantic_unknown = 0
    technical_failure = 0
    for item in results:
        if item.get("status") != "unknown":
            continue
        ident = item.get("identity") or item
        req_id = judge_mod.digest({"identity": ident})
        rec_dir = out_root / "AGNES" / "receipts" / stage / skey / req_id
        attempts = sorted(rec_dir.glob("ATTEMPT-????-RESPONSE.json"))
        resp = json.loads(attempts[-1].read_bytes()) if attempts else None
        cls = _classify_agnes_attempt(resp)
        if cls == "semantic_unknown":
            semantic_unknown += 1
        else:
            technical_failure += 1
    return semantic_unknown, technical_failure


def grade_agnes_pro5000(
    entries: list[dict[str, Any]],
    *,
    out_root: Path,
    repo_root: Path,
    key_file: Path,
    workers: int = 16,
    caller=None,
    window: SyncCongestionWindow | None = None,
    sleep_fn=None,
    max_tech_fail_rate: float = 0.01,
) -> dict[str, Any]:
    """Grade completed sources with unmodified agnes_judge.py under AIMD 1..16 congestion window."""
    if not 1 <= workers <= 16:
        raise ValueError("Agnes workers must be between 1 and 16")
    if key_file.is_symlink() or not key_file.is_file():
        raise ValueError("Invalid Agnes key file")
    key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("Empty Agnes key file")

    judge = load_verified_agnes_judge(repo_root / "evaluation" / "agnes_eval_v3" / "agnes_judge.py")
    locks = out_root / "AGNES" / "locks"
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    for wid in range(workers):
        (locks / f"w_{wid:02d}").mkdir(mode=0o700, parents=True, exist_ok=True)

    win = window if window is not None else SyncCongestionWindow(initial=1, maximum=workers)
    window_caller = AgnesWindowCaller(judge_mod=judge, inner_caller=caller, window=win)
    grade_sleep = sleep_fn if sleep_fn is not None else (lambda _s: None)

    gradable_entries: list[tuple[dict[str, Any], str, str, Path, Path]] = []
    for entry in entries:
        stage = STAGE_BY_DIFF[entry["difficulty"]]
        skey = entry["source_key"]
        src_dir = out_root / stage / "sources" / skey
        marker_path = src_dir / "RAW_COMPLETE.json"
        input_path = src_dir / "SCORE_INPUT.json"
        if marker_path.is_file() and input_path.is_file():
            gradable_entries.append((entry, stage, skey, marker_path, input_path))

    expected_total_calls = max(1, len(gradable_entries) * 32)
    stats: dict[str, Any] = {
        "graded_sources": 0,
        "skipped_existing": 0,
        "total_calls": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
        "semantic_unknown": 0,
        "technical_failures": 0,
        "technical_failure_rate": 0.0,
        "max_tech_fail_rate": max_tech_fail_rate,
        "aborted_over_1pct": False,
        "primary_status": "可用",
    }

    report_path = out_root / "AGNES" / "AGNES_GRADING_REPORT.json"

    def _write_report() -> dict[str, Any]:
        tot = int(stats["total_calls"])
        stats["technical_failure_rate"] = round((stats["technical_failures"] / tot), 6) if tot > 0 else 0.0
        stats["window"] = win.stats()
        report_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return stats

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, (entry, stage, skey, _marker_path, input_path) in enumerate(gradable_entries, 1):
            raw = input_path.read_bytes()
            rows = json.loads(raw)
            if not isinstance(rows, list) or len(rows) != 32:
                raise ValueError("Expected 32 rows in post-only SCORE_INPUT.json")
            input_hash = _sha256_bytes(raw)

            out_dir = out_root / "AGNES" / stage / "sources" / skey
            out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            out_path = out_dir / "RESULTS.json"
            if out_path.exists():
                prev = judge.read_json(out_path)
                if prev.get("input_sha256") != input_hash or prev.get("source_id") != entry["source_id"]:
                    raise ValueError("Existing RESULTS.json identity mismatch")
                results = prev["results"]
                stats["skipped_existing"] += 1
            else:
                rec_dir = out_root / "AGNES" / "receipts" / stage / skey
                futures = [
                    pool.submit(
                        judge.grade_one,
                        row,
                        key,
                        rec_dir,
                        locks / f"w_{r_idx % workers:02d}",
                        caller=window_caller,
                        sleep=grade_sleep,
                        window=None,
                    )
                    for r_idx, row in enumerate(rows)
                ]
                results = [fut.result() for fut in futures]
                doc = {
                    "source_id": entry["source_id"],
                    "input_sha256": input_hash,
                    "grader": "agnes_full_response_v2",
                    "completion_parser_version": judge.PARSER_VERSION,
                    "model": judge.MODEL,
                    "results": results,
                }
                judge.put_once(out_path, doc)
                stats["graded_sources"] += 1

            sem_u, tech_u = _count_source_unknowns(
                out_root=out_root, stage=stage, skey=skey, results=results, judge_mod=judge
            )
            stats["total_calls"] += len(results)
            stats["semantic_unknown"] += sem_u
            stats["technical_failures"] += tech_u
            for r in results:
                stats[r["status"]] += 1

            # Abort mid-run once technical failures already exceed 1% of expected total calls.
            if stats["technical_failures"] > max_tech_fail_rate * expected_total_calls:
                stats["aborted_over_1pct"] = True
                stats["primary_status"] = "不可用"
                _write_report()
                print(
                    f"[base4b] agnes_aborted_over_1pct tech_fail={stats['technical_failures']}/"
                    f"{stats['total_calls']} (expected={expected_total_calls}) primary_status=不可用",
                    flush=True,
                )
                raise AgnesTechnicalFailureLimitError("Agnes technical failure rate exceeded 1%")

            if idx % 10 == 0 or idx == len(gradable_entries):
                ws = win.stats()
                print(
                    f"[base4b] grade_progress={idx}/{len(gradable_entries)} graded={stats['graded_sources']} "
                    f"skipped={stats['skipped_existing']} correct={stats['correct']} "
                    f"incorrect={stats['incorrect']} unknown={stats['unknown']} "
                    f"tech_fail={stats['technical_failures']} sem_null={stats['semantic_unknown']} "
                    f"win={ws['final_limit']} max_win={ws['max_limit_reached']} "
                    f"drops={ws['window_drops']} http429={ws['http_429_events']}",
                    flush=True,
                )

    tot = int(stats["total_calls"])
    if tot > 0 and (stats["technical_failures"] / tot) > max_tech_fail_rate:
        stats["aborted_over_1pct"] = True
        stats["primary_status"] = "不可用"
        _write_report()
        print(
            f"[base4b] agnes_aborted_over_1pct tech_fail={stats['technical_failures']}/{tot} "
            "primary_status=不可用",
            flush=True,
        )
        raise AgnesTechnicalFailureLimitError("Agnes technical failure rate exceeded 1%")

    return _write_report()


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
    ap.add_argument("--workers", type=int, default=16, help="Max Agnes congestion window (1-16)")
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
        grade_stats = grade_agnes_pro5000(
            entries,
            out_root=args.out_root,
            repo_root=args.repo,
            key_file=args.key_file,
            workers=args.workers,
        )
        print(f"[base4b] agnes_grading_done {json.dumps(grade_stats, sort_keys=True, ensure_ascii=False)}", flush=True)

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
