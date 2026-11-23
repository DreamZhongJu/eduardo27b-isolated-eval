#!/usr/bin/env python3
"""Run 1-round untutored student control (ctl) on Pro 5000 using author's _solve_rate.

Calls ``eduardo.create_transfer_dataset._solve_rate`` directly (temperature=0.6,
k=16, max_tokens=1024, grade_math_answer) on both original and transfer problems.
Never prints problem text, answers, problem IDs, or student responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _ensure_author_repo_on_path(repo: Path) -> None:
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_control_rows(parquet_path: Path, num_problems: int = -1) -> list[dict[str, Any]]:
    """Load test.parquet rows matching eval/run_env_test_eval.py indexing."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if num_problems > 0:
        df = df.iloc[:num_problems]
    rows: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        extra = row["extra_info"]
        if not isinstance(extra, dict):
            raise ValueError(f"Row missing dict extra_info at position {idx}")
        problem = str(extra.get("problem") or "")
        answer = str(extra.get("answer") or "")
        transfer_problem = str(extra.get("transfer_problem") or "")
        transfer_answer = str(extra.get("transfer_answer") or "")
        if not (problem and answer and transfer_problem and transfer_answer):
            raise ValueError(f"Incomplete problem/transfer fields at position {idx}")
        pre_same = float(extra["pre_solve_rate"])
        pre_transfer = float(extra["pre_solve_rate_transfer_problem"])
        rows.append({
            "row_index": idx,
            "problem": problem,
            "answer": answer,
            "transfer_problem": transfer_problem,
            "transfer_answer": transfer_answer,
            "pre_solve_rate": pre_same,
            "pre_solve_rate_transfer": pre_transfer,
        })
    if not rows:
        raise ValueError("No rows loaded from test.parquet")
    return rows


async def run_control_round(
    rows: list[dict[str, Any]],
    *,
    client: Any,
    solve_rate_fn: Any,
    student_model: str,
    student_temperature: float = 0.6,
    student_max_tokens: int = 1024,
    k: int = 16,
    student_concurrency: int = 64,
    row_concurrency: int = 16,
    progress_cb: Any = None,
) -> list[dict[str, Any]]:
    """Execute 1 untutored control round (ctl) via author's _solve_rate."""
    solve_args = argparse.Namespace(
        student_model=student_model,
        student_temperature=student_temperature,
        student_max_tokens=student_max_tokens,
        k=k,
    )
    student_sem = asyncio.Semaphore(student_concurrency)
    row_sem = asyncio.Semaphore(row_concurrency)
    per_row: dict[int, dict[str, Any]] = {}
    done_count = 0
    lock = asyncio.Lock()
    total = len(rows)

    async def do_one(r: dict[str, Any]) -> None:
        nonlocal done_count
        async with row_sem:
            orig_rate, orig_scored, orig_failed = await solve_rate_fn(
                r["problem"], r["answer"], client, solve_args, student_sem
            )
            trans_rate, trans_scored, trans_failed = await solve_rate_fn(
                r["transfer_problem"], r["transfer_answer"], client, solve_args, student_sem
            )
        per_row[r["row_index"]] = {
            "row_index": r["row_index"],
            "pre_solve_rate": r["pre_solve_rate"],
            "pre_solve_rate_transfer": r["pre_solve_rate_transfer"],
            "ctl_same": float(orig_rate),
            "ctl_same_scored": int(orig_scored),
            "ctl_same_failed": int(orig_failed),
            "ctl_transfer": float(trans_rate),
            "ctl_transfer_scored": int(trans_scored),
            "ctl_transfer_failed": int(trans_failed),
        }
        async with lock:
            done_count += 1
            if progress_cb and (done_count % 50 == 0 or done_count == total):
                progress_cb("ctl", done_count, total)

    await asyncio.gather(*(do_one(r) for r in rows))
    return [per_row[r["row_index"]] for r in rows]


def build_control_doc(
    records: list[dict[str, Any]],
    *,
    tag: str,
    student_model: str,
    k: int,
    student_temperature: float,
    student_max_tokens: int,
    elapsed_seconds: float,
    timestamp: str,
) -> dict[str, Any]:
    """Assemble sanitized Pro 5000 control JSON document and failure rate diagnostics."""
    n_problems = len(records)
    total_expected = n_problems * 2 * k
    total_scored = sum(
        int(r[f"ctl_{kind}_scored"]) for r in records for kind in ("same", "transfer")
    )
    total_failed = sum(
        int(r[f"ctl_{kind}_failed"]) for r in records for kind in ("same", "transfer")
    )
    if total_scored + total_failed != total_expected:
        raise ValueError(
            f"Call count mismatch: scored={total_scored} + failed={total_failed} != expected={total_expected}"
        )
    fail_rate = (total_failed / total_expected) if total_expected else 0.0
    exceeds_1pct = fail_rate > 0.01 + 1e-12

    def mean_key(key: str) -> float:
        return sum(float(r[key]) for r in records) / n_problems if n_problems else 0.0

    return {
        "benchmark": "transfer_untutored_control_pro5000",
        "tag": tag,
        "timestamp": timestamp,
        "student_model": student_model,
        "k": k,
        "student_temperature": student_temperature,
        "student_max_tokens": student_max_tokens,
        "num_problems": n_problems,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "metrics": {
            "ctl_same": mean_key("ctl_same"),
            "ctl_transfer": mean_key("ctl_transfer"),
            "pre_solve_rate": mean_key("pre_solve_rate"),
            "pre_solve_rate_transfer": mean_key("pre_solve_rate_transfer"),
            "calls_expected": total_expected,
            "calls_scored": total_scored,
            "calls_failed": total_failed,
            "fail_rate": fail_rate,
            "exceeds_1pct": exceeds_1pct,
        },
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 1-round untutored control (ctl) via author _solve_rate.")
    ap.add_argument("--repo", type=Path, required=True, help="Path to unmodified upstream EduardoRL repo")
    ap.add_argument("--test-parquet", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tag", default="pro5000_author494_control")
    ap.add_argument("--num-problems", type=int, default=-1)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--student-api-base", required=True)
    ap.add_argument("--api-key-env", default="SERVING_API_KEY")
    ap.add_argument("--student-temperature", type=float, default=0.6)
    ap.add_argument("--student-max-tokens", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--student-concurrency", type=int, default=64)
    ap.add_argument("--row-concurrency", type=int, default=16)
    args = ap.parse_args()

    _ensure_author_repo_on_path(args.repo)
    from eduardo.create_transfer_dataset import _solve_rate  # noqa: E402
    from eduardo.utils import get_async_openai_client  # noqa: E402

    rows = load_control_rows(args.test_parquet, num_problems=args.num_problems)
    print(
        f"[control] loaded_rows={len(rows)} round=ctl k={args.k} expected_calls={len(rows) * 2 * args.k}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = args.output_dir / "control_progress.json"

    def on_progress(round_name: str, done: int, total: int) -> None:
        print(f"[control] round={round_name} done={done}/{total}", flush=True)
        tmp = progress_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"round": round_name, "done": done, "total": total, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(progress_file)

    async def _driver() -> list[dict[str, Any]]:
        client = get_async_openai_client(args.student_api_base, args.api_key_env)
        try:
            return await run_control_round(
                rows,
                client=client,
                solve_rate_fn=_solve_rate,
                student_model=args.student_model,
                student_temperature=args.student_temperature,
                student_max_tokens=args.student_max_tokens,
                k=args.k,
                student_concurrency=args.student_concurrency,
                row_concurrency=args.row_concurrency,
                progress_cb=on_progress,
            )
        finally:
            await client.close()

    t0 = time.monotonic()
    records = asyncio.run(_driver())
    elapsed = time.monotonic() - t0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = build_control_doc(
        records,
        tag=args.tag,
        student_model=args.student_model,
        k=args.k,
        student_temperature=args.student_temperature,
        student_max_tokens=args.student_max_tokens,
        elapsed_seconds=elapsed,
        timestamp=ts,
    )
    out_path = args.output_dir / f"control_eval_{args.tag}_{ts}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    m = doc["metrics"]
    print(
        f"[control] finished elapsed={elapsed:.1f}s problems={doc['num_problems']} "
        f"calls_expected={m['calls_expected']} calls_scored={m['calls_scored']} "
        f"calls_failed={m['calls_failed']} fail_rate={m['fail_rate']:.4%}",
        flush=True,
    )
    print(
        f"[control] means: pre_orig={m['pre_solve_rate']:.4f} ctl_orig={m['ctl_same']:.4f} | "
        f"pre_trans={m['pre_solve_rate_transfer']:.4f} ctl_trans={m['ctl_transfer']:.4f}",
        flush=True,
    )
    print(f"Control results written to: {out_path}", flush=True)
    if m["exceeds_1pct"]:
        print("ERROR: control student call fail rate exceeded 1%; stopping.", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
