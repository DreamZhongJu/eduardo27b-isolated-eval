#!/usr/bin/env python3
"""Stage 1 Item 4: Student precision comparison (float16 vs bfloat16) on 90 self-built sources.

Deterministically selects 30 sources per tier (seed=42, 90 sources total) via
``select_stack_check_entries``, runs untutored 16 orig + 16 transfer = 32 student calls
per source under both ``--dtype float16`` and ``--dtype bfloat16`` (exact T-021 student
params and ``math-verify==0.8.0`` grading), and reports paired difference (bf16 - fp16),
10,000-draw source-level bootstrap 95% CI (seed=42), and Pearson r.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import secrets
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AGROUP_SB_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(AGROUP_SB_DIR), str(BASE4B_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DEFAULT_BASE,
    build_clean_env,
    ensure_base_cache_dirs,
    format_machine_summary,
    locate_eduardo_repo,
    pick_free_ports,
    resolve_local_model,
    terminate_procs,
    wait_for_free_gpus,
    wait_for_health,
    write_json_atomic,
)
from import_bundle import load_and_verify_bundle  # noqa: E402
from ref27b_manifest import DIFFICULTY_ORDER  # noqa: E402
from stack_check_selfbuilt import (  # noqa: E402
    evaluate_stack_check,
    pearson_r,
    run_pro5000_pre_samples,
    select_stack_check_entries,
)


def _percentile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def paired_precision_stats(
    fp16_vals: list[float],
    bf16_vals: list[float],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    if len(fp16_vals) != len(bf16_vals) or not fp16_vals:
        raise ValueError("Paired lists must be non-empty and equal length")
    n = len(fp16_vals)
    diffs = [b - f for f, b in zip(fp16_vals, bf16_vals)]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(draws))
    r = pearson_r(fp16_vals, bf16_vals)
    return {
        "n_sources": n,
        "fp16_mean": round(statistics.fmean(fp16_vals), 6),
        "bf16_mean": round(statistics.fmean(bf16_vals), 6),
        "diff_bf16_minus_fp16": round(statistics.fmean(diffs), 6),
        "ci95_low": round(_percentile(boot, 0.025), 6),
        "ci95_high": round(_percentile(boot, 0.975), 6),
        "pearson_r": round(r, 6) if r is not None else None,
    }


def analyze_precision_compare(
    sampled_entries: list[dict[str, Any]],
    fp16_rates: dict[tuple[str, str], dict[str, float]],
    bf16_rates: dict[tuple[str, str], dict[str, float]],
    *,
    fp16_stu_fail: int,
    bf16_stu_fail: int,
    draws: int = 10000,
    seed: int = 42,
    out_csv: Optional[Path] = None,
) -> dict[str, Any]:
    if fp16_stu_fail < 0 or bf16_stu_fail < 0:
        raise ValueError("stu_fail must be non-negative")

    rows_csv: list[dict[str, Any]] = []
    fp16_orig, bf16_orig = [], []
    fp16_trans, bf16_trans = [], []
    fp16_all, bf16_all = [], []
    by_tier: dict[str, dict[str, list[float]]] = {
        d: {"fp16_orig": [], "bf16_orig": [], "fp16_trans": [], "bf16_trans": [], "fp16_all": [], "bf16_all": []}
        for d in DIFFICULTY_ORDER
    }

    for entry in sampled_entries:
        diff = entry["difficulty"]
        sid = entry["source_id"]
        k = (diff, sid)
        if k not in fp16_rates or k not in bf16_rates:
            raise ValueError(f"Missing fp16/bf16 rates for {diff}/{entry['source_key']}")
        f_o = float(fp16_rates[k]["orig_pre_rate"])
        f_t = float(fp16_rates[k]["trans_pre_rate"])
        b_o = float(bf16_rates[k]["orig_pre_rate"])
        b_t = float(bf16_rates[k]["trans_pre_rate"])
        f_a = 0.5 * (f_o + f_t)
        b_a = 0.5 * (b_o + b_t)

        fp16_orig.append(f_o)
        bf16_orig.append(b_o)
        fp16_trans.append(f_t)
        bf16_trans.append(b_t)
        fp16_all.append(f_a)
        bf16_all.append(b_a)
        by_tier[diff]["fp16_orig"].append(f_o)
        by_tier[diff]["bf16_orig"].append(b_o)
        by_tier[diff]["fp16_trans"].append(f_t)
        by_tier[diff]["bf16_trans"].append(b_t)
        by_tier[diff]["fp16_all"].append(f_a)
        by_tier[diff]["bf16_all"].append(b_a)

        rows_csv.append(
            {
                "difficulty": diff,
                "source_key": entry["source_key"],
                "fp16_orig_rate": f"{f_o:.6f}",
                "bf16_orig_rate": f"{b_o:.6f}",
                "fp16_trans_rate": f"{f_t:.6f}",
                "bf16_trans_rate": f"{b_t:.6f}",
                "fp16_all_rate": f"{f_a:.6f}",
                "bf16_all_rate": f"{b_a:.6f}",
            }
        )

    if out_csv is not None:
        out_csv.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_csv)

    per_tier_summary: dict[str, Any] = {}
    for d in DIFFICULTY_ORDER:
        td = by_tier[d]
        per_tier_summary[d] = {
            "all_32": paired_precision_stats(td["fp16_all"], td["bf16_all"], draws=draws, seed=seed),
            "orig_16": paired_precision_stats(td["fp16_orig"], td["bf16_orig"], draws=draws, seed=seed),
            "trans_16": paired_precision_stats(td["fp16_trans"], td["bf16_trans"], draws=draws, seed=seed),
        }

    vs_a100_fp16 = evaluate_stack_check(sampled_entries, fp16_rates, stu_fail=fp16_stu_fail)
    vs_a100_bf16 = evaluate_stack_check(sampled_entries, bf16_rates, stu_fail=bf16_stu_fail)
    passed = fp16_stu_fail == 0 and bf16_stu_fail == 0

    return {
        "status": "PRECISION_COMPARE_DONE" if passed else "PRECISION_COMPARE_FAILED",
        "rc": 0 if passed else 1,
        "passed": passed,
        "n_sources": len(sampled_entries),
        "calls_per_arm": len(sampled_entries) * 32,
        "total_calls": len(sampled_entries) * 64,
        "fp16_stu_fail": fp16_stu_fail,
        "bf16_stu_fail": bf16_stu_fail,
        "failures": fp16_stu_fail + bf16_stu_fail,
        "overall_all_32": paired_precision_stats(fp16_all, bf16_all, draws=draws, seed=seed),
        "orig_16": paired_precision_stats(fp16_orig, bf16_orig, draws=draws, seed=seed),
        "trans_16": paired_precision_stats(fp16_trans, bf16_trans, draws=draws, seed=seed),
        "per_tier": per_tier_summary,
        "stack_check_vs_a100_fp16": vs_a100_fp16,
        "stack_check_vs_a100_bf16": vs_a100_bf16,
    }


def _launch_student_and_run_arm(
    *,
    dtype: str,
    sampled: list[dict[str, Any]],
    out_dir: Path,
    base: Path,
    repo_root: Path,
    eduardo_repo: Path,
    student_model: Path,
    gpu_id: int,
    port: int,
    concurrency: int,
) -> tuple[dict[tuple[str, str], dict[str, float]], int]:
    vllm_bin = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    token = "loopback-only"
    env_base = build_clean_env(base, token=token, eduardo_repo=eduardo_repo)
    cmd = [
        vllm_bin,
        "serve",
        str(student_model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "Llama-3.1-8B-Instruct",
        "--api-key",
        token,
        "--dtype",
        dtype,
        "--max-model-len",
        "20000",
        "--gpu-memory-utilization",
        "0.90",
        "--seed",
        "42",
    ]
    log_path = out_dir / f"vllm_student_{dtype}.log"
    procs: list[tuple[str, subprocess.Popen]] = []
    with log_path.open("ab") as log_f:
        proc = subprocess.Popen(
            cmd,
            env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu_id)),
            cwd=out_dir,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    procs.append((f"student_{dtype}", proc))
    try:
        wait_for_health(f"http://127.0.0.1:{port}/health", proc, f"student_{dtype}", timeout_s=600.0)
        print(f"[prec_compare] student ({dtype}) healthy on GPU {gpu_id} port {port}", flush=True)
        arm_dir = out_dir / dtype
        arm_dir.mkdir(parents=True, exist_ok=True)
        return asyncio.run(
            run_pro5000_pre_samples(
                sampled,
                out_dir=arm_dir,
                repo_root=repo_root,
                student_url=f"http://127.0.0.1:{port}/v1",
                concurrency=concurrency,
            )
        )
    finally:
        terminate_procs(procs)
        time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Student fp16 vs bf16 precision comparison on 90 self-built sources.")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--student-gpu", type=int, default=None)
    ap.add_argument("--auto-wait", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--student-port", type=int, default=18722)
    ap.add_argument("--per-tier", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--fp16-url", default=None, help="Optional pre-started fp16 student /v1 URL")
    ap.add_argument("--bf16-url", default=None, help="Optional pre-started bf16 student /v1 URL")
    args = ap.parse_args()

    base = args.base.resolve()
    repo_root = args.repo.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    ensure_base_cache_dirs(base)

    bundle_dir = args.bundle_dir or (base / "t021_bundle")
    loaded = load_and_verify_bundle(bundle_dir, require_exact_totals=True)
    sampled = select_stack_check_entries(loaded["eligible"], per_tier=args.per_tier, seed=args.seed)

    eduardo_repo = locate_eduardo_repo(base)
    student_model = resolve_local_model(base, "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct"))

    t0 = time.monotonic()
    if args.fp16_url and args.bf16_url:
        fp16_rates, fp16_fail = asyncio.run(
            run_pro5000_pre_samples(
                sampled, out_dir=out_dir / "fp16", repo_root=repo_root,
                student_url=args.fp16_url, concurrency=args.concurrency,
            )
        )
        bf16_rates, bf16_fail = asyncio.run(
            run_pro5000_pre_samples(
                sampled, out_dir=out_dir / "bf16", repo_root=repo_root,
                student_url=args.bf16_url, concurrency=args.concurrency,
            )
        )
    else:
        gpu_id = args.student_gpu
        if gpu_id is None:
            if not args.auto_wait:
                raise SystemExit("Provide --student-gpu <id> or --auto-wait")
            gpu_id = wait_for_free_gpus(1, poll_seconds=args.poll_seconds)[0]
        port = pick_free_ports([args.student_port])[0]
        fp16_rates, fp16_fail = _launch_student_and_run_arm(
            dtype="float16", sampled=sampled, out_dir=out_dir, base=base,
            repo_root=repo_root, eduardo_repo=eduardo_repo,
            student_model=student_model, gpu_id=gpu_id, port=port, concurrency=args.concurrency,
        )
        bf16_rates, bf16_fail = _launch_student_and_run_arm(
            dtype="bfloat16", sampled=sampled, out_dir=out_dir, base=base,
            repo_root=repo_root, eduardo_repo=eduardo_repo,
            student_model=student_model, gpu_id=gpu_id, port=port, concurrency=args.concurrency,
        )

    summary = analyze_precision_compare(
        sampled,
        fp16_rates,
        bf16_rates,
        fp16_stu_fail=fp16_fail,
        bf16_stu_fail=bf16_fail,
        draws=args.draws,
        seed=args.seed,
        out_csv=out_dir / "precision_compare_per_source.csv",
    )
    summary["elapsed_s"] = round(time.monotonic() - t0, 2)
    write_json_atomic(out_dir / "precision_compare_summary.json", summary)

    ov = summary["overall_all_32"]
    og = summary["orig_16"]
    tr = summary["trans_16"]
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        format_machine_summary(
            {
                "status": summary["status"],
                "rc": summary["rc"],
                "calls": summary["total_calls"],
                "failures": summary["failures"],
                "prec_diff_all": f"{ov['diff_bf16_minus_fp16']:+.6f}",
                "prec_ci_all": f"[{ov['ci95_low']:+.6f},{ov['ci95_high']:+.6f}]",
                "prec_r_all": ov["pearson_r"],
                "prec_diff_orig": f"{og['diff_bf16_minus_fp16']:+.6f}",
                "prec_r_orig": og["pearson_r"],
                "prec_diff_trans": f"{tr['diff_bf16_minus_fp16']:+.6f}",
                "prec_r_trans": tr["pearson_r"],
            }
        ),
        flush=True,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
