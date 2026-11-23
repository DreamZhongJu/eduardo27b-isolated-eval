#!/usr/bin/env python3
"""Cross-machine Student stack check (Pro 5000 existing vLLM env vs A100 27B pre-test).

Samples 30 sources per tier (seed=42 hash of source_id, 90 sources = 2,880 student calls),
runs 16-attempt pre-tests for original and transfer problems under exact 27B student params
(float16 preferred, bfloat16 fallback logged as deviation if unsupported),
grades offline with Author-Rule (\boxed{} + math_verify), and verifies:
  - |mean_pro - mean_a100| <= 0.03 (both original and transfer)
  - pearson_r(pro_pre, a100_pre) >= pearson_r(a100_pre, a100_ctl) - 0.05 (both orig & trans)
  - student_calls_failed == 0
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
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
from ref27b_manifest import DIFFICULTY_ORDER  # noqa: E402
from summarize_base4b import STAGE_BY_DIFF, extract_safe_failure_summary  # noqa: E402

MAX_MEAN_DIFF = 0.03
CORR_SLACK = 0.05
_EPS = 1e-12


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    num = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.fsum((x - mx) ** 2 for x in xs)
    dy = math.fsum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx * dy)
    if denom <= _EPS:
        return None
    return max(-1.0, min(1.0, num / denom))


def select_stack_check_entries(
    entries: list[dict[str, Any]],
    *,
    per_tier: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically pick per_tier entries per difficulty by sha256(f'{seed}|{source_id}')."""
    if per_tier <= 0:
        raise ValueError("per_tier must be positive")
    chosen: list[dict[str, Any]] = []
    for diff in DIFFICULTY_ORDER:
        tier_entries = [e for e in entries if e["difficulty"] == diff]
        if len(tier_entries) < per_tier:
            raise ValueError(f"Not enough entries in {diff}: {len(tier_entries)} < {per_tier}")
        ordered = sorted(
            tier_entries,
            key=lambda e: (
                hashlib.sha256(f"{seed}|{e['source_id']}".encode("utf-8")).hexdigest(),
                e["source_key"],
            ),
        )
        chosen.extend(ordered[:per_tier])
    return chosen


def evaluate_stack_check(
    sampled_entries: list[dict[str, Any]],
    pro_rates: dict[tuple[str, str], dict[str, float]],
    *,
    stu_fail: int,
) -> dict[str, Any]:
    """Compare Pro 5000 90-source pre-test against A100 27B Author-Rule pre & ctl."""
    if stu_fail < 0:
        raise ValueError("stu_fail must be >= 0")

    pro_orig: list[float] = []
    pro_trans: list[float] = []
    a100_pre_orig: list[float] = []
    a100_pre_trans: list[float] = []
    a100_ctl_orig: list[float] = []
    a100_ctl_trans: list[float] = []

    for e in sampled_entries:
        k = (e["difficulty"], e["source_id"])
        if k not in pro_rates:
            raise ValueError(f"Missing Pro 5000 pre-test result for difficulty={e['difficulty']}")
        pr = pro_rates[k]
        ar_ref = e.get("ref_counts_ar") or {}
        pro_orig.append(float(pr["orig_pre_rate"]))
        pro_trans.append(float(pr["trans_pre_rate"]))
        a100_pre_orig.append(ar_ref["原题"]["pre"]["correct"] / 16.0)
        a100_pre_trans.append(ar_ref["迁移题"]["pre"]["correct"] / 16.0)
        a100_ctl_orig.append(ar_ref["原题"]["control"]["correct"] / 16.0)
        a100_ctl_trans.append(ar_ref["迁移题"]["control"]["correct"] / 16.0)

    mean_pro_orig = statistics.fmean(pro_orig)
    mean_a100_orig = statistics.fmean(a100_pre_orig)
    mean_pro_trans = statistics.fmean(pro_trans)
    mean_a100_trans = statistics.fmean(a100_pre_trans)
    diff_orig = abs(mean_pro_orig - mean_a100_orig)
    diff_trans = abs(mean_pro_trans - mean_a100_trans)

    r_pro_orig = pearson_r(pro_orig, a100_pre_orig)
    r_pro_trans = pearson_r(pro_trans, a100_pre_trans)
    r_bench_orig = pearson_r(a100_pre_orig, a100_ctl_orig)
    r_bench_trans = pearson_r(a100_pre_trans, a100_ctl_trans)

    min_r_orig = (r_bench_orig - CORR_SLACK) if r_bench_orig is not None else None
    min_r_trans = (r_bench_trans - CORR_SLACK) if r_bench_trans is not None else None

    mean_ok = (diff_orig <= MAX_MEAN_DIFF + _EPS) and (diff_trans <= MAX_MEAN_DIFF + _EPS)
    corr_ok = (
        r_pro_orig is not None
        and min_r_orig is not None
        and r_pro_orig >= min_r_orig - _EPS
        and r_pro_trans is not None
        and min_r_trans is not None
        and r_pro_trans >= min_r_trans - _EPS
    )
    passed = bool(stu_fail == 0 and len(sampled_entries) > 0 and mean_ok and corr_ok)

    return {
        "passed": passed,
        "verdict": "通过" if passed else "不通过",
        "n_sources": len(sampled_entries),
        "n_calls": len(sampled_entries) * 32,
        "student_calls_failed": stu_fail,
        "orig": {
            "pro5000_mean": round(mean_pro_orig, 6),
            "a100_27b_mean": round(mean_a100_orig, 6),
            "abs_mean_diff": round(diff_orig, 6),
            "max_mean_diff": MAX_MEAN_DIFF,
            "pearson_r_pro_vs_a100": round(r_pro_orig, 6) if r_pro_orig is not None else None,
            "pearson_r_a100_pre_vs_ctl": round(r_bench_orig, 6) if r_bench_orig is not None else None,
            "min_r_required": round(min_r_orig, 6) if min_r_orig is not None else None,
        },
        "trans": {
            "pro5000_mean": round(mean_pro_trans, 6),
            "a100_27b_mean": round(mean_a100_trans, 6),
            "abs_mean_diff": round(diff_trans, 6),
            "max_mean_diff": MAX_MEAN_DIFF,
            "pearson_r_pro_vs_a100": round(r_pro_trans, 6) if r_pro_trans is not None else None,
            "pearson_r_a100_pre_vs_ctl": round(r_bench_trans, 6) if r_bench_trans is not None else None,
            "min_r_required": round(min_r_trans, 6) if min_r_trans is not None else None,
        },
    }


async def run_pro5000_pre_samples(
    sampled_entries: list[dict[str, Any]],
    *,
    out_dir: Path,
    repo_root: Path,
    student_url: str,
    concurrency: int = 16,
    grade_fn=None,
) -> tuple[dict[tuple[str, str], dict[str, float]], int]:
    """Run 16 pre-test samples for orig & transfer on sampled_entries cleanly (resume-safe)."""
    app_dir = str(repo_root / "evaluation" / "app")
    off_dir = str(repo_root / "evaluation" / "official")
    for p in (off_dir, app_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import offline_eval  # noqa: E402
    import protocol as protocol_mod  # noqa: E402
    import scoring as scoring_mod  # noqa: E402

    os.environ["LOCAL_ONLY_KEY"] = "loopback-only"
    student_url = offline_eval.local_url(student_url)
    client = offline_eval.openai.AsyncClient(
        api_key="loopback-only", base_url=student_url, timeout=300, max_retries=0
    )
    sem = asyncio.Semaphore(concurrency)
    grader = grade_fn or grade_math_answer
    pro_rates: dict[tuple[str, str], dict[str, float]] = {}
    stu_fail = 0

    try:
        for idx, e in enumerate(sampled_entries, 1):
            stage = STAGE_BY_DIFF[e["difficulty"]]
            stage_dir = out_dir / "pre_runs" / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            os.environ["RESULTS_DIR"] = str(stage_dir)
            row = e["row"]
            source = {"source_id": row["source_id"], "problem": row["problem"], "answer": row["answer"]}
            folder = scoring_mod.source_dir(source)
            folder.mkdir(parents=True, exist_ok=True)
            res_json = folder / "STACK_PRE_RESULT.json"

            if res_json.is_file():
                cached = json.loads(res_json.read_bytes())
                pro_rates[(e["difficulty"], e["source_id"])] = {
                    "orig_pre_rate": float(cached["orig_pre_rate"]),
                    "trans_pre_rate": float(cached["trans_pre_rate"]),
                }
                stu_fail += int(cached.get("stu_fail", 0))
                continue

            orig_correct = 0
            trans_correct = 0
            local_fail = 0
            for tgt_name, prob, ans, is_orig in (
                ("source", row["problem"], row["answer"], True),
                (row["transfer_target"], row["transfer_problem"], row["transfer_answer"], False),
            ):
                prompt = [{"role": "user", "content": protocol_mod._SOLVE_PROMPT.format(problem=prob)}]
                tasks = [
                    offline_eval.student_sample(client, source, tgt_name, "pre", i, prompt, sem)
                    for i in range(16)
                ]
                samples = await asyncio.gather(*tasks, return_exceptions=True)
                for sp in samples:
                    if isinstance(sp, Exception):
                        local_fail += 1
                        continue
                    v = grader(sp["text"], ans)
                    if float(v.get("acc", 0.0)) >= 0.5:
                        if is_orig:
                            orig_correct += 1
                        else:
                            trans_correct += 1

            stu_fail += local_fail
            rec_out = {
                "orig_pre_correct": orig_correct,
                "trans_pre_correct": trans_correct,
                "orig_pre_rate": orig_correct / 16.0,
                "trans_pre_rate": trans_correct / 16.0,
                "stu_fail": local_fail,
            }
            scoring_mod.put_once(res_json, rec_out)
            pro_rates[(e["difficulty"], e["source_id"])] = {
                "orig_pre_rate": rec_out["orig_pre_rate"],
                "trans_pre_rate": rec_out["trans_pre_rate"],
            }
            if idx % 15 == 0 or idx == len(sampled_entries):
                print(
                    f"[stack_check] pre_progress={idx}/{len(sampled_entries)} stu_fail={stu_fail}",
                    flush=True,
                )
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            res = close_fn()
            if asyncio.iscoroutine(res):
                await res

    return pro_rates, stu_fail


def regrade_stack_check_dir(
    sampled_entries: list[dict[str, Any]],
    *,
    out_dir: Path,
    grade_fn=None,
) -> tuple[dict[tuple[str, str], dict[str, float]], int]:
    """Re-grade saved 2,880 pre-test responses offline using grade_fn (zero GPU calls)."""
    grader = grade_fn or grade_math_answer
    pro_rates: dict[tuple[str, str], dict[str, float]] = {}
    stu_fail = 0
    for e in sampled_entries:
        stage = STAGE_BY_DIFF[e["difficulty"]]
        key = e["source_key"]
        folder = out_dir / "pre_runs" / stage / "sources" / key
        res_json = folder / "STACK_PRE_RESULT.json"
        if res_json.is_file():
            cached = json.loads(res_json.read_bytes())
            stu_fail += int(cached.get("stu_fail", 0))
        row = e["row"]
        orig_correct = 0
        trans_correct = 0
        for tgt_name, ans, is_orig in (
            ("source", row["answer"], True),
            (row["transfer_target"], row["transfer_answer"], False),
        ):
            for i in range(16):
                resp_p = folder / "samples" / tgt_name / "pre" / f"{i:02d}" / "RESPONSE.json"
                if not resp_p.is_file():
                    raise FileNotFoundError(
                        f"Missing saved stack-check sample ({stage}/{tgt_name}/{i:02d})"
                    )
                rec = json.loads(resp_p.read_bytes())
                v = grader(rec["text"], ans)
                if float(v.get("acc", 0.0)) >= 0.5:
                    if is_orig:
                        orig_correct += 1
                    else:
                        trans_correct += 1
        pro_rates[(e["difficulty"], e["source_id"])] = {
            "orig_pre_rate": orig_correct / 16.0,
            "trans_pre_rate": trans_correct / 16.0,
        }
    return pro_rates, stu_fail


def format_stack_report(res: dict[str, Any]) -> str:
    lines = [
        "[base4b_summary] " + "=" * 80,
        f"[base4b_summary] Pro 5000 vs A100 Student Stack Check Verdict: {res['verdict']} (n={res['n_sources']}, calls={res['n_calls']}, stu_fail={res['student_calls_failed']}, math_verify={res.get('math_verify_version', 'unknown')})",
    ]
    rt = res.get("student_runtime")
    if isinstance(rt, dict):
        lines.append(
            f"[base4b_summary]   student_runtime: dtype={rt.get('student_dtype')} vllm={rt.get('vllm')} "
            f"torch={rt.get('torch')} transformers={rt.get('transformers')} math_verify={rt.get('math_verify')}"
        )
    for label, d in (("原题 pre", res["orig"]), ("迁移题 pre", res["trans"])):
        lines.append(
            f"[base4b_summary]   {label}: Pro5000={d['pro5000_mean']:.4f} vs A100={d['a100_27b_mean']:.4f} "
            f"|diff|={d['abs_mean_diff']:.4f} (<= {d['max_mean_diff']:.2f}) | "
            f"r={d['pearson_r_pro_vs_a100']} >= {d['min_r_required']} (bench={d['pearson_r_a100_pre_vs_ctl']})"
        )
    lines.append("[base4b_summary] " + "=" * 80)
    return "\n".join(lines)


def main() -> int:
    import importlib.metadata as md

    ap = argparse.ArgumentParser(description="Pro 5000 vs A100 cross-machine Student stack check.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--student-url", default="http://127.0.0.1:18702/v1")
    ap.add_argument("--student-runtime-json", type=Path, default=None)
    ap.add_argument("--regrade-only", action="store_true")
    ap.add_argument("--json-name", default=None)
    ap.add_argument("--per-tier", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    try:
        mv_ver = md.version("math-verify")
    except md.PackageNotFoundError:
        mv_ver = "NOT_INSTALLED"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_and_verify_bundle(args.bundle_dir, require_exact_totals=True)
    sampled = select_stack_check_entries(loaded["eligible"], per_tier=args.per_tier, seed=args.seed)
    if args.regrade_only:
        pro_rates, stu_fail = regrade_stack_check_dir(sampled, out_dir=args.out_dir)
    else:
        pro_rates, stu_fail = asyncio.run(
            run_pro5000_pre_samples(
                sampled,
                out_dir=args.out_dir,
                repo_root=args.repo,
                student_url=args.student_url,
                concurrency=args.concurrency,
            )
        )
    res = evaluate_stack_check(sampled, pro_rates, stu_fail=stu_fail)
    res["math_verify_version"] = mv_ver
    rt_path = args.student_runtime_json
    if rt_path is not None and rt_path.is_file():
        res["student_runtime"] = json.loads(rt_path.read_text(encoding="utf-8"))
    elif args.regrade_only and (args.out_dir / "STACK_CHECK.json").is_file():
        prev = json.loads((args.out_dir / "STACK_CHECK.json").read_text(encoding="utf-8"))
        if isinstance(prev.get("student_runtime"), dict):
            res["student_runtime"] = prev["student_runtime"]
    json_name = args.json_name or ("STACK_CHECK_MV090.json" if args.regrade_only else "STACK_CHECK.json")
    (args.out_dir / json_name).write_text(
        json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(format_stack_report(res), flush=True)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
