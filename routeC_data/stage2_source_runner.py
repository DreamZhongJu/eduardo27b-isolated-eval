#!/usr/bin/env python3
"""T-031 Stage 2 runner (Pro 5000): measure bf16 source pretest for all 2,100 sources.

Reuses T-023 Stage 2 exact sampling prompt/config and canonical `hybrid_grader.py`:
- Student: Llama-3.1-8B-Instruct (bf16), temp=0.6, max_tokens=1024, k=16
- Grading: `routeA_data/hybrid_grader.py` (sha256 7d2a5f46...) with math-verify==0.8.0
  + local Qwen3.6-27B (bf16, TP=2) equivalence fallback (AIMD window 1..16, >1% abort gate)
- Agnes: NOT used.
- Outputs `source_pretest_measured.jsonl` (IDs + counts + p_new/b_new ONLY, zero text)
  and final Route C routing inventory via `routeC_data.routeC_inventory`.

PRIVACY GUARANTEE:
- Never prints or logs problem statements, reference answers, or student responses.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from routeA_data.hybrid_grader import (
    DEFAULT_MAX_WINDOW,
    AsyncAdaptiveWindow,
    HybridGrader,
)
from routeA_data.hybrid_grader_caller import (
    CANONICAL_HYBRID_GRADER_SHA256,
    TechnicalFailureGateError,
    TechnicalFailureTracker,
    assert_canonical_hybrid_grader,
)
from routeA_data.rasch_ladder import compute_smoothed_p_and_b
from routeA_data.stage2_runner import (
    AUTHOR_SOLVE_PROMPT,
    _pearson,
    _quantiles,
    _sample_student_once,
    _spearman,
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    verify_required_pins,
)
from routeC_data.routeC_inventory import (
    EXPECTED_T023_MEASURED_SHA256,
    format_inventory_report,
    run_inventory,
)


def source_key_hash(source_id: str) -> str:
    return sha256_bytes(f"source::{source_id}".encode("utf-8"))[:24]


def load_measured_rows(
    path: Path, expected_sha256: Optional[str] = EXPECTED_T023_MEASURED_SHA256
) -> tuple[List[Dict[str, Any]], str]:
    actual_sha = sha256_file(path)
    if expected_sha256 and actual_sha != expected_sha256.strip().lower():
        raise ValueError(
            f"Input measured JSONL SHA-256 mismatch: actual={actual_sha} expected={expected_sha256}"
        )
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows, actual_sha


async def run_gpu_phase(
    rows: Sequence[Dict[str, Any]],
    *,
    run_dir: Path,
    student_client: Any,
    student_model: str,
    judge_client: Any,
    judge_model: str,
    k_samples: int = 16,
    student_temperature: float = 0.6,
    student_max_tokens: int = 1024,
    student_concurrency: int = 64,
    judge_max_window: int = DEFAULT_MAX_WINDOW,
    source_concurrency: int = 16,
    limit_sources: Optional[int] = None,
    rule_grader: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    llm_caller: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Generate k=16 student rollouts per source problem and grade with canonical HybridGrader."""
    assert_canonical_hybrid_grader()

    resp_dir = run_dir / "responses"
    grade_dir = run_dir / "hybrid_grades"
    resp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    grade_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    selected_rows = list(rows[:limit_sources]) if limit_sources is not None else list(rows)
    total_sources = len(selected_rows)
    student_sem = asyncio.Semaphore(student_concurrency)
    source_sem = asyncio.Semaphore(source_concurrency)
    judge_window = AsyncAdaptiveWindow(max_window=judge_max_window)

    grader_kwargs: Dict[str, Any] = {
        "client": judge_client if llm_caller is None else None,
        "model": judge_model,
        "llm_caller": llm_caller,
        "window": judge_window,
    }
    if rule_grader is not None:
        grader_kwargs["rule_grader"] = rule_grader
    grader = HybridGrader(**grader_kwargs)

    student_gate = TechnicalFailureTracker(max_percent=1, min_samples=200)
    judge_gate = TechnicalFailureTracker(max_percent=1, min_samples=100)
    state_lock = asyncio.Lock()

    stats: Dict[str, Any] = {
        "total_sources": total_sources,
        "sources_done": 0,
        "sources_resumed_resp": 0,
        "sources_resumed_grade": 0,
        "student_calls": 0,
        "student_failed": 0,
        "llm_calls": 0,
        "llm_tech_failed": 0,
        "llm_unknown": 0,
        "aborted": False,
        "abort_reason": None,
        "started_at": time.time(),
    }
    abort_event = asyncio.Event()

    async def process_one_source(row: Dict[str, Any]) -> None:
        if abort_event.is_set():
            return
        async with source_sem:
            if abort_event.is_set():
                return
            sid = str(row["source_id"])
            split = str(row["split"])
            skey = source_key_hash(sid)
            resp_path = resp_dir / f"{skey}.json"
            grade_path = grade_dir / f"{skey}.json"

            # 1. Student rollouts (resumable)
            if resp_path.is_file():
                resp_doc = json.loads(resp_path.read_text(encoding="utf-8"))
                rollouts = resp_doc["rollouts"]
                async with state_lock:
                    stats["sources_resumed_resp"] += 1
            else:
                msgs = [
                    {
                        "role": "user",
                        "content": AUTHOR_SOLVE_PROMPT.format(problem=row["source_problem"]),
                    }
                ]
                raw_results = await asyncio.gather(
                    *[
                        _sample_student_once(
                            student_client,
                            student_model,
                            msgs,
                            temperature=student_temperature,
                            max_tokens=student_max_tokens,
                            sem=student_sem,
                        )
                        for _ in range(k_samples)
                    ]
                )
                rollouts = []
                n_fail_local = 0
                for idx, (text, finish, att) in enumerate(raw_results):
                    failed = text is None
                    if failed:
                        n_fail_local += 1
                    rollouts.append(
                        {
                            "rollout_index": idx,
                            "failed": failed,
                            "finish_reason": finish,
                            "attempts": att,
                            "text": text,
                        }
                    )
                async with state_lock:
                    for r in rollouts:
                        stats["student_calls"] += 1
                        if r["failed"]:
                            stats["student_failed"] += 1
                        try:
                            student_gate.record(technical_failure=bool(r["failed"]))
                        except TechnicalFailureGateError as exc:
                            stats["aborted"] = True
                            stats["abort_reason"] = f"student_failure_gate: {exc}"
                            abort_event.set()
                            return

                resp_doc = {
                    "source_id": sid,
                    "split": split,
                    "k_samples": k_samples,
                    "student_model": student_model,
                    "temperature": student_temperature,
                    "max_tokens": student_max_tokens,
                    "n_failed": n_fail_local,
                    "rollouts": rollouts,
                }
                atomic_write_json(resp_path, resp_doc)

            # 2. Hybrid grading (resumable)
            if grade_path.is_file():
                async with state_lock:
                    stats["sources_resumed_grade"] += 1
                    stats["sources_done"] += 1
                return

            per_rollout: List[Dict[str, Any]] = []
            for r in rollouts:
                if abort_event.is_set():
                    return
                if r["failed"] or r["text"] is None:
                    per_rollout.append(
                        {
                            "rollout_index": r["rollout_index"],
                            "student_failed": True,
                            "rule_correct": None,
                            "hybrid_correct": None,
                            "llm_called": False,
                            "llm_unknown": False,
                            "llm_technical_failure": False,
                        }
                    )
                    continue

                gres = await grader.grade(row["source_problem"], row["source_answer"], r["text"])
                if gres.llm_called:
                    async with state_lock:
                        stats["llm_calls"] += 1
                        if gres.technical_failure:
                            stats["llm_tech_failed"] += 1
                        if gres.is_unknown:
                            stats["llm_unknown"] += 1
                        try:
                            judge_gate.record(technical_failure=gres.technical_failure)
                        except TechnicalFailureGateError as exc:
                            stats["aborted"] = True
                            stats["abort_reason"] = f"judge_27b_failure_gate: {exc}"
                            abort_event.set()
                            return

                per_rollout.append(
                    {
                        "rollout_index": r["rollout_index"],
                        "student_failed": False,
                        "rule_correct": bool(gres.rule_correct),
                        "rule_incorrect_format": int(gres.incorrect_format),
                        "hybrid_correct": None if gres.technical_failure else bool(gres.correct),
                        "hybrid_status": gres.status,
                        "hybrid_source": gres.source,
                        "llm_called": bool(gres.llm_called),
                        "llm_unknown": bool(gres.is_unknown),
                        "llm_technical_failure": bool(gres.technical_failure),
                        "attempts": int(gres.attempts),
                    }
                )

            n_failed = sum(
                1 for x in per_rollout if x["student_failed"] or x["llm_technical_failure"]
            )
            n_student_failed = sum(1 for x in per_rollout if x["student_failed"])
            n_llm_tech_failed = sum(1 for x in per_rollout if x["llm_technical_failure"])
            n_scored = len(per_rollout) - n_failed
            n_correct_rule = sum(
                1
                for x in per_rollout
                if (not x["student_failed"]) and (not x["llm_technical_failure"]) and x["rule_correct"]
            )
            n_correct_hybrid = sum(
                1
                for x in per_rollout
                if (not x["student_failed"]) and (not x["llm_technical_failure"]) and x["hybrid_correct"]
            )
            n_llm_called = sum(1 for x in per_rollout if x["llm_called"])
            n_unknown = sum(1 for x in per_rollout if x["llm_unknown"])

            grade_doc = {
                "source_id": sid,
                "split": split,
                "hybrid_grader_sha256": CANONICAL_HYBRID_GRADER_SHA256,
                "n_samples": len(per_rollout),
                "n_scored": n_scored,
                "n_failed": n_failed,
                "n_student_failed": n_student_failed,
                "n_llm_tech_failed": n_llm_tech_failed,
                "n_correct_rule": n_correct_rule,
                "n_correct_hybrid": n_correct_hybrid,
                "n_llm_called": n_llm_called,
                "n_unknown": n_unknown,
                "rollouts": per_rollout,
            }
            atomic_write_json(grade_path, grade_doc)

            async with state_lock:
                stats["sources_done"] += 1
                done = stats["sources_done"]
                if done % 25 == 0 or done == total_sources:
                    elapsed = max(0.001, time.time() - stats["started_at"])
                    rate = done / elapsed
                    eta_min = (total_sources - done) / rate / 60.0 if rate > 0 else 0.0
                    progress = {
                        **stats,
                        "judge_window": judge_window.window,
                        "elapsed_sec": round(elapsed, 1),
                        "sources_per_min": round(rate * 60.0, 2),
                        "eta_min": round(eta_min, 1),
                    }
                    atomic_write_json(run_dir / "gpu_phase_status.json", progress)
                    print(
                        f"[t031_gpu] {done}/{total_sources} sources "
                        f"({100.0 * done / total_sources:.1f}%) "
                        f"stu_fail={stats['student_failed']}/{stats['student_calls']} "
                        f"llm_fail={stats['llm_tech_failed']}/{stats['llm_calls']} "
                        f"win={judge_window.window} eta={eta_min:.1f}m",
                        flush=True,
                    )

    await asyncio.gather(*[process_one_source(row) for row in selected_rows])

    if stats["aborted"]:
        atomic_write_json(run_dir / "gpu_phase_status.json", stats)
        raise TechnicalFailureGateError(str(stats["abort_reason"]))

    student_gate.check_final()
    judge_gate.check_final()

    stats["completed_at"] = time.time()
    stats["elapsed_sec"] = round(stats["completed_at"] - stats["started_at"], 2)
    stats["status"] = "ok"
    atomic_write_json(run_dir / "gpu_phase_status.json", stats)
    (run_dir / "gpu_phase_done.flag").write_text("ok\n", encoding="utf-8")
    return stats


def run_summarize_phase(
    rows: Sequence[Dict[str, Any]],
    *,
    run_dir: Path,
    input_path: Path,
    input_sha256: str,
    allow_custom_sha256: bool = False,
    student_model_desc: str = "Meta-Llama-3.1-8B-Instruct (bf16)",
    judge_model_desc: str = "Qwen3.6-27B (bf16, TP=2)",
) -> Dict[str, Any]:
    """Assemble pure-number source pretest table, correlations, and final Route C inventory."""
    hybrid_sha = assert_canonical_hybrid_grader()
    grade_dir = run_dir / "hybrid_grades"

    pretest_rows: List[Dict[str, Any]] = []
    p_new_smooth: List[float] = []
    p_old_smooth: List[float] = []
    p_new_raw: List[float] = []
    p_old_raw: List[float] = []
    b_new_all: List[float] = []

    tot_samples = 0
    tot_scored = 0
    tot_failed = 0
    tot_student_failed = 0
    tot_llm_tech_failed = 0
    tot_rule_correct = 0
    tot_hybrid_correct = 0
    tot_llm_called = 0
    tot_unknown = 0

    for row in rows:
        sid = str(row["source_id"])
        split = str(row["split"])
        direction = str(row["direction"])
        skey = source_key_hash(sid)
        gpath = grade_dir / f"{skey}.json"
        if not gpath.is_file():
            raise FileNotFoundError(f"Missing hybrid grade file for source_id={sid}: {gpath}")
        gdoc = json.loads(gpath.read_text(encoding="utf-8"))

        n_samples = int(gdoc["n_samples"])
        n_scored = int(gdoc["n_scored"])
        n_failed = int(gdoc["n_failed"])
        n_stu_fail = int(gdoc.get("n_student_failed", 0))
        n_llm_fail = int(gdoc.get("n_llm_tech_failed", 0))
        n_rule = int(gdoc["n_correct_rule"])
        n_hyb = int(gdoc["n_correct_hybrid"])
        n_llm = int(gdoc["n_llm_called"])
        n_unk = int(gdoc["n_unknown"])

        p_new, b_new = compute_smoothed_p_and_b(n_hyb, n_scored)
        p_raw = (n_hyb / n_scored) if n_scored else 0.0

        old_rate = float(row["source_pre_solve_rate_old"])
        old_k16 = int(round(old_rate * 16))
        p_old_sm, _ = compute_smoothed_p_and_b(old_k16, 16)

        p_new_smooth.append(p_new)
        p_old_smooth.append(p_old_sm)
        p_new_raw.append(p_raw)
        p_old_raw.append(old_rate)
        b_new_all.append(b_new)

        tot_samples += n_samples
        tot_scored += n_scored
        tot_failed += n_failed
        tot_student_failed += n_stu_fail
        tot_llm_tech_failed += n_llm_fail
        tot_rule_correct += n_rule
        tot_hybrid_correct += n_hyb
        tot_llm_called += n_llm
        tot_unknown += n_unk

        pretest_rows.append(
            {
                "source_id": sid,
                "split": split,
                "direction": direction,
                "old_pre_solve_rate": old_rate,
                "old_correct_16": old_k16,
                "n_samples": n_samples,
                "n_scored": n_scored,
                "n_failed": n_failed,
                "n_correct_rule": n_rule,
                "n_correct_hybrid": n_hyb,
                "n_llm_called": n_llm,
                "n_unknown": n_unk,
                "p_raw_new": round(p_raw, 8),
                "p_new": round(p_new, 8),
                "b_new": round(b_new, 8),
            }
        )

    out_pretest_jsonl = run_dir / "source_pretest_measured.jsonl"
    with out_pretest_jsonl.open("w", encoding="utf-8") as f:
        for pr in pretest_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    os.chmod(out_pretest_jsonl, 0o600)
    out_pretest_sha = sha256_file(out_pretest_jsonl)
    (run_dir / "source_pretest_measured.jsonl.sha256").write_text(
        f"{out_pretest_sha}  {out_pretest_jsonl.name}\n", encoding="utf-8"
    )

    inv_dir = run_dir / "final_inventory"
    inv_summary = run_inventory(
        measured_jsonl=input_path,
        out_dir=inv_dir,
        source_pretest_jsonl=out_pretest_jsonl,
        expected_measured_sha256=None if allow_custom_sha256 else input_sha256,
        expected_source_pretest_sha256=out_pretest_sha,
        allow_custom_sha256=allow_custom_sha256,
    )

    n_src = len(pretest_rows)
    mean_diff_smooth = sum(a - b for a, b in zip(p_new_smooth, p_old_smooth)) / n_src if n_src else 0.0
    mae_smooth = sum(abs(a - b) for a, b in zip(p_new_smooth, p_old_smooth)) / n_src if n_src else 0.0
    mean_diff_raw = sum(a - b for a, b in zip(p_new_raw, p_old_raw)) / n_src if n_src else 0.0
    mae_raw = sum(abs(a - b) for a, b in zip(p_new_raw, p_old_raw)) / n_src if n_src else 0.0

    pearson_sm = _pearson(p_old_smooth, p_new_smooth)
    spearman_sm = _spearman(p_old_smooth, p_new_smooth)
    pearson_raw = _pearson(p_old_raw, p_new_raw)
    spearman_raw = _spearman(p_old_raw, p_new_raw)

    total_calls = tot_samples + tot_llm_called
    tech_fail_rate_overall = (tot_failed / tot_samples) if tot_samples else 0.0
    llm_fail_rate = (tot_llm_tech_failed / tot_llm_called) if tot_llm_called else 0.0
    unk_rate_of_llm = (tot_unknown / tot_llm_called) if tot_llm_called else 0.0
    unk_rate_of_scored = (tot_unknown / tot_scored) if tot_scored else 0.0

    summary: Dict[str, Any] = {
        "task": "T-031",
        "stage": "stage2",
        "card_version": 1,
        "student_model": student_model_desc,
        "judge_model": judge_model_desc,
        "hybrid_grader_sha256": hybrid_sha,
        "input_measured_jsonl_sha256": input_sha256,
        "source_pretest_measured_jsonl_sha256": out_pretest_sha,
        "source_pretest_measured_jsonl_bytes": out_pretest_jsonl.stat().st_size,
        "counts": {
            "n_sources": n_src,
            "n_samples": tot_samples,
            "n_scored": tot_scored,
            "n_failed": tot_failed,
            "n_student_failed": tot_student_failed,
            "n_llm_tech_failed": tot_llm_tech_failed,
            "tech_fail_rate_student": round(tot_student_failed / tot_samples, 6) if tot_samples else 0.0,
            "tech_fail_rate_llm": round(llm_fail_rate, 6),
            "tech_fail_rate_overall": round(tech_fail_rate_overall, 6),
            "n_correct_rule": tot_rule_correct,
            "n_correct_hybrid": tot_hybrid_correct,
            "promoted_rollouts": tot_hybrid_correct - tot_rule_correct,
            "n_llm_called": tot_llm_called,
            "n_unknown": tot_unknown,
            "unknown_rate_of_llm": round(unk_rate_of_llm, 6),
            "unknown_rate_of_scored": round(unk_rate_of_scored, 6),
        },
        "p_orig_new_smoothed": _quantiles(p_new_smooth),
        "p_orig_new_raw": _quantiles(p_new_raw),
        "b_orig_new": _quantiles(b_new_all),
        "vs_old_pre_solve_rate": {
            "n_pairs": n_src,
            "pearson_smoothed": round(pearson_sm, 6) if pearson_sm is not None else None,
            "spearman_smoothed": round(spearman_sm, 6) if spearman_sm is not None else None,
            "mean_diff_new_minus_old_smoothed": round(mean_diff_smooth, 6),
            "mae_smoothed": round(mae_smooth, 6),
            "pearson_raw": round(pearson_raw, 6) if pearson_raw is not None else None,
            "spearman_raw": round(spearman_raw, 6) if spearman_raw is not None else None,
            "mean_diff_new_minus_old_raw": round(mean_diff_raw, 6),
            "mae_raw": round(mae_raw, 6),
        },
        "final_inventory": inv_summary,
    }

    atomic_write_json(run_dir / "stage2_summary.json", summary)
    manifest = {
        "task": "T-031",
        "stage": "stage2",
        "input_measured_jsonl_sha256": input_sha256,
        "source_pretest_measured_jsonl_sha256": out_pretest_sha,
        "source_pretest_measured_jsonl_bytes": out_pretest_jsonl.stat().st_size,
        "hybrid_grader_sha256": hybrid_sha,
        "stage2_summary_sha256": sha256_file(run_dir / "stage2_summary.json"),
    }
    atomic_write_json(run_dir / "stage2_manifest.json", manifest)

    inv_txt = format_inventory_report(
        inv_summary,
        measured_sha256=input_sha256,
        source_pretest_sha256=out_pretest_sha,
        hybrid_grader_sha256=hybrid_sha,
    )
    hl = next(
        (r for r in inv_summary["routing_tables"]["train_t026"] if abs(r["tau"] - 0.4) < 1e-9 and abs(r["margin"] - 0.5) < 1e-9),
        inv_summary["routing_tables"]["train_t026"][0],
    )
    vs_old = summary["vs_old_pre_solve_rate"]
    lines = [
        "=== T-031 STAGE 2 SOURCE PRETEST SUMMARY (ZERO TEXT) ===",
        f"hybrid_grader_sha256={hybrid_sha}",
        f"input_measured_jsonl_sha256={input_sha256}",
        f"source_pretest_measured_jsonl_sha256={out_pretest_sha}",
        f"sources={n_src} samples={tot_samples} scored={tot_scored} failed={tot_failed} "
        f"(stu_fail={tot_student_failed}, llm_fail={tot_llm_tech_failed}, overall_fail_rate={tech_fail_rate_overall:.4%})",
        f"rule_correct={tot_rule_correct} hybrid_correct={tot_hybrid_correct} promoted=+{tot_hybrid_correct - tot_rule_correct} "
        f"llm_called={tot_llm_called} unknown={tot_unknown} (unk_of_llm={unk_rate_of_llm:.4%}, unk_of_scored={unk_rate_of_scored:.4%})",
        f"vs_old_pre_solve_rate={vs_old}",
        "",
        inv_txt.strip(),
        "",
        "status: STAGE2_DONE",
        "rc: 0",
        f"calls: {total_calls}",
        f"failures: {tot_failed}",
        f"tech_fail_rate_overall: {tech_fail_rate_overall:.6f}",
        f"llm_called: {tot_llm_called}",
        f"unknown: {tot_unknown}",
        f"unknown_rate_of_llm: {unk_rate_of_llm:.6f}",
        f"pearson_smoothed: {vs_old['pearson_smoothed']}",
        f"mean_diff_smoothed: {vs_old['mean_diff_new_minus_old_smoothed']}",
        f"source_pretest_sha256: {out_pretest_sha}",
        f"t026_train_easy_tau04_m05: {hl['easy_routed']}",
        f"t026_train_hard_tau04_m05: {hl['hard_routed']}",
        f"t026_train_fallback_tau04_m05: {hl['fallback_routeA']}",
    ]
    (run_dir / "stage2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "STAGE2_DONE.flag").write_text("ok\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T-031 Stage 2 source pretest runner (gpu / summarize).",
        allow_abbrev=False,
    )
    parser.add_argument("--phase", choices=["gpu", "summarize"], required=True)
    parser.add_argument("--measured-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", type=str, default=EXPECTED_T023_MEASURED_SHA256)
    parser.add_argument("--allow-custom-sha256", action="store_true")
    parser.add_argument("--student-api-base", type=str, default="http://127.0.0.1:18991/v1")
    parser.add_argument("--student-model", type=str, default="Llama-3.1-8B-Instruct-t031")
    parser.add_argument("--judge-api-base", type=str, default="http://127.0.0.1:18992/v1")
    parser.add_argument("--judge-model", type=str, default="judge-local")
    parser.add_argument("--serving-api-key", type=str, default=os.environ.get("SERVING_API_KEY", "EMPTY"))
    parser.add_argument("--k-samples", type=int, default=16)
    parser.add_argument("--student-concurrency", type=int, default=64)
    parser.add_argument("--judge-max-window", type=int, default=16)
    parser.add_argument("--source-concurrency", type=int, default=16)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--skip-pin-check", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows, input_sha = load_measured_rows(
        args.measured_jsonl,
        None if args.allow_custom_sha256 else args.expected_input_sha256,
    )

    if args.phase == "gpu":
        if not args.skip_pin_check:
            pins = verify_required_pins()
            print(f"[t031_stage2] verified pins={pins}", flush=True)
        import httpx
        import openai

        http_client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(180.0))
        student_client = openai.AsyncOpenAI(
            base_url=args.student_api_base,
            api_key=args.serving_api_key,
            http_client=http_client,
        )
        judge_client = openai.AsyncOpenAI(
            base_url=args.judge_api_base,
            api_key=args.serving_api_key,
            http_client=http_client,
        )

        async def _run() -> Dict[str, Any]:
            try:
                return await run_gpu_phase(
                    rows,
                    run_dir=args.run_dir,
                    student_client=student_client,
                    student_model=args.student_model,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    k_samples=args.k_samples,
                    student_concurrency=args.student_concurrency,
                    judge_max_window=args.judge_max_window,
                    source_concurrency=args.source_concurrency,
                    limit_sources=args.limit_sources,
                )
            finally:
                await http_client.aclose()

        res = asyncio.run(_run())
        print(json.dumps(res, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    if args.phase == "summarize":
        run_summarize_phase(
            rows,
            run_dir=args.run_dir,
            input_path=args.measured_jsonl,
            input_sha256=input_sha,
            allow_custom_sha256=args.allow_custom_sha256,
        )
        print((args.run_dir / "stage2_summary.txt").read_text(encoding="utf-8"), end="", flush=True)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
