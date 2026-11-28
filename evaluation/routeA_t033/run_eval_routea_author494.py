#!/usr/bin/env python3
"""T-033 Pipeline 1: Route A Step-100 evaluation on Author 494 test.parquet.

Wraps unmodified EduardoRL eval/run_env_test_eval.py using evaluation/author494's
ReplayStore + AgnesWindowClient (AIMD 1..max_agnes_window <= 16), enforces Tutor
config dtype == 'bfloat16', uses allow_abbrev=False, enforces Agnes technical
failure rate <= 1%, and exports sanitized per-problem CSV + diagnostics JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AUTHOR494_27B_DIR = REPO_ROOT / "evaluation" / "author494"
AUTHOR494_PRO_DIR = REPO_ROOT / "evaluation" / "author494_pro5000"
PAIRED_DIR = REPO_ROOT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(AUTHOR494_27B_DIR), str(AUTHOR494_PRO_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util  # noqa: E402

if "agnes_window" in sys.modules and not hasattr(sys.modules["agnes_window"], "AsyncCongestionWindow"):
    del sys.modules["agnes_window"]
if "agnes_window" not in sys.modules:
    _aw_spec = importlib.util.spec_from_file_location("agnes_window", AUTHOR494_27B_DIR / "agnes_window.py")
    assert _aw_spec and _aw_spec.loader
    _aw_mod = importlib.util.module_from_spec(_aw_spec)
    sys.modules["agnes_window"] = _aw_mod
    _aw_spec.loader.exec_module(_aw_mod)

from agnes_window import AsyncCongestionWindow, INNER_MAX_RETRIES  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_BASE,
    format_machine_summary,
    locate_eduardo_repo,
    verify_tutor_bfloat16,
    write_json_atomic,
)
from eval_hardening import ReplayStore, build_replay_manifest  # noqa: E402
from run_eval_27b_author494 import (  # noqa: E402
    _extract_flag_value,
    _resolve_git_commit,
    install_hardening_hooks,
)
from summarize_author494_pro5000 import (  # noqa: E402
    analyze_judge_failures,
    recompute_author_summary,
    validate_results_doc,
    verify_author_summary_match,
)

PER_PROBLEM_COLUMNS = (
    "row_index",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "post_same_correct",
    "post_same_total",
    "post_transfer_correct",
    "post_transfer_total",
    "delta_same",
    "delta_transfer",
    "delta_same_gated",
    "delta_transfer_gated",
    "leaked",
    "teaching_rejected",
    "eval_judge_pedagogy",
    "eval_judge_correctness",
    "num_teacher_turns",
    "num_blanked_teacher_turns",
    "unterminated_think_turns",
    "is_truncated",
    "eoc_used",
    "thinking_tokens_mean",
)


def strip_t033_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Strip wrapper flags with allow_abbrev=False (guards --student-model)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--max-agnes-window", type=int, default=16)
    ap.add_argument("--agnes-stats-out", default=None)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--replay-stats-out", default=None)
    ap.add_argument("--student-model-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_known_args(argv)


def _count_unterminated_think_turns(rec: dict[str, Any]) -> int:
    msgs = rec.get("reward_messages") or rec.get("messages") or []
    count = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("teacher", "assistant"):
            content = str(m.get("content") or "")
            if "</think>" not in content:
                count += 1
    return count


def export_sanitized_author494_outputs(
    result_json_path: Path,
    *,
    out_dir: Path,
    agnes_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = json.loads(result_json_path.read_text(encoding="utf-8"))
    metrics, records, judge_votes = validate_results_doc(doc)
    recomputed = recompute_author_summary(records)
    verify_author_summary_match(recomputed, metrics)
    jdiag = analyze_judge_failures(records, expected_votes=judge_votes)

    num_attempts = int((doc.get("args") or {}).get("num_student_attempts", 16))
    rows_sorted = sorted(records, key=lambda r: int(r["row_index"]))

    csv_rows: list[dict[str, Any]] = []
    total_teacher_turns = 0
    total_blanked_turns = 0
    total_unterm_turns = 0
    end_reasons: dict[str, int] = {}

    for rec in rows_sorted:
        ridx = int(rec["row_index"])
        n_turns = int(rec["num_teacher_turns"])
        n_blank = int(rec.get("num_blanked_teacher_turns", round(float(rec["blanked_teacher_turn_rate"]) * n_turns)))
        n_unterm = _count_unterminated_think_turns(rec)
        total_teacher_turns += n_turns
        total_blanked_turns += n_blank
        total_unterm_turns += n_unterm
        er = str(rec.get("end_reason") or "unknown")
        end_reasons[er] = end_reasons.get(er, 0) + 1

        t_toks = [float(x) for x in (rec.get("thinking_tokens") or [])]
        t_mean = statistics.fmean(t_toks) if t_toks else 0.0
        ps_same = int(round(float(rec["post_solve_same"]) * num_attempts))
        ps_trans = int(round(float(rec["post_solve_transfer"]) * num_attempts))

        csv_rows.append(
            {
                "row_index": ridx,
                "pre_solve_rate": f"{float(rec['pre_solve_rate']):.6f}",
                "pre_solve_rate_transfer": f"{float(rec['pre_solve_rate_transfer']):.6f}",
                "post_same_correct": ps_same,
                "post_same_total": num_attempts,
                "post_transfer_correct": ps_trans,
                "post_transfer_total": num_attempts,
                "delta_same": f"{float(rec['delta_same']):.6f}",
                "delta_transfer": f"{float(rec['delta_transfer']):.6f}",
                "delta_same_gated": f"{float(rec['delta_same_gated']):.6f}",
                "delta_transfer_gated": f"{float(rec['delta_transfer_gated']):.6f}",
                "leaked": int(bool(rec["leaked"])),
                "teaching_rejected": int(bool(rec["teaching_rejected"])),
                "eval_judge_pedagogy": "" if rec["eval_judge_pedagogy"] is None else f"{float(rec['eval_judge_pedagogy']):.4f}",
                "eval_judge_correctness": "" if rec["eval_judge_correctness"] is None else f"{float(rec['eval_judge_correctness']):.4f}",
                "num_teacher_turns": n_turns,
                "num_blanked_teacher_turns": n_blank,
                "unterminated_think_turns": n_unterm,
                "is_truncated": 1 if er == "token_budget" else 0,
                "eoc_used": int(bool(rec["eoc_used"])),
                "thinking_tokens_mean": f"{t_mean:.4f}",
            }
        )

    csv_path = out_dir / "author494_per_problem.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PER_PROBLEM_COLUMNS))
        writer.writeheader()
        writer.writerows(csv_rows)

    unterm_rate = (total_unterm_turns / total_teacher_turns) if total_teacher_turns else 0.0
    tech_fail = int(jdiag["retry_failed_votes"])
    total_votes = int(jdiag["total_votes"])
    eval_missing = int(jdiag["eval_pedagogy_missing"]) + int(jdiag["eval_correctness_missing"])
    agnes_ok = bool(jdiag["judge_available"])

    summary = {
        "status": "AUTHOR494_DONE" if agnes_ok else "AUTHOR494_AGNES_OVER_1PCT",
        "rc": 0 if agnes_ok else 2,
        "num_problems": int(metrics["num_problems"]),
        "num_conversations": int(metrics["num_conversations"]),
        "result_json": result_json_path.name,
        "per_problem_csv": csv_path.name,
        "metrics": {
            "delta_transfer": float(metrics["delta_transfer"]),
            "delta_same": float(metrics["delta_same"]),
            "delta_transfer_gated": float(metrics["delta_transfer_gated"]),
            "delta_same_gated": float(metrics["delta_same_gated"]),
            "leak_rate": float(metrics["leak_rate"]),
            "teaching_reject_rate": float(metrics["teaching_reject_rate"]),
            "eval_judge_pedagogy": float(metrics["eval_judge_pedagogy"]),
            "eval_judge_correctness": float(metrics["eval_judge_correctness"]),
            "eval_judge_ok": float(metrics["eval_judge_ok"]),
            "n_student_calls_failed": int(metrics["n_student_calls_failed"]),
            "transfer_tested_rate": float(metrics["transfer_tested_rate"]),
            "assume_thinking": bool(metrics["assume_thinking"]),
        },
        "diagnostics": {
            "total_teacher_turns": total_teacher_turns,
            "blanked_turns": total_blanked_turns,
            "blanked_rate": float(metrics["blanked_teacher_turn_rate"]),
            "unterminated_think_turns": total_unterm_turns,
            "unterminated_think_rate": unterm_rate,
            "token_budget_truncated_rate": float(metrics["token_budget_truncated_rate"]),
            "eoc_rate": float(metrics["eoc_rate"]),
            "thinking_tokens_avg_per_turn": float(metrics["thinking_tokens_avg_per_turn"]),
            "end_reasons": end_reasons,
        },
        "agnes_judge": {
            "total_votes": total_votes,
            "retry_failed_votes": tech_fail,
            "retry_fail_rate": float(jdiag["retry_fail_rate"]),
            "parse_failed_votes": int(jdiag["parse_failed_votes"]),
            "regex_recovered_votes": int(jdiag["regex_recovered_votes"]),
            "eval_pedagogy_missing": int(jdiag["eval_pedagogy_missing"]),
            "eval_correctness_missing": int(jdiag["eval_correctness_missing"]),
            "eval_missing_total": eval_missing,
            "judge_available": agnes_ok,
        },
        "agnes_window": agnes_stats or {},
    }
    write_json_atomic(out_dir / "author494_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    w_args, pass_argv = strip_t033_wrapper_args(raw_argv)

    if not (1 <= int(w_args.max_agnes_window) <= 16):
        raise SystemExit(f"ERROR: --max-agnes-window must be in 1..16, got {w_args.max_agnes_window}")

    base = Path(os.environ.get("PRO5000_BASE") or DEFAULT_BASE).resolve()
    eduardo_repo = locate_eduardo_repo(base)
    if str(eduardo_repo) not in sys.path:
        sys.path.insert(0, str(eduardo_repo))

    model_val = _extract_flag_value(pass_argv, "--model", "")
    if not model_val:
        raise SystemExit("ERROR: --model is required")
    tutor_dtype = verify_tutor_bfloat16(Path(model_val))
    print(f"[t033_author494] tutor_config_dtype={tutor_dtype} max_agnes_window={w_args.max_agnes_window}", flush=True)

    import eval.run_env_test_eval as upstream_eval  # noqa: E402
    import eval.run_middleturn_eval as upstream_middleturn  # noqa: E402

    old_argv = sys.argv
    try:
        sys.argv = ["eval.run_env_test_eval", *pass_argv]
        parsed_eval = upstream_eval.parse_args()
    finally:
        sys.argv = old_argv

    if w_args.student_model_dir and not Path(w_args.student_model_dir).is_dir():
        raise SystemExit(f"ERROR: --student-model-dir missing: {w_args.student_model_dir}")

    out_dir = Path(parsed_eval.output_dir or "./eval_results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = Path(w_args.replay_dir) if w_args.replay_dir else (out_dir / "replay_cache")

    eval_args_manifest = {
        k: v for k, v in vars(parsed_eval).items() if k not in ("output_dir", "conversations")
    }
    manifest = build_replay_manifest(
        upstream_repo=eduardo_repo,
        upstream_commit=_resolve_git_commit(eduardo_repo),
        wrapper_files=[
            AUTHOR494_27B_DIR / "agnes_window.py",
            AUTHOR494_27B_DIR / "eval_hardening.py",
            HERE / "run_eval_routea_author494.py",
        ],
        wrapper_base_dir=REPO_ROOT,
        tutor_model_path=parsed_eval.model,
        student_model_path=w_args.student_model_dir,
        test_parquet_path=parsed_eval.test_parquet,
        eval_args={**eval_args_manifest, "max_agnes_window": w_args.max_agnes_window},
    )

    store = ReplayStore(replay_dir, manifest=manifest, resume=w_args.resume)
    outer_retries = int(parsed_eval.num_retries)
    placeholder = install_hardening_hooks(
        upstream_eval,
        upstream_middleturn,
        store=store,
        agnes_api_base=parsed_eval.gemini_api_base,
        agnes_api_key_env=parsed_eval.gemini_api_key_env,
        student_api_base=parsed_eval.student_api_base,
        student_api_key_env=parsed_eval.student_api_key_env,
        outer_num_retries=outer_retries,
    )
    placeholder.window.maximum = int(w_args.max_agnes_window)

    stats: dict[str, Any] = {}
    try:
        sys.argv = ["eval.run_env_test_eval", *pass_argv]
        upstream_eval.main()
    finally:
        sys.argv = old_argv
        stats = placeholder.stats(outer_num_retries=outer_retries)
        rstats = store.stats()
        store.close()
        if w_args.agnes_stats_out:
            write_json_atomic(Path(w_args.agnes_stats_out), stats)
        if w_args.replay_stats_out:
            write_json_atomic(Path(w_args.replay_stats_out), rstats)

    res_jsons = sorted(out_dir.glob("transfer_eval_*.json"))
    if not res_jsons:
        raise SystemExit("ERROR: transfer_eval_*.json not generated")
    summary = export_sanitized_author494_outputs(res_jsons[-1], out_dir=out_dir, agnes_stats=stats)
    print(
        format_machine_summary(
            {
                "status": summary["status"],
                "rc": summary["rc"],
                "calls": stats.get("logical_calls", summary["agnes_judge"]["total_votes"]),
                "failures": summary["agnes_judge"]["retry_failed_votes"] + summary["metrics"]["n_student_calls_failed"],
                "num_problems": summary["num_problems"],
                "delta_transfer": f"{summary['metrics']['delta_transfer']:+.6f}",
                "delta_same": f"{summary['metrics']['delta_same']:+.6f}",
                "max_window": stats.get("max_limit_reached", 0),
                "http_429": stats.get("http_429_events", 0),
            }
        ),
        flush=True,
    )
    return int(summary["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
