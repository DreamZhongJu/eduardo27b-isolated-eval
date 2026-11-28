#!/usr/bin/env python3
"""T-033 Pipeline 2: Route A Step-100 evaluation on Self-built 3-tier (bf16 Student & Tutor).

Reuses frozen evaluation/app + base4b_selfbuilt/run_post_eval.py +
agroup_selfbuilt_pro5000/{agnes_window.py, import_bundle.py} and unmodified
evaluation/agnes_eval_v3/agnes_judge.py (SHA-256 2ffe7464...).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AGROUP_SB_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(AGROUP_SB_DIR), str(BASE4B_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    AGNES_JUDGE_SHA256,
    DEFAULT_BASE,
    format_machine_summary,
    sha256_file,
    verify_tutor_bfloat16,
    write_json_atomic,
)
from export_27b_selfbuilt import grade_math_answer  # noqa: E402
from import_bundle import load_and_verify_bundle  # noqa: E402
from ref27b_manifest import DIFFICULTY_ORDER, select_pilot_entries  # noqa: E402
from run_post_eval import (  # noqa: E402
    generate_entries,
    grade_author_rule_entries,
)
import importlib.util  # noqa: E402

if "agnes_window" in sys.modules and not hasattr(sys.modules["agnes_window"], "AgnesWindowCaller"):
    del sys.modules["agnes_window"]
if "agnes_window" not in sys.modules:
    _sb_spec = importlib.util.spec_from_file_location("agnes_window", AGROUP_SB_DIR / "agnes_window.py")
    assert _sb_spec and _sb_spec.loader
    _sb_mod = importlib.util.module_from_spec(_sb_spec)
    sys.modules["agnes_window"] = _sb_mod
    _sb_spec.loader.exec_module(_sb_mod)

from run_post_eval_pro5000 import (  # noqa: E402
    AgnesTechnicalFailureLimitError,
    grade_agnes_pro5000,
)
from summarize_base4b import (  # noqa: E402
    STAGE_BY_DIFF,
    collect_base4b_records,
    extract_safe_failure_summary,
)


def select_smoke_entries(
    loaded: dict[str, Any],
    *,
    smoke_total: int = 16,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically pick smoke_total entries across 3 tiers (6 easy + 5 medium + 5 hard for 16)."""
    if smoke_total <= 0:
        return list(loaded["eligible"])
    base_k = smoke_total // len(DIFFICULTY_ORDER)
    rem = smoke_total % len(DIFFICULTY_ORDER)
    chosen: list[dict[str, Any]] = []
    for idx, diff in enumerate(DIFFICULTY_ORDER):
        take = base_k + (1 if idx < rem else 0)
        tier = [e for e in loaded["eligible"] if e["difficulty"] == diff]
        ordered = sorted(
            tier,
            key=lambda e: (
                hashlib.sha256(f"{seed}|{e['source_id']}".encode("utf-8")).hexdigest(),
                e["source_key"],
            ),
        )
        if len(ordered) < take:
            raise ValueError(f"Not enough entries in {diff}: {len(ordered)} < {take}")
        chosen.extend(ordered[:take])
    return chosen


def shard_entries(
    entries: list[dict[str, Any]],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"Invalid shard ({shard_index}/{num_shards})")
    if num_shards == 1:
        return entries
    return [e for i, e in enumerate(entries) if (i % num_shards) == shard_index]


def check_agnes_guard(workers: int, *, proc_dir: Path = Path("/proc")) -> None:
    if not (1 <= workers <= 16):
        raise ValueError(f"Pro 5000 Agnes workers must be in 1..16, got {workers}")
    if proc_dir.is_dir():
        for p in proc_dir.iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if any(tok in cmd for tok in ("agnes_judge.py", "run_eval_routea_author494.py", "run_post_eval_routea.py")):
                raise RuntimeError("Conflicting active Agnes grading process detected on host")


def export_selfbuilt_outputs(
    loaded: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    out_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    subset_view = {**loaded, "eligible": entries}
    agnes_records, agnes_diag = collect_base4b_records(subset_view, out_root, repo_root, grade_subdir="AGNES")
    ar_records, _ = collect_base4b_records(subset_view, out_root, repo_root, grade_subdir="AUTHOR_RULE")

    agnes_map = {(r["difficulty"], r["source_key"]): r for r in agnes_records}
    ar_map = {(r["difficulty"], r["source_key"]): r for r in ar_records}

    app_dir = str(repo_root / "evaluation" / "app")
    off_dir = str(repo_root / "evaluation" / "official")
    for p in (off_dir, app_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import protocol as protocol_mod  # noqa: E402

    csv_rows: list[dict[str, Any]] = []
    for e in entries:
        diff = e["difficulty"]
        skey = e["source_key"]
        k = (diff, skey)
        ag = agnes_map.get(k)
        ar = ar_map.get(k)
        if ag is None and ar is None:
            continue
        stage = STAGE_BY_DIFF[diff]
        roll_p = out_root / stage / "sources" / skey / "dialogue" / "ROLLOUT_RESULT.json"
        n_turns = 0
        n_blank = 0
        n_unterm = 0
        eoc = 0
        t_mean = 0.0
        end_reason = "unknown"
        if roll_p.is_file():
            roll = json.loads(roll_p.read_bytes())
            assume = bool(roll.get("assume_thinking", True))
            end_reason = str(roll.get("end_reason") or "unknown")
            eoc = int(bool(roll.get("eoc_used")))
            toks = [float(x) for x in (roll.get("thinking_tokens") or [])]
            t_mean = statistics.fmean(toks) if toks else 0.0
            for m in roll.get("raw_messages") or []:
                if m.get("role") == "teacher":
                    n_turns += 1
                    raw_c = str(m.get("content") or "")
                    if not protocol_mod.strip_teacher_controls(raw_c, assume_thinking=assume).strip():
                        n_blank += 1
                    if "</think>" not in raw_c:
                        n_unterm += 1
        row_out = {
            "difficulty": diff,
            "source_key": skey,
            "row_index": int(e["row_index"]),
            "agnes_orig_post_correct": ag["counts"]["原题"]["post"]["correct"] if ag else "",
            "agnes_orig_post_incorrect": ag["counts"]["原题"]["post"]["incorrect"] if ag else "",
            "agnes_orig_post_unknown": ag["counts"]["原题"]["post"]["unknown"] if ag else "",
            "agnes_trans_post_correct": ag["counts"]["迁移题"]["post"]["correct"] if ag else "",
            "agnes_trans_post_incorrect": ag["counts"]["迁移题"]["post"]["incorrect"] if ag else "",
            "agnes_trans_post_unknown": ag["counts"]["迁移题"]["post"]["unknown"] if ag else "",
            "ar_orig_post_correct": ar["counts"]["原题"]["post"]["correct"] if ar else "",
            "ar_trans_post_correct": ar["counts"]["迁移题"]["post"]["correct"] if ar else "",
            "teacher_turns": n_turns,
            "blanked_turns": n_blank,
            "unterminated_think_turns": n_unterm,
            "eoc_used": eoc,
            "is_truncated": 1 if end_reason == "token_budget" else 0,
            "thinking_tokens_mean": f"{t_mean:.4f}",
        }
        csv_rows.append(row_out)

    csv_path = out_root / "selfbuilt_per_problem.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    per_tier = agnes_diag.get("per_tier") or {}
    total_complete = sum(int(d.get("gen_complete", 0)) for d in per_tier.values())
    total_held = sum(int(d.get("held", 0)) for d in per_tier.values())
    total_stu_fail = sum(int(d.get("student_calls_failed", 0)) for d in per_tier.values())
    total_turns = sum(int(d.get("teacher_turns", 0)) for d in per_tier.values())
    total_blanked = sum(int(d.get("blanked_turns", 0)) for d in per_tier.values())
    total_unterm = sum(int(d.get("unterminated_think_turns", 0)) for d in per_tier.values())
    total_eoc = sum(int(d.get("eoc_used", 0)) for d in per_tier.values())
    tok_sum = sum(int(d.get("thinking_tokens_sum", 0)) for d in per_tier.values())
    tok_cnt = sum(int(d.get("thinking_tokens_count", 0)) for d in per_tier.values())
    end_reasons = agnes_diag.get("end_reasons") or {}
    trunc_n = int(end_reasons.get("token_budget", 0))

    agnes_rep_p = out_root / "AGNES" / "AGNES_GRADING_REPORT.json"
    agnes_rep = json.loads(agnes_rep_p.read_text(encoding="utf-8")) if agnes_rep_p.is_file() else {}
    agnes_ok = (not agnes_rep) or (agnes_rep.get("primary_status") == "可用" and not agnes_rep.get("aborted_over_1pct"))

    summary = {
        "status": "SELFBUILT_DONE" if (agnes_ok and total_held == 0 and total_stu_fail == 0) else "SELFBUILT_WARN",
        "rc": 0 if (agnes_ok and total_stu_fail == 0) else 2,
        "selected_entries": len(entries),
        "gen_complete": total_complete,
        "held": total_held,
        "student_calls_failed": total_stu_fail,
        "graded_agnes": len(agnes_records),
        "graded_author_rule": len(ar_records),
        "per_problem_csv": csv_path.name if csv_rows else None,
        "diagnostics": {
            "assume_thinking_values": agnes_diag.get("assume_thinking_values"),
            "teacher_turns": total_turns,
            "blanked_turns": total_blanked,
            "empty_rate": (total_blanked / total_turns) if total_turns else 0.0,
            "unterminated_think_turns": total_unterm,
            "unterminated_think_rate": (total_unterm / total_turns) if total_turns else 0.0,
            "eoc_rate": (total_eoc / total_complete) if total_complete else 0.0,
            "seq_trunc_rate": (trunc_n / total_complete) if total_complete else 0.0,
            "thinking_tokens_avg_per_turn": (tok_sum / tok_cnt) if tok_cnt else 0.0,
            "end_reasons": end_reasons,
        },
        "agnes_report": agnes_rep,
    }
    write_json_atomic(out_root / "selfbuilt_summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="T-033 Route A Step-100 self-built 3-tier runner.")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--teacher-served-name", default="RouteA-step100")
    ap.add_argument("--teacher-url", default="http://127.0.0.1:18711/v1")
    ap.add_argument("--student-url", default="http://127.0.0.1:18712/v1")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".secrets" / "agnes_api_key")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--gate-first-n", type=int, default=0)
    ap.add_argument("--max-est-days", type=float, default=3.0)
    ap.add_argument("--pilot-per-tier", type=int, default=0)
    ap.add_argument("--smoke-total", type=int, default=0, help="If >0, select exact smoke_total sources (e.g. 16)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--run-grading", action="store_true")
    ap.add_argument("--run-author-rule", action="store_true")
    ap.add_argument("--export-summary", action="store_true")
    args = ap.parse_args()

    judge_p = args.repo / "evaluation" / "agnes_eval_v3" / "agnes_judge.py"
    if sha256_file(judge_p) != AGNES_JUDGE_SHA256:
        raise RuntimeError("Frozen agnes_judge.py SHA-256 mismatch")

    loaded = load_and_verify_bundle(args.bundle_dir, require_exact_totals=True)
    if args.smoke_total > 0:
        entries = select_smoke_entries(loaded, smoke_total=args.smoke_total, seed=42)
    else:
        entries = select_pilot_entries(loaded, per_tier=args.pilot_per_tier)
    entries = shard_entries(entries, shard_index=args.shard_index, num_shards=args.num_shards)

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"[t033_selfbuilt] selected_entries={len(entries)} shard={args.shard_index}/{args.num_shards}", flush=True)

    if args.run_generation:
        if not args.teacher_model:
            raise ValueError("--teacher-model required for generation")
        verify_tutor_bfloat16(Path(args.teacher_model))
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
        print(f"[t033_selfbuilt] generation_done {json.dumps(gen_stats, sort_keys=True)}", flush=True)

    if args.run_author_rule:
        ar_stats = grade_author_rule_entries(
            entries,
            out_root=args.out_root,
            repo_root=args.repo,
            grade_math_fn=grade_math_answer,
        )
        print(f"[t033_selfbuilt] author_rule_done {json.dumps(ar_stats, sort_keys=True)}", flush=True)

    if args.run_grading:
        check_agnes_guard(args.workers)
        try:
            grade_stats = grade_agnes_pro5000(
                entries,
                out_root=args.out_root,
                repo_root=args.repo,
                key_file=args.key_file,
                workers=args.workers,
            )
            print(f"[t033_selfbuilt] agnes_grading_done {json.dumps(grade_stats, sort_keys=True, ensure_ascii=False)}", flush=True)
        except AgnesTechnicalFailureLimitError:
            export_selfbuilt_outputs(loaded, entries, out_root=args.out_root, repo_root=args.repo)
            return 2

    if args.export_summary or args.run_grading or args.run_author_rule or args.run_generation:
        summary = export_selfbuilt_outputs(loaded, entries, out_root=args.out_root, repo_root=args.repo)
        agnes_rep = summary.get("agnes_report") or {}
        print(
            format_machine_summary(
                {
                    "status": summary["status"],
                    "rc": summary["rc"],
                    "calls": agnes_rep.get("total_calls", summary["gen_complete"] * 32),
                    "failures": int(agnes_rep.get("technical_failures", 0)) + int(summary["student_calls_failed"]),
                    "gen_complete": summary["gen_complete"],
                    "held": summary["held"],
                    "graded_agnes": summary["graded_agnes"],
                }
            ),
            flush=True,
        )
        return int(summary["rc"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(os.linesep.join([""])), file=sys.stderr)
        raise
