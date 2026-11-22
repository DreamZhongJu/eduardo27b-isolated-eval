#!/usr/bin/env python3
"""Post-only dialogue + student sampling + Agnes/author-rule grading for base-4B eval.

Reuses frozen evaluation/app/{dialogue,offline_eval,protocol,scoring}.py and
unmodified evaluation/agnes_eval_v3/agnes_judge.py (verified by SHA-256).
Skips pre and control sampling because those are reused from the verified 27B records.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ref27b_manifest import DIFFICULTY_ORDER, build_manifest, select_pilot_entries
from summarize_base4b import STAGE_BY_DIFF, extract_safe_failure_summary

AGNES_V3_JUDGE_SHA = "2ffe746444939555b85e3557d2ddc3b61e3cd59d7997263909668876b924c871"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_agnes_judge(judge_path: Path):
    if not judge_path.is_absolute() or judge_path.is_symlink() or not judge_path.is_file():
        raise ValueError("Invalid agnes_judge.py path")
    if _sha256_bytes(judge_path.read_bytes()) != AGNES_V3_JUDGE_SHA:
        raise ValueError("agnes_eval_v3/agnes_judge.py SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("verified_agnes_judge_v3", judge_path)
    if spec is None or spec.loader is None:
        raise ValueError("Failed to load agnes_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_frozen_app(repo_root: Path):
    app_dir = str(repo_root / "evaluation" / "app")
    off_dir = str(repo_root / "evaluation" / "official")
    for p in (off_dir, app_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import dialogue  # noqa: E402
    import offline_eval  # noqa: E402
    import protocol  # noqa: E402
    import scoring  # noqa: E402

    return offline_eval, dialogue, protocol, scoring


async def run_one_post_only(
    row: dict[str, Any],
    *,
    teacher_model_name: str,
    teacher_path: str,
    teacher_url: str,
    student_url: str,
    student_client: Any,
    offline_eval: Any,
    dialogue_mod: Any,
    protocol_mod: Any,
    scoring_mod: Any,
) -> None:
    """Generate dialogue and 32 post-test student samples for one eligible row."""
    source = {"source_id": row["source_id"], "problem": row["problem"], "answer": row["answer"]}
    folder = scoring_mod.source_dir(source)
    folder.mkdir(parents=True, exist_ok=True)
    source_hash = scoring_mod.sha(scoring_mod.stable(row))
    src_json = folder / "SOURCE.json"
    if src_json.exists():
        if scoring_mod.read(src_json)["sha256"] != source_hash:
            raise ValueError("Saved SOURCE.json sha256 mismatch")
    else:
        identity = {"sha256": source_hash, "source_id": row["source_id"]}
        if row.get("transfer_target", "").startswith("ours_"):
            identity.update(
                transfer_target=row["transfer_target"],
                candidate_id=row["transfer_candidate_id"],
                position_by_student_correct=row["transfer_position"],
                historical_student_correct=row["student_correct"],
                historical_student_n_scored=row["student_n_scored"],
                historical_group_score_tie=row["score_tie"],
            )
        scoring_mod.put_once(src_json, identity)

    official_row = protocol_mod._row(
        row["problem"], row["answer"], 0.0, row["transfer_problem"], row["transfer_answer"], 0.0, "test"
    )
    # Exact 27B Tutor and Student rollout hyperparameters (only Tutor model name/path swapped).
    args = SimpleNamespace(
        model=teacher_model_name,
        teacher_tokenizer_path=teacher_path,
        teacher_api_key_env="LOCAL_ONLY_KEY",
        teacher_api_base=teacher_url,
        student_api_key_env="LOCAL_ONLY_KEY",
        student_api_base=student_url,
        student_model=offline_eval.MODEL,
        student_concurrency=1,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        student_temperature=0.6,
        max_tokens_per_turn=4192,
        student_max_tokens=1024,
        max_conversation_tokens=16384,
        max_teacher_turns=11,
        min_teacher_turns_before_eoc=3,
        enable_thinking=True,
        num_samples_per_problem=1,
        assume_thinking="auto",
        seed=42,
        num_retries=0,
    )
    dialogue = await dialogue_mod.make_dialogue(source, official_row, args)

    sem = asyncio.Semaphore(4)
    transfer_target = row.get("transfer_target", "paper")
    targets = [
        ("source", row["problem"], row["answer"]),
        (transfer_target, row["transfer_problem"], row["transfer_answer"]),
    ]
    collected = []
    for name, problem, answer in targets:
        messages = protocol_mod._build_final_attempt_messages(
            dialogue["reward_messages"],
            source["problem"],
            transfer_problem=problem if name != "source" else None,
            omit_teacher_turns=True,
        )
        samples = await asyncio.gather(
            *(offline_eval.student_sample(student_client, source, name, "post", i, messages, sem) for i in range(16))
        )
        collected.extend((name, "post", problem, answer, i, sample) for i, sample in enumerate(samples))

    if len(collected) != 32:
        raise ValueError("Expected 32 post-test samples")

    score_input = [
        {
            "source_id": row["source_id"],
            "target": target,
            "phase": phase,
            "index": index,
            "problem": problem,
            "reference_answer": answer,
            "student_response": sample["text"],
            "finish_reason": sample["finish_reason"],
            "grading_status": "ungraded",
            **(
                {
                    "candidate_id": row["transfer_candidate_id"],
                    "difficulty": transfer_target[5:],
                    "position_by_student_correct": row["transfer_position"],
                    "historical_student_correct": row["student_correct"],
                    "historical_student_n_scored": row["student_n_scored"],
                    "historical_group_score_tie": row["score_tie"],
                }
                if target == transfer_target and transfer_target.startswith("ours_")
                else {}
            ),
        }
        for target, phase, problem, answer, index, sample in collected
    ]

    score_bytes = scoring_mod.stable(score_input)
    score_input_sha = scoring_mod.sha(score_bytes)
    score_path = folder / "SCORE_INPUT.json"
    if score_path.exists():
        if scoring_mod.read(score_path) != score_input:
            raise ValueError("Saved SCORE_INPUT.json differs")
    else:
        scoring_mod.put_once(score_path, score_input)

    raw_complete = folder / "RAW_COMPLETE.json"
    if not raw_complete.exists():
        scoring_mod.put_once(
            raw_complete,
            {
                "source_id": row["source_id"],
                "samples": 32,
                "post_only": True,
                "grading_status": "ungraded",
                "source_sha256": source_hash,
                "score_input_sha256": score_input_sha,
            },
        )


async def generate_entries(
    entries: list[dict[str, Any]],
    *,
    out_root: Path,
    repo_root: Path,
    teacher_model_name: str,
    teacher_path: str,
    teacher_url: str,
    student_url: str,
) -> dict[str, int]:
    """Generate dialogue + post samples for selected eligible entries across tiers."""
    offline_eval, dialogue_mod, protocol_mod, scoring_mod = _load_frozen_app(repo_root)
    os.environ["LOCAL_ONLY_KEY"] = "loopback-only"
    student_url = offline_eval.local_url(student_url)
    teacher_url = offline_eval.local_url(teacher_url)
    student_client = offline_eval.openai.AsyncClient(
        api_key="loopback-only", base_url=student_url, timeout=300, max_retries=0
    )
    stats = {"complete": 0, "skipped_existing": 0, "held": 0}
    try:
        for idx, entry in enumerate(entries, 1):
            stage = STAGE_BY_DIFF[entry["difficulty"]]
            stage_dir = out_root / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            os.environ["RESULTS_DIR"] = str(stage_dir)
            row = entry["row"]
            folder = scoring_mod.source_dir(row)
            if (folder / "RAW_COMPLETE.json").is_file():
                stats["skipped_existing"] += 1
                continue
            if (folder / "SOURCE_HELD.json").is_file():
                stats["held"] += 1
                continue
            try:
                await run_one_post_only(
                    row,
                    teacher_model_name=teacher_model_name,
                    teacher_path=teacher_path,
                    teacher_url=teacher_url,
                    student_url=student_url,
                    student_client=student_client,
                    offline_eval=offline_eval,
                    dialogue_mod=dialogue_mod,
                    protocol_mod=protocol_mod,
                    scoring_mod=scoring_mod,
                )
                stats["complete"] += 1
            except Exception as exc:
                marker = folder / "SOURCE_HELD.json"
                if not marker.exists() and not (folder / "RAW_COMPLETE.json").exists():
                    scoring_mod.put_once(
                        marker,
                        {"source_id": row["source_id"], "status": "held", "error_type": type(exc).__name__},
                    )
                stats["held"] += 1
            if idx % 10 == 0 or idx == len(entries):
                print(
                    f"[base4b] gen_progress={idx}/{len(entries)} complete={stats['complete']} "
                    f"skipped={stats['skipped_existing']} held={stats['held']}",
                    flush=True,
                )
    finally:
        close_fn = getattr(student_client, "close", None)
        if callable(close_fn):
            res = close_fn()
            if inspect.isawaitable(res):
                await res
    return stats


def grade_agnes_entries(
    entries: list[dict[str, Any]],
    *,
    out_root: Path,
    repo_root: Path,
    key_file: Path,
    workers: int = 5,
    caller=None,
) -> dict[str, int]:
    """Grade completed post-only sources using unmodified agnes_eval_v3/agnes_judge.py."""
    if not 1 <= workers <= 5:
        raise ValueError("Agnes workers must be between 1 and 5")
    if key_file.is_symlink() or not key_file.is_file():
        raise ValueError("Invalid Agnes key file")
    key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("Empty Agnes key file")

    judge = load_verified_agnes_judge(repo_root / "evaluation" / "agnes_eval_v3" / "agnes_judge.py")
    locks = out_root / "AGNES" / "locks"
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    window = judge.AdaptiveWindow(workers)
    stats = {"graded_sources": 0, "skipped_existing": 0, "correct": 0, "incorrect": 0, "unknown": 0}

    for idx, entry in enumerate(entries, 1):
        stage = STAGE_BY_DIFF[entry["difficulty"]]
        skey = entry["source_key"]
        src_dir = out_root / stage / "sources" / skey
        marker_path = src_dir / "RAW_COMPLETE.json"
        input_path = src_dir / "SCORE_INPUT.json"
        if not marker_path.is_file() or not input_path.is_file():
            continue
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
            stats["skipped_existing"] += 1
            for r in prev["results"]:
                stats[r["status"]] += 1
            continue

        rec_dir = out_root / "AGNES" / "receipts" / stage / skey
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            if caller is None:
                results = list(
                    pool.map(lambda r: judge.grade_one(r, key, rec_dir, locks, window=window), rows)
                )
            else:
                results = list(
                    pool.map(
                        lambda r: judge.grade_one(r, key, rec_dir, locks, caller=caller, window=window),
                        rows,
                    )
                )

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
        for r in results:
            stats[r["status"]] += 1
        if idx % 10 == 0 or idx == len(entries):
            print(
                f"[base4b] grade_progress={idx}/{len(entries)} graded={stats['graded_sources']} "
                f"correct={stats['correct']} incorrect={stats['incorrect']} unknown={stats['unknown']}",
                flush=True,
            )
    return stats


def grade_author_rule_entries(
    entries: list[dict[str, Any]],
    *,
    out_root: Path,
    repo_root: Path,
    grade_math_fn=None,
) -> dict[str, int]:
    """Offline sensitivity grading using author's boxed + math_verify rule (no API calls)."""
    if grade_math_fn is None:
        # Import grade_math_answer directly if available, or fall back to protocol boxed extraction.
        _, _, protocol_mod, _ = _load_frozen_app(repo_root)

        def _fallback_grade(sol: str, gold: str) -> dict[str, Any]:
            boxed = protocol_mod.last_boxed_only_string(sol[-200:]) or protocol_mod.last_boxed_only_string(sol)
            pred = protocol_mod._remove_boxed(boxed) if boxed else ""
            if pred == "":
                return {"acc": 0.0}
            if pred.strip() == gold.strip():
                return {"acc": 1.0}
            try:
                from math_verify import parse as mv_parse, verify as mv_verify

                return {"acc": 1.0 if bool(mv_verify(mv_parse(gold), mv_parse(pred))) else 0.0}
            except Exception:
                return {"acc": 0.0}

        grade_math_fn = _fallback_grade

    stats = {"graded_sources": 0, "correct": 0, "incorrect": 0}
    for entry in entries:
        stage = STAGE_BY_DIFF[entry["difficulty"]]
        skey = entry["source_key"]
        input_path = out_root / stage / "sources" / skey / "SCORE_INPUT.json"
        if not input_path.is_file():
            continue
        raw = input_path.read_bytes()
        rows = json.loads(raw)
        out_dir = out_root / "AUTHOR_RULE" / stage / "sources" / skey
        out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        out_path = out_dir / "RESULTS.json"
        if out_path.exists():
            continue
        results = []
        for r in rows:
            verdict = grade_math_fn(r["student_response"], r["reference_answer"])
            status = "correct" if float(verdict.get("acc", 0.0)) >= 0.5 else "incorrect"
            stats[status] += 1
            results.append(
                {
                    "identity": {
                        "source_id": r["source_id"],
                        "target": r["target"],
                        "phase": r["phase"],
                        "index": r["index"],
                    },
                    "status": status,
                }
            )
        doc = {"source_id": entry["source_id"], "input_sha256": _sha256_bytes(raw), "results": results}
        out_path.write_text(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        stats["graded_sources"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Run base-4B post-only self-built 3-tier evaluation.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref-root", type=Path, default=Path("/workspace/results/own-test-499-v1"))
    ap.add_argument("--overlay", type=Path, default=Path("/workspace/results/agnes-topbool-recovery-v1"))
    ap.add_argument("--decisions", type=Path, default=Path("/workspace/results/agnes-rejudge-v1"))
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--teacher-served-name", default="Qwen3.5-4B")
    ap.add_argument("--teacher-url", default="http://127.0.0.1:18701/v1")
    ap.add_argument("--student-url", default="http://127.0.0.1:18702/v1")
    ap.add_argument("--key-file", type=Path, default=Path("/workspace/.secrets/agnes_api_key"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--pilot-per-tier", type=int, default=3, help="0 means all 1,489 eligible sources")
    ap.add_argument("--check-manifest", action="store_true")
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--run-grading", action="store_true")
    ap.add_argument("--run-author-rule", action="store_true")
    args = ap.parse_args()

    manifest = build_manifest(
        args.data,
        args.ref_root,
        args.overlay,
        args.decisions,
        repo_root=args.repo,
        require_exact_totals=True,
        show_progress=True,
    )
    print(
        f"[base4b] manifest_ok eligible={manifest['eligible_count']} skipped={manifest['skipped_count']} "
        f"tiers={manifest['per_tier_counts']}",
        flush=True,
    )
    if args.check_manifest:
        return 0

    if args.out_root is None:
        raise ValueError("--out-root is required")
    args.out_root.mkdir(parents=True, exist_ok=True)
    entries = select_pilot_entries(manifest, per_tier=args.pilot_per_tier)
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
            )
        )
        print(f"[base4b] generation_done {json.dumps(gen_stats, sort_keys=True)}", flush=True)

    if args.run_grading:
        grade_stats = grade_agnes_entries(
            entries, out_root=args.out_root, repo_root=args.repo, key_file=args.key_file, workers=args.workers
        )
        print(f"[base4b] agnes_grading_done {json.dumps(grade_stats, sort_keys=True)}", flush=True)

    if args.run_author_rule:
        ar_stats = grade_author_rule_entries(entries, out_root=args.out_root, repo_root=args.repo)
        print(f"[base4b] author_rule_done {json.dumps(ar_stats, sort_keys=True)}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
