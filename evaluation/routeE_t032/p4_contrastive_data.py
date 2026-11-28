#!/usr/bin/env python3
"""P4 Contrastive Pair Builder & Contamination Guard.

Builds 300-1,000 contrastive reply pairs from A-group step_100 dialogues on the
author's official TRAIN split (4,712 problems; NEVER touches val.parquet or
test.parquet for fitting):
- Picks non-empty teacher replies strictly before student's first correct turn.
- Prompts local Qwen3.6-27B (enable_thinking=False) to rewrite the SAME reply into:
  (1) good_reply: reworded Socratic hint without leakage;
  (2) bad_reply: reworded reply with direct answer / key step takeover.
- Filters strictly to +/-15% token-length difference (retries up to 4x, else drops).
- Splits 80/20 into fit / heldout grouped by source_problem_hash (zero overlap).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from common_utils import (
    AUTHOR_DATASET_FILES,
    AUTHOR_DATASET_REV,
    TEACHER_SYSTEM_PROMPT,
    atomic_write_json,
    atomic_write_jsonl,
    download_verified_file,
    print_machine_summary,
    sha256_text,
    strip_teacher_controls,
)
from leakage_judges import (
    AsyncJudgeClient,
    Layer1RuleScreener,
    _strip_json_fences,
    find_first_student_correct_turn,
)


class ContaminationError(RuntimeError):
    """Raised when train problems overlap with val or test splits."""


REWRITE_PAIR_PROMPT = (
    "You are constructing controlled contrastive pairs for tutoring research.\n"
    "Given the math problem, its ground-truth answer, the dialogue context so far, "
    "and an original tutor reply, rewrite the SAME tutor reply into TWO versions "
    "that keep the same conversational tone and VERY SIMILAR LENGTH (within +/-15% tokens):\n"
    "1. `good_reply`: Keep it as helpful Socratic guidance (question or small hint) "
    "WITHOUT stating the final answer or solving the key steps for the student.\n"
    "2. `bad_reply`: Modify the reply so it DOES leak the solution — either directly "
    "stating the final answer or writing out the final 1-2 key equations/steps so the "
    "student can copy the answer.\n\n"
    "Problem: {problem}\n"
    "Correct Answer: {answer}\n"
    "==CONTEXT SO FAR==\n{context}\n"
    "==ORIGINAL TUTOR REPLY==\n{orig_reply}\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"good_reply": "...", "bad_reply": "..."}}'
)


def ensure_author_train_val_parquets(
    data_dir: Path,
    *,
    num_train: int = 4712,
    num_val: int = 100,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Ensure train.parquet and val.parquet exist from official EduardoRL data (never test.parquet)."""
    data_dir = Path(data_dir)
    train_pq = data_dir / "train.parquet"
    val_pq = data_dir / "val.parquet"
    if train_pq.is_file() and val_pq.is_file():
        return train_pq, val_pq

    import pandas as pd
    import pyarrow.parquet as pq

    raw_dir = data_dir / "official_raw"
    raw_train = download_verified_file(
        [
            f"https://hf-mirror.com/datasets/dmacjam/eduardoRL-transfer-dataset/resolve/{AUTHOR_DATASET_REV}/data/train-00000-of-00001.parquet"
        ],
        raw_dir / "train-00000-of-00001.parquet",
        algo="sha256",
        expected_hash=AUTHOR_DATASET_FILES["data/train-00000-of-00001.parquet"],
    )
    raw_test = download_verified_file(
        [
            f"https://hf-mirror.com/datasets/dmacjam/eduardoRL-transfer-dataset/resolve/{AUTHOR_DATASET_REV}/data/test-00000-of-00001.parquet"
        ],
        raw_dir / "test-00000-of-00001.parquet",
        algo="sha256",
        expected_hash=AUTHOR_DATASET_FILES["data/test-00000-of-00001.parquet"],
    )
    test_keys = {
        (str(r.get("problem", "")), str(r.get("answer", "")))
        for r in pq.read_table(raw_test).to_pylist()
    }
    rows = pq.read_table(raw_train).to_pylist()

    filtered = []
    for x in rows:
        tp = x.get("transfer_problem")
        r1 = x.get("pre_solve_rate")
        r2 = x.get("pre_solve_rate_transfer_problem")
        if not tp or r1 is None or r2 is None:
            continue
        if not (0.02 <= float(r1) <= 0.65 and 0.02 <= float(r2) <= 0.65):
            continue
        if (str(x.get("problem", "")), str(x.get("answer", ""))) in test_keys:
            continue
        filtered.append(x)

    rng = random.Random(seed)
    indices = list(range(len(filtered)))
    rng.shuffle(indices)
    shuffled = [filtered[i] for i in indices]
    val_rows = shuffled[:num_val]
    train_rows = shuffled[num_val : num_val + num_train]

    def _to_verl_row(x: dict[str, Any], split: str) -> dict[str, Any]:
        prob = str(x["problem"])
        ans = str(x["answer"])
        return {
            "data_source": "bigmath",
            "prompt": [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT.format(problem=prob)},
                {"role": "user", "content": f"This is the problem I am trying to understand: {prob}"},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ans},
            "extra_info": {
                "problem": prob,
                "answer": ans,
                "split": split,
                "solve_rate": float(x["pre_solve_rate"]),
                "pre_solve_rate": float(x["pre_solve_rate"]),
                "transfer_problem": str(x["transfer_problem"]),
                "transfer_answer": str(x["transfer_answer"]),
                "pre_solve_rate_transfer_problem": float(x["pre_solve_rate_transfer_problem"]),
                "interaction_kwargs": {
                    "name": "eduardo",
                    "problem": prob,
                    "answer": ans,
                    "ground_truth": ans,
                },
            },
            "agent_name": "eduardo_agent",
        }

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_to_verl_row(r, "val") for r in val_rows]).to_parquet(val_pq, index=False)
    pd.DataFrame([_to_verl_row(r, "train") for r in train_rows]).to_parquet(train_pq, index=False)
    return train_pq, val_pq


def _extract_problem_hashes(parquet_path: Path) -> set[str]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    out: set[str] = set()
    for r in rows:
        extra = r.get("extra_info") or {}
        prob = str(extra.get("problem") or r.get("problem") or "").strip()
        if prob:
            out.add(sha256_text(prob))
    return out


def assert_zero_split_overlap(
    train_problems: Sequence[str],
    *,
    val_parquet: Path | None = None,
    test_parquet: Path | None = None,
) -> None:
    train_hashes = {sha256_text(p.strip()) for p in train_problems if p.strip()}
    if val_parquet and Path(val_parquet).is_file():
        val_hashes = _extract_problem_hashes(Path(val_parquet))
        overlap = train_hashes & val_hashes
        if overlap:
            raise ContaminationError(f"Train overlaps val.parquet on {len(overlap)} problem(s)")
    if test_parquet and Path(test_parquet).is_file():
        test_hashes = _extract_problem_hashes(Path(test_parquet))
        overlap = train_hashes & test_hashes
        if overlap:
            raise ContaminationError(f"Train overlaps test.parquet on {len(overlap)} problem(s)")


def default_token_counter(text: str, tokenizer: Any | None = None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len([tok for tok in text.strip().split() if tok])


def length_diff_within_tolerance(
    good_text: str,
    bad_text: str,
    *,
    max_rel_diff: float = 0.15,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[bool, int, int, float]:
    counter = token_counter or default_token_counter
    lg = counter(good_text)
    lb = counter(bad_text)
    if lg <= 0 or lb <= 0:
        return False, lg, lb, 1.0
    rel = abs(lg - lb) / min(lg, lb)
    rel_max = abs(lg - lb) / max(lg, lb)
    # Require both symmetric ratios to stay <= max_rel_diff
    ok = (rel <= max_rel_diff + 1e-12) and (rel_max <= max_rel_diff + 1e-12)
    return ok, lg, lb, rel


def extract_candidate_teacher_turns(
    conversations: Sequence[dict[str, Any]],
    *,
    max_turns_per_conv: int = 3,
    require_math_verify: bool = False,
) -> list[dict[str, Any]]:
    """Extract non-empty teacher turns strictly before student's first correct turn."""
    candidates: list[dict[str, Any]] = []
    for conv_idx, rec in enumerate(conversations):
        problem = str(rec.get("problem", ""))
        answer = str(rec.get("answer", ""))
        assume = bool(rec.get("assume_thinking", True))
        raw_msgs = rec.get("raw_messages") or rec.get("reward_messages") or rec.get("messages") or []
        already_vis = not ("raw_messages" in rec or "reward_messages" in rec)

        first_correct = find_first_student_correct_turn(
            raw_msgs, answer, require_math_verify=require_math_verify
        )
        context_msgs: list[dict[str, str]] = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT.format(problem=problem)}
        ]
        taken = 0
        teacher_turn_num = 0
        for msg_idx, msg in enumerate(raw_msgs):
            role = msg.get("role")
            if role == "system":
                continue
            raw_content = str(msg.get("content") or "")
            if role in ("teacher", "assistant"):
                teacher_turn_num += 1
                if first_correct is not None and msg_idx >= first_correct:
                    break
                vis = (
                    raw_content.replace("<end_of_conversation>", "").strip()
                    if already_vis
                    else strip_teacher_controls(raw_content, assume_thinking=assume)
                )
                if vis and len(vis) >= 12 and taken < max_turns_per_conv:
                    candidates.append(
                        {
                            "pair_seed_id": f"c{conv_idx:04d}_t{teacher_turn_num:02d}",
                            "source_problem_hash": sha256_text(problem.strip()),
                            "problem": problem,
                            "answer": answer,
                            "teacher_turn_idx": teacher_turn_num,
                            "context_messages": [dict(m) for m in context_msgs],
                            "orig_reply": vis,
                        }
                    )
                    taken += 1
                context_msgs.append({"role": "assistant", "content": vis or ""})
            else:
                context_msgs.append({"role": "user", "content": raw_content.strip()})
    return candidates


def _format_context_for_rewriter(context_messages: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    for m in context_messages:
        if m["role"] == "system":
            continue
        lbl = "Tutor" if m["role"] == "assistant" else "Student"
        lines.append(f"- {lbl}: {m['content']}")
    return "\n".join(lines)


def _parse_rewrite_json(raw_text: str) -> tuple[str, str] | None:
    cleaned = _strip_json_fences(raw_text)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end])
    except Exception:
        return None
    good = str(data.get("good_reply") or "").strip()
    bad = str(data.get("bad_reply") or "").strip()
    if not good or not bad:
        return None
    return good, bad


def split_pairs_by_source_problem(
    pairs: Sequence[dict[str, Any]],
    *,
    fit_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs 80/20 grouped strictly by source_problem_hash."""
    groups = sorted({p["source_problem_hash"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_fit = max(1, min(len(groups) - 1, int(round(len(groups) * fit_ratio)))) if len(groups) >= 2 else len(groups)
    fit_hashes = set(groups[:n_fit])
    fit_pairs = [p for p in pairs if p["source_problem_hash"] in fit_hashes]
    held_pairs = [p for p in pairs if p["source_problem_hash"] not in fit_hashes]

    overlap = {p["source_problem_hash"] for p in fit_pairs} & {
        p["source_problem_hash"] for p in held_pairs
    }
    if overlap:
        raise AssertionError(f"Group split leaked {len(overlap)} problem hashes")
    return fit_pairs, held_pairs


async def build_contrastive_pairs(
    *,
    conversations: Sequence[dict[str, Any]],
    rewriter_client: AsyncJudgeClient,
    val_parquet: Path | None = None,
    test_parquet: Path | None = None,
    tokenizer: Any | None = None,
    target_pairs: int = 600,
    min_pairs: int = 300,
    max_pairs: int = 1000,
    max_rel_length_diff: float = 0.15,
    max_retries_per_candidate: int = 4,
    require_math_verify: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    assert_zero_split_overlap(
        [str(c.get("problem", "")) for c in conversations],
        val_parquet=val_parquet,
        test_parquet=test_parquet,
    )
    candidates = extract_candidate_teacher_turns(
        conversations, max_turns_per_conv=3, require_math_verify=require_math_verify
    )
    rng = random.Random(seed)
    rng.shuffle(candidates)

    screener = Layer1RuleScreener(require_math_verify=require_math_verify)
    tok_counter = (lambda s: default_token_counter(s, tokenizer=tokenizer))

    accepted: list[dict[str, Any]] = []
    stats = {
        "candidates_total": len(candidates),
        "rewrite_calls": 0,
        "rewrite_failures": 0,
        "dropped_parse": 0,
        "dropped_length": 0,
        "dropped_leak_check": 0,
    }

    async def process_candidate(cand: dict[str, Any]) -> dict[str, Any] | None:
        ctx_str = _format_context_for_rewriter(cand["context_messages"])
        prompt = REWRITE_PAIR_PROMPT.format(
            problem=cand["problem"],
            answer=cand["answer"],
            context=ctx_str,
            orig_reply=cand["orig_reply"],
        )
        for _ in range(max_retries_per_candidate):
            stats["rewrite_calls"] += 1
            try:
                raw = await rewriter_client.complete(prompt)
            except Exception:
                stats["rewrite_failures"] += 1
                continue
            parsed = _parse_rewrite_json(raw)
            if parsed is None:
                stats["dropped_parse"] += 1
                continue
            good_reply, bad_reply = parsed
            len_ok, lg, lb, rel = length_diff_within_tolerance(
                good_reply,
                bad_reply,
                max_rel_diff=max_rel_length_diff,
                token_counter=tok_counter,
            )
            if not len_ok:
                stats["dropped_length"] += 1
                continue
            # Check good_reply does not trigger Layer 1 leak
            good_msgs = cand["context_messages"] + [{"role": "assistant", "content": good_reply}]
            g_l1 = screener.screen(
                good_msgs,
                problem=cand["problem"],
                answer=cand["answer"],
                already_visible=True,
            )
            if g_l1["leaked"] == 1:
                stats["dropped_leak_check"] += 1
                continue
            return {
                "pair_id": cand["pair_seed_id"],
                "source_problem_hash": cand["source_problem_hash"],
                "problem": cand["problem"],
                "answer": cand["answer"],
                "context_messages": cand["context_messages"],
                "orig_reply": cand["orig_reply"],
                "good_reply": good_reply,
                "bad_reply": bad_reply,
                "good_tokens": lg,
                "bad_tokens": lb,
                "rel_length_diff": rel,
            }
        return None

    cap = min(max_pairs, target_pairs)
    batch_size = 32
    for start in range(0, len(candidates), batch_size):
        if len(accepted) >= cap:
            break
        batch = candidates[start : start + batch_size]
        res_list = await asyncio.gather(*[process_candidate(c) for c in batch])
        for item in res_list:
            if item is not None:
                accepted.append(item)
                if len(accepted) >= cap:
                    break

    if len(accepted) < min_pairs:
        raise RuntimeError(
            f"Only collected {len(accepted)} valid contrastive pairs (< min_pairs={min_pairs})"
        )

    fit_pairs, heldout_pairs = split_pairs_by_source_problem(accepted, fit_ratio=0.8, seed=seed)
    return {
        "fit": fit_pairs,
        "heldout": heldout_pairs,
        "all": accepted,
        "stats": {
            **stats,
            "accepted_total": len(accepted),
            "fit_count": len(fit_pairs),
            "heldout_count": len(heldout_pairs),
            "fit_unique_problems": len({p["source_problem_hash"] for p in fit_pairs}),
            "heldout_unique_problems": len({p["source_problem_hash"] for p in heldout_pairs}),
        },
    }


def _load_or_rollout_train_conversations(
    *,
    train_parquet: Path,
    conversations_jsonl: Path | None,
    tutor_model: str | None,
    student_api_base: str | None,
    student_model: str,
    num_train_problems: int,
    seed: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    if conversations_jsonl and Path(conversations_jsonl).is_file():
        with Path(conversations_jsonl).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if not tutor_model or not student_api_base:
        raise ValueError(
            "Either --train-conversations or (--tutor-model and --student-api-base) must be provided"
        )

    # Rollout on a fixed-seed subset of train.parquet using unmodified EduardoRL LocalVLLMTeacher
    import pyarrow.parquet as pq
    from p4_eval_val100 import rollout_conversations_vllm

    rows = pq.read_table(train_parquet).to_pylist()
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    chosen = [rows[i] for i in idxs[:num_train_problems]]

    rollout_out = out_dir / "train_rollouts_agroup.jsonl"
    records = asyncio.run(
        rollout_conversations_vllm(
            rows=chosen,
            tutor_model=tutor_model,
            student_api_base=student_api_base,
            student_model=student_model,
            seed=seed,
        )
    )
    atomic_write_jsonl(rollout_out, records, mode=0o600)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 contrastive pair builder on author train split")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rewriter-api-base", required=True)
    ap.add_argument("--rewriter-model", default="Qwen3.6-27B")
    ap.add_argument("--rewriter-concurrency", type=int, default=16)
    ap.add_argument("--train-conversations", default=None)
    ap.add_argument("--tutor-model", default=None)
    ap.add_argument("--student-api-base", default=None)
    ap.add_argument("--student-model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--num-train-problems", type=int, default=260)
    ap.add_argument("--target-pairs", type=int, default=600)
    ap.add_argument("--min-pairs", type=int, default=300)
    ap.add_argument("--max-pairs", type=int, default=1000)
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--test-parquet-guard", default=None, help="Optional path to test.parquet hash-guard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-math-verify", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pq, val_pq = ensure_author_train_val_parquets(Path(args.data_dir), seed=args.seed)

    convs = _load_or_rollout_train_conversations(
        train_parquet=train_pq,
        conversations_jsonl=Path(args.train_conversations) if args.train_conversations else None,
        tutor_model=args.tutor_model,
        student_api_base=args.student_api_base,
        student_model=args.student_model,
        num_train_problems=args.num_train_problems,
        seed=args.seed,
        out_dir=out_dir,
    )

    tokenizer = None
    if args.tokenizer_path or args.tutor_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.tutor_model, trust_remote_code=True)

    rewriter = AsyncJudgeClient(
        api_base=args.rewriter_api_base,
        model=args.rewriter_model,
        concurrency=args.rewriter_concurrency,
        temperature=0.4,
        max_tokens=1536,
    )
    res = asyncio.run(
        build_contrastive_pairs(
            conversations=convs,
            rewriter_client=rewriter,
            val_parquet=val_pq,
            test_parquet=Path(args.test_parquet_guard) if args.test_parquet_guard else None,
            tokenizer=tokenizer,
            target_pairs=args.target_pairs,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
            require_math_verify=args.require_math_verify,
            seed=args.seed,
        )
    )

    atomic_write_jsonl(out_dir / "contrastive_fit.jsonl", res["fit"], mode=0o600)
    atomic_write_jsonl(out_dir / "contrastive_heldout.jsonl", res["heldout"], mode=0o600)
    atomic_write_json(out_dir / "contrastive_stats.json", res["stats"])

    st = res["stats"]
    print_machine_summary(
        {
            "status": "P4_CONTRASTIVE_DONE",
            "rc": 0,
            "calls": st["rewrite_calls"],
            "failures": st["rewrite_failures"],
            "accepted_pairs": st["accepted_total"],
            "fit_pairs": st["fit_count"],
            "heldout_pairs": st["heldout_count"],
        }
    )


if __name__ == "__main__":
    main()
