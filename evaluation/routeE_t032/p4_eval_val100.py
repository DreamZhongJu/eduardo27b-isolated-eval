#!/usr/bin/env python3
"""P4 E0 Evaluation on author validation split (val.parquet, 100 problems).

Strictly NEVER touches test.parquet (494 problems).
Evaluates unmodified bases + all 6 E0 methods on each base:
- Calibrated leak rate (`combined_4layer`) & relative drop vs same unmodified base
- Empty reply rate (`empty_turn_rate_conv_mean`)
- `delta_transfer` & `delta_same` + paired diff vs same unmodified base (10,000 bootstrap CIs)
- `MATH500` accuracy & pp delta vs same unmodified base

Outputs objective numeric comparison table ONLY (no pass/fail verdict — Claude decides).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from common_utils import (
    STUDENT_FINAL_PROMPT,
    STUDENT_FINAL_TRANSFER_PROMPT,
    STUDENT_OPENING_TEMPLATE,
    STUDENT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    atomic_write_json,
    bootstrap_mean_ci,
    check_grading_packages,
    format_ci,
    grade_math_response,
    normalized_gain,
    paired_bootstrap_diff_ci,
    print_machine_summary,
    strip_teacher_controls,
)
from leakage_judges import AsyncJudgeClient, JudgeCache, Layer1RuleScreener, evaluate_combined_4layer
from p1_math_checkup import VLLMBatchGenerator, load_benchmark_items, score_benchmark_items
from p4_e0_methods import ModificationController, export_merged_bf16_for_vllm


def assert_val_parquet_safe(val_parquet: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    val_parquet = Path(val_parquet)
    if "test.parquet" in val_parquet.name:
        raise RuntimeError(f"Forbidden test.parquet passed to P4 val evaluation: {val_parquet}")
    rows = pq.read_table(val_parquet).to_pylist()
    if len(rows) > 150:
        raise RuntimeError(f"Expected ~100 val rows, got {len(rows)} in {val_parquet}")
    return rows


def _render_teacher_messages(problem: str, messages: Sequence[dict[str, str]], prompt_suffix: str | None = None) -> list[dict[str, str]]:
    sys_txt = TEACHER_SYSTEM_PROMPT.format(problem=problem)
    if prompt_suffix:
        head, tail = sys_txt.split("\nProblem:", 1)
        sys_txt = f"{head}\n{prompt_suffix}\nProblem:{tail}"
    out = [{"role": "system", "content": sys_txt}]
    for m in messages:
        role = "assistant" if m["role"] == "teacher" else "user"
        out.append({"role": role, "content": m["content"]})
    return out


def _render_student_messages(problem: str, messages: Sequence[dict[str, str]], assume_thinking: bool = True) -> list[dict[str, str]]:
    out = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT.format(problem=problem)}]
    for m in messages:
        if m["role"] == "teacher":
            out.append({"role": "user", "content": strip_teacher_controls(m["content"], assume_thinking=assume_thinking)})
        else:
            out.append({"role": "assistant", "content": m["content"]})
    return out


def _build_masked_final_messages(
    problem: str,
    messages: Sequence[dict[str, str]],
    transfer_problem: str | None = None,
) -> list[dict[str, str]]:
    out = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT.format(problem=problem)}]
    for m in messages:
        if m["role"] == "teacher":
            out.append({"role": "user", "content": "(hidden)"})
        else:
            out.append({"role": "assistant", "content": m["content"]})
    if transfer_problem:
        final_p = STUDENT_FINAL_TRANSFER_PROMPT.replace("{transfer_problem}", transfer_problem)
    else:
        final_p = STUDENT_FINAL_PROMPT
    out.append({"role": "user", "content": final_p})
    return out


async def rollout_conversations_vllm(
    *,
    rows: Sequence[dict[str, Any]],
    tutor_model: str,
    student_api_base: str,
    student_model: str,
    prompt_suffix: str | None = None,
    max_teacher_turns: int = 11,
    min_teacher_turns_before_eoc: int = 3,
    max_tokens_per_turn: int = 4192,
    max_conversation_tokens: int = 16384,
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
) -> list[dict[str, Any]]:
    """Multi-turn rollout via vLLM (bf16) + student OpenAI server."""
    import openai
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(tutor_model, trust_remote_code=True)
    llm = LLM(
        model=tutor_model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=20000,
        trust_remote_code=True,
        seed=seed,
    )
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tokens_per_turn)
    student_client = openai.AsyncClient(api_key="EMPTY", base_url=student_api_base, timeout=300.0, max_retries=2)
    sem = asyncio.Semaphore(64)

    convs: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        extra = row["extra_info"]
        prob = str(extra["problem"])
        convs.append(
            {
                "row_index": idx,
                "problem": prob,
                "answer": str(extra["answer"]),
                "transfer_problem": str(extra.get("transfer_problem") or ""),
                "transfer_answer": str(extra.get("transfer_answer") or ""),
                "pre_solve_rate": float(extra.get("pre_solve_rate") or 0.0),
                "pre_solve_rate_transfer": float(
                    extra.get("pre_solve_rate_transfer_problem")
                    if extra.get("pre_solve_rate_transfer_problem") is not None
                    else extra.get("pre_solve_rate") or 0.0
                ),
                "messages": [{"role": "student", "content": STUDENT_OPENING_TEMPLATE.format(problem=prob)}],
                "done": False,
                "end_reason": None,
                "assume_thinking": True,
            }
        )

    async def student_turn(c: dict[str, Any]) -> None:
        async with sem:
            s_msgs = _render_student_messages(c["problem"], c["messages"], assume_thinking=True)
            resp = await student_client.chat.completions.create(
                model=student_model, messages=s_msgs, temperature=0.6, max_tokens=500
            )
            c["messages"].append({"role": "student", "content": resp.choices[0].message.content or ""})

    try:
        for _turn in range(max_teacher_turns):
            active = [c for c in convs if not c["done"]]
            if not active:
                break
            prompts = []
            for c in active:
                t_msgs = _render_teacher_messages(c["problem"], c["messages"], prompt_suffix=prompt_suffix)
                try:
                    prompts.append(
                        tok.apply_chat_template(
                            t_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                        )
                    )
                except TypeError:
                    prompts.append(
                        tok.apply_chat_template(t_msgs, tokenize=False, add_generation_prompt=True)
                    )
            outs = llm.generate(prompts, sp)
            for c, out in zip(active, outs):
                txt = out.outputs[0].text.strip()
                c["messages"].append({"role": "teacher", "content": txt})
                n_t = sum(1 for m in c["messages"] if m["role"] == "teacher")
                total_tok = sum(len(tok.encode(m["content"])) for m in c["messages"])
                if not txt:
                    c["done"], c["end_reason"] = True, "teacher_failed"
                elif "<end_of_conversation>" in txt and (n_t - 1) >= min_teacher_turns_before_eoc:
                    c["done"], c["end_reason"] = True, "end_of_conversation"
                elif n_t >= max_teacher_turns:
                    c["done"], c["end_reason"] = True, "max_teacher_turns"
                elif total_tok > max_conversation_tokens:
                    c["done"], c["end_reason"] = True, "token_budget"

            active2 = [c for c in convs if not c["done"]]
            if not active2:
                break
            await asyncio.gather(*(student_turn(c) for c in active2))
            for c in active2:
                if sum(len(tok.encode(m["content"])) for m in c["messages"]) > max_conversation_tokens:
                    c["done"], c["end_reason"] = True, "token_budget"
    finally:
        await student_client.close()
        del llm
    return convs


async def score_val_conversations(
    *,
    convs: Sequence[dict[str, Any]],
    student_api_base: str,
    student_model: str,
    judge_client: AsyncJudgeClient,
    num_student_attempts: int = 16,
    require_math_verify: bool = False,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    import openai

    student_client = openai.AsyncClient(api_key="EMPTY", base_url=student_api_base, timeout=300.0, max_retries=2)
    s_sem = asyncio.Semaphore(64)
    screener = Layer1RuleScreener(require_math_verify=require_math_verify)
    cache = JudgeCache(cache_path)

    async def post_attempts(msgs: list[dict[str, str]], gold: str) -> tuple[float, int]:
        async def one() -> int | None:
            async with s_sem:
                try:
                    resp = await student_client.chat.completions.create(
                        model=student_model, messages=msgs, temperature=0.6, max_tokens=800
                    )
                    txt = resp.choices[0].message.content or ""
                    return grade_math_response(
                        txt, gold, mode="boxed", assume_thinking=False, require_math_verify=require_math_verify
                    )["correct"]
                except Exception:
                    return None

        vals = await asyncio.gather(*(one() for _ in range(num_student_attempts)))
        ok_vals = [v for v in vals if v is not None]
        n_fail = len(vals) - len(ok_vals)
        return (sum(ok_vals) / len(ok_vals)) if ok_vals else 0.0, n_fail

    async def score_one(c: dict[str, Any]) -> dict[str, Any]:
        msgs = c["messages"]
        prob = c["problem"]
        ans = c["answer"]
        t_prob = c.get("transfer_problem") or ""
        t_ans = c.get("transfer_answer") or ""
        use_t = bool(t_prob and t_ans)

        t_msgs = _build_masked_final_messages(prob, msgs, transfer_problem=t_prob if use_t else None)
        s_msgs = _build_masked_final_messages(prob, msgs, transfer_problem=None)

        (post_t, fail_t), (post_s, fail_s), jres = await asyncio.gather(
            post_attempts(t_msgs, t_ans if use_t else ans),
            post_attempts(s_msgs, ans),
            evaluate_combined_4layer(
                screener=screener,
                judge_client=judge_client,
                messages=msgs,
                problem=prob,
                answer=ans,
                assume_thinking=bool(c.get("assume_thinking", True)),
                cache=cache,
            ),
        )
        pre_t = float(c["pre_solve_rate_transfer"] if use_t else c["pre_solve_rate"])
        pre_s = float(c["pre_solve_rate"])
        teacher_turns = [m for m in msgs if m["role"] == "teacher"]
        blanked = sum(
            1 for m in teacher_turns if not strip_teacher_controls(m["content"], assume_thinking=True)
        )
        return {
            "row_index": int(c["row_index"]),
            "delta_transfer": normalized_gain(pre_t, post_t),
            "delta_same": normalized_gain(pre_s, post_s),
            "leaked": int(jres["leaked"]),
            "empty_turn_rate": (blanked / len(teacher_turns)) if teacher_turns else 0.0,
            "student_failed": fail_t + fail_s,
            "judge_calls": jres["calls"],
            "judge_failures": jres["failures"],
        }

    try:
        return list(await asyncio.gather(*(score_one(c) for c in convs)))
    finally:
        await student_client.close()


def summarize_against_base(
    *,
    arm_name: str,
    base_label: str,
    arm_items: Sequence[dict[str, Any]],
    base_items: Sequence[dict[str, Any]],
    arm_math500_vec: Sequence[int],
    base_math500_vec: Sequence[int],
    engine: str = "vllm",
    paired_against: str | None = None,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute exact Criterion #1-#4 numbers paired against the same-engine unmodified base."""
    dt = paired_bootstrap_diff_ci(
        [r["delta_transfer"] for r in base_items],
        [r["delta_transfer"] for r in arm_items],
        draws=draws,
        seed=seed,
    )
    ds = paired_bootstrap_diff_ci(
        [r["delta_same"] for r in base_items],
        [r["delta_same"] for r in arm_items],
        draws=draws,
        seed=seed,
    )
    leak_arm = bootstrap_mean_ci([r["leaked"] for r in arm_items], draws=draws, seed=seed)
    leak_base = bootstrap_mean_ci([r["leaked"] for r in base_items], draws=draws, seed=seed)
    empty_arm = bootstrap_mean_ci([r["empty_turn_rate"] for r in arm_items], draws=draws, seed=seed)
    m500 = paired_bootstrap_diff_ci(base_math500_vec, arm_math500_vec, draws=draws, seed=seed)

    rel_leak_drop = (
        (leak_base["mean"] - leak_arm["mean"]) / leak_base["mean"]
        if leak_base["mean"] > 0
        else 0.0
    )
    return {
        "arm": arm_name,
        "base_label": base_label,
        "engine": engine,
        "paired_against": paired_against or f"unmodified_{engine}",
        "n_val": len(arm_items),
        "leak_rate": leak_arm["mean"],
        "leak_rate_ci": [leak_arm["ci_low"], leak_arm["ci_high"]],
        "leak_rel_drop_vs_base": rel_leak_drop,
        "empty_turn_rate": empty_arm["mean"],
        "empty_turn_rate_ci": [empty_arm["ci_low"], empty_arm["ci_high"]],
        "delta_transfer": dt["target_mean"],
        "delta_transfer_ci": [dt["target_ci_low"], dt["target_ci_high"]],
        "delta_transfer_paired_diff": dt["diff_mean"],
        "delta_transfer_paired_diff_ci": [dt["diff_ci_low"], dt["diff_ci_high"]],
        "delta_same": ds["target_mean"],
        "delta_same_ci": [ds["target_ci_low"], ds["target_ci_high"]],
        "delta_same_paired_diff": ds["diff_mean"],
        "delta_same_paired_diff_ci": [ds["diff_ci_low"], ds["diff_ci_high"]],
        "math500_acc": m500["target_mean"],
        "math500_ci": [m500["target_ci_low"], m500["target_ci_high"]],
        "math500_diff_vs_base_pp": m500["diff_mean"] * 100.0,
        "math500_diff_vs_base_pp_ci": [m500["diff_ci_low"] * 100.0, m500["diff_ci_high"] * 100.0],
    }


def select_paired_baseline(
    *,
    kind: str,
    vllm_baseline: tuple[Sequence[dict[str, Any]], Sequence[int]],
    hf_baseline: tuple[Sequence[dict[str, Any]], Sequence[int]] | None,
) -> tuple[str, str, Sequence[dict[str, Any]], Sequence[int]]:
    """Select same-engine unmodified baseline (HF for steering_hook; vLLM for all other arms)."""
    if kind == "steering_hook":
        if hf_baseline is None:
            raise RuntimeError("HF unmodified baseline is required for steering_hook paired comparison")
        return "hf", "unmodified_hf", hf_baseline[0], hf_baseline[1]
    return "vllm", "unmodified_vllm", vllm_baseline[0], vllm_baseline[1]


def render_val100_markdown_table(summary_rows: Sequence[dict[str, Any]]) -> str:
    """Render objective numeric summary table (no pass/fail verdict)."""
    lines = [
        "| Base | Arm | Engine | Paired Against | Leak Rate | Rel Drop | Empty% | Δ_transfer (paired diff) | Δ_same (paired diff) | MATH500 (pp diff) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        leak_s = format_ci(r["leak_rate"], *r["leak_rate_ci"], pct=True)
        rel_s = f"{r['leak_rel_drop_vs_base'] * 100.0:+.1f}%"
        emp_s = format_ci(r["empty_turn_rate"], *r["empty_turn_rate_ci"], pct=True)
        dt_s = f"{r['delta_transfer']:+.4f} ({r['delta_transfer_paired_diff']:+.4f})"
        ds_s = f"{r['delta_same']:+.4f} ({r['delta_same_paired_diff']:+.4f})"
        m5_s = f"{r['math500_acc'] * 100.0:.2f}% ({r['math500_diff_vs_base_pp']:+.2f}pp)"
        lines.append(
            f"| {r['base_label']} | {r['arm']} | {r.get('engine', 'vllm')} | {r.get('paired_against', 'unmodified_vllm')} "
            f"| {leak_s} | {rel_s} | {emp_s} | {dt_s} | {ds_s} | {m5_s} |"
        )
    return "\n".join(lines) + "\n"


def _hf_rollout_sync(
    *,
    rows: Sequence[dict[str, Any]],
    base_model_path: Path,
    artifact_dir: Path | None,
    student_api_base: str,
    student_model: str,
    math500_items: Sequence[dict[str, Any]],
    max_teacher_turns: int = 11,
    min_teacher_turns_before_eoc: int = 3,
    require_math_verify: bool = False,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Run HF bfloat16 rollout + MATH500 (unmodified when artifact_dir is None, or steering_hook)."""
    import openai
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, PretrainedConfig

    configs = PretrainedConfig.get_config_dict(str(base_model_path))
    cls = AutoModelForImageTextToText if any("vision_config" in c for c in configs) else AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    model = cls.from_pretrained(str(base_model_path), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if artifact_dir is not None:
        sv = torch.load(artifact_dir / "steering_vectors.pt", map_location="cpu", weights_only=True)
        ctrl = ModificationController(
            model=model,
            tokenizer=tok,
            kind="steering_hook",
            steering_vectors=sv["unit_vectors"],
            steering_layers=sv["layers"],
            steering_alpha=float(sv["alpha"]),
        )
    else:
        ctrl = ModificationController(model=model, tokenizer=tok, kind="none")

    async def _rollout() -> list[dict[str, Any]]:
        s_client = openai.AsyncClient(api_key="EMPTY", base_url=student_api_base, timeout=300.0, max_retries=2)
        sem = asyncio.Semaphore(32)
        convs = []
        for idx, row in enumerate(rows):
            extra = row["extra_info"]
            prob = str(extra["problem"])
            convs.append(
                {
                    "row_index": idx,
                    "problem": prob,
                    "answer": str(extra["answer"]),
                    "transfer_problem": str(extra.get("transfer_problem") or ""),
                    "transfer_answer": str(extra.get("transfer_answer") or ""),
                    "pre_solve_rate": float(extra.get("pre_solve_rate") or 0.0),
                    "pre_solve_rate_transfer": float(
                        extra.get("pre_solve_rate_transfer_problem")
                        if extra.get("pre_solve_rate_transfer_problem") is not None
                        else extra.get("pre_solve_rate") or 0.0
                    ),
                    "messages": [{"role": "student", "content": STUDENT_OPENING_TEMPLATE.format(problem=prob)}],
                    "done": False,
                    "end_reason": None,
                    "assume_thinking": True,
                }
            )

        async def s_turn(c: dict[str, Any]) -> None:
            async with sem:
                resp = await s_client.chat.completions.create(
                    model=student_model,
                    messages=_render_student_messages(c["problem"], c["messages"], assume_thinking=True),
                    temperature=0.6,
                    max_tokens=500,
                )
                c["messages"].append({"role": "student", "content": resp.choices[0].message.content or ""})

        try:
            with ctrl.plugged():
                for _t in range(max_teacher_turns):
                    active = [c for c in convs if not c["done"]]
                    if not active:
                        break
                    t_msgs_list = [_render_teacher_messages(c["problem"], c["messages"]) for c in active]
                    gen_ids_list = ctrl.sample_generate_ids(
                        t_msgs_list, max_new_tokens=4192, temperature=0.6, top_p=0.95, seed=seed
                    )
                    for c, g_ids in zip(active, gen_ids_list):
                        txt = tok.decode(g_ids, skip_special_tokens=False).strip()
                        c["messages"].append({"role": "teacher", "content": txt})
                        n_t = sum(1 for m in c["messages"] if m["role"] == "teacher")
                        if not txt:
                            c["done"], c["end_reason"] = True, "teacher_failed"
                        elif "<end_of_conversation>" in txt and (n_t - 1) >= min_teacher_turns_before_eoc:
                            c["done"], c["end_reason"] = True, "end_of_conversation"
                        elif n_t >= max_teacher_turns:
                            c["done"], c["end_reason"] = True, "max_teacher_turns"
                    active2 = [c for c in convs if not c["done"]]
                    if not active2:
                        break
                    await asyncio.gather(*(s_turn(c) for c in active2))
        finally:
            await s_client.close()
        return convs

    convs = asyncio.run(_rollout())
    with ctrl.plugged():
        m_prompts = [[{"role": "user", "content": it["prompt_text"]}] for it in math500_items]
        m_gen = ctrl.sample_generate_ids(m_prompts, max_new_tokens=8192, temperature=0.6, top_p=0.95, seed=seed)
        m_texts = [(tok.decode(g, skip_special_tokens=False), len(g)) for g in m_gen]
    m_res = score_benchmark_items(
        math500_items,
        lambda _p: m_texts,
        benchmark="math500",
        require_math_verify=require_math_verify,
    )
    del model
    return convs, m_res["correct_vec"]


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 E0 evaluation on author val.parquet (100 problems)")
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base4b-path", required=True)
    ap.add_argument("--agroup-path", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--student-api-base", required=True)
    ap.add_argument("--student-model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--judge-api-base", required=True)
    ap.add_argument("--judge-model", default="Qwen3.6-27B")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--num-student-attempts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--require-math-verify", action="store_true", default=False)
    args = ap.parse_args()

    check_grading_packages(require_exact=args.require_math_verify)
    val_rows = assert_val_parquet_safe(Path(args.val_parquet))
    math500_items = load_benchmark_items("math500", Path(args.data_dir), seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_merge_root = Path(os.environ.get("TMPDIR", str(out_dir))) / "tmp_merged_vllm"

    judge_client = AsyncJudgeClient(
        api_base=args.judge_api_base,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
    )

    bases = [("base_4b", Path(args.base4b_path)), ("a_group_step100", Path(args.agroup_path))]
    summary_rows: list[dict[str, Any]] = []
    total_calls = 0
    total_failures = 0

    for base_label, base_path in bases:
        base_convs = asyncio.run(
            rollout_conversations_vllm(
                rows=val_rows,
                tutor_model=str(base_path),
                student_api_base=args.student_api_base,
                student_model=args.student_model,
                seed=args.seed,
            )
        )
        base_scored = asyncio.run(
            score_val_conversations(
                convs=base_convs,
                student_api_base=args.student_api_base,
                student_model=args.student_model,
                judge_client=judge_client,
                num_student_attempts=args.num_student_attempts,
                require_math_verify=args.require_math_verify,
                cache_path=out_dir / f"{base_label}_unmodified_judge_cache.jsonl",
            )
        )
        gen = VLLMBatchGenerator(str(base_path), max_model_len=16384, seed=args.seed)
        try:
            base_m500 = score_benchmark_items(
                math500_items,
                gen.generate_chat,
                benchmark="math500",
                require_math_verify=args.require_math_verify,
                seed=args.seed,
            )
        finally:
            gen.close()

        summary_rows.append(
            summarize_against_base(
                arm_name="unmodified_vllm",
                base_label=base_label,
                arm_items=base_scored,
                base_items=base_scored,
                arm_math500_vec=base_m500["correct_vec"],
                base_math500_vec=base_m500["correct_vec"],
                engine="vllm",
                paired_against="unmodified_vllm",
                draws=args.draws,
                seed=args.seed,
            )
        )

        # Always run unmodified HF baseline with identical sampling so steering_hook (Method 3)
        # is paired against HF unmodified and engine difference (HF vs vLLM) is explicitly reported.
        base_hf_convs, base_hf_m500_vec = _hf_rollout_sync(
            rows=val_rows,
            base_model_path=base_path,
            artifact_dir=None,
            student_api_base=args.student_api_base,
            student_model=args.student_model,
            math500_items=math500_items,
            require_math_verify=args.require_math_verify,
            seed=args.seed,
        )
        base_hf_scored = asyncio.run(
            score_val_conversations(
                convs=base_hf_convs,
                student_api_base=args.student_api_base,
                student_model=args.student_model,
                judge_client=judge_client,
                num_student_attempts=args.num_student_attempts,
                require_math_verify=args.require_math_verify,
                cache_path=out_dir / f"{base_label}_unmodified_hf_judge_cache.jsonl",
            )
        )
        total_calls += sum(r["judge_calls"] for r in base_hf_scored) + len(val_rows)
        total_failures += sum(r["student_failed"] + r["judge_failures"] for r in base_hf_scored)
        summary_rows.append(
            summarize_against_base(
                arm_name="unmodified_hf",
                base_label=base_label,
                arm_items=base_hf_scored,
                base_items=base_scored,
                arm_math500_vec=base_hf_m500_vec,
                base_math500_vec=base_m500["correct_vec"],
                engine="hf",
                paired_against="unmodified_vllm",
                draws=args.draws,
                seed=args.seed,
            )
        )

        fit_summary_path = Path(args.artifacts_dir) / base_label / "fit_summary.json"
        if not fit_summary_path.is_file():
            continue
        fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
        for mname, mmanifest in fit_summary["methods"].items():
            mdir = Path(args.artifacts_dir) / base_label / mmanifest["method"]
            kind = mmanifest["kind"]
            if kind == "peft_lora":
                merged_dir = export_merged_bf16_for_vllm(
                    base_model_path=base_path,
                    artifact_dir=mdir,
                    merged_out_dir=tmp_merge_root / f"{base_label}_{mname}",
                )
                try:
                    arm_convs = asyncio.run(
                        rollout_conversations_vllm(
                            rows=val_rows,
                            tutor_model=str(merged_dir),
                            student_api_base=args.student_api_base,
                            student_model=args.student_model,
                            seed=args.seed,
                        )
                    )
                    m_gen = VLLMBatchGenerator(str(merged_dir), max_model_len=16384, seed=args.seed)
                    try:
                        arm_m500 = score_benchmark_items(
                            math500_items,
                            m_gen.generate_chat,
                            benchmark="math500",
                            require_math_verify=args.require_math_verify,
                            seed=args.seed,
                        )
                    finally:
                        m_gen.close()
                    arm_m500_vec = arm_m500["correct_vec"]
                finally:
                    shutil.rmtree(merged_dir, ignore_errors=True)
            elif kind == "prompt_only":
                arm_convs = asyncio.run(
                    rollout_conversations_vllm(
                        rows=val_rows,
                        tutor_model=str(base_path),
                        student_api_base=args.student_api_base,
                        student_model=args.student_model,
                        prompt_suffix=mmanifest["prompt_suffix"],
                        seed=args.seed,
                    )
                )
                # Weight-free prompt_only does not alter standalone math solving weights
                arm_m500_vec = base_m500["correct_vec"]
            elif kind == "steering_hook":
                arm_convs, arm_m500_vec = _hf_rollout_sync(
                    rows=val_rows,
                    base_model_path=base_path,
                    artifact_dir=mdir,
                    student_api_base=args.student_api_base,
                    student_model=args.student_model,
                    math500_items=math500_items,
                    require_math_verify=args.require_math_verify,
                    seed=args.seed,
                )
            else:
                raise ValueError(f"Unsupported kind {kind}")

            arm_scored = asyncio.run(
                score_val_conversations(
                    convs=arm_convs,
                    student_api_base=args.student_api_base,
                    student_model=args.student_model,
                    judge_client=judge_client,
                    num_student_attempts=args.num_student_attempts,
                    require_math_verify=args.require_math_verify,
                    cache_path=out_dir / f"{base_label}_{mname}_judge_cache.jsonl",
                )
            )
            total_calls += sum(r["judge_calls"] for r in arm_scored) + len(val_rows)
            total_failures += sum(r["student_failed"] + r["judge_failures"] for r in arm_scored)

            engine_name, paired_ref, ref_items, ref_m500_vec = select_paired_baseline(
                kind=kind,
                vllm_baseline=(base_scored, base_m500["correct_vec"]),
                hf_baseline=(base_hf_scored, base_hf_m500_vec),
            )
            row_summary = summarize_against_base(
                arm_name=mmanifest["method"],
                base_label=base_label,
                arm_items=arm_scored,
                base_items=ref_items,
                arm_math500_vec=arm_m500_vec,
                base_math500_vec=ref_m500_vec,
                engine=engine_name,
                paired_against=paired_ref,
                draws=args.draws,
                seed=args.seed,
            )
            row_summary["unplug_check"] = mmanifest.get("unplug_check")
            summary_rows.append(row_summary)

    atomic_write_json(out_dir / "p4_val100_table.json", {"rows": summary_rows})
    (out_dir / "p4_val100_table.md").write_text(render_val100_markdown_table(summary_rows), encoding="utf-8")
    print_machine_summary(
        {
            "status": "P4_EVAL_VAL100_DONE",
            "rc": 0,
            "calls": total_calls,
            "failures": total_failures,
            "arms_evaluated": len(summary_rows),
        }
    )


if __name__ == "__main__":
    main()
