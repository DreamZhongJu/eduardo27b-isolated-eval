#!/usr/bin/env python3
"""P1: Reasoning sanity check on MATH500 (0-shot) & GSM8K test (4-shot).

Follows PedagogicalRL (EMNLP 2025, arXiv:2505.15607) Table 3 / README:
- Benchmarks: lighteval|math_500|0|0 and lighteval|gsm8k|4|0
- Prompts: exact LightEval math_500.py and gsm8k.py templates with --use-chat-template
- Decoding: dtype=bfloat16, temperature=0.6, top_p=0.95, max_new_tokens=8192, n=1
- Context cap: max_model_len=16384 (so full 8192 generation budget remains after 4-shot prompt)
- Grading: math-verify==0.8.0 / latex2sympy2_extended==1.10.2 (after stripping <think>)

Never prints problem text, answers, or model completions to stdout or summary JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from common_utils import (
    GSM8K_FILES,
    GSM8K_REV,
    MATH500_FILES,
    MATH500_REV,
    atomic_write_json,
    bootstrap_mean_ci,
    check_grading_packages,
    download_verified_file,
    format_ci,
    grade_math_response,
    paired_bootstrap_diff_ci,
    print_machine_summary,
    verify_model_bfloat16,
)

# Exact LightEval templates (src/lighteval/tasks/tasks/{math_500,gsm8k}.py)
MATH500_QUERY_TEMPLATE = (
    "Solve the following problem. The final line of your response MUST be of the following format:\n"
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.\n\n'
    "{prompt}"
)

GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your\n"
    'response should be of the form "ANSWER: $ANSWER" (without quotes)\n'
    "where $ANSWER is the answer to the problem.\n\n"
    "{prompt}\n\n"
    "Remember to put your answer on its own line at the end in the form\n"
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to\n'
    "the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)


def _split_gsm8k_answer(raw_answer: str) -> tuple[str, str]:
    if "####" in raw_answer:
        parts = raw_answer.split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        return reasoning, target
    return "", raw_answer.strip()


def format_gsm8k_4shot_prompt(question: str, shots: Sequence[dict[str, str]]) -> str:
    blocks: list[str] = []
    for shot in shots:
        reasoning, target = _split_gsm8k_answer(shot["answer"])
        q_block = GSM8K_PROMPT_TEMPLATE.format(prompt=shot["question"])
        blocks.append(f"{q_block}\n{reasoning}\n\nANSWER: {target}")
    blocks.append(GSM8K_PROMPT_TEMPLATE.format(prompt=question))
    return "\n\n".join(blocks)


def format_math500_prompt(problem: str) -> str:
    return MATH500_QUERY_TEMPLATE.format(prompt=problem)


def ensure_math_benchmarks_data(data_dir: Path) -> dict[str, Path]:
    """Download & hash-verify MATH-500 and GSM8K from hf-mirror (no token)."""
    data_dir = Path(data_dir)
    out: dict[str, Path] = {}

    m_dir = data_dir / "MATH-500"
    for rel, (algo, _, exp_hash) in MATH500_FILES.items():
        url = f"https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/{MATH500_REV}/{rel}"
        out["math500_test"] = download_verified_file([url], m_dir / rel, algo=algo, expected_hash=exp_hash)

    g_dir = data_dir / "gsm8k"
    for rel, (algo, _, exp_hash) in GSM8K_FILES.items():
        url = f"https://hf-mirror.com/datasets/openai/gsm8k/resolve/{GSM8K_REV}/{rel}"
        key = "gsm8k_train" if "train" in rel else "gsm8k_test"
        out[key] = download_verified_file([url], g_dir / rel, algo=algo, expected_hash=exp_hash)

    return out


def load_benchmark_items(
    benchmark: str,
    data_dir: Path,
    *,
    num_problems: int = -1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    paths = ensure_math_benchmarks_data(data_dir)
    items: list[dict[str, Any]] = []
    if benchmark == "math500":
        with paths["math500_test"].open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for idx, row in enumerate(rows):
            items.append(
                {
                    "idx": idx,
                    "benchmark": "math500",
                    "prompt_text": format_math500_prompt(row["problem"]),
                    "answer": str(row["answer"]),
                }
            )
    elif benchmark == "gsm8k":
        import pyarrow.parquet as pq

        train_rows = pq.read_table(paths["gsm8k_train"]).to_pylist()
        test_rows = pq.read_table(paths["gsm8k_test"]).to_pylist()
        rng = random.Random(seed)
        shots = rng.sample(train_rows, 4)
        for idx, row in enumerate(test_rows):
            _, target = _split_gsm8k_answer(str(row["answer"]))
            items.append(
                {
                    "idx": idx,
                    "benchmark": "gsm8k",
                    "prompt_text": format_gsm8k_4shot_prompt(row["question"], shots),
                    "answer": target,
                }
            )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    if num_problems > 0:
        items = items[:num_problems]
    return items


class VLLMBatchGenerator:
    """vLLM bfloat16 generator following PedagogicalRL Table 3 sampling config."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.85,
        seed: int = 42,
        tensor_parallel_size: int = 1,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        verify_model_bfloat16(Path(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            seed=seed,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=8192,
            n=1,
        )

    def generate_chat(self, prompt_texts: Sequence[str]) -> list[tuple[str, int]]:
        rendered: list[str] = []
        for p in prompt_texts:
            msgs = [{"role": "user", "content": p}]
            try:
                txt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            except TypeError:
                txt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            rendered.append(txt)
        outputs = self.llm.generate(rendered, self.sampling_params)
        results: list[tuple[str, int]] = []
        for out in outputs:
            cand = out.outputs[0]
            results.append((cand.text, len(cand.token_ids)))
        return results

    def close(self) -> None:
        try:
            del self.llm
        except Exception:
            pass
        try:
            import gc, torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def score_benchmark_items(
    items: Sequence[dict[str, Any]],
    generate_fn: Callable[[Sequence[str]], Sequence[tuple[str, int]]],
    *,
    benchmark: str,
    assume_thinking: bool = True,
    require_math_verify: bool = False,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score items via ``generate_fn`` and return aggregate stats + 0/1 vector only."""
    prompts = [it["prompt_text"] for it in items]
    generations = list(generate_fn(prompts))
    if len(generations) != len(items):
        raise RuntimeError(f"Generation count mismatch: {len(generations)} vs {len(items)}")

    records: list[dict[str, Any]] = []
    correct_vec: list[int] = []
    n_format = 0
    n_unclosed = 0
    total_tokens = 0
    for it, (text, n_tok) in zip(items, generations):
        g = grade_math_response(
            text,
            it["answer"],
            mode=benchmark,
            assume_thinking=assume_thinking,
            require_math_verify=require_math_verify,
        )
        correct_vec.append(g["correct"])
        n_format += g["incorrect_format"]
        n_unclosed += g["unclosed_thinking"]
        total_tokens += int(n_tok)
        records.append(
            {
                "problem_idx": int(it["idx"]),
                "correct": int(g["correct"]),
                "incorrect_format": int(g["incorrect_format"]),
                "unclosed_thinking": int(g["unclosed_thinking"]),
                "gen_tokens": int(n_tok),
            }
        )

    ci = bootstrap_mean_ci(correct_vec, draws=draws, seed=seed)
    n = len(items)
    return {
        "benchmark": benchmark,
        "n_problems": n,
        "accuracy": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "incorrect_format_rate": (n_format / n) if n else 0.0,
        "unclosed_thinking_rate": (n_unclosed / n) if n else 0.0,
        "avg_gen_tokens": (total_tokens / n) if n else 0.0,
        "correct_vec": correct_vec,
        "per_problem": records,
    }


def compare_model_results(
    base_res: dict[str, Any],
    target_res: dict[str, Any],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    paired = paired_bootstrap_diff_ci(
        base_res["correct_vec"],
        target_res["correct_vec"],
        draws=draws,
        seed=seed,
    )
    return paired


def _parse_models_arg(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected label=path in --models, got {item!r}")
        label, path = item.split("=", 1)
        pairs.append((label.strip(), path.strip()))
    if not pairs:
        raise ValueError("--models produced empty model list")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="P1 MATH500/GSM8K checkup (bf16, PedagogicalRL Table 3)")
    ap.add_argument("--models", required=True, help="Comma-separated label=model_path pairs")
    ap.add_argument("--benchmarks", default="math500,gsm8k")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-problems", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--require-math-verify", action="store_true", default=False)
    ap.add_argument("--no-assume-thinking", dest="assume_thinking", action="store_false")
    ap.set_defaults(assume_thinking=True)
    args = ap.parse_args()

    pkg_versions = check_grading_packages(require_exact=args.require_math_verify)
    model_specs = _parse_models_arg(args.models)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    items_by_bench = {
        b: load_benchmark_items(b, data_dir, num_problems=args.num_problems, seed=args.seed)
        for b in benchmarks
    }

    results_by_model: dict[str, dict[str, Any]] = {}
    calls = 0
    for label, model_path in model_specs:
        gen = VLLMBatchGenerator(
            model_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        try:
            results_by_model[label] = {}
            for b in benchmarks:
                res = score_benchmark_items(
                    items_by_bench[b],
                    gen.generate_chat,
                    benchmark=b,
                    assume_thinking=args.assume_thinking,
                    require_math_verify=args.require_math_verify,
                    draws=args.draws,
                    seed=args.seed,
                )
                calls += res["n_problems"]
                atomic_write_json(out_dir / f"{label}_{b}_scores.json", res["per_problem"])
                res_summary = {k: v for k, v in res.items() if k not in ("per_problem", "correct_vec")}
                results_by_model[label][b] = {
                    **res_summary,
                    "correct_vec": res["correct_vec"],
                }
        finally:
            gen.close()

    paired_comparisons: dict[str, Any] = {}
    if len(model_specs) >= 2:
        base_label = model_specs[0][0]
        for target_label, _ in model_specs[1:]:
            paired_comparisons[f"{target_label}_minus_{base_label}"] = {
                b: compare_model_results(
                    results_by_model[base_label][b],
                    results_by_model[target_label][b],
                    draws=args.draws,
                    seed=args.seed,
                )
                for b in benchmarks
            }

    clean_models = {
        label: {
            b: {k: v for k, v in bres.items() if k != "correct_vec"}
            for b, bres in mres.items()
        }
        for label, mres in results_by_model.items()
    }
    summary_payload = {
        "protocol": {
            "paper": "PedagogicalRL (EMNLP 2025, arXiv:2505.15607) Table 3",
            "dtype": "bfloat16",
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 8192,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "draws": args.draws,
            "grading_packages": pkg_versions,
        },
        "models": clean_models,
        "paired_comparisons": paired_comparisons,
    }
    atomic_write_json(out_dir / "p1_checkup_summary.json", summary_payload)

    summary_fields: dict[str, Any] = {
        "status": "P1_CHECKUP_DONE",
        "rc": 0,
        "calls": calls,
        "failures": 0,
    }
    for label, mres in clean_models.items():
        for b, bres in mres.items():
            summary_fields[f"{b}_{label}_acc"] = format_ci(
                bres["accuracy"], bres["ci_low"], bres["ci_high"], pct=True
            )
    for cmp_name, bmap in paired_comparisons.items():
        for b, pstats in bmap.items():
            summary_fields[f"{b}_{cmp_name}_diff"] = format_ci(
                pstats["diff_mean"], pstats["diff_ci_low"], pstats["diff_ci_high"], pct=True
            )
    print_machine_summary(summary_fields)


if __name__ == "__main__":
    main()
