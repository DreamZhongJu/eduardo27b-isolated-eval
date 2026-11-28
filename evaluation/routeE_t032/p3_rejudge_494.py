#!/usr/bin/env python3
"""P3: Re-judge full 494 conversations for A-group step_100 and Eduardo-27B.

Reports per model (with 10,000 problem-level bootstrap 95% CIs):
- leak_rate_all (overall over 494 conversations)
- empty_turn_rate_conv_mean (mean per-conversation blanked teacher turn rate)
- empty_turn_rate_micro (blanked teacher turns / total teacher turns)
- all_empty_conv_rate (fraction of conversations with 0 non-empty teacher turns)
- leak_rate_nonempty_convs (leak rate among conversations with >=1 non-empty teacher turn)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from common_utils import (
    atomic_write_json,
    bootstrap_mean_ci,
    format_ci,
    print_machine_summary,
    strip_teacher_controls,
)
from leakage_judges import (
    AsyncJudgeClient,
    JudgeCache,
    Layer1RuleScreener,
    evaluate_combined_4layer,
    run_llm_judge,
)


def summarize_rejudged_records(
    item_rows: Sequence[dict[str, Any]],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    n = len(item_rows)
    leak_all = [int(r["leaked"]) for r in item_rows]
    empty_rates = [float(r["empty_turn_rate"]) for r in item_rows]
    total_turns = sum(int(r["num_teacher_turns"]) for r in item_rows)
    blanked_turns = sum(int(r["num_blanked_turns"]) for r in item_rows)
    nonempty_rows = [r for r in item_rows if int(r["num_nonempty_turns"]) > 0]
    leak_nonempty = [int(r["leaked"]) for r in nonempty_rows]
    all_empty_flags = [int(r["num_nonempty_turns"] == 0) for r in item_rows]

    ci_leak_all = bootstrap_mean_ci(leak_all, draws=draws, seed=seed)
    ci_empty_conv = bootstrap_mean_ci(empty_rates, draws=draws, seed=seed)
    ci_leak_ne = bootstrap_mean_ci(leak_nonempty, draws=draws, seed=seed)

    return {
        "n_conversations": n,
        "n_nonempty_conversations": len(nonempty_rows),
        "n_all_empty_conversations": sum(all_empty_flags),
        "all_empty_conv_rate": (sum(all_empty_flags) / n) if n else 0.0,
        "total_teacher_turns": total_turns,
        "blanked_teacher_turns": blanked_turns,
        "empty_turn_rate_micro": (blanked_turns / total_turns) if total_turns else 0.0,
        "empty_turn_rate_conv_mean": ci_empty_conv["mean"],
        "empty_turn_rate_conv_ci": [ci_empty_conv["ci_low"], ci_empty_conv["ci_high"]],
        "leak_rate_all": ci_leak_all["mean"],
        "leak_rate_all_ci": [ci_leak_all["ci_low"], ci_leak_all["ci_high"]],
        "leak_rate_nonempty_convs": ci_leak_ne["mean"],
        "leak_rate_nonempty_convs_ci": [ci_leak_ne["ci_low"], ci_leak_ne["ci_high"]],
        "layer1_hit_rate": (sum(int(r["layer1_leaked"]) for r in item_rows) / n) if n else 0.0,
    }


async def rejudge_model_file(
    *,
    label: str,
    eval_json_path: Path,
    judge_client: AsyncJudgeClient | None,
    judge_mode: str = "combined_4layer",
    out_dir: Path,
    require_math_verify: bool = False,
    num_votes: int = 3,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    data = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    records = list(data["conversations"])
    screener = Layer1RuleScreener(require_math_verify=require_math_verify)
    cache = JudgeCache(out_dir / f"{label}_rejudge_cache.jsonl")

    async def proc_one(rec: dict[str, Any]) -> dict[str, Any]:
        raw_msgs = rec.get("raw_messages") or rec.get("reward_messages") or rec.get("messages") or []
        assume = bool(rec.get("assume_thinking", True))
        already_vis = not ("raw_messages" in rec or "reward_messages" in rec)

        teacher_msgs = [m for m in raw_msgs if m.get("role") in ("teacher", "assistant")]
        n_turns = len(teacher_msgs)
        n_blank = 0
        for m in teacher_msgs:
            txt = str(m.get("content") or "")
            vis = txt.replace("<end_of_conversation>", "").strip() if already_vis else strip_teacher_controls(txt, assume_thinking=assume)
            if not vis:
                n_blank += 1
        n_nonempty = n_turns - n_blank

        if judge_mode == "combined_4layer":
            jres = await evaluate_combined_4layer(
                screener=screener,
                judge_client=judge_client,
                messages=raw_msgs,
                problem=str(rec.get("problem", "")),
                answer=str(rec.get("answer", "")),
                assume_thinking=assume,
                already_visible=already_vis,
                num_votes=num_votes,
                cache=cache,
            )
        else:
            assert judge_client is not None
            jres = await run_llm_judge(
                mode=judge_mode,
                judge_client=judge_client,
                messages=raw_msgs,
                problem=str(rec.get("problem", "")),
                answer=str(rec.get("answer", "")),
                assume_thinking=assume,
                already_visible=already_vis,
                num_votes=num_votes,
                cache=cache,
            )
            jres["layer1_leaked"] = 0

        return {
            "row_index": int(rec.get("row_index", -1)),
            "leaked": int(jres["leaked"]),
            "layer1_leaked": int(jres.get("layer1_leaked", 0)),
            "num_teacher_turns": n_turns,
            "num_blanked_turns": n_blank,
            "num_nonempty_turns": n_nonempty,
            "empty_turn_rate": (n_blank / n_turns) if n_turns else 0.0,
            "stored_leaked": int(rec.get("leaked", 0)),
            "calls": int(jres.get("calls", 0)),
            "failures": int(jres.get("failures", 0)),
        }

    item_rows = await asyncio.gather(*[proc_one(r) for r in records])
    summary = summarize_rejudged_records(item_rows, draws=draws, seed=seed)
    summary["calls"] = sum(r["calls"] for r in item_rows)
    summary["failures"] = sum(r["failures"] for r in item_rows)
    summary["judge_mode"] = judge_mode
    atomic_write_json(
        out_dir / f"{label}_p3_rejudge_items.json",
        [{k: v for k, v in r.items() if k not in ("calls", "failures")} for r in item_rows],
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 rejudge 494 conversations for A-group & Eduardo-27B")
    ap.add_argument("--agroup-json", required=True)
    ap.add_argument("--model27b-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--judge-api-base", required=True)
    ap.add_argument("--judge-model", default="Qwen3.6-27B")
    ap.add_argument("--judge-mode", default="combined_4layer")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-votes", type=int, default=3)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-math-verify", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncJudgeClient(
        api_base=args.judge_api_base,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
    )

    async def _run() -> dict[str, Any]:
        a_sum = await rejudge_model_file(
            label="a_group_step100",
            eval_json_path=Path(args.agroup_json),
            judge_client=client,
            judge_mode=args.judge_mode,
            out_dir=out_dir,
            require_math_verify=args.require_math_verify,
            num_votes=args.judge_votes,
            draws=args.draws,
            seed=args.seed,
        )
        b_sum = await rejudge_model_file(
            label="eduardo_27b",
            eval_json_path=Path(args.model27b_json),
            judge_client=client,
            judge_mode=args.judge_mode,
            out_dir=out_dir,
            require_math_verify=args.require_math_verify,
            num_votes=args.judge_votes,
            draws=args.draws,
            seed=args.seed,
        )
        return {"a_group_step100": a_sum, "eduardo_27b": b_sum}

    rep = asyncio.run(_run())
    atomic_write_json(out_dir / "p3_rejudge_summary.json", rep)

    a = rep["a_group_step100"]
    b = rep["eduardo_27b"]
    print_machine_summary(
        {
            "status": "P3_REJUDGE_DONE",
            "rc": 0,
            "calls": a["calls"] + b["calls"],
            "failures": a["failures"] + b["failures"],
            "agroup_leak_rate_all": format_ci(a["leak_rate_all"], *a["leak_rate_all_ci"], pct=True),
            "agroup_empty_turn_rate": format_ci(a["empty_turn_rate_conv_mean"], *a["empty_turn_rate_conv_ci"], pct=True),
            "agroup_leak_rate_nonempty": format_ci(a["leak_rate_nonempty_convs"], *a["leak_rate_nonempty_convs_ci"], pct=True),
            "m27b_leak_rate_all": format_ci(b["leak_rate_all"], *b["leak_rate_all_ci"], pct=True),
            "m27b_empty_turn_rate": format_ci(b["empty_turn_rate_conv_mean"], *b["empty_turn_rate_conv_ci"], pct=True),
            "m27b_leak_rate_nonempty": format_ci(b["leak_rate_nonempty_convs"], *b["leak_rate_nonempty_convs_ci"], pct=True),
        }
    )


if __name__ == "__main__":
    main()
