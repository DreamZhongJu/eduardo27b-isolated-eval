#!/usr/bin/env python3
"""P2 Calibration & Inter-Rater Consistency (F1 / Recall / Cohen's kappa).

Must pass the hard annotation gate (`unseal_mapping_key`) requiring 100/100 user
annotations and 30/30 Claude annotations before unsealing the model mapping key.
Reports Precision, Recall, F1 (with 10,000 conversation-level bootstrap 95% CIs)
across:
  - layer1_only
  - layer2_checklist
  - combined_4layer (primary candidate; requires F1 >= 0.70 and Recall >= 0.70)
  - naive_single_sentence (baseline)
  - author_judge_leakage (baseline)
  - stored_judge (original JSON verdicts from T-020 / T-022)
And Cohen's kappa (binary + 4-way) with 10,000 bootstrap 95% CIs on the 30 items.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common_utils import atomic_write_json, format_ci, print_machine_summary
from leakage_judges import (
    AsyncJudgeClient,
    JudgeCache,
    Layer1RuleScreener,
    evaluate_combined_4layer,
    run_llm_judge,
)
from p2_gold_sampling import VALID_CATEGORIES, unseal_mapping_key


def compute_prf1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }


def bootstrap_prf1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    n = len(yt)
    point = compute_prf1(y_true, y_pred)
    rng = np.random.default_rng(seed)

    precs = np.empty(draws, dtype=np.float64)
    recs = np.empty(draws, dtype=np.float64)
    f1s = np.empty(draws, dtype=np.float64)
    chunk_size = max(1, min(draws, 5_000_000 // max(1, n)))
    offset = 0
    while offset < draws:
        cur = min(chunk_size, draws - offset)
        idx = rng.integers(0, n, size=(cur, n))
        bt = yt[idx]
        bp = yp[idx]
        tp = ((bt == 1) & (bp == 1)).sum(axis=1).astype(np.float64)
        fp = ((bt == 0) & (bp == 1)).sum(axis=1).astype(np.float64)
        fn = ((bt == 1) & (bp == 0)).sum(axis=1).astype(np.float64)
        den_p = tp + fp
        den_r = tp + fn
        p = np.divide(tp, den_p, out=np.zeros_like(tp), where=den_p > 0)
        r = np.divide(tp, den_r, out=np.zeros_like(tp), where=den_r > 0)
        den_f = p + r
        f = np.divide(2.0 * p * r, den_f, out=np.zeros_like(p), where=den_f > 0)
        precs[offset : offset + cur] = p
        recs[offset : offset + cur] = r
        f1s[offset : offset + cur] = f
        offset += cur

    p_lo, p_hi = np.percentile(precs, [2.5, 97.5])
    r_lo, r_hi = np.percentile(recs, [2.5, 97.5])
    f_lo, f_hi = np.percentile(f1s, [2.5, 97.5])
    return {
        **point,
        "precision_ci": [float(p_lo), float(p_hi)],
        "recall_ci": [float(r_lo), float(r_hi)],
        "f1_ci": [float(f_lo), float(f_hi)],
    }


def compute_cohens_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> dict[str, float]:
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        raise ValueError("Rater label sequences must be non-empty and equal length")
    cats = sorted(set(labels_a) | set(labels_b))
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for a in labels_a if a == c) / n
        pb = sum(1 for b in labels_b if b == c) / n
        pe += pa * pb
    kappa = (po - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else (1.0 if po == 1.0 else 0.0)
    return {"n": n, "po": float(po), "pe": float(pe), "kappa": float(kappa)}


def bootstrap_kappa_ci(
    labels_a: Sequence[Any],
    labels_b: Sequence[Any],
    *,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    point = compute_cohens_kappa(labels_a, labels_b)
    n = len(labels_a)
    rng = np.random.default_rng(seed)
    kappas = np.empty(draws, dtype=np.float64)
    pos = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        idx = rng.integers(0, n, size=n)
        sa = [labels_a[i] for i in idx]
        sb = [labels_b[i] for i in idx]
        res = compute_cohens_kappa(sa, sb)
        kappas[d] = res["kappa"]
        pos[d] = res["po"]
    k_lo, k_hi = np.percentile(kappas, [2.5, 97.5])
    p_lo, p_hi = np.percentile(pos, [2.5, 97.5])
    return {
        **point,
        "kappa_ci": [float(k_lo), float(k_hi)],
        "po_ci": [float(p_lo), float(p_hi)],
    }


async def run_calibration(
    *,
    gold_dir: Path,
    user_csv: Path,
    claude_csv: Path,
    out_dir: Path,
    judge_client: AsyncJudgeClient | None,
    require_math_verify: bool = False,
    num_votes: int = 3,
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    gold_dir = Path(gold_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard unseal gate: fails immediately if user_100 or claude_30 is incomplete
    unsealed = unseal_mapping_key(
        sealed_path=gold_dir / "mapping_key.sealed.json",
        user_csv_path=user_csv,
        claude_ids_path=gold_dir / "claude_30_ids.json",
        claude_csv_path=claude_csv,
        require_claude30=True,
        receipt_path=out_dir / "annotation_lock_receipt.json",
    )

    mapping = unsealed["mapping"]
    user_ann = unsealed["user_annotations"]
    claude_ann = unsealed["claude_annotations"]
    claude_ids = unsealed["claude_ids"]

    restricted_path = gold_dir / "internal_eval_inputs_100.Restricted.jsonl"
    items_by_id: dict[str, dict[str, Any]] = {}
    with restricted_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                items_by_id[obj["neutral_id"]] = obj

    ordered_ids = sorted(items_by_id.keys())
    y_true = [int(user_ann[nid]["leaked"]) for nid in ordered_ids]

    screener = Layer1RuleScreener(require_math_verify=require_math_verify)
    cache = JudgeCache(out_dir / "judge_calibration_cache.jsonl")

    async def eval_one(nid: str) -> dict[str, Any]:
        it = items_by_id[nid]
        msgs = it["raw_messages"]
        prob = it["problem"]
        ans = it["answer"]
        assume = bool(it.get("assume_thinking", True))

        comb = await evaluate_combined_4layer(
            screener=screener,
            judge_client=judge_client,
            messages=msgs,
            problem=prob,
            answer=ans,
            assume_thinking=assume,
            num_votes=num_votes,
            cache=cache,
            run_layer2_even_if_l1_hits=True,
        )
        if judge_client is not None:
            naive = await run_llm_judge(
                mode="naive_single_sentence",
                judge_client=judge_client,
                messages=msgs,
                problem=prob,
                answer=ans,
                assume_thinking=assume,
                num_votes=num_votes,
                cache=cache,
            )
            author = await run_llm_judge(
                mode="author_judge_leakage",
                judge_client=judge_client,
                messages=msgs,
                problem=prob,
                answer=ans,
                assume_thinking=assume,
                num_votes=num_votes,
                cache=cache,
            )
        else:
            naive = {"leaked": 0, "calls": 0, "failures": 0}
            author = {"leaked": 0, "calls": 0, "failures": 0}

        return {
            "neutral_id": nid,
            "model": mapping[nid]["model"],
            "user_leaked": int(user_ann[nid]["leaked"]),
            "user_category": user_ann[nid]["category"],
            "layer1_only": int(comb["layer1_leaked"]),
            "layer2_checklist": int(comb["layer2_leaked"] or 0),
            "combined_4layer": int(comb["leaked"]),
            "naive_single_sentence": int(naive["leaked"]),
            "author_judge_leakage": int(author["leaked"]),
            "stored_judge": int(it.get("stored_leaked", 0)),
            "calls": comb["calls"] + naive["calls"] + author["calls"],
            "failures": comb["failures"] + naive["failures"] + author["failures"],
        }

    per_item = await asyncio.gather(*[eval_one(nid) for nid in ordered_ids])
    total_calls = sum(r["calls"] for r in per_item)
    total_failures = sum(r["failures"] for r in per_item)

    judge_names = [
        "layer1_only",
        "layer2_checklist",
        "combined_4layer",
        "naive_single_sentence",
        "author_judge_leakage",
        "stored_judge",
    ]
    metrics_by_judge: dict[str, Any] = {}
    for jname in judge_names:
        preds = [r[jname] for r in per_item]
        metrics_by_judge[jname] = bootstrap_prf1_ci(y_true, preds, draws=draws, seed=seed)

    # Breakdown by model
    by_model: dict[str, Any] = {}
    for mtag in sorted({r["model"] for r in per_item}):
        sub = [r for r in per_item if r["model"] == mtag]
        yt_sub = [r["user_leaked"] for r in sub]
        yp_sub = [r["combined_4layer"] for r in sub]
        by_model[mtag] = {
            "n": len(sub),
            "gold_leak_rate": sum(yt_sub) / len(sub),
            "combined_4layer_leak_rate": sum(yp_sub) / len(sub),
            "combined_4layer_prf1": compute_prf1(yt_sub, yp_sub),
        }

    # Inter-rater agreement on the 30 dual-annotated items
    u_bin = [int(user_ann[nid]["leaked"]) for nid in claude_ids]
    c_bin = [int(claude_ann[nid]["leaked"]) for nid in claude_ids]
    u_4way = [VALID_CATEGORIES.index(user_ann[nid]["category"]) for nid in claude_ids]
    c_4way = [VALID_CATEGORIES.index(claude_ann[nid]["category"]) for nid in claude_ids]

    kappa_binary = bootstrap_kappa_ci(u_bin, c_bin, draws=draws, seed=seed)
    kappa_4way = bootstrap_kappa_ci(u_4way, c_4way, draws=draws, seed=seed)

    comb_metrics = metrics_by_judge["combined_4layer"]
    gate_pass = bool(comb_metrics["f1"] >= 0.70 and comb_metrics["recall"] >= 0.70)

    report = {
        "n_gold": len(ordered_ids),
        "gold_leak_count": sum(y_true),
        "gold_leak_rate": sum(y_true) / len(y_true),
        "calibration_gate_threshold": {"f1_min": 0.70, "recall_min": 0.70},
        "calibration_gate_pass": gate_pass,
        "judges": metrics_by_judge,
        "by_model": by_model,
        "inter_rater_30": {
            "binary": kappa_binary,
            "four_way": kappa_4way,
        },
        "calls": total_calls,
        "failures": total_failures,
        "annotation_lock_receipt": unsealed["receipt"],
    }
    atomic_write_json(out_dir / "p2_calibration_report.json", report)
    # Save de-identified per-item predictions (no text/answers)
    clean_rows = [
        {k: v for k, v in r.items() if k not in ("calls", "failures")}
        for r in per_item
    ]
    atomic_write_json(out_dir / "p2_calibration_per_item.json", clean_rows)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 calibration & inter-rater kappa (after 100+30 annotations)")
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--user-csv", required=True)
    ap.add_argument("--claude-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--judge-api-base", required=True)
    ap.add_argument("--judge-model", default="Qwen3.6-27B")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-votes", type=int, default=3)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-math-verify", action="store_true", default=False)
    args = ap.parse_args()

    client = AsyncJudgeClient(
        api_base=args.judge_api_base,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
    )
    rep = asyncio.run(
        run_calibration(
            gold_dir=Path(args.gold_dir),
            user_csv=Path(args.user_csv),
            claude_csv=Path(args.claude_csv),
            out_dir=Path(args.out_dir),
            judge_client=client,
            require_math_verify=args.require_math_verify,
            num_votes=args.judge_votes,
            draws=args.draws,
            seed=args.seed,
        )
    )

    comb = rep["judges"]["combined_4layer"]
    naive = rep["judges"]["naive_single_sentence"]
    author = rep["judges"]["author_judge_leakage"]
    kb = rep["inter_rater_30"]["binary"]
    print_machine_summary(
        {
            "status": "P2_CALIBRATION_DONE",
            "rc": 0,
            "calls": rep["calls"],
            "failures": rep["failures"],
            "calibration_gate_pass": rep["calibration_gate_pass"],
            "combined_4layer_f1": format_ci(comb["f1"], comb["f1_ci"][0], comb["f1_ci"][1]),
            "combined_4layer_recall": format_ci(comb["recall"], comb["recall_ci"][0], comb["recall_ci"][1]),
            "combined_4layer_precision": format_ci(comb["precision"], comb["precision_ci"][0], comb["precision_ci"][1]),
            "naive_single_sentence_f1": format_ci(naive["f1"], naive["f1_ci"][0], naive["f1_ci"][1]),
            "author_judge_leakage_f1": format_ci(author["f1"], author["f1_ci"][0], author["f1_ci"][1]),
            "kappa_binary_30": format_ci(kb["kappa"], kb["kappa_ci"][0], kb["kappa_ci"][1]),
        }
    )


if __name__ == "__main__":
    main()
