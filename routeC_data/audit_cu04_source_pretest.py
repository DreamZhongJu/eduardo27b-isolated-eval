#!/usr/bin/env python3
"""Read-only cu04 audit of original source pretest records (counts, rates, hashes only).

PRIVACY GUARANTEE:
- Never reads or prints problem statements, reference answers, or student responses.
- Computes and prints only aggregate counts, distributions, quantiles, and SHA-256 digests.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _quantiles(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n

    def q(prob: float) -> float:
        idx = prob * (n - 1)
        lo = int(math.floor(idx))
        hi = min(n - 1, lo + 1)
        frac = idx - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(math.sqrt(var), 6),
        "min": round(s[0], 6),
        "p10": round(q(0.10), 6),
        "p25": round(q(0.25), 6),
        "p50": round(q(0.50), 6),
        "p75": round(q(0.75), 6),
        "p90": round(q(0.90), 6),
        "max": round(s[-1], 6),
    }


def compute_source_pretest_table_sha256(rows: Sequence[Tuple[str, int, int]]) -> str:
    """Compute canonical SHA-256 of sorted `(source_id, n_correct, n_scored)` pure-number table."""
    lines = [
        f"{sid}\t{int(nc)}\t{int(ns)}\n"
        for sid, nc, ns in sorted(rows, key=lambda x: x[0])
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def audit_exports_and_selected(
    export_dir: Path,
    incomplete_zip: Path,
    stage1_v2_jsonl: Path,
) -> Dict[str, Any]:
    # 1. Completed export tars (一号站 + 二号站)
    completed: Dict[str, Dict[str, Any]] = {}
    alias_conflicts = 0
    for tar_name, station in (("一号站.tar.gz", "one"), ("二号站.tar.gz", "two")):
        with tarfile.open(export_dir / tar_name, "r:gz") as tf:
            for m in tf.getmembers():
                if not (m.isfile() and m.name.endswith(".jsonl")):
                    continue
                for line in tf.extractfile(m).read().decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    src = json.loads(line)["source"]
                    sid = str(src["source_id"])
                    if sid in completed:
                        continue
                    ars = src.get("author_rates") or []
                    rates_set = {a.get("rate") for a in ars if isinstance(a, dict)}
                    has_conflict = len(ars) > 1 and len(rates_set) > 1
                    if has_conflict:
                        alias_conflicts += 1
                    completed[sid] = {
                        "station": station,
                        "direction_band": src.get("direction_band"),
                        "direction_basis": src.get("direction_basis"),
                        "pre_solve_rate": src.get("pre_solve_rate"),
                        "pre_solve_rate_n_scored": src.get("pre_solve_rate_n_scored"),
                        "llama8b_solve_rate": src.get("llama8b_solve_rate"),
                        "author_rates_origins": sorted(
                            {str(a.get("origin")) for a in ars if isinstance(a, dict)}
                        ),
                        "author_rates_roles": sorted(
                            {str(a.get("role")) for a in ars if isinstance(a, dict)}
                        ),
                        "n_aliases": len(ars),
                        "alias_rate_conflict": has_conflict,
                    }

    # 2. Incomplete sources zip
    inc_pre: List[float] = []
    inc_l8: List[float] = []
    inc_n_scored: collections.Counter[str] = collections.Counter()
    inc_overlap = 0
    inc_aliases_multi = 0
    inc_alias_conflicts = 0
    with zipfile.ZipFile(incomplete_zip) as zf:
        with zf.open("sources.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                src = json.loads(line)
                sid = str(src["source_id"])
                if sid in completed:
                    inc_overlap += 1
                inc_n_scored[str(src.get("pre_solve_rate_n_scored"))] += 1
                if src.get("pre_solve_rate") is not None:
                    inc_pre.append(float(src["pre_solve_rate"]))
                if src.get("llama8b_solve_rate") is not None:
                    inc_l8.append(float(src["llama8b_solve_rate"]))
                ars = src.get("author_rates") or []
                if len(ars) > 1:
                    inc_aliases_multi += 1
                    if len({a.get("rate") for a in ars if isinstance(a, dict)}) > 1:
                        inc_alias_conflicts += 1

    # 3. Selected 2,100 sources (T-023 Stage 1 v2)
    sel_rows: List[Dict[str, Any]] = []
    sel_table_tuples: List[Tuple[str, int, int]] = []
    sel_pre_train: List[float] = []
    sel_pre_dev: List[float] = []
    sel_l8_all: List[float] = []
    sel_alias_multi = 0
    sel_alias_conflicts = 0
    k16_hist: collections.Counter[int] = collections.Counter()
    by_split_floor_le1: collections.Counter[str] = collections.Counter()
    by_split_ceil_ge15: collections.Counter[str] = collections.Counter()
    dir_vs_pre: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    with stage1_v2_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r["source_id"])
            split = str(r["split"])
            direction = str(r["direction"])
            pr = r.get("source_pre_solve_rate_old")
            l8 = r.get("source_llama8b_solve_rate_old")
            if pr is None:
                raise ValueError(f"Missing source_pre_solve_rate_old for {sid}")
            pr_f = float(pr)
            k16 = int(round(pr_f * 16))
            if abs(pr_f - k16 / 16.0) > 1e-9:
                raise ValueError(f"Non k/16 pre_solve_rate for {sid}: {pr_f}")
            exp_rec = completed.get(sid)
            if exp_rec is None:
                raise ValueError(f"Selected source_id {sid} missing from export tars")
            if abs(float(exp_rec["pre_solve_rate"]) - pr_f) > 1e-12:
                raise ValueError(f"Export vs Stage1 pre_solve_rate mismatch for {sid}")
            if exp_rec["n_aliases"] > 1:
                sel_alias_multi += 1
            if exp_rec["alias_rate_conflict"]:
                sel_alias_conflicts += 1

            sel_rows.append(r)
            sel_table_tuples.append((sid, k16, 16))
            k16_hist[k16] += 1
            if split == "train":
                sel_pre_train.append(pr_f)
            else:
                sel_pre_dev.append(pr_f)
            if l8 is not None:
                sel_l8_all.append(float(l8))
            if k16 <= 1:
                by_split_floor_le1[split] += 1
            if k16 >= 15:
                by_split_ceil_ge15[split] += 1
            dir_vs_pre[direction]["total"] += 1
            if k16 <= 4:
                dir_vs_pre[direction]["k_0_to_4"] += 1
            elif k16 <= 11:
                dir_vs_pre[direction]["k_5_to_11"] += 1
            else:
                dir_vs_pre[direction]["k_12_to_16"] += 1

    table_sha256 = compute_source_pretest_table_sha256(sel_table_tuples)

    sel_pre_all = sel_pre_train + sel_pre_dev
    comp_pre_all = [float(v["pre_solve_rate"]) for v in completed.values()]
    comp_l8_all = [float(v["llama8b_solve_rate"]) for v in completed.values()]

    # Stage 2 resource estimate (2,100 sources x 16 = 33,600 rollouts)
    # T-023 actuals: 3,736 variants (59,776 rollouts), hybrid_correct=22,026 (36.85%),
    # non-dev 27B fallback calls = 38,607 (64.59% of rollouts), total GPU phase = 10,950.86s.
    total_rollouts_stage2 = len(sel_rows) * 16
    mean_old_p = sum(sel_pre_all) / len(sel_pre_all)
    # Rough expectation: rule-pass ~ old_pre_solve_rate (actually slightly lower or higher depending on bf16/0.8.0),
    # so 27B fallback ~ (1 - rule_pass) * 33,600, plus ~2pp conservative margin.
    est_27b_calls_from_old_p = int(round(total_rollouts_stage2 * (1.0 - mean_old_p)))
    est_27b_calls_t023_ratio = int(round(total_rollouts_stage2 * (38607 / 59776)))
    # T-023 GPU phase throughput: 59,776 rollouts + 39,661 27B calls in 10,950.86s (~182.5 min)
    # Ratio of rollouts is 33600 / 59776 = 0.5621; without 1,054 dev audit calls, ~95-110 min with DP=2 + TP=2.
    est_minutes_dp2_tp2 = round((10950.86 / 60.0) * (total_rollouts_stage2 / 59776.0), 1)

    return {
        "completed_exports": {
            "n_unique_sources": len(completed),
            "n_multi_alias_sources": sum(1 for v in completed.values() if v["n_aliases"] > 1),
            "n_alias_rate_conflicts": alias_conflicts,
            "pre_solve_rate_n_scored_values": dict(
                collections.Counter(str(v["pre_solve_rate_n_scored"]) for v in completed.values())
            ),
            "pre_solve_rate_quantiles": _quantiles(comp_pre_all),
            "llama8b_solve_rate_quantiles": _quantiles(comp_l8_all),
        },
        "incomplete_sources_zip": {
            "n_sources": len(inc_pre),
            "overlap_with_completed": inc_overlap,
            "n_multi_alias_sources": inc_aliases_multi,
            "n_alias_rate_conflicts": inc_alias_conflicts,
            "pre_solve_rate_n_scored_values": dict(inc_n_scored),
            "pre_solve_rate_quantiles": _quantiles(inc_pre),
            "llama8b_solve_rate_quantiles": _quantiles(inc_l8),
        },
        "selected_2100": {
            "n_sources": len(sel_rows),
            "n_train": len(sel_pre_train),
            "n_dev": len(sel_pre_dev),
            "n_multi_alias_sources": sel_alias_multi,
            "n_alias_rate_conflicts": sel_alias_conflicts,
            "old_source_pretest_table_sha256": table_sha256,
            "pre_solve_rate_all_quantiles": _quantiles(sel_pre_all),
            "pre_solve_rate_train_quantiles": _quantiles(sel_pre_train),
            "pre_solve_rate_dev_quantiles": _quantiles(sel_pre_dev),
            "llama8b_solve_rate_all_quantiles": _quantiles(sel_l8_all),
            "k16_histogram": {str(k): k16_hist[k] for k in range(17)},
            "floor_le_1_over_16_by_split": dict(by_split_floor_le1),
            "ceiling_ge_15_over_16_by_split": dict(by_split_ceil_ge15),
            "direction_band_vs_k16": {k: dict(v) for k, v in sorted(dir_vs_pre.items())},
        },
        "stage2_estimates": {
            "n_sources": len(sel_rows),
            "k_samples": 16,
            "total_student_rollouts": total_rollouts_stage2,
            "est_27b_calls_from_old_p": est_27b_calls_from_old_p,
            "est_27b_calls_t023_ratio": est_27b_calls_t023_ratio,
            "est_gpu_minutes_at_t023_dp2_tp2_rate": est_minutes_dp2_tp2,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit cu04 original source pretest availability (zero text).")
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/home/m20265227048/项目/数学迁移/数据集/T023_private_sparse/data/训练集正式出口"),
    )
    ap.add_argument(
        "--incomplete-zip",
        type=Path,
        default=Path("/home/m20265227048/项目/数学迁移/数据集/T023_private_sparse/data/训练来源/训练集未完成来源_7246_入站包.zip"),
    )
    ap.add_argument(
        "--stage1-v2-jsonl",
        type=Path,
        default=Path("/home/m20265227048/项目/数学迁移/数据集/routeA_stage1_pack_v2_20261125/routeA_stage1_dataset.jsonl"),
    )
    args = ap.parse_args()
    res = audit_exports_and_selected(args.export_dir, args.incomplete_zip, args.stage1_v2_jsonl)
    print(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
