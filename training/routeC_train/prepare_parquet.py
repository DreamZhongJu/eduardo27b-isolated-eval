"""Build T-034 Route C train/val parquet files on Pro 5000.

Reuses T-031 routing/pretest helpers (``routeC_data.routeC_inventory``,
``routeA_data.rasch_ladder``) and T-026 row builder (``training.routeA_train.prepare_parquet``)
so ``fallback`` rows match T-026 field-for-field and the source set, row count,
and row order stay identical to T-026.

Privacy guarantee:
    Never prints or writes problem statements, answers, or student responses
    to stdout, stderr, or summary JSON files. Only numeric counts, branch
    distributions, and SHA-256 hashes are emitted.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from routeA_data.rasch_ladder import (
    compute_smoothed_p_and_b,
    infer_generation_tier,
    sort_ladder_variants,
)
from routeC_data.routeC_inventory import (
    _load_source_pretest_map,
    is_t026_ceiling_group,
)
from training.routeA_train.prepare_parquet import (
    EXPECTED_T023_JSONL_SHA256,
    _STUDENT_OPENING_TEMPLATE,
    _compact_variants,
    _select_middle_variant_dict,
    atomic_write_json,
    build_parquet_row,
    sha256_file,
)

EXPECTED_T031_SOURCE_PRETEST_SHA256 = (
    "a031276fe89069854a59f2782def149964ca5d52626b67f7648bce7f7f1e3702"
)
EXPECTED_T026_TRAIN_PARQUET_SHA256 = (
    "1e85fd0f2d1282b409e8617b62591fd157b165b397aa3ad514ba3a95d1adee22"
)
EXPECTED_T026_VAL_PARQUET_SHA256 = (
    "9a6808eacdf3facfdeb39566dce26f13f6b46c8b9c332e1fd2853709f398355f"
)

DEFAULT_T023_JSONL = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/"
    "t023_stage2_20261125_163319/routeA_stage2_measured.jsonl"
)
DEFAULT_T031_SOURCE_PRETEST = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/"
    "t031_stage2_20261128_095549/source_pretest_measured.jsonl"
)
DEFAULT_T026_TRAIN_PARQUET = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/data/routeA_t026/train.parquet"
)
DEFAULT_T026_VAL_PARQUET = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/data/routeA_t026/val.parquet"
)

ROUTE_TAU = 0.4
ROUTE_MARGIN = 1.0
EXPECTED_ROUTEC_TRAIN_COUNTS = {"easy": 289, "hard": 801, "fallback": 805}
EXPECTED_ROUTEC_VAL_COUNTS = {"easy": 15, "hard": 46, "fallback": 39}


def _require_non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty required text field: {label}")
    return value


def _build_routed_row(
    *,
    source_id: str,
    taught_id: str,
    route_branch: str,
    taught_problem: str,
    taught_answer: str,
    taught_pre_rate: float,
    direction: str,
    ladder_variants: Sequence[Dict[str, Any]],
    teacher_system_prompt: str,
    parquet_split: str,
) -> Dict[str, Any]:
    compact_ladder = _compact_variants(ladder_variants)
    mid_var = _select_middle_variant_dict(compact_ladder)
    mid_p = float(mid_var["p_new"]) if mid_var.get("p_new") is not None else 0.5

    prompt = [
        {
            "role": "system",
            "content": teacher_system_prompt.format(problem=taught_problem),
        },
        {
            "role": "user",
            "content": _STUDENT_OPENING_TEMPLATE.format(problem=taught_problem),
        },
    ]

    return {
        "data_source": "bigmath",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": taught_answer,
        },
        "extra_info": {
            "problem": taught_problem,
            "answer": taught_answer,
            "source_id": source_id,
            "taught_id": taught_id,
            "route_branch": route_branch,
            "split": parquet_split,
            "solve_rate": float(taught_pre_rate),
            "pre_solve_rate": float(taught_pre_rate),
            "transfer_problem": str(mid_var["problem"]),
            "transfer_answer": str(mid_var["answer"]),
            "pre_solve_rate_transfer_problem": float(mid_p),
            "variants": json.dumps(compact_ladder, ensure_ascii=False, separators=(",", ":")),
            "num_variants": len(compact_ladder),
            "direction": direction,
            "interaction_kwargs": {
                "name": "eduardo",
                "problem": taught_problem,
                "answer": taught_answer,
                "ground_truth": taught_answer,
            },
        },
        "agent_name": "eduardo_agent",
    }


def route_and_build_parquet_row(
    row: Dict[str, Any],
    *,
    source_pretest: Dict[str, Dict[str, Any]],
    teacher_system_prompt: str,
    parquet_split: str,
    tau: float = ROUTE_TAU,
    margin: float = ROUTE_MARGIN,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply Route C routing (T-034) and build one veRL-compatible parquet row.

    Returns ``(parquet_row, meta_without_text)``.
    """
    source_id = _require_non_empty_text(str(row.get("source_id") or ""), "source_id")
    source_problem = _require_non_empty_text(row.get("source_problem"), f"{source_id}.source_problem")
    source_answer = _require_non_empty_text(row.get("source_answer"), f"{source_id}.source_answer")
    direction = str(row.get("direction") or "")

    if source_id not in source_pretest:
        raise ValueError(f"source_id missing from T-031 source pretest: {source_id}")
    sp = source_pretest[source_id]
    n_orig = int(sp["n_correct"])
    n_orig_scored = int(sp["n_scored"])
    p_orig = float(sp["p_orig"])
    b_orig = float(sp["b_orig"])

    raw_variants = row.get("variants") or []
    if not raw_variants:
        raise ValueError(f"source group {source_id} has no variants")

    prepared_vars: List[Dict[str, Any]] = []
    for idx, var in enumerate(raw_variants):
        vid = _require_non_empty_text(str(var.get("variant_id") or f"{source_id}_v{idx}"), "variant_id")
        v_prob = _require_non_empty_text(var.get("problem"), f"{source_id}/{vid}.problem")
        v_ans = _require_non_empty_text(var.get("answer"), f"{source_id}/{vid}.answer")
        n_hyb = int(var["n_correct_hybrid"])
        n_scored = int(var.get("n_scored", 16))
        p_var, b_var = compute_smoothed_p_and_b(n_hyb, n_scored)
        if var.get("p_new") is not None and abs(float(var["p_new"]) - p_var) > 1e-5:
            raise ValueError(
                f"Variant p_new mismatch for {source_id}/{vid}: {var.get('p_new')} vs {p_var}"
            )
        origin = str(var.get("origin") or "")
        gen_tier = str(
            var.get("generation_tier")
            or infer_generation_tier(origin, var.get("generation_tag"))
        )
        prepared_vars.append(
            {
                "variant_id": vid,
                "problem": v_prob,
                "answer": v_ans,
                "n_correct_hybrid": n_hyb,
                "n_scored": n_scored,
                "p_new": p_var,
                "b": b_var,
                "delta_b": b_var - b_orig,
                "origin": origin,
                "generation_tier": gen_tier,
                "generation_tag": var.get("generation_tag"),
            }
        )

    # Branch decision matching T-031 / T-034 specification
    if p_orig >= tau:
        qualifying_easy = [v for v in prepared_vars if v["delta_b"] >= margin]
        if qualifying_easy:
            ladder_pool = [v for v in prepared_vars if v["delta_b"] > -margin]
            if not ladder_pool:
                raise ValueError(f"easy branch post-test ladder unexpectedly empty for {source_id}")
            sorted_ladder, _ = sort_ladder_variants(ladder_pool)
            out_row = _build_routed_row(
                source_id=source_id,
                taught_id=source_id,
                route_branch="easy",
                taught_problem=source_problem,
                taught_answer=source_answer,
                taught_pre_rate=p_orig,
                direction=direction,
                ladder_variants=sorted_ladder,
                teacher_system_prompt=teacher_system_prompt,
                parquet_split=parquet_split,
            )
            return out_row, {
                "source_id": source_id,
                "route_branch": "easy",
                "taught_id": source_id,
                "num_posttest_levels": len(sorted_ladder),
            }
    else:
        qualifying_hard = [
            (idx, v) for idx, v in enumerate(prepared_vars) if v["delta_b"] <= -margin
        ]
        if qualifying_hard:
            # Pick variant with delta_b <= -m closest to original (max b, tie-break smallest variant_id).
            taught_idx, v_taught = min(
                qualifying_hard,
                key=lambda pair: (-float(pair[1]["b"]), str(pair[1]["variant_id"]), pair[0]),
            )
            orig_rung = {
                "variant_id": source_id,
                "problem": source_problem,
                "answer": source_answer,
                "n_correct_hybrid": n_orig,
                "n_scored": n_orig_scored,
                "p_new": p_orig,
                "b": b_orig,
                "origin": "original",
                "generation_tier": "original",
                "generation_tag": "original",
            }
            untaught = [v for idx, v in enumerate(prepared_vars) if idx != taught_idx]
            sorted_ladder, _ = sort_ladder_variants([orig_rung, *untaught])
            taught_id = str(v_taught["variant_id"])
            out_row = _build_routed_row(
                source_id=source_id,
                taught_id=taught_id,
                route_branch="hard",
                taught_problem=str(v_taught["problem"]),
                taught_answer=str(v_taught["answer"]),
                taught_pre_rate=float(v_taught["p_new"]),
                direction=direction,
                ladder_variants=sorted_ladder,
                teacher_system_prompt=teacher_system_prompt,
                parquet_split=parquet_split,
            )
            return out_row, {
                "source_id": source_id,
                "route_branch": "hard",
                "taught_id": taught_id,
                "num_posttest_levels": len(sorted_ladder),
            }

    # Fallback: field-for-field identical to T-026 build_parquet_row, plus route_branch/taught_id
    fallback_row = build_parquet_row(
        row,
        teacher_system_prompt=teacher_system_prompt,
        parquet_split=parquet_split,
    )
    fallback_row["extra_info"]["route_branch"] = "fallback"
    fallback_row["extra_info"]["taught_id"] = source_id
    return fallback_row, {
        "source_id": source_id,
        "route_branch": "fallback",
        "taught_id": source_id,
        "num_posttest_levels": int(fallback_row["extra_info"]["num_variants"]),
    }


def _summarize_split_routing(metas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    branch_counts = collections.Counter(m["route_branch"] for m in metas)
    hard_levels = collections.Counter()
    easy_levels = collections.Counter()
    for m in metas:
        lv = int(m["num_posttest_levels"])
        bucket = "3+" if lv >= 3 else str(lv)
        if m["route_branch"] == "hard":
            hard_levels[bucket] += 1
        elif m["route_branch"] == "easy":
            easy_levels[bucket] += 1

    def _level_dict(c: collections.Counter[str]) -> Dict[str, int]:
        return {
            "1": int(c.get("1", 0)),
            "2": int(c.get("2", 0)),
            "3+": int(c.get("3+", 0)),
        }

    return {
        "n_sources": len(metas),
        "branch_counts": {
            "easy": int(branch_counts.get("easy", 0)),
            "hard": int(branch_counts.get("hard", 0)),
            "fallback": int(branch_counts.get("fallback", 0)),
        },
        "hard_posttest_levels": _level_dict(hard_levels),
        "easy_posttest_levels": _level_dict(easy_levels),
    }


def _verify_against_t026_parquet(
    *,
    routec_rows: Sequence[Dict[str, Any]],
    t026_parquet_path: Path,
    expected_t026_sha256: Optional[str],
) -> Dict[str, Any]:
    import pandas as pd

    if not t026_parquet_path.is_file():
        raise FileNotFoundError(f"T-026 parquet not found: {t026_parquet_path}")
    actual_sha = sha256_file(t026_parquet_path)
    if expected_t026_sha256 and actual_sha != expected_t026_sha256.lower():
        raise ValueError(
            f"T-026 parquet sha256 mismatch ({t026_parquet_path.name}): "
            f"actual={actual_sha} expected={expected_t026_sha256}"
        )
    df_ref = pd.read_parquet(t026_parquet_path)
    ref_records = df_ref.to_dict(orient="records")
    if len(routec_rows) != len(ref_records):
        raise ValueError(
            f"Row count mismatch vs T-026 ({t026_parquet_path.name}): "
            f"{len(routec_rows)} != {len(ref_records)}"
        )

    fallback_checked = 0
    for idx, (rc_row, ref_row) in enumerate(zip(routec_rows, ref_records)):
        rc_ex = rc_row["extra_info"]
        ref_ex = ref_row["extra_info"]
        if str(rc_ex["source_id"]) != str(ref_ex["source_id"]):
            raise ValueError(
                f"Order/source_id mismatch vs T-026 at row {idx}: "
                f"{rc_ex['source_id']} != {ref_ex['source_id']}"
            )
        if rc_ex["route_branch"] == "fallback":
            fallback_checked += 1
            if list(rc_row["prompt"]) != list(ref_row["prompt"]):
                raise ValueError(f"Fallback prompt mismatch vs T-026 at row {idx}")
            if dict(rc_row["reward_model"]) != dict(ref_row["reward_model"]):
                raise ValueError(f"Fallback reward_model mismatch vs T-026 at row {idx}")
            for field in (
                "problem",
                "answer",
                "source_id",
                "split",
                "variants",
                "num_variants",
                "transfer_problem",
                "transfer_answer",
            ):
                if rc_ex[field] != ref_ex[field]:
                    raise ValueError(f"Fallback extra_info.{field} mismatch vs T-026 at row {idx}")
            if dict(rc_ex["interaction_kwargs"]) != dict(ref_ex["interaction_kwargs"]):
                raise ValueError(f"Fallback interaction_kwargs mismatch vs T-026 at row {idx}")

    return {
        "t026_parquet_sha256": actual_sha,
        "rows": len(routec_rows),
        "order_match": True,
        "fallback_exact_match": True,
        "fallback_checked": fallback_checked,
    }


def prepare_routec_parquets(
    *,
    input_jsonl: Path,
    source_pretest_jsonl: Path,
    out_dir: Path,
    t026_train_parquet: Optional[Path] = DEFAULT_T026_TRAIN_PARQUET,
    t026_val_parquet: Optional[Path] = DEFAULT_T026_VAL_PARQUET,
    expected_t023_sha256: Optional[str] = EXPECTED_T023_JSONL_SHA256,
    expected_source_pretest_sha256: Optional[str] = EXPECTED_T031_SOURCE_PRETEST_SHA256,
    expected_t026_train_sha256: Optional[str] = EXPECTED_T026_TRAIN_PARQUET_SHA256,
    expected_t026_val_sha256: Optional[str] = EXPECTED_T026_VAL_PARQUET_SHA256,
    teacher_system_prompt: Optional[str] = None,
    tau: float = ROUTE_TAU,
    margin: float = ROUTE_MARGIN,
    enforce_t031_counts: bool = True,
) -> Dict[str, Any]:
    """Generate Route C train/val parquets and verify parity against T-031 & T-026."""
    import pandas as pd

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"T-023 measured JSONL not found: {input_jsonl}")
    t023_sha = sha256_file(input_jsonl)
    if expected_t023_sha256 and t023_sha != expected_t023_sha256.lower():
        raise ValueError(f"T-023 JSONL sha256 mismatch: actual={t023_sha} expected={expected_t023_sha256}")

    if not source_pretest_jsonl.is_file():
        raise FileNotFoundError(f"T-031 source pretest JSONL not found: {source_pretest_jsonl}")
    sp_map, sp_sha = _load_source_pretest_map(
        source_pretest_jsonl,
        expected_sha256=expected_source_pretest_sha256,
    )

    if teacher_system_prompt is None:
        from eduardo.utils import load_prompt

        teacher_system_prompt = load_prompt("teacher_prompt")

    train_kept: List[Dict[str, Any]] = []
    train_removed_ceiling = 0
    dev_raw: List[Dict[str, Any]] = []
    other_split_count = 0

    with input_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            split = str(rec.get("split") or "")
            if split == "train":
                if is_t026_ceiling_group(rec.get("variants", [])):
                    train_removed_ceiling += 1
                else:
                    train_kept.append(rec)
            elif split == "dev":
                dev_raw.append(rec)
            else:
                other_split_count += 1

    if not train_kept:
        raise ValueError("No training rows remained after ceiling filtering")
    if not dev_raw:
        raise ValueError("No dev rows found in T-023 JSONL")

    train_out_rows: List[Dict[str, Any]] = []
    train_metas: List[Dict[str, Any]] = []
    for rec in train_kept:
        out_r, meta = route_and_build_parquet_row(
            rec,
            source_pretest=sp_map,
            teacher_system_prompt=teacher_system_prompt,
            parquet_split="train",
            tau=tau,
            margin=margin,
        )
        train_out_rows.append(out_r)
        train_metas.append(meta)

    val_out_rows: List[Dict[str, Any]] = []
    val_metas: List[Dict[str, Any]] = []
    for rec in dev_raw:
        out_r, meta = route_and_build_parquet_row(
            rec,
            source_pretest=sp_map,
            teacher_system_prompt=teacher_system_prompt,
            parquet_split="val",
            tau=tau,
            margin=margin,
        )
        val_out_rows.append(out_r)
        val_metas.append(meta)

    train_stats = _summarize_split_routing(train_metas)
    val_stats = _summarize_split_routing(val_metas)

    if enforce_t031_counts:
        if train_stats["branch_counts"] != EXPECTED_ROUTEC_TRAIN_COUNTS:
            raise ValueError(
                f"Train branch counts mismatch vs T-031: "
                f"got={train_stats['branch_counts']} expected={EXPECTED_ROUTEC_TRAIN_COUNTS}"
            )
        if val_stats["branch_counts"] != EXPECTED_ROUTEC_VAL_COUNTS:
            raise ValueError(
                f"Val branch counts mismatch vs T-031: "
                f"got={val_stats['branch_counts']} expected={EXPECTED_ROUTEC_VAL_COUNTS}"
            )

    t026_parity: Dict[str, Any] = {}
    if t026_train_parquet is not None:
        t026_parity["train"] = _verify_against_t026_parquet(
            routec_rows=train_out_rows,
            t026_parquet_path=t026_train_parquet,
            expected_t026_sha256=expected_t026_train_sha256,
        )
    if t026_val_parquet is not None:
        t026_parity["val"] = _verify_against_t026_parquet(
            routec_rows=val_out_rows,
            t026_parquet_path=t026_val_parquet,
            expected_t026_sha256=expected_t026_val_sha256,
        )

    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    def _write_parquet(records: List[Dict[str, Any]], dest: Path) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=".parquet")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            pd.DataFrame(records).to_parquet(tmp, index=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    _write_parquet(train_out_rows, train_path)
    _write_parquet(val_out_rows, val_path)

    summary = {
        "input_jsonl": str(input_jsonl),
        "input_jsonl_sha256": t023_sha,
        "source_pretest_jsonl": str(source_pretest_jsonl),
        "source_pretest_sha256": sp_sha,
        "routing_params": {"tau": float(tau), "margin": float(margin)},
        "train_removed_ceiling": train_removed_ceiling,
        "other_split_ignored": other_split_count,
        "missing_variant_fields": 0,
        "train": train_stats,
        "val": val_stats,
        "t026_parity": t026_parity,
        "files": {
            "train.parquet": {
                "path": str(train_path),
                "rows": len(train_out_rows),
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
            },
            "val.parquet": {
                "path": str(val_path),
                "rows": len(val_out_rows),
                "bytes": val_path.stat().st_size,
                "sha256": sha256_file(val_path),
            },
        },
    }
    atomic_write_json(out_dir / "dataset_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_T023_JSONL)
    parser.add_argument("--source-pretest-jsonl", type=Path, default=DEFAULT_T031_SOURCE_PRETEST)
    parser.add_argument("--t026-train-parquet", type=Path, default=DEFAULT_T026_TRAIN_PARQUET)
    parser.add_argument("--t026-val-parquet", type=Path, default=DEFAULT_T026_VAL_PARQUET)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skip-sha-check", action="store_true")
    parser.add_argument("--skip-t026-parity", action="store_true")
    parser.add_argument("--skip-t031-count-check", action="store_true")
    args = parser.parse_args()

    summary = prepare_routec_parquets(
        input_jsonl=args.input_jsonl,
        source_pretest_jsonl=args.source_pretest_jsonl,
        out_dir=args.out_dir,
        t026_train_parquet=None if args.skip_t026_parity else args.t026_train_parquet,
        t026_val_parquet=None if args.skip_t026_parity else args.t026_val_parquet,
        expected_t023_sha256=None if args.skip_sha_check else EXPECTED_T023_JSONL_SHA256,
        expected_source_pretest_sha256=None if args.skip_sha_check else EXPECTED_T031_SOURCE_PRETEST_SHA256,
        expected_t026_train_sha256=None if args.skip_sha_check else EXPECTED_T026_TRAIN_PARQUET_SHA256,
        expected_t026_val_sha256=None if args.skip_sha_check else EXPECTED_T026_VAL_PARQUET_SHA256,
        enforce_t031_counts=not args.skip_t031_count_check,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
