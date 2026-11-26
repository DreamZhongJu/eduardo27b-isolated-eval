"""Convert T-023 measured JSONL into veRL-compatible train/val parquet files on Pro 5000.

Privacy guarantee:
    Never prints or writes problem statements, reference answers, or student responses
    to stdout, stderr, or summary JSON files. Only numeric counts and SHA-256 hashes
    are emitted.
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

EXPECTED_T023_JSONL_SHA256 = (
    "9ff8449b781906290f9b671bc5be1982ca31e5401ebfba485474d475a60752ce"
)
DEFAULT_T023_JSONL = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/"
    "t023_stage2_20261125_163319/routeA_stage2_measured.jsonl"
)
CEILING_MIN_CORRECT = 15

_STUDENT_OPENING_TEMPLATE = (
    "This is the problem I am trying to understand: {problem}"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def is_ceiling_group(variants: Sequence[Dict[str, Any]]) -> bool:
    """Return True iff every variant in the source group has n_correct_hybrid >= 15."""
    if not variants:
        return False
    return all(int(v["n_correct_hybrid"]) >= CEILING_MIN_CORRECT for v in variants)


def _select_middle_variant_dict(variants: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(
        enumerate(variants),
        key=lambda pair: (int(pair[1]["ladder_rank"]), pair[0]),
    )
    sorted_vars = [v for _, v in ordered]
    n = len(sorted_vars)
    if n == 0:
        raise ValueError("source group has no variants")
    if n % 2 == 1:
        return sorted_vars[n // 2]
    return sorted_vars[(n - 1) // 2]


def _resolve_source_pre_rate(row: Dict[str, Any], mid_var: Dict[str, Any]) -> float:
    for key in ("source_pre_solve_rate_old", "source_llama8b_solve_rate_old", "source_pre_solve_rate"):
        val = row.get(key)
        if val is not None:
            return float(val)
    if mid_var.get("p_new") is not None:
        return float(mid_var["p_new"])
    return 0.0


def _compact_variants(variants: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        enumerate(variants),
        key=lambda pair: (int(pair[1]["ladder_rank"]), pair[0]),
    )
    compact: List[Dict[str, Any]] = []
    for _, v in ordered:
        compact.append(
            {
                "variant_id": str(v["variant_id"]),
                "problem": str(v["problem"]),
                "answer": str(v["answer"]),
                "b": float(v["b"]),
                "ladder_rank": int(v["ladder_rank"]),
                "p_new": float(v["p_new"]) if v.get("p_new") is not None else None,
                "n_correct_hybrid": int(v["n_correct_hybrid"]),
            }
        )
    return compact


def build_parquet_row(
    row: Dict[str, Any],
    *,
    teacher_system_prompt: str,
    parquet_split: str,
) -> Dict[str, Any]:
    """Build one veRL-compatible parquet row matching eduardo/process_dataset.py:_row."""
    source_problem = str(row["source_problem"])
    source_answer = str(row["source_answer"])
    source_id = str(row["source_id"])
    direction = str(row.get("direction") or "")
    variants = _compact_variants(row["variants"])
    mid_var = _select_middle_variant_dict(variants)
    source_pre = _resolve_source_pre_rate(row, mid_var)
    mid_p = float(mid_var["p_new"]) if mid_var.get("p_new") is not None else 0.5

    prompt = [
        {
            "role": "system",
            "content": teacher_system_prompt.format(problem=source_problem),
        },
        {
            "role": "user",
            "content": _STUDENT_OPENING_TEMPLATE.format(problem=source_problem),
        },
    ]

    return {
        "data_source": "bigmath",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": source_answer,
        },
        "extra_info": {
            "problem": source_problem,
            "answer": source_answer,
            "source_id": source_id,
            "split": parquet_split,
            "solve_rate": float(source_pre),
            "pre_solve_rate": float(source_pre),
            "transfer_problem": str(mid_var["problem"]),
            "transfer_answer": str(mid_var["answer"]),
            "pre_solve_rate_transfer_problem": float(mid_p),
            # Serialized JSON keeps schema flat for PyArrow/datasets & ladder.extract_ladder_variants.
            "variants": json.dumps(variants, ensure_ascii=False, separators=(",", ":")),
            "num_variants": len(variants),
            "direction": direction,
            "interaction_kwargs": {
                "name": "eduardo",
                "problem": source_problem,
                "answer": source_answer,
                "ground_truth": source_answer,
            },
        },
        "agent_name": "eduardo_agent",
    }


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_n_vars: collections.Counter[str] = collections.Counter()
    by_dir: collections.Counter[str] = collections.Counter()
    by_stratum: collections.Counter[str] = collections.Counter()
    n_variants_total = 0
    for r in rows:
        nv = len(r["variants"])
        direction = str(r.get("direction") or "unknown")
        n_variants_total += nv
        by_n_vars[str(nv)] += 1
        by_dir[direction] += 1
        by_stratum[f"{nv}|{direction}"] += 1
    return {
        "n_sources": len(rows),
        "n_variants": n_variants_total,
        "by_num_variants": dict(sorted(by_n_vars.items(), key=lambda kv: int(kv[0]))),
        "by_direction": dict(sorted(by_dir.items())),
        "by_stratum_num_variants_direction": dict(sorted(by_stratum.items())),
    }


def prepare_routea_parquets(
    *,
    input_jsonl: Path,
    out_dir: Path,
    expected_sha256: Optional[str] = EXPECTED_T023_JSONL_SHA256,
    teacher_system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Read T-023 measured JSONL, filter train ceiling groups, and write train/val parquets."""
    import pandas as pd

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"T-023 measured JSONL not found: {input_jsonl}")
    actual_sha = sha256_file(input_jsonl)
    if expected_sha256 and actual_sha != expected_sha256.lower():
        raise ValueError(
            f"T-023 JSONL sha256 mismatch: actual={actual_sha} expected={expected_sha256}"
        )

    if teacher_system_prompt is None:
        from eduardo.utils import load_prompt

        teacher_system_prompt = load_prompt("teacher_prompt")

    train_raw: List[Dict[str, Any]] = []
    train_kept: List[Dict[str, Any]] = []
    train_removed_ceiling: List[Dict[str, Any]] = []
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
                train_raw.append(rec)
                if is_ceiling_group(rec.get("variants", [])):
                    train_removed_ceiling.append(rec)
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

    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    train_out_rows = [
        build_parquet_row(r, teacher_system_prompt=teacher_system_prompt, parquet_split="train")
        for r in train_kept
    ]
    val_out_rows = [
        build_parquet_row(r, teacher_system_prompt=teacher_system_prompt, parquet_split="val")
        for r in dev_raw
    ]

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

    dev_ceiling_count = sum(1 for r in dev_raw if is_ceiling_group(r.get("variants", [])))
    summary = {
        "input_jsonl": str(input_jsonl),
        "input_jsonl_sha256": actual_sha,
        "ceiling_rule": f"all variants n_correct_hybrid >= {CEILING_MIN_CORRECT}",
        "other_split_ignored": other_split_count,
        "train_before_filter": _summarize_rows(train_raw),
        "train_removed_ceiling": _summarize_rows(train_removed_ceiling),
        "train_after_filter": _summarize_rows(train_kept),
        "val_dev": {
            **_summarize_rows(dev_raw),
            "ceiling_groups_kept": dev_ceiling_count,
        },
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=EXPECTED_T023_JSONL_SHA256,
    )
    parser.add_argument("--skip-sha-check", action="store_true")
    args = parser.parse_args()

    exp_sha = None if args.skip_sha_check else args.expected_sha256
    summary = prepare_routea_parquets(
        input_jsonl=args.input_jsonl,
        out_dir=args.out_dir,
        expected_sha256=exp_sha,
    )
    tr_before = summary["train_before_filter"]
    tr_rm = summary["train_removed_ceiling"]
    tr_after = summary["train_after_filter"]
    val = summary["val_dev"]
    print(
        f"PARQUET_READY train={tr_after['n_sources']}/{tr_before['n_sources']} "
        f"(removed_ceiling={tr_rm['n_sources']}, variants={tr_after['n_variants']}) "
        f"val={val['n_sources']} (variants={val['n_variants']})"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
