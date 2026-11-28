#!/usr/bin/env python3
"""P2: Stratified blind sampling of 100 conversations + sealed key gate.

Rules enforced:
- Samples 50 from A-group (T-020) + 50 from Eduardo-27B (T-022), stratified across
  5 pre_solve_rate quintiles (10 per stratum per model), preferring disjoint row_index
  so raters cannot compare the same problem across models.
- Assigns neutral IDs D001..D100 after global shuffle (seed=42).
- Draws 30 IDs stratified across (model, stratum) for Claude's independent dual-annotation.
- Writes blind dialogues (no model name, no judge verdicts, no problem metadata) and
  sealed mapping key (`mapping_key.sealed.json`).
- `unseal_mapping_key` hard-refuses to unseal until 100/100 user annotations and
  30/30 Claude annotations are valid and complete.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from common_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    print_machine_summary,
    sha256_file,
    strip_teacher_controls,
)

VALID_CATEGORIES: tuple[str, ...] = (
    "硬报答案",
    "整段代劳",
    "回声洗白",
    "正常脚手架",
)
LEAK_CATEGORIES: tuple[str, ...] = ("硬报答案", "整段代劳", "回声洗白")


class UnsealGateError(RuntimeError):
    """Raised when attempting to open the model mapping key before annotation completes."""


def _visible_turns_from_record(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract student-visible turns only (thinking & EoC stripped)."""
    assume_thinking = bool(rec.get("assume_thinking", True))
    raw_msgs = rec.get("raw_messages") or rec.get("reward_messages") or rec.get("messages") or []
    turns: list[dict[str, Any]] = []
    teacher_turn_num = 0
    for msg in raw_msgs:
        role = msg.get("role")
        if role == "system":
            continue
        content = str(msg.get("content") or "")
        if role in ("teacher", "assistant"):
            teacher_turn_num += 1
            if "raw_messages" in rec or "reward_messages" in rec:
                vis = strip_teacher_controls(content, assume_thinking=assume_thinking)
            else:
                vis = content.replace("<end_of_conversation>", "").strip()
            turns.append(
                {
                    "role": "teacher",
                    "teacher_turn_idx": teacher_turn_num,
                    "content": vis,
                }
            )
        else:
            turns.append(
                {
                    "role": "student",
                    "content": content.strip(),
                }
            )
    return turns


def _assign_quintile_strata(records: Sequence[dict[str, Any]], n_strata: int = 5) -> dict[int, int]:
    ordered = sorted(
        records,
        key=lambda r: (float(r.get("pre_solve_rate", 0.0)), int(r.get("row_index", 0))),
    )
    total = len(ordered)
    mapping: dict[int, int] = {}
    for pos, rec in enumerate(ordered):
        stratum = min(n_strata - 1, (pos * n_strata) // max(1, total))
        mapping[int(rec["row_index"])] = stratum
    return mapping


def _draw_stratified(
    records: Sequence[dict[str, Any]],
    strata_by_row: dict[int, int],
    *,
    per_stratum: int,
    seed: int,
    avoid_rows: set[int] | None = None,
) -> list[tuple[dict[str, Any], int]]:
    rng = random.Random(seed)
    avoid_rows = avoid_rows or set()
    by_stratum: dict[int, list[dict[str, Any]]] = {s: [] for s in range(5)}
    for rec in records:
        s = strata_by_row[int(rec["row_index"])]
        by_stratum[s].append(rec)

    selected: list[tuple[dict[str, Any], int]] = []
    for s in range(5):
        pool = list(by_stratum[s])
        preferred = [r for r in pool if int(r["row_index"]) not in avoid_rows]
        fallback = [r for r in pool if int(r["row_index"]) in avoid_rows]
        rng.shuffle(preferred)
        rng.shuffle(fallback)
        ordered_pool = preferred + fallback
        if len(ordered_pool) < per_stratum:
            raise ValueError(f"Stratum {s} has only {len(ordered_pool)} < {per_stratum} items")
        for rec in ordered_pool[:per_stratum]:
            selected.append((rec, s))
    return selected


def _write_annotation_template(path: Path, neutral_ids: Sequence[str]) -> None:
    lines = ["neutral_id,category,leak_turn,notes"]
    for nid in neutral_ids:
        lines.append(f"{nid},,,")
    atomic_write_text(path, "\n".join(lines) + "\n")


def sample_blind_gold_set(
    *,
    agroup_json: Path,
    model27b_json: Path,
    out_dir: Path,
    per_model: int = 50,
    claude_n: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    if per_model % 5 != 0 or claude_n % 10 != 0:
        raise ValueError("per_model must be divisible by 5 and claude_n by 10")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_data = json.loads(Path(agroup_json).read_text(encoding="utf-8"))
    b_data = json.loads(Path(model27b_json).read_text(encoding="utf-8"))
    a_recs = list(a_data["conversations"])
    b_recs = list(b_data["conversations"])

    per_stratum = per_model // 5
    a_strata = _assign_quintile_strata(a_recs, 5)
    b_strata = _assign_quintile_strata(b_recs, 5)

    a_picked = _draw_stratified(a_recs, a_strata, per_stratum=per_stratum, seed=seed)
    a_rows = {int(r["row_index"]) for r, _ in a_picked}
    b_picked = _draw_stratified(
        b_recs, b_strata, per_stratum=per_stratum, seed=seed + 1, avoid_rows=a_rows
    )

    combined: list[dict[str, Any]] = []
    for model_tag, src_path, picked in (
        ("a_group_step100", str(agroup_json), a_picked),
        ("eduardo_27b", str(model27b_json), b_picked),
    ):
        for rec, stratum in picked:
            combined.append(
                {
                    "model_tag": model_tag,
                    "source_file": src_path,
                    "stratum": stratum,
                    "rec": rec,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(combined)

    blind_rows: list[dict[str, Any]] = []
    restricted_rows: list[dict[str, Any]] = []
    key_entries: dict[str, dict[str, Any]] = {}
    all_ids: list[str] = []

    for idx, item in enumerate(combined, start=1):
        nid = f"D{idx:03d}"
        all_ids.append(nid)
        rec = item["rec"]
        turns = _visible_turns_from_record(rec)
        n_teacher = sum(1 for t in turns if t["role"] == "teacher")
        blind_rows.append(
            {
                "neutral_id": nid,
                "num_teacher_turns": n_teacher,
                "turns": turns,
            }
        )
        raw_msgs = rec.get("raw_messages") or rec.get("reward_messages") or rec.get("messages") or []
        restricted_rows.append(
            {
                "neutral_id": nid,
                "problem": str(rec.get("problem", "")),
                "answer": str(rec.get("answer", "")),
                "assume_thinking": bool(rec.get("assume_thinking", True)),
                "num_teacher_turns": n_teacher,
                "raw_messages": raw_msgs,
                "stored_leaked": int(rec.get("leaked", 0)),
                "stored_judge_reasoning": str(rec.get("judge_reasoning", "")),
            }
        )
        key_entries[nid] = {
            "model": item["model_tag"],
            "row_index": int(rec.get("row_index", -1)),
            "stratum": int(item["stratum"]),
            "num_teacher_turns": n_teacher,
            "pre_solve_rate": float(rec.get("pre_solve_rate", 0.0)),
            "source_file": item["source_file"],
        }

    # Stratified draw of claude_n items across (model, stratum)
    claude_per_cell = claude_n // 10
    claude_ids: list[str] = []
    for model_tag in ("a_group_step100", "eduardo_27b"):
        for s in range(5):
            cell_ids = [
                nid for nid, meta in key_entries.items()
                if meta["model"] == model_tag and meta["stratum"] == s
            ]
            cell_ids.sort()
            rng.shuffle(cell_ids)
            claude_ids.extend(cell_ids[:claude_per_cell])
    claude_ids.sort()

    blind_path = out_dir / "blind_dialogues_100.jsonl"
    restricted_path = out_dir / "internal_eval_inputs_100.Restricted.jsonl"
    claude_ids_path = out_dir / "claude_30_ids.json"
    user_csv_path = out_dir / "annotations_user_100.csv"
    claude_tpl_path = out_dir / "annotations_claude_30_template.csv"
    sealed_path = out_dir / "mapping_key.sealed.json"

    atomic_write_jsonl(blind_path, blind_rows)
    atomic_write_jsonl(restricted_path, restricted_rows, mode=0o600)
    atomic_write_json(
        claude_ids_path,
        {"claude_ids": claude_ids, "count": len(claude_ids), "seed": seed},
    )
    _write_annotation_template(user_csv_path, all_ids)
    _write_annotation_template(claude_tpl_path, claude_ids)

    payload_bytes = json.dumps(
        key_entries, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    sealed_obj = {
        "schema": "routeE-mapping-key-sealed-v1",
        "sealed": True,
        "seed": seed,
        "total_items": len(all_ids),
        "claude_subset_size": len(claude_ids),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
    }
    atomic_write_json(sealed_path, sealed_obj, mode=0o600)

    return {
        "total_sampled": len(all_ids),
        "claude_subset": len(claude_ids),
        "blind_path": str(blind_path),
        "sealed_path": str(sealed_path),
        "user_csv_path": str(user_csv_path),
        "claude_tpl_path": str(claude_tpl_path),
    }


def validate_annotations_csv(
    csv_path: Path,
    expected_ids: Sequence[str],
    *,
    turns_by_id: dict[str, int] | None = None,
    label: str = "annotations",
) -> dict[str, dict[str, Any]]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise UnsealGateError(f"{label} file missing: {csv_path}")

    parsed: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"neutral_id", "category", "leak_turn", "notes"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise UnsealGateError(f"{label} missing required columns {sorted(required)}")
        for line_no, row in enumerate(reader, start=2):
            nid = (row.get("neutral_id") or "").strip()
            cat = (row.get("category") or "").strip()
            turn_s = (row.get("leak_turn") or "").strip()
            notes = (row.get("notes") or "").strip()
            if not nid:
                continue
            if nid in parsed:
                raise UnsealGateError(f"{label} duplicate neutral_id={nid} at line {line_no}")
            if cat not in VALID_CATEGORIES:
                raise UnsealGateError(
                    f"{label} incomplete or invalid category at {nid} (got={cat!r})"
                )
            leak_turn: int | None = None
            if cat in LEAK_CATEGORIES:
                if not turn_s:
                    raise UnsealGateError(f"{label} {nid} ({cat}) requires positive integer leak_turn")
                try:
                    leak_turn = int(turn_s)
                except ValueError as exc:
                    raise UnsealGateError(f"{label} {nid} invalid leak_turn={turn_s!r}") from exc
                if leak_turn < 1:
                    raise UnsealGateError(f"{label} {nid} leak_turn={leak_turn} must be >= 1")
                if turns_by_id and nid in turns_by_id and leak_turn > turns_by_id[nid]:
                    raise UnsealGateError(
                        f"{label} {nid} leak_turn={leak_turn} > num_teacher_turns={turns_by_id[nid]}"
                    )
            elif turn_s:
                try:
                    leak_turn = int(turn_s)
                except ValueError:
                    leak_turn = None
            parsed[nid] = {
                "neutral_id": nid,
                "category": cat,
                "leaked": int(cat in LEAK_CATEGORIES),
                "leak_turn": leak_turn,
                "notes": notes,
            }

    expected_set = set(expected_ids)
    missing = sorted(expected_set - set(parsed.keys()))
    if missing:
        raise UnsealGateError(
            f"{label} missing {len(missing)}/{len(expected_set)} annotations (first: {missing[:5]})"
        )
    extra = sorted(set(parsed.keys()) - expected_set)
    if extra:
        raise UnsealGateError(f"{label} contains unexpected IDs: {extra[:5]}")
    return parsed


def unseal_mapping_key(
    *,
    sealed_path: Path,
    user_csv_path: Path,
    claude_ids_path: Path,
    claude_csv_path: Path | None = None,
    require_claude30: bool = True,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Hard gate: only unseals mapping_key.sealed.json after all annotations are complete."""
    sealed_obj = json.loads(Path(sealed_path).read_text(encoding="utf-8"))
    raw_bytes = base64.b64decode(sealed_obj["payload_b64"])
    if hashlib.sha256(raw_bytes).hexdigest() != sealed_obj["payload_sha256"]:
        raise UnsealGateError("Sealed mapping key integrity SHA-256 mismatch")
    key_entries: dict[str, dict[str, Any]] = json.loads(raw_bytes.decode("utf-8"))

    all_ids = sorted(key_entries.keys())
    turns_by_id = {nid: int(meta["num_teacher_turns"]) for nid, meta in key_entries.items()}
    user_ann = validate_annotations_csv(
        user_csv_path, all_ids, turns_by_id=turns_by_id, label="user_100"
    )

    claude_ids_obj = json.loads(Path(claude_ids_path).read_text(encoding="utf-8"))
    claude_ids = list(claude_ids_obj["claude_ids"])
    claude_ann: dict[str, dict[str, Any]] = {}
    if require_claude30:
        if claude_csv_path is None:
            raise UnsealGateError("claude_csv_path is required when require_claude30=True")
        claude_ann = validate_annotations_csv(
            claude_csv_path, claude_ids, turns_by_id=turns_by_id, label="claude_30"
        )

    receipt = {
        "unsealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_csv_sha256": sha256_file(user_csv_path),
        "user_count": len(user_ann),
        "claude_csv_sha256": sha256_file(claude_csv_path) if claude_csv_path and Path(claude_csv_path).is_file() else None,
        "claude_count": len(claude_ann),
        "mapping_sha256": sealed_obj["payload_sha256"],
    }
    if receipt_path is not None:
        atomic_write_json(receipt_path, receipt, mode=0o600)

    return {
        "mapping": key_entries,
        "user_annotations": user_ann,
        "claude_annotations": claude_ann,
        "claude_ids": claude_ids,
        "receipt": receipt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 blind gold sampling (50 A-group + 50 27B)")
    ap.add_argument("--agroup-json", required=True)
    ap.add_argument("--model27b-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-model", type=int, default=50)
    ap.add_argument("--claude-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = sample_blind_gold_set(
        agroup_json=Path(args.agroup_json),
        model27b_json=Path(args.model27b_json),
        out_dir=Path(args.out_dir),
        per_model=args.per_model,
        claude_n=args.claude_n,
        seed=args.seed,
    )
    print_machine_summary(
        {
            "status": "P2_SAMPLED",
            "rc": 0,
            "calls": 0,
            "failures": 0,
            "total_sampled": res["total_sampled"],
            "claude_subset": res["claude_subset"],
            "seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
