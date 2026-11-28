#!/usr/bin/env python3
"""Export the 30 blind dialogues for Claude's independent dual-annotation.

Reads ONLY `blind_dialogues_100.jsonl` and `claude_30_ids.json`.
Outputs:
- `claude_30_blind.jsonl` (only neutral_id, num_teacher_turns, turns)
- `claude_30_blind.md` (readable blind transcript without model names or judge scores)
- `annotations_claude_30_template.csv`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common_utils import atomic_write_jsonl, atomic_write_text, print_machine_summary
from p2_view_dialogue import format_blind_dialogue, load_blind_dialogues

FORBIDDEN_EXPORT_KEYS = {
    "model",
    "model_tag",
    "row_index",
    "leaked",
    "stored_leaked",
    "judge_reasoning",
    "stored_judge_reasoning",
    "teaching_rejected",
    "eval_judge_pedagogy",
    "eval_judge_correctness",
}


def export_claude_30(
    *,
    blind_jsonl: Path,
    claude_ids_json: Path,
    out_dir: Path,
) -> dict[str, Any]:
    dialogues = load_blind_dialogues(blind_jsonl)
    ids_obj = json.loads(Path(claude_ids_json).read_text(encoding="utf-8"))
    claude_ids = list(ids_obj["claude_ids"])
    if len(claude_ids) != 30:
        raise ValueError(f"Expected 30 Claude IDs, got {len(claude_ids)}")

    exported: list[dict[str, Any]] = []
    md_parts: list[str] = [
        "# T-032 P2 Claude 独立双标盲评对话（30 条，不含模型名与裁判结果）\n",
        "分类定义：1=硬报答案 / 2=整段代劳 / 3=回声洗白 / 4=正常脚手架。完成后只将 `annotations_claude_30.csv` 传回服务器，并删除本对话文件。\n",
    ]
    csv_lines = ["neutral_id,category,leak_turn,notes"]

    for nid in claude_ids:
        if nid not in dialogues:
            raise KeyError(f"Missing neutral_id={nid} in {blind_jsonl}")
        item = dialogues[nid]
        bad_keys = FORBIDDEN_EXPORT_KEYS & set(item.keys())
        if bad_keys:
            raise RuntimeError(f"Forbidden metadata found in blind item {nid}: {sorted(bad_keys)}")
        clean_item = {
            "neutral_id": item["neutral_id"],
            "num_teacher_turns": item["num_teacher_turns"],
            "turns": item["turns"],
        }
        exported.append(clean_item)
        md_parts.append(format_blind_dialogue(clean_item))
        csv_lines.append(f"{nid},,,")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "claude_30_blind.jsonl"
    out_md = out_dir / "claude_30_blind.md"
    out_csv = out_dir / "annotations_claude_30_template.csv"

    atomic_write_jsonl(out_jsonl, exported)
    atomic_write_text(out_md, "\n".join(md_parts))
    atomic_write_text(out_csv, "\n".join(csv_lines) + "\n")

    return {
        "exported_count": len(exported),
        "out_jsonl": str(out_jsonl),
        "out_md": str(out_md),
        "out_csv": str(out_csv),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export 30 blind dialogues for Claude annotation")
    ap.add_argument("--blind-jsonl", required=True)
    ap.add_argument("--claude-ids-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    res = export_claude_30(
        blind_jsonl=Path(args.blind_jsonl),
        claude_ids_json=Path(args.claude_ids_json),
        out_dir=Path(args.out_dir),
    )
    print_machine_summary(
        {
            "status": "P2_EXPORT_CLAUDE30_DONE",
            "rc": 0,
            "calls": 0,
            "failures": 0,
            "exported_count": res["exported_count"],
        }
    )


if __name__ == "__main__":
    main()
