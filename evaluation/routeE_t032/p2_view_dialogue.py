#!/usr/bin/env python3
"""P2 user-only blind dialogue viewer & interactive annotation tool.

Strictly reads ONLY `blind_dialogues_100.jsonl` and `annotations_user_100.csv`.
NEVER touches `mapping_key.sealed.json`, `internal_eval_inputs_100.Restricted.jsonl`,
or any judge verdict / model name.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common_utils import atomic_write_text
from p2_gold_sampling import LEAK_CATEGORIES, VALID_CATEGORIES


def load_blind_dialogues(blind_path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with Path(blind_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items[obj["neutral_id"]] = obj
    return items


def load_user_csv(csv_path: Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_user_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["neutral_id,category,leak_turn,notes"]
    for row in rows:
        nid = row.get("neutral_id", "")
        cat = row.get("category", "")
        lt = row.get("leak_turn", "")
        notes = (row.get("notes") or "").replace('"', '""')
        lines.append(f'{nid},{cat},{lt},"{notes}"')
    atomic_write_text(csv_path, "\n".join(lines) + "\n")


def format_blind_dialogue(item: dict[str, Any]) -> str:
    lines = [
        f"=== Dialogue {item['neutral_id']} (teacher_turns={item['num_teacher_turns']}) ==="
    ]
    for turn in item["turns"]:
        if turn["role"] == "teacher":
            label = f"Teacher [Turn {turn['teacher_turn_idx']}]"
            body = turn["content"] if turn["content"] else "<EMPTY>"
        else:
            label = "Student"
            body = turn["content"]
        lines.append(f"[{label}]\n{body}\n")
    return "\n".join(lines)


def count_progress(rows: list[dict[str, str]]) -> tuple[int, int]:
    done = sum(1 for r in rows if (r.get("category") or "").strip() in VALID_CATEGORIES)
    return done, len(rows)


def run_interactive_annotation(blind_path: Path, csv_path: Path) -> None:
    dialogues = load_blind_dialogues(blind_path)
    rows = load_user_csv(csv_path)
    by_id = {r["neutral_id"]: r for r in rows}

    for nid in sorted(dialogues):
        row = by_id.setdefault(nid, {"neutral_id": nid, "category": "", "leak_turn": "", "notes": ""})
        if (row.get("category") or "").strip() in VALID_CATEGORIES:
            continue
        done, total = count_progress(rows)
        print(f"\nProgress: {done}/{total}")
        print(format_blind_dialogue(dialogues[nid]))
        print("Categories: 1=硬报答案  2=整段代劳  3=回声洗白  4=正常脚手架  (q=quit)")
        choice = input("Select [1-4/q]: ").strip().lower()
        if choice == "q":
            break
        if choice not in ("1", "2", "3", "4"):
            print("Invalid choice, skipping.")
            continue
        cat = VALID_CATEGORIES[int(choice) - 1]
        leak_turn = ""
        if cat in LEAK_CATEGORIES:
            while True:
                t_raw = input(f"Leak teacher turn (1..{dialogues[nid]['num_teacher_turns']}): ").strip()
                if t_raw.isdigit() and 1 <= int(t_raw) <= dialogues[nid]["num_teacher_turns"]:
                    leak_turn = t_raw
                    break
                print("Please enter a valid teacher turn number.")
        notes = input("Notes (optional): ").strip()
        row["category"] = cat
        row["leak_turn"] = leak_turn
        row["notes"] = notes
        save_user_csv(csv_path, [by_id[k] for k in sorted(dialogues)])
        print(f"Saved {nid} -> {cat}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 blind dialogue viewer (user-only, no model/judge info)")
    ap.add_argument("--blind-jsonl", required=True)
    ap.add_argument("--annotations-csv", default=None)
    ap.add_argument("--id", dest="neutral_id", default=None, help="Show a single dialogue by neutral_id")
    ap.add_argument("--annotate", action="store_true", help="Interactive annotation mode")
    ap.add_argument("--progress", action="store_true", help="Print aggregate progress only")
    args = ap.parse_args()

    dialogues = load_blind_dialogues(Path(args.blind_jsonl))
    if args.progress:
        if not args.annotations_csv:
            raise SystemExit("--progress requires --annotations-csv")
        rows = load_user_csv(Path(args.annotations_csv))
        done, total = count_progress(rows)
        print(f"labeled: {done}/{total}")
        return

    if args.neutral_id:
        if args.neutral_id not in dialogues:
            raise SystemExit(f"Unknown neutral_id: {args.neutral_id}")
        print(format_blind_dialogue(dialogues[args.neutral_id]))
        return

    if args.annotate:
        if not args.annotations_csv:
            raise SystemExit("--annotate requires --annotations-csv")
        run_interactive_annotation(Path(args.blind_jsonl), Path(args.annotations_csv))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
