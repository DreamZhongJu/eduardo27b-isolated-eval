#!/usr/bin/env python3
"""Combine sanitized Pro 5000 summary JSON and A100 counts JSON into unified T-025 verdicts.

Runs locally on cu04 after the user executes the read-only Pro 5000 and A100 commands.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import zlib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from attribution_core import (  # noqa: E402
    assert_output_sanitized,
    judge_cause_c,
    judge_cause_d,
    judge_cause_e,
    judge_cause_g,
)


def decode_b64_zlib_json(b64_text: str) -> dict[str, Any]:
    raw = zlib.decompress(base64.b64decode(b64_text.strip()))
    return json.loads(raw.decode("utf-8"))


def merge_pro5000_and_a100(pro_doc: dict[str, Any], a100_doc: dict[str, Any]) -> dict[str, Any]:
    """Merge A100 format/length/config counts into Pro 5000 attribution results and finalize all 7 verdicts."""
    if pro_doc.get("schema") != "t025_pro5000_summary_v1":
        raise ValueError("Unexpected Pro 5000 JSON schema")
    if a100_doc.get("schema") != "t025_a100_counts_v1":
        raise ValueError("Unexpected A100 JSON schema")

    d27_a100 = a100_doc["eduardo27b"]
    fmt_check = d27_a100["length_and_format"]

    # Finalize Cause C with A100 format pre-check
    cause_c_decision = judge_cause_c(pro_doc["cause_c"]["eduardo27b_ar_minus_agnes"], fmt_check)

    # Finalize Cause D across 27B (A100), base (A100), and A-group (Pro 5000)
    models_d = {
        "eduardo27b": fmt_check,
        "a_group": pro_doc["cause_d_pro5000_agroup"]["length_and_format"],
    }
    if a100_doc.get("base4b"):
        models_d["base"] = a100_doc["base4b"]["length_and_format"]
    cause_d_decision = judge_cause_d(models_d)

    # Finalize Cause E1 with verified A100 config
    cause_e_decision = judge_cause_e(
        bool(d27_a100.get("pre_post_identical_student_and_grader", True)),
        pro_doc["cause_e"]["overlap_orig_pre_ours_ar_minus_author"],
        pro_doc["cause_e"]["overlap_orig_pre_ours_agnes_minus_author"],
    )

    # Finalize Cause G with verified A100 27B rollout diagnostics + T-022 summary
    cause_g_decision = judge_cause_g(pro_doc["t022_summary"], d27_a100)

    verdicts_summary = {
        "A": pro_doc["cause_a"]["decision"],
        "B": pro_doc["cause_b"]["decision"],
        "C": cause_c_decision,
        "D": cause_d_decision,
        "E": cause_e_decision,
        "F": pro_doc["cause_f"]["decision"],
        "G": cause_g_decision,
    }

    return {
        "schema": "t025_combined_verdicts_v1",
        "verdicts": verdicts_summary,
        "pro5000": pro_doc,
        "a100": a100_doc,
        "cause_d_models": models_d,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge Pro 5000 and A100 T-025 JSONs.")
    ap.add_argument("--pro5000-json", type=Path, required=True)
    ap.add_argument("--a100-json", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    pro_doc = json.loads(args.pro5000_json.read_text(encoding="utf-8"))
    a100_doc = json.loads(args.a100_json.read_text(encoding="utf-8"))
    combined = merge_pro5000_and_a100(pro_doc, a100_doc)
    out_text = json.dumps(combined, ensure_ascii=False, indent=2) + "\n"
    assert_output_sanitized(out_text)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(out_text, encoding="utf-8")

    for k in ("A", "B", "C", "D", "E", "F", "G"):
        print(f"Cause {k}: verdict={combined['verdicts'][k]['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
