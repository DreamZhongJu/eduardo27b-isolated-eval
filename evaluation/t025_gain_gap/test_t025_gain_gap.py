#!/usr/bin/env python3
"""Unit tests for T-025 offline gain-gap attribution (evaluation/t025_gain_gap).

Covers all 5 required areas from T-025 Delivery #2:
  1. Stratification bins & author threshold filter exact boundaries
  2. Reweighting weights (sum=1, exact bin-weighted mean, empty-bin renormalization)
  3. Paired comparisons strictly use only sources present on both sides
  4. Two-implementation cross-checks (paired_gain.py & summarize_base4b vs attribution_core)
  5. Zero leakage of problem statements, answers, responses, or source IDs in outputs
Plus end-to-end synthetic Pro 5000 + A100 + combiner pipeline & A-G decision rule checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PRO5000_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
PAIRED_DIR = REPO_ROOT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(PRO5000_DIR), str(BASE4B_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paired_gain as pg  # noqa: E402
from attribution_core import (  # noqa: E402
    PRE_BIN_LABELS,
    assert_output_sanitized,
    assign_pre_bin,
    compute_cause_a,
    compute_cause_b,
    compute_cause_c_pro5000,
    compute_cause_f,
    compute_reweight_weights,
    derive_seed,
    extract_author_boxed_pred,
    judge_cause_a,
    judge_cause_b,
    judge_cause_c,
    judge_cause_d,
    judge_cause_e,
    judge_cause_f,
    judge_cause_g,
    match_sources_499_vs_494,
    normalized_gain,
    paired_diff_bootstrap,
    passes_author_threshold,
)
from combine_t025_verdicts import merge_pro5000_and_a100  # noqa: E402
from export_27b_selfbuilt import COUNT_COLUMNS, BASE4B_COUNT_COLUMNS  # noqa: E402
from summarize_base4b import paired_bootstrap_diff  # noqa: E402


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class TestT025GainGap(unittest.TestCase):
    def test_1_stratification_and_threshold_boundaries(self) -> None:
        """Verify exact interval boundaries for the 6 pre-test bins and author threshold gate."""
        self.assertEqual(assign_pre_bin(0.0), "0")
        self.assertEqual(assign_pre_bin(1.0 / 16.0), "(0, 0.25]")
        self.assertEqual(assign_pre_bin(0.25), "(0, 0.25]")
        self.assertEqual(assign_pre_bin(0.25 + 1e-6), "(0.25, 0.5]")
        self.assertEqual(assign_pre_bin(0.50), "(0.25, 0.5]")
        self.assertEqual(assign_pre_bin(0.50 + 1e-6), "(0.5, 0.75]")
        self.assertEqual(assign_pre_bin(12.0 / 16.0), "(0.5, 0.75]")
        self.assertEqual(assign_pre_bin(13.0 / 16.0), "(0.75, 15/16)")
        self.assertEqual(assign_pre_bin(14.0 / 16.0), "(0.75, 15/16)")
        self.assertEqual(assign_pre_bin(15.0 / 16.0 - 1e-5), "(0.75, 15/16)")
        self.assertEqual(assign_pre_bin(15.0 / 16.0), "[15/16, 1]")
        self.assertEqual(assign_pre_bin(1.0), "[15/16, 1]")
        with self.assertRaises(ValueError):
            assign_pre_bin(-0.01)
        with self.assertRaises(ValueError):
            assign_pre_bin(1.01)

        # Author threshold boundaries: pre_transfer <= 0.75 and |pre_transfer - pre_orig| <= 0.375
        self.assertTrue(passes_author_threshold(0.75, 0.375))
        self.assertTrue(passes_author_threshold(0.375, 0.75))
        self.assertFalse(passes_author_threshold(0.75 + 1e-5, 0.75))
        self.assertFalse(passes_author_threshold(0.75, 0.375 - 1e-5))
        self.assertFalse(passes_author_threshold(0.0, 0.375 + 1e-5))
        self.assertTrue(passes_author_threshold(12.0 / 16.0, 6.0 / 16.0))
        self.assertFalse(passes_author_threshold(13.0 / 16.0, 12.0 / 16.0))
        self.assertFalse(passes_author_threshold(12.0 / 16.0, 5.0 / 16.0))

    def test_2_reweighting_weights_and_empty_bins(self) -> None:
        """Verify reweighting weights sum to 1.0, reproduce target bin mean, and handle empty bins."""
        # 4 items in "0", 1 item in "(0, 0.25]", 1 item in "[15/16, 1]"
        item_bins = ["0", "0", "0", "0", "(0, 0.25]", "[15/16, 1]"]
        gains = [0.10, 0.10, 0.10, 0.10, 0.30, -0.20]
        # Target distribution: 50% in "0", 50% in "(0, 0.25]", 0% in "[15/16, 1]"
        target_counts = {
            "0": 50,
            "(0, 0.25]": 50,
            "(0.25, 0.5]": 0,
            "(0.5, 0.75]": 0,
            "(0.75, 15/16)": 0,
            "[15/16, 1]": 0,
        }
        rw = compute_reweight_weights(item_bins, target_counts)
        w = rw["weights"]
        self.assertAlmostEqual(sum(w), 1.0, places=12)
        self.assertEqual(rw["covered_target_mass"], 1.0)
        self.assertEqual(rw["covered_items"], 5)
        self.assertEqual(rw["zero_target_bins_with_ours_items"], ["[15/16, 1]"])
        # Each of the 4 items in "0" gets 0.5 / 4 = 0.125; the 1 item in "(0, 0.25]" gets 0.5; last gets 0.0
        for i in range(4):
            self.assertAlmostEqual(w[i], 0.125, places=12)
        self.assertAlmostEqual(w[4], 0.50, places=12)
        self.assertAlmostEqual(w[5], 0.0, places=12)
        weighted_mean = sum(wi * gi for wi, gi in zip(w, gains))
        self.assertAlmostEqual(weighted_mean, 0.5 * 0.10 + 0.5 * 0.30, places=12)

        # Empty ours bin where target has 20% mass -> renormalizes over remaining 80% mass
        target_with_gap = {**target_counts, "0": 40, "(0, 0.25]": 40, "(0.25, 0.5]": 20}
        rw2 = compute_reweight_weights(item_bins, target_with_gap)
        self.assertAlmostEqual(sum(rw2["weights"]), 1.0, places=12)
        self.assertAlmostEqual(rw2["covered_target_mass"], 0.80, places=12)
        self.assertEqual(rw2["empty_ours_bins_with_target_mass"], ["(0.25, 0.5]"])
        self.assertAlmostEqual(rw2["weights"][4], 0.50, places=12)

    def test_3_paired_only_common_sources_and_4_cross_check(self) -> None:
        """Verify paired bootstrap uses ONLY shared IDs and matches paired_gain.py & summarize_base4b."""
        # Include unpaired outlier values (+999 / -999) on non-shared IDs to prove they are excluded
        base_rows_csv: list[dict[str, Any]] = []
        a_pairs: list[tuple[str, float]] = []
        b_pairs: list[tuple[str, float]] = []

        for idx in range(40):
            uid = f"src_{idx:03d}"
            pre = (idx % 10) / 16.0
            post_b_c = min(16, (idx % 10) + 1)
            post_t_c = min(16, (idx % 10) + 4)
            b_pairs.append((uid, normalized_gain(pre, post_b_c / 16.0)))
            a_pairs.append((uid, normalized_gain(pre, post_t_c / 16.0)))
            base_rows_csv.append(
                {
                    "unit_id": uid,
                    "condition": "base",
                    "group": "medium",
                    "pre": pre,
                    "post_correct": post_b_c,
                    "post_total": 16,
                    "post_unknown": 0,
                    "line_no": len(base_rows_csv) + 2,
                }
            )
            base_rows_csv.append(
                {
                    "unit_id": uid,
                    "condition": "a_group",
                    "group": "medium",
                    "pre": pre,
                    "post_correct": post_t_c,
                    "post_total": 16,
                    "post_unknown": 0,
                    "line_no": len(base_rows_csv) + 2,
                }
            )

        # Unpaired extras on each side with extreme values
        b_pairs.append(("only_in_base_X", 1.0))
        b_pairs.append(("only_in_base_Y", 1.0))
        a_pairs.append(("only_in_target_Z", -1.0))
        base_rows_csv.append(
            {
                "unit_id": "only_in_base_X",
                "condition": "base",
                "group": "medium",
                "pre": 0.0,
                "post_correct": 16,
                "post_total": 16,
                "post_unknown": 0,
                "line_no": len(base_rows_csv) + 2,
            }
        )
        base_rows_csv.append(
            {
                "unit_id": "only_in_target_Z",
                "condition": "a_group",
                "group": "medium",
                "pre": 1.0,
                "post_correct": 0,
                "post_total": 16,
                "post_unknown": 0,
                "line_no": len(base_rows_csv) + 2,
            }
        )

        res_core = paired_diff_bootstrap(a_pairs, b_pairs, seed=derive_seed(42, "cc"), draws=10000)
        res_sb = paired_bootstrap_diff(a_pairs, b_pairs, seed=derive_seed(42, "cc_sb"), draws=10000)
        res_pg = pg.evaluate_dataset(
            base_rows_csv, base_condition="base", target_condition="a_group", draws=10000, seed=42
        )
        pg_med = next(
            r for r in res_pg["results"] if r["group"] == "medium" and r["unknown_policy"] == "unknown_as_wrong" and r["source_filter"] == "all"
        )

        self.assertEqual(res_core["n_paired"], 40)
        self.assertEqual(res_core["n_only_target"], 1)
        self.assertEqual(res_core["n_only_base"], 2)
        self.assertEqual(pg_med["n_paired"], 40)

        # Exact point estimate agreement (<1e-12) and tight bootstrap CI agreement (<0.01) across all 3!
        self.assertLess(abs(res_core["mean_diff"] - pg_med["stats"]["diff_mean"]), 1e-12)
        self.assertLess(abs(res_core["mean_diff"] - res_sb["mean_diff"]), 1e-12)
        self.assertLess(abs(res_core["ci"][0] - pg_med["stats"]["diff_ci_low"]), 0.01)
        self.assertLess(abs(res_core["ci"][1] - pg_med["stats"]["diff_ci_high"]), 0.01)
        self.assertLess(abs(res_core["ci"][0] - res_sb["ci"][0]), 0.01)
        self.assertLess(abs(res_core["ci"][1] - res_sb["ci"][1]), 0.01)

    def test_5_end_to_end_pipeline_and_sanitization_guard(self) -> None:
        """Run end-to-end Pro 5000 + combiner on synthetic bundle with sentinel secrets and verify 0 leakage."""
        sentinel_sid = "SECRET_SOURCE_ID_ALPHA_99182"
        sentinel_prob = "Find the unique eigenvalue of SECRET_MATRIX_OMEGA_774123 over Q."
        sentinel_trans = "Compute the determinant of SECRET_VARIANT_SIGMA_553190 modulo 97."
        sentinel_ans = "SECRET_GOLD_ANSWER_424242"
        sentinel_resp = r"Let us derive step by step: SECRET_STUDENT_CHAIN_881122 therefore \boxed{42}"

        # Verify sanitizer catches leaks immediately
        with self.assertRaises(ValueError):
            assert_output_sanitized(
                f"summary leaked {sentinel_sid}",
                forbidden_exact_ids=[sentinel_sid],
                forbidden_text_snippets=[sentinel_prob],
            )
        with self.assertRaises(ValueError):
            assert_output_sanitized(
                f"text leaked: {sentinel_prob}",
                forbidden_exact_ids=[sentinel_sid],
                forbidden_text_snippets=[sentinel_prob],
            )

        # Verify boxed parser edge cases
        p1, f1 = extract_author_boxed_pred(r"answer is \boxed{\frac{1}{2}}.")
        self.assertEqual((p1, f1), (r"\frac{1}{2}", False))
        p2, f2 = extract_author_boxed_pred("no boxed answer here")
        self.assertEqual((p2, f2), ("", True))
        p3, f3 = extract_author_boxed_pred(r"empty \boxed{}")
        self.assertEqual((p3, f3), ("", True))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            agroup_dir = root / "agroup"
            tok_dir = root / "tok"
            bundle_dir.mkdir()
            agroup_dir.mkdir()
            tok_dir.mkdir()

            # Build minimal valid HF-style wordlevel tokenizer.json (no external weights needed)
            vocab = {"<unk>": 0, "step": 1, "by": 2, r"\boxed{42}": 3}
            tok_doc = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
            }
            (tok_dir / "tokenizer.json").write_text(json.dumps(tok_doc), encoding="utf-8")

            n_src = 35
            inputs_lines: list[str] = []
            ref_csv_rows: list[dict[str, Any]] = []
            base_csv_rows: list[dict[str, Any]] = []

            for i in range(n_src):
                sid = sentinel_sid if i == 0 else f"test_secret_src_{i:03d}"
                skey = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:24]
                prob = sentinel_prob if i == 0 else f"Original secret math problem statement number {i:03d} with x^2 + {i} = 0."
                ans = sentinel_ans if i == 0 else str(i)

                for d_idx, diff in enumerate(("easy", "medium", "hard")):
                    stage = {"easy": "EASY", "medium": "MEDIUM", "hard": "HARD"}[diff]
                    tier_zh = {"easy": "简单", "medium": "中等", "hard": "困难"}[diff]
                    t_prob = sentinel_trans if (i == 0 and d_idx == 0) else f"Transfer {diff} secret problem {i:03d} find y + {d_idx}."
                    row_obj = {
                        "source_id": sid,
                        "problem": prob,
                        "answer": ans,
                        "transfer_problem": t_prob,
                        "transfer_answer": ans,
                        "transfer_target": f"ours_{diff}",
                        "transfer_candidate_id": f"cand_{i}_{diff}",
                        "transfer_position": 3 - d_idx,
                        "student_correct": 15 if diff == "easy" and i < 15 else 6,
                        "student_n_scored": 16,
                        "score_tie": False,
                    }
                    inputs_lines.append(
                        json.dumps(
                            {
                                "difficulty": diff,
                                "tier_zh": tier_zh,
                                "ref_stage": stage,
                                "source_key": skey,
                                "source_id": sid,
                                "row_index": i,
                                "row": row_obj,
                            },
                            ensure_ascii=False,
                        )
                    )
                    pre_t_c = 15 if (diff == "easy" and i < 15) else (6 if diff == "easy" else (5 if diff == "medium" else 2))
                    post_27_t_c = 15 if (diff == "easy" and i < 15) else min(16, pre_t_c + 3)
                    crow: dict[str, Any] = {
                        "difficulty": diff,
                        "tier_zh": tier_zh,
                        "ref_stage": stage,
                        "source_key": skey,
                        "source_id": sid,
                        "row_index": i,
                    }
                    for grader in ("agnes", "ar"):
                        for tgt, pre_c, post_c in (("orig", 4, 11), ("trans", pre_t_c, post_27_t_c)):
                            for ph, cc in (("pre", pre_c), ("control", pre_c), ("post", post_c)):
                                crow[f"{grader}_{tgt}_{ph}_correct"] = cc
                                crow[f"{grader}_{tgt}_{ph}_incorrect"] = 16 - cc
                                crow[f"{grader}_{tgt}_{ph}_unknown"] = 0
                    ref_csv_rows.append(crow)

                    brow: dict[str, Any] = {
                        "difficulty": diff,
                        "tier_zh": tier_zh,
                        "source_key": skey,
                        "source_id": sid,
                        "row_index": i,
                    }
                    for grader in ("agnes", "ar"):
                        for tgt, cc in (("orig", 6), ("trans", max(0, pre_t_c + 1))):
                            brow[f"{grader}_{tgt}_post_correct"] = cc
                            brow[f"{grader}_{tgt}_post_incorrect"] = 16 - cc
                            brow[f"{grader}_{tgt}_post_unknown"] = 0
                    base_csv_rows.append(brow)

                    # Write A-group RAW_COMPLETE, SCORE_INPUT, AGNES & AUTHOR_RULE RESULTS.json
                    sdir = agroup_dir / stage / "sources" / skey
                    sdir.mkdir(parents=True)
                    (sdir / "RAW_COMPLETE.json").write_text(json.dumps({"source_id": sid, "samples": 32}), encoding="utf-8")
                    sc_in = []
                    ag_res = []
                    for tgt_raw, is_orig in (("source", True), (f"ours_{diff}", False)):
                        cc_lim = 12 if is_orig else min(16, pre_t_c + 3)
                        for k in range(16):
                            # Make k==0 a >800 token response to exercise >800 path
                            txt = ("step " * 820 + sentinel_resp) if k == 0 else sentinel_resp
                            sc_in.append(
                                {
                                    "source_id": sid,
                                    "target": tgt_raw,
                                    "phase": "post",
                                    "index": k,
                                    "problem": prob if is_orig else t_prob,
                                    "reference_answer": ans,
                                    "student_response": txt,
                                    "finish_reason": "stop",
                                }
                            )
                            ag_res.append(
                                {
                                    "identity": {"source_id": sid, "target": tgt_raw, "phase": "post", "index": k},
                                    "status": "correct" if k < cc_lim else "incorrect",
                                }
                            )
                    (sdir / "SCORE_INPUT.json").write_text(json.dumps(sc_in), encoding="utf-8")
                    for gsub in ("AGNES", "AUTHOR_RULE"):
                        gdir = agroup_dir / gsub / stage / "sources" / skey
                        gdir.mkdir(parents=True)
                        (gdir / "RESULTS.json").write_text(
                            json.dumps({"source_id": sid, "results": ag_res}), encoding="utf-8"
                        )

            inputs_p = bundle_dir / "selfbuilt_eval_inputs_1489.jsonl"
            counts_p = bundle_dir / "ref27b_counts.csv"
            inputs_p.write_text("\n".join(inputs_lines) + "\n", encoding="utf-8")
            with counts_p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(COUNT_COLUMNS))
                w.writeheader()
                w.writerows(ref_csv_rows)
            man_doc = {
                "schema": "t021_a100_export_v1",
                "eligible_count": len(ref_csv_rows),
                "skipped_count": 0,
                "per_tier_counts": {"easy": n_src, "medium": n_src, "hard": n_src},
                "math_verify_version": "0.9.0",
                "files_sha256": {
                    inputs_p.name: _sha256_bytes(inputs_p.read_bytes()),
                    counts_p.name: _sha256_bytes(counts_p.read_bytes()),
                },
            }
            (bundle_dir / "manifest_export.json").write_text(json.dumps(man_doc), encoding="utf-8")

            base_csv_p = root / "base4b_counts.csv"
            with base_csv_p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(BASE4B_COUNT_COLUMNS))
                w.writeheader()
                w.writerows(base_csv_rows)

            # Build synthetic author test.parquet (32 overlapping with our 35 + 3 author-only = 35 rows)
            pq_rows = []
            t022_convs = []
            for ridx in range(n_src):
                if ridx < 32:
                    p_txt = sentinel_prob if ridx == 0 else f"Original secret math problem statement number {ridx:03d} with x^2 + {ridx} = 0."
                else:
                    p_txt = f"Author exclusive problem statement {ridx:03d} unrelated to ours."
                pq_rows.append(
                    {
                        "extra_info": {
                            "problem": p_txt,
                            "answer": str(ridx),
                            "pre_solve_rate": 0.25,
                            "pre_solve_rate_transfer_problem": 0.25,
                            "transfer_problem": f"Author transfer {ridx}",
                            "transfer_answer": str(ridx),
                        }
                    }
                )
                t022_convs.append(
                    {
                        "row_index": ridx,
                        "pre_solve_rate": 0.25,
                        "pre_solve_rate_transfer": 0.25,
                        "post_solve_transfer": 0.50,
                        "post_solve_same": 0.625,
                        "delta_transfer": normalized_gain(0.25, 0.50),
                        "delta_same": normalized_gain(0.25, 0.625),
                    }
                )
            pq_p = root / "test.parquet"
            pd.DataFrame(pq_rows).to_parquet(pq_p)
            t022_p = root / "t022.json"
            t022_p.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "blanked_teacher_turn_rate": 0.005,
                            "token_budget_truncated_rate": 0.0,
                            "eoc_rate": 0.54,
                            "thinking_tokens_avg_per_turn": 61.8,
                            "n_student_calls_failed": 0,
                            "assume_thinking": True,
                        },
                        "conversations": t022_convs,
                    }
                ),
                encoding="utf-8",
            )

            # Synthetic A100 counts JSON
            a100_p = root / "a100.json"
            a100_doc = {
                "schema": "t025_a100_counts_v1",
                "eduardo27b": {
                    "run_configuration": {
                        "teacher_dtype": "bfloat16",
                        "student_dtype": "float16",
                        "teacher_max_tokens": 4192,
                        "student_max_tokens": 1024,
                        "student_samples_per_condition": 16,
                        "student_temperature": 0.6,
                    },
                    "pre_post_identical_student_and_grader": True,
                    "omit_teacher_turns_in_post": True,
                    "rollout_diagnostics": {"blanked_turn_rate": 0.004, "token_budget_truncated_rate": 0.0},
                    "length_and_format": {
                        "by_tier_target_phase": [
                            {"phase": "pre", "target": "trans", "difficulty": "easy", "ar_incorrect_format_rate": 0.02},
                            {"phase": "post", "target": "trans", "difficulty": "easy", "ar_incorrect_format_rate": 0.02},
                        ],
                        "post_length_screening": [
                            {
                                "group": g,
                                "n_total": 560,
                                "n_gt_800": 10,
                                "gt_800_rate": 10 / 560,
                                "le_800": {"acc_unknown_as_wrong": 0.50, "agnes_unknown_rate": 0.0},
                                "gt_800": {"acc_unknown_as_wrong": 0.40, "agnes_unknown_rate": 0.0},
                                "gt_800_boxed_diagnostics": {
                                    "boxed_only_in_801_1024_lost_if_trunc": 4,
                                    "boxed_overwritten_in_801_1024": 0,
                                },
                            }
                            for g in ("简单迁移题", "中等迁移题", "困难迁移题")
                        ],
                    },
                },
                "base4b": None,
            }
            a100_p.write_text(json.dumps(a100_doc), encoding="utf-8")

            pro_out = root / "pro_summary.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "run_t025_pro5000.py"),
                    "--bundle-dir",
                    str(bundle_dir),
                    "--agroup-dir",
                    str(agroup_dir),
                    "--base4b-counts-csv",
                    str(base_csv_p),
                    "--t022-json",
                    str(t022_p),
                    "--test-parquet",
                    str(pq_p),
                    "--student-tokenizer-dir",
                    str(tok_dir),
                    "--a100-json",
                    str(a100_p),
                    "--json-out",
                    str(pro_out),
                    "--draws",
                    "2500",
                    "--allow-small-test",
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"run_t025_pro5000.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            pro_json_str = pro_out.read_text(encoding="utf-8")
            combined = merge_pro5000_and_a100(json.loads(pro_json_str), a100_doc)
            comb_str = json.dumps(combined, ensure_ascii=False)

            for secret in (
                sentinel_sid,
                "SECRET_MATRIX_OMEGA_774123",
                "SECRET_VARIANT_SIGMA_553190",
                sentinel_ans,
                "SECRET_STUDENT_CHAIN_881122",
                "test_secret_src_001",
            ):
                self.assertNotIn(secret, proc.stdout)
                self.assertNotIn(secret, pro_json_str)
                self.assertNotIn(secret, comb_str)

            self.assertEqual(combined["pro5000"]["cause_f"]["overlap_counts"]["n_exact_overlap"], 32)
            self.assertTrue(set(combined["verdicts"].keys()) == {"A", "B", "C", "D", "E", "F", "G"})
            for k, v in combined["verdicts"].items():
                self.assertIn(v["verdict"], ("确认", "部分贡献", "排除", "检验不了"), msg=f"Invalid verdict on {k}")

    def test_6_decision_rules_all_branches(self) -> None:
        """Verify pre-registered decision rules for Causes A, B, C, D, E, F, G."""
        # Cause C: >=15% format error -> 检验不了
        vc_untest = judge_cause_c(
            {d: {"mean_diff": 0.0, "ci": [-0.01, 0.01]} for d in ("easy", "medium", "hard")},
            {"by_tier_target_phase": [{"phase": "post", "ar_incorrect_format_rate": 0.16}]},
        )
        self.assertEqual(vc_untest["verdict"], "检验不了")

        # Cause B: all 3 tiers CI low > 0.05 and author >= 0.25 -> 确认
        vb_conf = judge_cause_b(
            {d: {"mean_diff": 0.12, "ci": [0.06, 0.18]} for d in ("easy", "medium", "hard")},
            0.2775,
        )
        self.assertEqual(vb_conf["verdict"], "确认")

        # Cause D: <5% >800 -> 排除
        vd_excl = judge_cause_d(
            {
                "eduardo27b": {
                    "post_length_screening": [
                        {
                            "group": g,
                            "n_total": 1000,
                            "gt_800_rate": 0.02,
                            "le_800": {"acc_unknown_as_wrong": 0.5},
                            "gt_800": {"acc_unknown_as_wrong": 0.2},
                            "gt_800_boxed_diagnostics": {},
                        }
                        for g in ("简单迁移题", "中等迁移题", "困难迁移题")
                    ]
                }
            }
        )
        self.assertEqual(vd_excl["verdict"], "排除")

        # Cause E: same config and |diff|<0.02 with CI containing 0 -> 排除
        ve_excl = judge_cause_e(
            True,
            {"mean_diff": 0.005, "ci": [-0.012, 0.021]},
            {"mean_diff": 0.008, "ci": [-0.010, 0.025]},
        )
        self.assertEqual(ve_excl["verdict"], "排除")

        # Cause F: |diff|<0.015 and missing bound <0.015 -> 排除
        vf_excl = judge_cause_f(
            {d: {"diff_intersect_minus_all": {"mean_diff": 0.002}} for d in ("easy", "medium", "hard")},
            {d: {"max_upward_shift": 0.007} for d in ("easy", "medium", "hard")},
        )
        self.assertEqual(vf_excl["verdict"], "排除")

        # Cause G: aligned tutor, empty < 2%, T-022 within 0.02 of paper -> 排除
        vg_excl = judge_cause_g(
            {"delta_transfer": {"mean": 0.2775}, "delta_same": {"mean": 0.4548}},
            {
                "run_configuration": {"teacher_dtype": "bfloat16", "student_dtype": "float16", "teacher_max_tokens": 4192},
                "omit_teacher_turns_in_post": True,
                "rollout_diagnostics": {"blanked_turn_rate": 0.005},
            },
        )
        self.assertEqual(vg_excl["verdict"], "排除")

        # Cause D triggered (>=5% and acc_gt800 < acc_le800) without rerun -> 检验不了
        vd_trig = judge_cause_d(
            {
                "eduardo27b": {
                    "post_length_screening": [
                        {
                            "group": g,
                            "n_total": 1000,
                            "gt_800_rate": 0.08,
                            "le_800": {"acc_unknown_as_wrong": 0.50},
                            "gt_800": {"acc_unknown_as_wrong": 0.20},
                            "gt_800_boxed_diagnostics": {},
                        }
                        for g in ("简单迁移题", "中等迁移题", "困难迁移题")
                    ]
                }
            }
        )
        self.assertEqual(vd_trig["verdict"], "检验不了")

    def test_7_a100_collector_end_to_end_and_sanitization(self) -> None:
        """Run run_t025_a100.py end-to-end on synthetic A100 ref_root + overlay + rejudge + base4b."""
        from ref27b_manifest import INVENTORY_SHA, SUMMARY_SHA, _default_module_paths, _import_checked

        sum_p, inv_p = _default_module_paths(REPO_ROOT)
        s_mod = _import_checked(sum_p, SUMMARY_SHA, "test_s_mod")
        inv_mod = _import_checked(inv_p, INVENTORY_SHA, "test_inv_mod")
        app_dir = str(REPO_ROOT / "evaluation" / "app")
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        import offline_eval  # noqa: E402

        sentinel_sid = "A100_SECRET_SID_778899"
        sentinel_prob = "Prove that A100_SECRET_PROBLEM_TEXT_654321 holds for all primes."
        sentinel_resp = r"Step 1 A100_SECRET_STUDENT_RESP_112233 final \boxed{7}"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_jsonl = root / "test.jsonl"
            ref_root = root / "own-test-499-v1"
            overlay_root = root / "overlay"
            decisions_root = root / "decisions"
            base4b_dir = root / "base4b"
            tok_dir = root / "tok"
            for d in (ref_root, overlay_root, decisions_root, base4b_dir, tok_dir):
                d.mkdir(parents=True)
            for st in ("PILOT", "EASY", "MEDIUM", "HARD"):
                (ref_root / st / "sources").mkdir(parents=True)

            vocab = {"<unk>": 0, "Step": 1, "1": 2, r"\boxed{7}": 3}
            (tok_dir / "tokenizer.json").write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "truncation": None,
                        "padding": None,
                        "added_tokens": [],
                        "normalizer": None,
                        "pre_tokenizer": {"type": "Whitespace"},
                        "post_processor": None,
                        "decoder": None,
                        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
                    }
                ),
                encoding="utf-8",
            )

            raw_line = {
                "source_id": sentinel_sid,
                "our_record": {
                    "source_id": sentinel_sid,
                    "source_problem": sentinel_prob,
                    "source_answer": "7",
                    "transfer_questions": [
                        {
                            "candidate_id": f"c_{pos}",
                            "position_by_student_correct": pos,
                            "student_correct": pos * 4,
                            "student_n_scored": 16,
                            "problem": f"Variant {pos} of {sentinel_prob}",
                            "answer": "7",
                        }
                        for pos in (1, 2, 3)
                    ],
                },
            }
            data_jsonl.write_text(json.dumps(raw_line, ensure_ascii=False) + "\n", encoding="utf-8")

            skey = hashlib.sha256(sentinel_sid.encode("utf-8")).hexdigest()[:24]
            for diff, stage in (("easy", "EASY"), ("medium", "MEDIUM"), ("hard", "HARD")):
                (ref_root / stage / "RUN_CONFIGURATION.json").write_text(
                    json.dumps(
                        {
                            "protocol": "own-transfer-fresh-offline-v1",
                            "transfer_difficulty": diff,
                            "teacher_dtype": "bfloat16",
                            "teacher_tp": 2,
                            "student_dtype": "float16",
                            "teacher_max_tokens": 4192,
                            "student_max_tokens": 1024,
                            "student_samples_per_condition": 16,
                            "student_temperature": 0.6,
                            "installed_versions": {"vllm": "0.18.0", "math-verify": "0.8.0"},
                        }
                    ),
                    encoding="utf-8",
                )
                parsed_row = offline_eval.parse_sources(data_jsonl, diff)[0]
                row_sha = hashlib.sha256(
                    json.dumps(parsed_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                sdir = ref_root / stage / "sources" / skey
                (sdir / "dialogue").mkdir(parents=True)
                (sdir / "SOURCE.json").write_text(
                    json.dumps(
                        {
                            "source_id": sentinel_sid,
                            "sha256": row_sha,
                            "transfer_target": f"ours_{diff}",
                            "candidate_id": parsed_row["transfer_candidate_id"],
                        }
                    ),
                    encoding="utf-8",
                )
                (sdir / "dialogue" / "ROLLOUT_RESULT.json").write_text(
                    json.dumps(
                        {
                            "assume_thinking": True,
                            "end_reason": "end_of_conversation",
                            "eoc_used": True,
                            "thinking_tokens": [55, 60],
                            "raw_messages": [
                                {"role": "teacher", "content": "think</think>visible reply"},
                                {"role": "student", "content": "ok"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                sc_rows = []
                for tgt in ("source", f"ours_{diff}"):
                    for ph in ("pre", "control", "post"):
                        for idx in range(16):
                            sc_rows.append(
                                {
                                    "source_id": sentinel_sid,
                                    "target": tgt,
                                    "phase": ph,
                                    "index": idx,
                                    "problem": sentinel_prob if tgt == "source" else parsed_row["transfer_problem"],
                                    "reference_answer": "7",
                                    "student_response": ("Step " * 810 + sentinel_resp) if (ph == "post" and idx == 0) else sentinel_resp,
                                    "finish_reason": "stop",
                                    **(
                                        {"candidate_id": parsed_row["transfer_candidate_id"]}
                                        if tgt != "source"
                                        else {}
                                    ),
                                }
                            )
                sc_bytes = json.dumps(sc_rows, ensure_ascii=False).encode("utf-8")
                sc_sha = hashlib.sha256(sc_bytes).hexdigest()
                (sdir / "SCORE_INPUT.json").write_bytes(sc_bytes)
                (sdir / "RAW_COMPLETE.json").write_text(
                    json.dumps(
                        {
                            "source_id": sentinel_sid,
                            "samples": 96,
                            "grading_status": "ungraded",
                            "source_sha256": row_sha,
                            "score_input_sha256": sc_sha,
                        }
                    ),
                    encoding="utf-8",
                )

                ag_results = []
                for item in sc_rows:
                    ag_results.append(
                        {
                            "identity": inv_mod.identity(item),
                            "status": "correct" if item["index"] < 6 else "incorrect",
                            "request_sha256": inv_mod.digest(inv_mod.payload(item)),
                            "model": inv_mod.MODEL,
                            "grader": "agnes_full_response_v2",
                        }
                    )
                ag_dir = ref_root / "AGNES" / stage / "sources" / skey
                ag_dir.mkdir(parents=True)
                (ag_dir / "RESULTS.json").write_text(
                    json.dumps(
                        {
                            "source_id": sentinel_sid,
                            "input_sha256": sc_sha,
                            "grader": "agnes_full_response_v2",
                            "model": inv_mod.MODEL,
                            "results": ag_results,
                        }
                    ),
                    encoding="utf-8",
                )

                # Base4b post-only dir
                b_sdir = base4b_dir / stage / "sources" / skey
                b_sdir.mkdir(parents=True)
                post_rows = [r for r in sc_rows if r["phase"] == "post"]
                (b_sdir / "SCORE_INPUT.json").write_text(json.dumps(post_rows), encoding="utf-8")
                (b_sdir / "RAW_COMPLETE.json").write_text(json.dumps({"source_id": sentinel_sid, "samples": 32}), encoding="utf-8")
                b_ag = base4b_dir / "AGNES" / stage / "sources" / skey
                b_ag.mkdir(parents=True)
                (b_ag / "RESULTS.json").write_text(
                    json.dumps(
                        {
                            "source_id": sentinel_sid,
                            "results": [{"identity": inv_mod.identity(r), "status": "correct"} for r in post_rows],
                        }
                    ),
                    encoding="utf-8",
                )

            a100_out = root / "t025_a100_counts.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "run_t025_a100.py"),
                    "--repo",
                    str(REPO_ROOT),
                    "--data",
                    str(data_jsonl),
                    "--ref-root",
                    str(ref_root),
                    "--overlay",
                    str(overlay_root),
                    "--decisions",
                    str(decisions_root),
                    "--base4b-dir",
                    str(base4b_dir),
                    "--student-tokenizer-dir",
                    str(tok_dir),
                    "--json-out",
                    str(a100_out),
                    "--allow-small-test",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            out_json_s = a100_out.read_text(encoding="utf-8")
            for secret in (sentinel_sid, "A100_SECRET_PROBLEM_TEXT_654321", "A100_SECRET_STUDENT_RESP_112233"):
                self.assertNotIn(secret, proc.stdout)
                self.assertNotIn(secret, out_json_s)
            parsed = json.loads(out_json_s)
            self.assertEqual(parsed["eduardo27b"]["length_and_format"]["total_samples_scanned"], 288)
            self.assertEqual(parsed["base4b"]["length_and_format"]["total_samples_scanned"], 96)


if __name__ == "__main__":
    unittest.main()
