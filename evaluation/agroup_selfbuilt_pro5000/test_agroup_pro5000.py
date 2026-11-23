#!/usr/bin/env python3
"""Unit tests for T-021 (Pro 5000 A-group step_100 x self-built 3-tier) using synthetic data only."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
PAIRED_DIR = REPO_ROOT / "evaluation" / "paired_gain"
for _p in (str(HERE), str(BASE4B_DIR), str(PAIRED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import export_27b_selfbuilt as exp27  # noqa: E402
import import_bundle as ib  # noqa: E402
import make_selfbuilt_paired_csv as mcsv  # noqa: E402
import paired_gain as pg  # noqa: E402
import run_post_eval_pro5000 as rpp  # noqa: E402
import stack_check_selfbuilt as sc  # noqa: E402
import summarize_agroup_pro5000 as sap  # noqa: E402
import summarize_base4b as sb  # noqa: E402
import test_base4b_selfbuilt as tbase  # noqa: E402
import wait_gpus_agroup_pro5000 as wgp  # noqa: E402


class TestT021AGroupPro5000(unittest.TestCase):
    def test_export_import_alignment_and_mismatch_failure(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sids = [f"T021_SID_{i}" for i in range(4)]
            incomplete = {("easy", "T021_SID_1"), ("hard", "T021_SID_3")}
            data_path, ref_root, overlay_root, decisions_root = tbase._build_synthetic_27b_env(
                tmp, sids, incomplete
            )
            exp_dir = tmp / "export_out"
            man = exp27.export_ref27b(
                repo_root=REPO_ROOT,
                data_path=data_path,
                ref_root=ref_root,
                overlay_root=overlay_root,
                decisions_root=decisions_root,
                out_dir=exp_dir,
                require_exact_totals=False,
                verify_27b_r1=False,
                grade_fn=lambda sol, gold: {"acc": 1.0},
            )
            self.assertEqual(man["eligible_count"], 10)  # 4*3 - 2
            self.assertEqual(man["skipped_count"], 2)
            self.assertTrue((exp_dir / "t021_a100_export_bundle.tar.gz").is_file())
            self.assertTrue((exp_dir / "t021_a100_export_bundle.tar.gz.sha256").is_file())

            # Zero 27B response strings leaked into selfbuilt_eval_inputs_1489.jsonl
            inputs_txt = (exp_dir / "selfbuilt_eval_inputs_1489.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("SYNTHETIC_RESPONSE", inputs_txt)

            loaded = ib.load_and_verify_bundle(exp_dir, require_exact_totals=False)
            self.assertEqual(loaded["eligible_count"], 10)
            pairs = {(e["difficulty"], e["source_id"]) for e in loaded["eligible"]}
            for bad in incomplete:
                self.assertNotIn(bad, pairs)

            # Corrupting ref27b_counts.csv must fail SHA-256 check on import
            counts_p = exp_dir / "ref27b_counts.csv"
            orig_csv = counts_p.read_text(encoding="utf-8")
            counts_p.write_text(orig_csv + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ib.load_and_verify_bundle(exp_dir, require_exact_totals=False)

            # Restoring SHA-256 matches, while changing row_index without updating manifest vs mismatched row_index fails
            counts_p.write_text(orig_csv.replace("T021_SID_0", "WRONG_SID_X", 1), encoding="utf-8")
            man_p = exp_dir / "manifest_export.json"
            m_doc = json.loads(man_p.read_text(encoding="utf-8"))
            m_doc["files_sha256"]["ref27b_counts.csv"] = exp27._sha256_file(counts_p)
            man_p.write_text(json.dumps(m_doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                ib.load_and_verify_bundle(exp_dir, require_exact_totals=False)

    def test_stack_check_selection_and_boundaries(self):
        # Build synthetic 35 sources -> 30 per tier deterministic hash selection
        entries = []
        pro_ok = {}
        for diff in ("easy", "medium", "hard"):
            for i in range(35):
                sid = f"SC_SID_{i}"
                pre_c = (i % 13) + 2
                ctl_c = (i % 13) + 2 if i % 5 != 0 else max(1, (i % 13) + 1)
                ar_ref = {
                    "原题": {"pre": {"correct": pre_c}, "control": {"correct": ctl_c}},
                    "迁移题": {"pre": {"correct": pre_c}, "control": {"correct": ctl_c}},
                }
                entries.append(
                    {
                        "difficulty": diff,
                        "source_id": sid,
                        "source_key": hashlib.sha256(sid.encode()).hexdigest()[:24],
                        "ref_counts_ar": ar_ref,
                    }
                )
                pro_ok[(diff, sid)] = {
                    "orig_pre_rate": pre_c / 16.0,
                    "trans_pre_rate": pre_c / 16.0,
                }
        s1 = sc.select_stack_check_entries(entries, per_tier=30, seed=42)
        s2 = sc.select_stack_check_entries(list(reversed(entries)), per_tier=30, seed=42)
        self.assertEqual([(e["difficulty"], e["source_id"]) for e in s1], [(e["difficulty"], e["source_id"]) for e in s2])
        self.assertEqual(len(s1), 90)

        res_ok = sc.evaluate_stack_check(s1, {k: pro_ok[k] for k in [(e["difficulty"], e["source_id"]) for e in s1]}, stu_fail=0)
        self.assertTrue(res_ok["passed"])

        # Boundary: mean diff > 0.03 fails
        pro_mean_bad = {
            (e["difficulty"], e["source_id"]): {
                "orig_pre_rate": min(1.0, pro_ok[(e["difficulty"], e["source_id"])]["orig_pre_rate"] + 0.05),
                "trans_pre_rate": pro_ok[(e["difficulty"], e["source_id"])]["trans_pre_rate"],
            }
            for e in s1
        }
        self.assertFalse(sc.evaluate_stack_check(s1, pro_mean_bad, stu_fail=0)["passed"])

        # Boundary: correlation inverted -> fails
        pro_corr_bad = {
            (e["difficulty"], e["source_id"]): {
                "orig_pre_rate": 1.0 - pro_ok[(e["difficulty"], e["source_id"])]["orig_pre_rate"],
                "trans_pre_rate": 1.0 - pro_ok[(e["difficulty"], e["source_id"])]["trans_pre_rate"],
            }
            for e in s1
        }
        self.assertFalse(sc.evaluate_stack_check(s1, pro_corr_bad, stu_fail=0)["passed"])

        # Boundary: stu_fail > 0 fails even with identical rates
        self.assertFalse(sc.evaluate_stack_check(s1, pro_ok, stu_fail=1)["passed"])

        # No source_id in stack report; embeds student_runtime metadata cleanly
        res_ok["student_runtime"] = {
            "student_dtype": "float16",
            "vllm": "0.19.1",
            "torch": "2.10.0+cu128",
            "transformers": "5.7.0",
            "math_verify": "0.8.0",
        }
        rep = sc.format_stack_report(res_ok)
        self.assertNotIn("SC_SID_", rep)
        self.assertIn("dtype=float16", rep)

    def test_t020_agnes_guard_and_max_3_workers(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "a_group_pro5000"
            runs = base / "runs"
            t020 = runs / "eval_author494_full_20261123"
            t020.mkdir(parents=True)
            fake_proc = Path(td) / "proc"
            fake_proc.mkdir()

            # Workers > 3 forbidden
            with self.assertRaises(ValueError):
                rpp.check_t020_finished(t020_base=base, workers=4, proc_dir=fake_proc)

            # Missing result JSON -> fails closed
            with self.assertRaises(RuntimeError):
                rpp.check_t020_finished(t020_base=base, workers=3, proc_dir=fake_proc)

            # Complete JSON written, but runner.pid alive in fake_proc -> fails closed
            (t020 / "transfer_eval_a.json").write_text(
                json.dumps({"records": [{"i": i} for i in range(494)], "metrics": {"delta_transfer": 0.25}}),
                encoding="utf-8",
            )
            (t020 / "runner.pid").write_text("999999\n", encoding="utf-8")
            (fake_proc / "999999").mkdir()
            (fake_proc / "999999" / "cmdline").write_bytes(b"bash\0run_author494_pro5000.sh\0")
            with self.assertRaises(RuntimeError):
                rpp.check_t020_finished(t020_base=base, workers=3, proc_dir=fake_proc)

            # Once PID exited -> passes cleanly
            (fake_proc / "999999" / "cmdline").unlink()
            (fake_proc / "999999").rmdir()
            info = rpp.check_t020_finished(t020_base=base, workers=3, proc_dir=fake_proc)
            self.assertEqual(info["records"], 494)
            self.assertEqual(info["workers"], 3)

    def test_paired_csv_crosscheck_and_sanitized_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sids = [f"PAIR_SID_{i}" for i in range(8)]
            incomplete = {("hard", "PAIR_SID_7")}
            data_path, ref_root, overlay_root, decisions_root = tbase._build_synthetic_27b_env(
                tmp, sids, incomplete
            )
            exp_dir = tmp / "exp"
            exp27.export_ref27b(
                repo_root=REPO_ROOT,
                data_path=data_path,
                ref_root=ref_root,
                overlay_root=overlay_root,
                decisions_root=decisions_root,
                out_dir=exp_dir,
                require_exact_totals=False,
                verify_27b_r1=False,
                grade_fn=lambda sol, gold: {"acc": 1.0},
            )
            loaded = ib.load_and_verify_bundle(exp_dir, require_exact_totals=False)

            # Build synthetic Pro 5000 A-group results for ONLY the first 6 sources (2 per tier)
            ag_root = tmp / "ag_out"
            for e in loaded["eligible"]:
                if int(e["source_id"].split("_")[-1]) >= 2:
                    continue
                diff = e["difficulty"]
                stage = sb.STAGE_BY_DIFF[diff]
                key = e["source_key"]
                s_dir = ag_root / stage / "sources" / key
                s_dir.mkdir(parents=True)
                (s_dir / "RAW_COMPLETE.json").write_text('{"samples":32}', encoding="utf-8")
                res_dir = ag_root / "AGNES" / stage / "sources" / key
                res_dir.mkdir(parents=True)
                results = []
                for tgt in ("source", e["row"]["transfer_target"]):
                    for i in range(16):
                        results.append(
                            {
                                "identity": {"source_id": e["source_id"], "target": tgt, "phase": "post", "index": i},
                                "status": "correct" if i < 12 else "incorrect",
                            }
                        )
                (res_dir / "RESULTS.json").write_text(
                    json.dumps({"source_id": e["source_id"], "results": results}),
                    encoding="utf-8",
                )

            out_csv = tmp / "paired.csv"
            mcsv.build_selfbuilt_paired_csv(
                loaded_bundle=loaded,
                a_group_root=ag_root,
                base4b_counts_csv=None,
                out_csv=out_csv,
                grade_subdir="AGNES",
            )
            csv_rows = pg.load_csv(out_csv)

            # Paired evaluation against 27B should use ONLY the 2 completed A-group sources per tier
            records, diag = sb.collect_base4b_records(loaded, ag_root, REPO_ROOT, grade_subdir="AGNES")
            all_27b = [{"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]} for e in loaded["eligible"]]
            cc = sap.cross_check_paired_diff(
                csv_rows, records, all_27b, base_name="eduardo27b", target_name="a_group", draws=2000, seed=42
            )
            self.assertEqual(cc["comparisons"]["未决按错.easy"]["n_paired_pg"], 2)
            self.assertEqual(cc["comparisons"]["未决按错.orig"]["n_paired_pg"], 2)

            # Check sanitization on markdown report & safe logs/exceptions
            md = pg.render_markdown(cc["pg_report"])
            for sid in sids:
                self.assertNotIn(sid, md)
            self.assertNotIn("SYNTHETIC_ORIG_PROBLEM", md)
            fake_log = (
                "[base4b] bundle_ok eligible=1489\n"
                "[WARNING] agnes: FAKE_WARNING_SECRET_PROBLEM\n"
                '  File "/tmp/run.py", line 10, in go\n'
                "ValueError: UNQUOTED_PROBLEM_IN_EXC 2+2=4\n"
            )
            safe_lines = "\n".join(sb.extract_safe_log_lines(fake_log))
            safe_fail = sb.extract_safe_failure_summary(fake_log)
            self.assertNotIn("FAKE_WARNING_SECRET_PROBLEM", safe_lines)
            self.assertNotIn("UNQUOTED_PROBLEM_IN_EXC", safe_fail)
            self.assertNotIn("2+2=4", safe_fail)
            self.assertNotIn("FAKE_WARNING_SECRET_PROBLEM", safe_fail)

    def test_pro5000_wait_gpus_handles_gpu6_quirk(self):
        # GPU 6 shows util=100 while used=0 & 0 apps; should prefer GPU 0 first then GPU 6 as quirk fallback
        snap1 = wgp.parse_gpu_csv(
            "0, UUID-0, 71000, 73415, 0\n"
            "6, UUID-6, 73000, 73415, 100\n",
            "",
        )
        snap2 = wgp.parse_gpu_csv(
            "0, UUID-0, 71000, 73415, 0\n"
            "6, UUID-6, 73000, 73415, 100\n",
            "",
        )
        with tempfile.TemporaryDirectory() as td:
            snaps = iter([snap1, snap2])
            t_gpu, s_gpus = wgp.wait_for_gpus(
                status_path=Path(td) / "w.json",
                log_path=Path(td) / "w.log",
                ports=[18701, 18702],
                num_gpus=2,
                min_free_mib=70000,
                snapshot_fn=lambda: next(snaps),
                ports_fn=lambda _p: True,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(t_gpu, 0)
            self.assertEqual(s_gpus, [6])

    def test_preflight_model_hashes_are_64_hex(self):
        import re

        txt = (HERE / "preflight_agroup_pro5000.sh").read_text(encoding="utf-8")
        hashes = re.findall(r':\s*"([0-9a-f]+)"', txt)
        self.assertEqual(len(hashes), 14)
        for h in hashes:
            self.assertEqual(len(h), 64, f"Expected 64-char sha256, got {len(h)}: {h}")


if __name__ == "__main__":
    unittest.main()
