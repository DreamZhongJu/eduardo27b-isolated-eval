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

import agnes_window as aw  # noqa: E402
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

    def test_t020_agnes_guard_and_max_16_workers(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "a_group_pro5000"
            runs = base / "runs"
            t020 = runs / "eval_author494_agroup_full_20261123"
            t020_sub = t020 / "a_group_step100"
            t020_sub.mkdir(parents=True)
            fake_proc = Path(td) / "proc"
            fake_proc.mkdir()

            # Workers > 16 forbidden
            with self.assertRaises(ValueError):
                rpp.check_t020_finished(t020_base=base, workers=17, proc_dir=fake_proc)

            # Missing result JSON -> fails closed
            with self.assertRaises(RuntimeError):
                rpp.check_t020_finished(t020_base=base, workers=16, proc_dir=fake_proc)

            # Complete JSON written under subdirectory with conversations key, but runner.pid alive -> fails closed
            (t020_sub / "transfer_eval_a.json").write_text(
                json.dumps({"conversations": [{"i": i} for i in range(494)], "metrics": {"delta_transfer": 0.25}}),
                encoding="utf-8",
            )
            (t020 / "runner.pid").write_text("999999\n", encoding="utf-8")
            (fake_proc / "999999").mkdir()
            (fake_proc / "999999" / "cmdline").write_bytes(b"bash\0run_author494_pro5000.sh\0")
            with self.assertRaises(RuntimeError):
                rpp.check_t020_finished(t020_base=base, workers=16, proc_dir=fake_proc)

            # Once PID exited -> passes cleanly up to workers=16
            (fake_proc / "999999" / "cmdline").unlink()
            (fake_proc / "999999").rmdir()
            info = rpp.check_t020_finished(t020_base=base, workers=16, proc_dir=fake_proc)
            self.assertEqual(info["records"], 494)
            self.assertEqual(info["workers"], 16)

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
            p_info = mcsv.build_selfbuilt_paired_csv(
                loaded_bundle=loaded,
                a_group_root=ag_root,
                base4b_counts_csv=None,
                out_csv=out_csv,
                grade_subdir="AGNES",
            )
            self.assertTrue(p_info["emit_combined_orig"])
            csv_rows = pg.load_csv(out_csv)

            # Paired evaluation against 27B should use ONLY the 2 completed A-group sources per tier
            records, diag = sb.collect_base4b_records(loaded, ag_root, REPO_ROOT, grade_subdir="AGNES")
            all_27b = [{"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]} for e in loaded["eligible"]]
            cc = sap.cross_check_paired_diff(
                csv_rows, records, all_27b, base_name="eduardo27b", target_name="a_group", draws=2000, seed=42
            )
            self.assertEqual(cc["comparisons"]["未决按错.easy"]["n_paired_pg"], 2)
            self.assertEqual(cc["comparisons"]["未决按错.orig"]["n_paired_pg"], 2)

            # When 27B EASY vs MEDIUM orig pre differs for the same source_id, CSV build omits
            # conflicting combined "orig" instead of failing validate_pre_consistency, while keeping
            # per-tier orig_easy/orig_medium/orig_hard and easy/medium/hard intact.
            loaded_diff_orig = json.loads(json.dumps(loaded))
            for item in loaded_diff_orig["eligible"]:
                if item["source_id"] == "PAIR_SID_0" and item["difficulty"] == "easy":
                    item["ref_counts"]["原题"]["pre"] = {"correct": 12, "incorrect": 4, "unknown": 0}
                    item["ref_counts"]["原题"]["post"] = {"correct": 15, "incorrect": 1, "unknown": 0}
                elif item["source_id"] == "PAIR_SID_0" and item["difficulty"] == "medium":
                    item["ref_counts"]["原题"]["pre"] = {"correct": 4, "incorrect": 12, "unknown": 0}
                    item["ref_counts"]["原题"]["post"] = {"correct": 10, "incorrect": 6, "unknown": 0}
            out_csv_diff = tmp / "paired_diff_orig.csv"
            p_diff = mcsv.build_selfbuilt_paired_csv(
                loaded_bundle=loaded_diff_orig,
                a_group_root=ag_root,
                base4b_counts_csv=None,
                out_csv=out_csv_diff,
                grade_subdir="AGNES",
            )
            self.assertFalse(p_diff["emit_combined_orig"])
            self.assertEqual(p_diff["orig_inconsistent_sources"], 1)
            diff_rows = pg.load_csv(out_csv_diff)
            self.assertFalse(any(r["group"] == "orig" for r in diff_rows))
            all_27b_diff = [{"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]} for e in loaded_diff_orig["eligible"]]
            recs_diff, _ = sb.collect_base4b_records(loaded_diff_orig, ag_root, REPO_ROOT, grade_subdir="AGNES")
            cc_diff = sap.cross_check_paired_diff(
                diff_rows, recs_diff, all_27b_diff, base_name="eduardo27b", target_name="a_group", draws=2000, seed=42
            )
            self.assertEqual(cc_diff["comparisons"]["未决按错.orig_easy"]["n_paired_pg"], 2)
            self.assertEqual(cc_diff["comparisons"]["未决按错.easy"]["n_paired_pg"], 2)

            # When 27B pre itself has Agnes unknowns, summarize_base4b excludes pre_unknown under
            # "排除未决" while paired_gain CSV keeps scalar pre=correct/16; cross_check aligns scalar pre
            # without crashing while preserving strict equality on "未决按错".
            loaded_pre_unk = json.loads(json.dumps(loaded))
            for item in loaded_pre_unk["eligible"]:
                if item["source_id"] == "PAIR_SID_0" and item["difficulty"] == "easy":
                    item["ref_counts"]["迁移题"]["pre"] = {"correct": 8, "incorrect": 6, "unknown": 2}
                    item["ref_counts"]["迁移题"]["post"] = {"correct": 12, "incorrect": 3, "unknown": 1}
            out_csv_pre_unk = tmp / "paired_pre_unk.csv"
            mcsv.build_selfbuilt_paired_csv(
                loaded_bundle=loaded_pre_unk,
                a_group_root=ag_root,
                base4b_counts_csv=None,
                out_csv=out_csv_pre_unk,
                grade_subdir="AGNES",
            )
            unk_rows = pg.load_csv(out_csv_pre_unk)
            all_27b_unk = [{"tier_zh": e["tier_zh"], "source_id": e["source_id"], "ref_counts": e["ref_counts"]} for e in loaded_pre_unk["eligible"]]
            recs_unk, _ = sb.collect_base4b_records(loaded_pre_unk, ag_root, REPO_ROOT, grade_subdir="AGNES")
            cc_unk = sap.cross_check_paired_diff(
                unk_rows, recs_unk, all_27b_unk, base_name="eduardo27b", target_name="a_group", draws=2000, seed=42
            )
            self.assertLessEqual(cc_unk["comparisons"]["排除未决.easy"]["pt_diff"], 1e-12)
            self.assertNotAlmostEqual(
                cc_unk["comparisons"]["排除未决.easy"]["sb_diff"],
                cc_unk["comparisons"]["排除未决.easy"]["sb_diff_pre_excl_unk"],
                places=4,
            )

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

    def test_agnes_window_aimd_cap_and_429_recovery(self):
        clock = [100.0]

        def now():
            return clock[0]

        def advance(s):
            clock[0] += float(s)

        win = aw.SyncCongestionWindow(
            initial=1,
            maximum=16,
            backoff_base=2.0,
            backoff_cap=60.0,
            jitter_low=1.0,
            jitter_high=1.0,
            monotonic_fn=now,
            sleep_fn=advance,
            rand_fn=lambda a, b: a,
        )
        # Window grows 1 -> 16 and caps cleanly at 16
        for _ in range(25):
            lease = win.acquire()
            win.report_success(lease)
            win.release()
        self.assertEqual(win.limit, 16)
        self.assertEqual(win.max_limit_reached, 16)

        # Consecutive 429s drop 16 -> 8 -> 4 -> 2 -> 1 -> 1 with capped exponential backoff
        delays = []
        for _ in range(5):
            lease = win.acquire()
            delays.append(win.report_congestion(event="http_429"))
            win.release()
        self.assertEqual(win.limit, 1)
        self.assertEqual(delays, [2.0, 4.0, 8.0, 16.0, 32.0])
        self.assertEqual(win.http_429_events, 5)
        self.assertEqual(win.window_drops, 4)

        # One more 429 hits 60s cap and stays floored at 1
        lease = win.acquire()
        d6 = win.report_congestion(event="http_429")
        win.release()
        self.assertEqual(d6, 60.0)
        self.assertEqual(win.limit, 1)

        # Next acquire honors backoff cooldown via fake sleep, then valid successes climb back up
        for expected in (2, 3, 4, 5):
            lease = win.acquire()
            win.report_success(lease)
            win.release()
            self.assertEqual(win.limit, expected)

        # Verify caller treats semantic null as neutral while ambiguous HTTP 200 triggers congestion
        judge = rpp.load_verified_agnes_judge(REPO_ROOT / "evaluation" / "agnes_eval_v3" / "agnes_judge.py")
        responses = iter([
            (200, {"model": judge.MODEL, "choices": [{"finish_reason": "stop", "message": {"content": '{"correct": null}'}}]}),
            (200, {"model": judge.MODEL, "choices": [{"finish_reason": "stop", "message": {"content": "NOT_JSON"}}]}),
        ])
        caller = aw.AgnesWindowCaller(judge_mod=judge, inner_caller=lambda b, k: next(responses), window=win)
        caller({}, "k")
        self.assertEqual(win.limit, 5)
        self.assertEqual(win.semantic_neutral_calls, 1)
        self.assertEqual(win.ambiguous_events, 0)

        caller({}, "k")
        self.assertEqual(win.limit, 2)  # 5 // 2 = 2
        self.assertEqual(win.ambiguous_events, 1)

    def test_grade_agnes_pro5000_counts_and_1pct_abort(self):
        judge = rpp.load_verified_agnes_judge(REPO_ROOT / "evaluation" / "agnes_eval_v3" / "agnes_judge.py")

        def _make_source_env(tmp_dir: Path):
            sid = "WIN_SID_0"
            key = hashlib.sha256(f"easy:{sid}".encode()).hexdigest()[:24]
            out_root = tmp_dir / "out"
            src_dir = out_root / "EASY" / "sources" / key
            src_dir.mkdir(parents=True)
            rows = []
            for idx in range(32):
                tgt = "source" if idx < 16 else "ours_easy"
                rows.append(
                    {
                        "source_id": sid,
                        "target": tgt,
                        "phase": "post",
                        "index": idx % 16,
                        "problem": "P",
                        "reference_answer": "2",
                        "student_response": f"resp-{idx}",
                    }
                )
            (src_dir / "RAW_COMPLETE.json").write_text('{"samples":32}', encoding="utf-8")
            (src_dir / "SCORE_INPUT.json").write_text(json.dumps(rows), encoding="utf-8")
            key_file = tmp_dir / "agnes.key"
            key_file.write_text("dummy-key\n", encoding="utf-8")
            entries = [
                {
                    "difficulty": "easy",
                    "source_id": sid,
                    "source_key": key,
                    "row": {"transfer_target": "ours_easy"},
                }
            ]
            return out_root, key_file, entries

        # Case A: 31 valid + 1 semantic null -> tech_fail=0, primary_status=可用
        with tempfile.TemporaryDirectory() as td:
            out_root, key_file, entries = _make_source_env(Path(td))
            clock = [0.0]
            win = aw.SyncCongestionWindow(
                initial=1,
                maximum=16,
                monotonic_fn=lambda: clock[0],
                sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s),
                rand_fn=lambda a, b: a,
            )

            def ok_caller(body, k):
                content = body["messages"][1]["content"]
                is_null = "resp-0" in content
                ans = '{"correct": null}' if is_null else '{"correct": true}'
                return 200, {"model": judge.MODEL, "choices": [{"finish_reason": "stop", "message": {"content": ans}}]}

            rep = rpp.grade_agnes_pro5000(
                entries,
                out_root=out_root,
                repo_root=REPO_ROOT,
                key_file=key_file,
                workers=16,
                caller=ok_caller,
                window=win,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(rep["primary_status"], "可用")
            self.assertFalse(rep["aborted_over_1pct"])
            self.assertEqual(rep["total_calls"], 32)
            self.assertEqual(rep["correct"], 31)
            self.assertEqual(rep["semantic_unknown"], 1)
            self.assertEqual(rep["technical_failures"], 0)
            self.assertGreater(rep["window"]["max_limit_reached"], 1)
            self.assertTrue((out_root / "AGNES" / "AGNES_GRADING_REPORT.json").is_file())

        # Case B: 2/32 permanent 429 (> 1%) -> stops with AgnesTechnicalFailureLimitError & 不可用
        with tempfile.TemporaryDirectory() as td:
            out_root, key_file, entries = _make_source_env(Path(td))
            clock = [0.0]
            win = aw.SyncCongestionWindow(
                initial=1,
                maximum=16,
                monotonic_fn=lambda: clock[0],
                sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s),
                rand_fn=lambda a, b: a,
            )

            def fail_caller(body, k):
                sr = json.loads(body["messages"][1]["content"])["student_response"]
                if sr in ("resp-0", "resp-1"):
                    return 429, None
                return 200, {"model": judge.MODEL, "choices": [{"finish_reason": "stop", "message": {"content": '{"correct": true}'}}]}

            with self.assertRaises(rpp.AgnesTechnicalFailureLimitError):
                rpp.grade_agnes_pro5000(
                    entries,
                    out_root=out_root,
                    repo_root=REPO_ROOT,
                    key_file=key_file,
                    workers=16,
                    caller=fail_caller,
                    window=win,
                    sleep_fn=lambda _s: None,
                )
            rep = json.loads((out_root / "AGNES" / "AGNES_GRADING_REPORT.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["primary_status"], "不可用")
            self.assertTrue(rep["aborted_over_1pct"])
            self.assertEqual(rep["technical_failures"], 2)
            self.assertEqual(rep["window"]["http_429_events"], 8)  # 2 samples * 4 MAX_ATTEMPTS each


if __name__ == "__main__":
    unittest.main()
