#!/usr/bin/env python3
"""Unit tests for Pro 5000 Eduardo-27B adaptation (synthetic data only)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _find_author_repo() -> Path:
    candidates = []
    if os.environ.get("EDUARDO_REPO"):
        candidates.append(Path(os.environ["EDUARDO_REPO"]))
    candidates.extend([
        HERE.parents[2] / "EduardoRL_A组源码_20261117",
        Path("/workspace/A_group_repro_20261117/source/EduardoRL"),
    ])
    for cand in candidates:
        if (cand / "eval" / "run_env_test_eval.py").is_file():
            return cand
    raise RuntimeError("Could not locate upstream EduardoRL repo")


AUTHOR_REPO = _find_author_repo()
if str(AUTHOR_REPO) not in sys.path:
    sys.path.insert(0, str(AUTHOR_REPO))

import eval.run_env_test_eval as upstream_eval  # noqa: E402
import eval.run_middleturn_eval as upstream_middleturn  # noqa: E402
from eval.run_env_test_eval import compute_summary as author_compute_summary  # noqa: E402

import download_27b_pro5000 as dl27b  # noqa: E402
import run_eval_27b_author494 as runner27b  # noqa: E402
import summarize_author494 as s  # noqa: E402
import wait_gpus_27b_pro5000 as wg  # noqa: E402


def _make_gpu_row(idx: int, *, free_mib: int = 72000, total_mib: int = 73415, util: int = 0, apps: list | None = None) -> dict:
    return {
        "index": idx,
        "uuid": f"GPU-{idx:04d}",
        "free_mib": free_mib,
        "total_mib": total_mib,
        "used_mib": total_mib - free_mib,
        "util_pct": util,
        "apps": apps or [],
    }


def _make_record(row_index: int, *, blanked_rate: float = 0.01, retry_fail: bool = False, student_failed: int = 0, transfer_used: int = 1) -> dict:
    no_leak = "[Vote 1: OK] ok | [Vote 2: REJECT] Judge call failed after retries | [Vote 3: OK] ok" if retry_fail else "[Vote 1: OK] ok | [Vote 2: OK] ok | [Vote 3: OK] ok"
    return {
        "row_index": row_index,
        "sample_index": 0,
        "problem": f"SYNTHETIC_P_{row_index}",
        "answer": "42",
        "messages": [{"role": "teacher", "content": "SYNTHETIC_T"}],
        "pre_solve_rate": 0.25,
        "pre_solve_rate_transfer": 0.25,
        "post_solve_transfer": 0.50,
        "post_solve_same": 0.55,
        "delta_transfer": 1 / 3,
        "delta_same": 0.4,
        "delta_transfer_raw": 0.25,
        "delta_same_raw": 0.30,
        "delta_transfer_gated": 1 / 3,
        "delta_same_gated": 0.4,
        "transfer_gap": 0.4 - (1 / 3),
        "leaked": 0,
        "teaching_rejected": 0,
        "judge_reasoning": {"no_leak": no_leak, "teaching": "[Vote 1: OK] ok | [Vote 2: OK] ok | [Vote 3: OK] ok"},
        "eval_judge_pedagogy": 4.0,
        "eval_judge_correctness": 4.5,
        "eval_judge_ok": 1,
        "blanked_teacher_turn_rate": blanked_rate,
        "end_reason": "end_of_conversation",
        "eoc_used": True,
        "num_teacher_turns": 4,
        "thinking_tokens": [50, 60],
        "used_transfer_problem": transfer_used,
        "n_student_calls_failed": student_failed,
        "assume_thinking": True,
    }


class TestPro5000WaitGpus(unittest.TestCase):
    def test_gpu6_quirk_and_preference_order(self):
        snap1 = [
            _make_gpu_row(0, free_mib=72000, util=0),
            _make_gpu_row(1, free_mib=72000, util=0),
            _make_gpu_row(2, free_mib=20000, util=90, apps=[("123", "50000")]),
            _make_gpu_row(3, free_mib=72000, util=0),
            _make_gpu_row(4, free_mib=72000, util=0),
            _make_gpu_row(6, free_mib=73000, total_mib=73415, util=100),  # quirk idle (<500 MiB used, 0 apps)
            _make_gpu_row(7, free_mib=72000, util=0),
        ]
        snap2 = list(snap1)
        # 5 preferred GPUs (0,1,3,4,7) come before quirk GPU 6
        chosen5 = wg.select_confirmed_gpus(snap1, snap2, num_gpus=5, fallback_min_gpus=4)
        self.assertEqual(chosen5, [0, 1, 3, 4, 7])

        # With GPU 7 busy, GPU 6 quirk supplies the 5th GPU
        snap_no7 = [g for g in snap1 if g["index"] != 7]
        chosen5_with_6 = wg.select_confirmed_gpus(snap_no7, snap_no7, num_gpus=5, fallback_min_gpus=4)
        self.assertEqual(chosen5_with_6, [0, 1, 3, 4, 6])

        # If GPU 6 has >= 500 MiB used, it is NOT eligible -> falls back to 4 GPUs (0,1,3,4)
        snap_bad6 = [
            _make_gpu_row(6, free_mib=72800, total_mib=73415, util=100) if g["index"] == 6 else g
            for g in snap_no7
        ]
        chosen4 = wg.select_confirmed_gpus(snap_bad6, snap_bad6, num_gpus=5, fallback_min_gpus=4)
        self.assertEqual(chosen4, [0, 1, 3, 4])

        # Strict 5-GPU mode (fallback_min_gpus=0) returns None when only 4 are idle
        self.assertIsNone(wg.select_confirmed_gpus(snap_bad6, snap_bad6, num_gpus=5, fallback_min_gpus=0))

    def test_wait_for_gpus_flexible_and_find_replacement_count2(self):
        snap_4 = [_make_gpu_row(i) for i in (0, 2, 5, 7)]
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            chosen = wg.wait_for_gpus(
                status_path=td / "wait_status.json",
                log_path=td / "wait.log",
                port=18971,
                num_gpus=5,
                fallback_min_gpus=4,
                snapshot_fn=lambda: snap_4,
                port_fn=lambda _p: True,
                sleep_fn=lambda _dt: None,
            )
            self.assertEqual(chosen, [0, 2, 5, 7])
            st = json.loads((td / "wait_status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["state"], "ACQUIRED")
            self.assertEqual(st["num_gpus"], 4)

        snap_repl = [_make_gpu_row(i, free_mib=(73000 if i == 6 else 72000)) for i in (0, 1, 3, 4, 6)]
        reps = wg.find_replacement_gpus(snap_repl, snap_repl, exclude={0, 1, 3}, count=2)
        self.assertEqual(reps, [4, 6])
        self.assertIsNone(wg.find_replacement_gpus(snap_repl, snap_repl, exclude={0, 1, 3, 4}, count=2))


class TestPro5000DownloadAndVerify(unittest.TestCase):
    def test_manifest_completeness_and_local_verification(self):
        for fname in dl27b.LOCKED_SIX_FILES:
            self.assertIn(fname, dl27b.MODEL_FILES)
        self.assertIn("model.safetensors.index.json", dl27b.MODEL_FILES)
        for fname, (size, sha) in dl27b.MODEL_FILES.items():
            self.assertGreater(size, 0, fname)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", fname)

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            orig_files = dl27b.MODEL_FILES
            try:
                fake_bytes = b'{"torch_dtype": "bfloat16"}'
                fake_sha = hashlib.sha256(fake_bytes).hexdigest()
                dl27b.MODEL_FILES = {"config.json": (len(fake_bytes), fake_sha)}
                (td / "config.json").write_bytes(fake_bytes)

                res = dl27b.download_and_verify_model(td, verify_only=False)
                self.assertEqual(res, {"config.json": "SKIP_VERIFIED"})

                (td / "config.json").write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    dl27b.verify_model_dir(td)
            finally:
                dl27b.MODEL_FILES = orig_files

    def test_clean_env_strips_ld_library_path_and_rc4_falls_back_to_urllib(self):
        orig_env = dict(os.environ)
        orig_run = dl27b.subprocess.run
        orig_find = dl27b._find_system_curl
        orig_url = dl27b._download_with_urllib
        try:
            os.environ["LD_LIBRARY_PATH"] = "/tmp/conda/lib"
            os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9999"
            os.environ["HF_TOKEN"] = "secret"
            cleaned = dl27b._clean_env()
            self.assertNotIn("LD_LIBRARY_PATH", cleaned)
            self.assertNotIn("HTTPS_PROXY", cleaned)
            self.assertNotIn("HF_TOKEN", cleaned)

            with tempfile.TemporaryDirectory() as tmpdir:
                td = Path(tmpdir)
                payload = b'{"torch_dtype": "bfloat16"}'
                sha = hashlib.sha256(payload).hexdigest()
                called = {"urllib": 0}
                dl27b._find_system_curl = lambda: "/usr/bin/curl"
                dl27b.subprocess.run = lambda *a, **kw: SimpleNamespace(returncode=4, stderr="curl: (4) unsupported")

                def fake_urllib(_url, part_path: Path):
                    called["urllib"] += 1
                    part_path.write_bytes(payload)

                dl27b._download_with_urllib = fake_urllib
                st = dl27b.download_one_file("https://hf-mirror.com/x", td / "config.json", len(payload), sha)
                self.assertEqual(st, "DOWNLOADED_VERIFIED")
                self.assertEqual(called["urllib"], 1)
        finally:
            os.environ.clear()
            os.environ.update(orig_env)
            dl27b.subprocess.run = orig_run
            dl27b._find_system_curl = orig_find
            dl27b._download_with_urllib = orig_url


class TestPro5000SmokeGateAndTP2Resume(unittest.TestCase):
    def test_smoke_gate_criteria_and_ours_label(self):
        good_recs = [_make_record(i, blanked_rate=0.01) for i in range(20)]
        doc = {
            "model": "/dev/shm/zhujie_a_group/Eduardo-27b-3bbf17e",
            "tag": "smoke20",
            "judge_model": "agnes-3.0-flash",
            "args": {"judge_votes": 3},
            "metrics": author_compute_summary(good_recs),
            "conversations": good_recs,
        }
        summary = s.build_summary(doc, seed=42, draws=200, paper_baseline="eduardo-27b")
        gate_ok = s.evaluate_smoke_gate(summary)
        self.assertTrue(gate_ok["passed"])
        report = s.format_report(summary, ours_label="Ours (Pro 5000 + Agnes)")
        self.assertIn("Ours (Pro 5000 + Agnes)", report)

        # Fail 1: student call failed > 0
        r_stu = [_make_record(i, student_failed=(1 if i == 0 else 0)) for i in range(20)]
        doc_stu = {**doc, "metrics": author_compute_summary(r_stu), "conversations": r_stu}
        self.assertFalse(s.evaluate_smoke_gate(s.build_summary(doc_stu, seed=42, draws=100))["passed"])

        # Fail 2: transfer not 100% tested
        r_tr = [_make_record(i, transfer_used=(0 if i == 0 else 1)) for i in range(20)]
        doc_tr = {**doc, "metrics": author_compute_summary(r_tr), "conversations": r_tr}
        self.assertFalse(s.evaluate_smoke_gate(s.build_summary(doc_tr, seed=42, draws=100))["passed"])

        # Fail 3: Agnes retry failure > 1% (1 failed vote out of 120 = 0.83% OK; 2 failed votes = 1.67% FAIL)
        r_ag = [_make_record(i, retry_fail=(i < 2)) for i in range(20)]
        doc_ag = {**doc, "metrics": author_compute_summary(r_ag), "conversations": r_ag}
        self.assertFalse(s.evaluate_smoke_gate(s.build_summary(doc_ag, seed=42, draws=100))["passed"])

        # Fail 4: Empty rate > 5%
        r_emp = [_make_record(i, blanked_rate=0.06) for i in range(20)]
        doc_emp = {**doc, "metrics": author_compute_summary(r_emp), "conversations": r_emp}
        self.assertFalse(s.evaluate_smoke_gate(s.build_summary(doc_emp, seed=42, draws=100))["passed"])

    def test_tp2_manifest_binding_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tutor_dir = root / "tutor"
            student_dir = root / "student"
            tutor_dir.mkdir()
            student_dir.mkdir()
            (tutor_dir / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
            (student_dir / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")

            pq_path = root / "test.parquet"
            pd.DataFrame([{
                "prompt": [{"role": "system", "content": "Teach SYNTH_0"}, {"role": "user", "content": "Help SYNTH_0"}],
                "extra_info": {
                    "problem": "SYNTH_0",
                    "answer": "5",
                    "transfer_problem": "SYNTH_T0",
                    "transfer_answer": "6",
                    "pre_solve_rate": 0.25,
                    "pre_solve_rate_transfer_problem": 0.25,
                },
            }]).to_parquet(pq_path)

            class _FakeTok:
                def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                    return "|".join(m["content"] for m in messages) + "|<think>"

                def encode(self, text: str) -> list[int]:
                    return list(range(max(1, len(text) // 4))) if text else []

            class _FakeTeacher:
                def __init__(self, args):
                    self.tp = args.tensor_parallel_size
                    self.tokenizer = _FakeTok()

                async def generate(self, convs):
                    return [f"<think>tp={self.tp}</think>done <end_of_conversation>" for _ in convs]

                def count_tokens(self, conv):
                    return 10

                def thinking_tokens(self, _i, _t):
                    return 2, "tokenizer"

                def close(self):
                    pass

            seen_models: list[str] = []

            class _FakeComp:
                async def create(self, **kw):
                    seen_models.append(str(kw.get("model", "")))
                    return SimpleNamespace(
                        id="1", model="m",
                        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"reasoning":"ok","decision":"OK","score":4}\\boxed{5}\\boxed{6}'))],
                    )

            import transformers
            orig_tok = transformers.AutoTokenizer.from_pretrained
            orig_mt = upstream_eval.make_teacher
            orig_mj = upstream_eval.make_judge_client
            orig_gc = upstream_eval.get_async_openai_client
            orig_mgc = upstream_middleturn.get_async_openai_client
            orig_dt = upstream_eval.datetime

            class _FixedDt:
                @staticmethod
                def now():
                    from datetime import datetime as dt
                    return dt(2026, 11, 24, 9, 0, 0)

            out_dir = root / "eval_tp2"
            try:
                transformers.AutoTokenizer.from_pretrained = lambda *_a, **_kw: _FakeTok()
                upstream_eval.datetime = _FixedDt
                upstream_eval.make_teacher = lambda a: _FakeTeacher(a)
                upstream_eval.make_judge_client = lambda a: (SimpleNamespace(chat=SimpleNamespace(completions=_FakeComp())), a.gemini_model, {})
                upstream_eval.get_async_openai_client = lambda *_a, **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_FakeComp()))

                os.environ["AGNES_API_KEY"] = "fake"
                os.environ["SERVING_API_KEY"] = "fake"
                base_argv = [
                    "--student-model-dir", str(student_dir),
                    "--model", str(tutor_dir),
                    "--tensor-parallel-size", "2",
                    "--test-parquet", str(pq_path),
                    "--num-problems", "1",
                    "--num-student-attempts", "2",
                    "--enable-thinking",
                    "--student-model", "stu",
                    "--student-api-base", "http://127.0.0.1:18971/v1",
                    "--gemini",
                    "--gemini-api-base", "https://api.agnes-ai.cn/v1",
                    "--gemini-model", "agnes-3.0-flash",
                    "--gemini-api-key-env", "AGNES_API_KEY",
                    "--eval-judge-model", "agnes-3.0-flash",
                    "--judge-concurrency", "12",
                    "--eval-judge-concurrency", "4",
                    "--output-dir", str(out_dir),
                    "--tag", "tp2_test",
                ]
                runner27b.main(base_argv)
                manifest = json.loads((out_dir / "replay_cache" / "replay_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["eval_args"]["tensor_parallel_size"], 2)
                self.assertEqual(manifest["eval_args"]["student_model"], "stu")
                self.assertIsInstance(manifest["student_model_sha256"], dict)
                self.assertIn("config.json", manifest["student_model_sha256"])
                self.assertIn("stu", seen_models)

                # Changing --tensor-parallel-size to 1 on --resume must fail manifest verification
                bad_argv = ["--resume" if x == "--enable-thinking" else x for x in base_argv]
                tp1_argv = ["--resume"] + [x if x != "2" else "1" for x in base_argv]
                # Replace only the value right after --tensor-parallel-size
                tp_idx = tp1_argv.index("--tensor-parallel-size") + 1
                tp1_argv[tp_idx] = "1"
                with self.assertRaises(ValueError):
                    runner27b.main(tp1_argv)
            finally:
                transformers.AutoTokenizer.from_pretrained = orig_tok
                upstream_eval.datetime = orig_dt
                upstream_eval.make_teacher = orig_mt
                upstream_eval.make_judge_client = orig_mj
                upstream_eval.get_async_openai_client = orig_gc
                upstream_middleturn.get_async_openai_client = orig_mgc


    def test_strip_wrapper_args_preserves_student_model(self):
        wargs, pass_argv = runner27b._strip_wrapper_args([
            "--student-model-dir", "/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct",
            "--student-model", "Llama-3.1-8B-Instruct-zhujie-author494-27b",
            "--resume",
            "--tensor-parallel-size", "2",
        ])
        self.assertEqual(wargs.student_model_dir, "/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct")
        self.assertTrue(wargs.resume)
        self.assertEqual(
            pass_argv,
            ["--student-model", "Llama-3.1-8B-Instruct-zhujie-author494-27b", "--tensor-parallel-size", "2"],
        )


if __name__ == "__main__":
    unittest.main()
