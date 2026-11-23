#!/usr/bin/env python3
"""Unit tests for author494 idempotent record/replay hardening layer (synthetic data only)."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
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

from eduardo.metrics import correctness  # noqa: E402
import eval.run_env_test_eval as upstream_eval  # noqa: E402
import eval.run_middleturn_eval as upstream_middleturn  # noqa: E402

import agnes_window as aw  # noqa: E402
import eval_hardening as eh  # noqa: E402
import run_eval_27b_author494 as runner27b  # noqa: E402


def _fake_completion(content: str, *, model: str = "fake-model", cid: str = "id-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        model=model,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
    )


def _make_manifest(tmp: Path, *, seed: int = 42, parquet_sha: str = "abc123") -> dict:
    return {
        "schema": eh.MANIFEST_SCHEMA,
        "upstream_commit": "e8c89b7",
        "upstream_files_sha256": {"eval/run_env_test_eval.py": "deadbeef"},
        "wrapper_files_sha256": {"agnes_window.py": "cafebabe"},
        "tutor_model_sha256": {"config.json": "1111"},
        "student_model_sha256": {"config.json": "2222"},
        "test_parquet_sha256": parquet_sha,
        "eval_args": {"seed": seed, "teacher_temperature": 0.6},
    }


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        rendered = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return rendered + ("|<think>" if enable_thinking else "|")

    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text) // 4))) if text else []


class _CrashNow(BaseException):
    pass


class TestEvalHardening(unittest.TestCase):
    def test_16_identical_requests_preserve_16_distinct_responses_on_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            manifest = _make_manifest(Path(tmpdir))
            store1 = eh.ReplayStore(cache_dir, manifest=manifest, resume=False)

            calls = 0

            class _FakeStudent:
                async def create(self, *, model, messages, temperature, max_tokens, **_kw):
                    nonlocal calls
                    idx = calls
                    calls += 1
                    # First 8 return \boxed{7} (correct), next 8 return \boxed{0..7} (distinct strings)
                    ans = "7" if idx < 8 else str(idx)
                    return _fake_completion(f"Attempt {idx}: \\boxed{{{ans}}}", model=model, cid=f"s-{idx}")

            raw = SimpleNamespace(chat=SimpleNamespace(completions=_FakeStudent()))
            client1 = eh.ReplayOpenAIClient(raw, store=store1, api_base="http://127.0.0.1:18971/v1", role="student")

            frac1, fmt1, fail1 = asyncio.run(
                correctness.compute_fraction(
                    messages=[{"role": "user", "content": "SYNTHETIC_Q"}, {"role": "assistant", "content": "SYNTHETIC_T"}],
                    ground_truth="7",
                    problem="SYNTHETIC_Q",
                    client=client1,
                    model="Llama-fake",
                    num_attempts=16,
                    temperature=0.6,
                    max_tokens=800,
                )
            )
            store1.close()
            self.assertEqual(calls, 16)
            self.assertEqual(fail1, 0)
            self.assertEqual(fmt1, 0)
            self.assertAlmostEqual(frac1, 8 / 16)

            # Verify 16 distinct records stored on disk
            lines = (cache_dir / "replay_records.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 16)
            contents = [json.loads(ln)["response"]["content"] for ln in lines]
            self.assertEqual(len(set(contents)), 16)

            # Now --resume with a client that raises if called live
            class _BoomStudent:
                async def create(self, **_kw):
                    raise AssertionError("Live client should not be called on full cache hit")

            store2 = eh.ReplayStore(cache_dir, manifest=manifest, resume=True)
            client2 = eh.ReplayOpenAIClient(
                SimpleNamespace(chat=SimpleNamespace(completions=_BoomStudent())),
                store=store2,
                api_base="http://127.0.0.1:18971/v1",
                role="student",
            )
            frac2, fmt2, fail2 = asyncio.run(
                correctness.compute_fraction(
                    messages=[{"role": "user", "content": "SYNTHETIC_Q"}, {"role": "assistant", "content": "SYNTHETIC_T"}],
                    ground_truth="7",
                    problem="SYNTHETIC_Q",
                    client=client2,
                    model="Llama-fake",
                    num_attempts=16,
                    temperature=0.6,
                    max_tokens=800,
                )
            )
            st2 = store2.stats()
            store2.close()
            self.assertEqual(frac1, frac2)
            self.assertEqual(fmt1, fmt2)
            self.assertEqual(fail1, fail2)
            self.assertEqual(st2["total_hits"], 16)
            self.assertEqual(st2["total_live"], 0)

    def test_corrupt_trailing_record_is_dropped_and_resent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            manifest = _make_manifest(Path(tmpdir))
            store1 = eh.ReplayStore(cache_dir, manifest=manifest, resume=False)

            live_calls = 0

            class _FakeComp:
                async def create(self, *, messages, **_kw):
                    nonlocal live_calls
                    live_calls += 1
                    return _fake_completion(f"reply-{live_calls}-{messages[0]['content']}")

            client1 = eh.ReplayOpenAIClient(
                SimpleNamespace(chat=SimpleNamespace(completions=_FakeComp())),
                store=store1,
                api_base="http://127.0.0.1:18971/v1",
                role="student",
            )
            asyncio.run(client1.chat.completions.create(model="m", messages=[{"role": "user", "content": "p1"}]))
            asyncio.run(client1.chat.completions.create(model="m", messages=[{"role": "user", "content": "p2"}]))
            store1.close()

            # Append truncated/corrupt trailing line without newline
            rec_path = cache_dir / "replay_records.jsonl"
            orig_bytes = rec_path.read_bytes()
            rec_path.write_bytes(orig_bytes + b'{"schema":"author494-replay-v1","request_hash":"broken')

            store2 = eh.ReplayStore(cache_dir, manifest=manifest, resume=True)
            self.assertEqual(store2.dropped_corrupt_tail, 1)
            self.assertEqual(store2.loaded_records, 2)
            self.assertEqual(rec_path.read_bytes(), orig_bytes)
            store2.close()

            # Also test trailing line WITH newline but corrupted integrity checksum
            lines = orig_bytes.splitlines(keepends=True)
            bad_last = json.loads(lines[1].decode("utf-8"))
            bad_last["response"]["content"] = "TAMPERED"
            rec_path.write_bytes(lines[0] + (json.dumps(bad_last) + "\n").encode("utf-8"))

            store3 = eh.ReplayStore(cache_dir, manifest=manifest, resume=True)
            self.assertEqual(store3.dropped_corrupt_tail, 1)
            self.assertEqual(store3.loaded_records, 1)

            live_calls = 0
            client3 = eh.ReplayOpenAIClient(
                SimpleNamespace(chat=SimpleNamespace(completions=_FakeComp())),
                store=store3,
                api_base="http://127.0.0.1:18971/v1",
                role="student",
            )
            r1 = asyncio.run(client3.chat.completions.create(model="m", messages=[{"role": "user", "content": "p1"}]))
            r2 = asyncio.run(client3.chat.completions.create(model="m", messages=[{"role": "user", "content": "p2"}]))
            self.assertEqual(r1.choices[0].message.content, "reply-1-p1")  # replayed
            self.assertEqual(r2.choices[0].message.content, "reply-1-p2")  # resent live!
            self.assertEqual(store3.stats()["total_hits"], 1)
            self.assertEqual(store3.stats()["total_live"], 1)
            store3.close()

    def test_manifest_mismatch_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            m0 = _make_manifest(Path(tmpdir))
            s0 = eh.ReplayStore(cache_dir, manifest=m0, resume=False)
            s0.close()

            for field, bad_val in (
                ("upstream_commit", "other_commit"),
                ("tutor_model_sha256", {"config.json": "9999"}),
                ("student_model_sha256", {"config.json": "8888"}),
                ("test_parquet_sha256", "different_parquet"),
                ("eval_args", {"seed": 43, "teacher_temperature": 0.6}),
            ):
                bad_m = copy.deepcopy(m0)
                bad_m[field] = bad_val
                with self.assertRaises(ValueError):
                    eh.ReplayStore(cache_dir, manifest=bad_m, resume=True)

    def test_technical_failures_are_never_recorded(self):
        now = [0.0]

        async def fake_sleep(dt: float) -> None:
            now[0] += dt

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            manifest = _make_manifest(Path(tmpdir))
            store1 = eh.ReplayStore(cache_dir, manifest=manifest, resume=False)

            state = {"mode": "429"}

            class _FlakyAgnes:
                async def create(self, **_kw):
                    if state["mode"] == "429":
                        raise openai.RateLimitError(
                            "limit",
                            response=httpx.Response(429, request=httpx.Request("POST", "https://api.agnes-ai.cn/v1/chat/completions")),
                            body=None,
                        )
                    if state["mode"] == "ambig":
                        return SimpleNamespace(choices=[])
                    return _fake_completion('{"reasoning": "ok", "decision": "OK"}')

            win = aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0)
            agnes_win_client = aw.AgnesWindowClient(
                SimpleNamespace(chat=SimpleNamespace(completions=_FlakyAgnes())),
                window=win,
                inner_max_retries=1,
            )
            replay_client = eh.ReplayOpenAIClient(
                agnes_win_client, store=store1, api_base="https://api.agnes-ai.cn/v1", role="judge"
            )

            with self.assertRaises(openai.RateLimitError):
                asyncio.run(replay_client.chat.completions.create(model="agnes", messages=[{"role": "user", "content": "q"}]))
            state["mode"] = "ambig"
            with self.assertRaises(openai.APIConnectionError):
                asyncio.run(replay_client.chat.completions.create(model="agnes", messages=[{"role": "user", "content": "q"}]))

            # Neither technical failure was recorded!
            self.assertEqual((cache_dir / "replay_records.jsonl").read_bytes(), b"")
            self.assertEqual(store1.stats()["total_recorded"], 0)
            store1.close()

            # Resume when Agnes recovers: sends live and records 1 valid item
            state["mode"] = "ok"
            store2 = eh.ReplayStore(cache_dir, manifest=manifest, resume=True)
            win2 = aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0)
            rc2 = eh.ReplayOpenAIClient(
                aw.AgnesWindowClient(SimpleNamespace(chat=SimpleNamespace(completions=_FlakyAgnes())), window=win2),
                store=store2,
                api_base="https://api.agnes-ai.cn/v1",
                role="judge",
            )
            resp = asyncio.run(rc2.chat.completions.create(model="agnes", messages=[{"role": "user", "content": "q"}]))
            self.assertIn("OK", resp.choices[0].message.content)
            self.assertEqual(store2.stats()["total_recorded"], 1)
            store2.close()

    def test_end_to_end_kill_mid_rollout_and_mid_scoring_resumes_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tutor_dir = root / "tutor_model"
            student_dir = root / "student_model"
            tutor_dir.mkdir()
            student_dir.mkdir()
            (tutor_dir / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
            (student_dir / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")

            parquet_path = root / "test.parquet"
            rows = []
            for i in range(2):
                rows.append({
                    "prompt": [
                        {"role": "system", "content": f"Teach SYNTHETIC_PROB_{i}"},
                        {"role": "user", "content": f"Please help with SYNTHETIC_PROB_{i}"},
                    ],
                    "extra_info": {
                        "problem": f"SYNTHETIC_PROB_{i}",
                        "answer": str(i + 10),
                        "transfer_problem": f"SYNTHETIC_TRANS_{i}",
                        "transfer_answer": str(i + 20),
                        "pre_solve_rate": 0.25,
                        "pre_solve_rate_transfer_problem": 0.125,
                    },
                })
            pd.DataFrame(rows).to_parquet(parquet_path)

            def run_eval_with_hooks(out_dir: Path, *, resume: bool, crash_config: dict | None = None):
                crash_cfg = crash_config or {}
                teacher_calls = 0
                student_calls = 0
                agnes_calls = 0
                teacher_inits = 0

                class _FakeTeacher:
                    def __init__(self, _args):
                        nonlocal teacher_inits
                        teacher_inits += 1
                        self.tokenizer = _FakeTokenizer()
                        self._verified = False

                    async def generate(self, convs):
                        nonlocal teacher_calls
                        teacher_calls += 1
                        out = []
                        for c in convs:
                            turn = sum(1 for m in c.messages if m["role"] == "teacher")
                            if turn >= 3:
                                out.append(f"<think>t{turn}</think>Summary for {c.problem} <end_of_conversation>")
                            else:
                                out.append(f"<think>t{turn}</think>Hint {turn} for {c.problem}")
                        return out

                    def count_tokens(self, conv):
                        return sum(len(self.tokenizer.encode(m["content"])) for m in conv.messages)

                    def thinking_tokens(self, _idx, text):
                        tt = upstream_middleturn.thinking_text(text)
                        return (len(self.tokenizer.encode(tt)) if tt else 0), "tokenizer"

                    def close(self):
                        pass

                class _FakeStudentCompletions:
                    async def create(self, *, messages, **_kw):
                        nonlocal student_calls
                        student_calls += 1
                        if crash_cfg.get("crash_on_student_call") == student_calls:
                            raise _CrashNow("simulated crash in student")
                        last_msg = messages[-1]["content"]
                        # Deterministic answer keyed off message content + student_calls?
                        # Note: must depend ONLY on request content (and sequence) so fresh vs resumed match!
                        if "SYNTHETIC_TRANS_0" in last_msg:
                            return _fake_completion(r"\boxed{20}", model="stu")
                        if "SYNTHETIC_PROB_0" in last_msg:
                            return _fake_completion(r"\boxed{10}", model="stu")
                        if "SYNTHETIC_TRANS_1" in last_msg:
                            return _fake_completion(r"\boxed{999}", model="stu")
                        if "SYNTHETIC_PROB_1" in last_msg:
                            return _fake_completion(r"\boxed{11}", model="stu")
                        return _fake_completion(f"Student step on {last_msg[:20]}", model="stu")

                class _FakeAgnesCompletions:
                    async def create(self, *, messages, max_tokens, **_kw):
                        nonlocal agnes_calls
                        agnes_calls += 1
                        if crash_cfg.get("crash_on_agnes_call") == agnes_calls:
                            raise _CrashNow("simulated crash in agnes")
                        prompt = messages[0]["content"]
                        if "1-5" in prompt or max_tokens == 16384 and "score" in prompt.lower() and "decision" not in prompt.lower():
                            return _fake_completion('{"reasoning": "SYNTHETIC_EVAL_REASON", "score": 4}', model="agnes")
                        return _fake_completion('{"reasoning": "SYNTHETIC_GATE_REASON", "decision": "OK"}', model="agnes")

                orig_make_teacher = upstream_eval.make_teacher
                orig_make_judge = upstream_eval.make_judge_client
                orig_get_client = upstream_eval.get_async_openai_client
                orig_mid_get_client = upstream_middleturn.get_async_openai_client
                orig_datetime = upstream_eval.datetime
                orig_auto_tok = None
                import transformers
                orig_auto_tok = transformers.AutoTokenizer.from_pretrained

                class _FixedDatetime:
                    @staticmethod
                    def now():
                        from datetime import datetime as real_dt
                        return real_dt(2026, 11, 23, 12, 0, 0)

                try:
                    transformers.AutoTokenizer.from_pretrained = lambda *_a, **_kw: _FakeTokenizer()
                    upstream_eval.datetime = _FixedDatetime
                    upstream_eval.make_teacher = lambda a: _FakeTeacher(a)
                    upstream_eval.make_judge_client = lambda a: (
                        SimpleNamespace(chat=SimpleNamespace(completions=_FakeAgnesCompletions())),
                        a.gemini_model,
                        {},
                    )
                    upstream_eval.get_async_openai_client = lambda base, key, timeout=300.0: (
                        SimpleNamespace(chat=SimpleNamespace(completions=_FakeAgnesCompletions()))
                        if key == "AGNES_API_KEY"
                        else SimpleNamespace(chat=SimpleNamespace(completions=_FakeStudentCompletions()))
                    )

                    argv = [
                        "--student-model-dir", str(student_dir),
                        "--agnes-stats-out", str(out_dir / "agnes_window_stats.json"),
                        "--replay-stats-out", str(out_dir / "replay_stats.json"),
                        *(["--resume"] if resume else []),
                        "--model", str(tutor_dir),
                        "--test-parquet", str(parquet_path),
                        "--num-problems", "2",
                        "--num-student-attempts", "4",
                        "--enable-thinking",
                        "--student-model", "stu",
                        "--student-api-base", "http://127.0.0.1:18971/v1",
                        "--student-api-key-env", "SERVING_API_KEY",
                        "--gemini",
                        "--gemini-api-base", "https://api.agnes-ai.cn/v1",
                        "--gemini-model", "agnes-3.0-flash",
                        "--gemini-api-key-env", "AGNES_API_KEY",
                        "--eval-judge-model", "agnes-3.0-flash",
                        "--judge-concurrency", "12",
                        "--eval-judge-concurrency", "4",
                        "--output-dir", str(out_dir),
                        "--tag", "synth_test",
                    ]
                    os.environ["AGNES_API_KEY"] = "fake-agnes-key"
                    os.environ["SERVING_API_KEY"] = "fake-stu-key"
                    runner27b.main(argv)
                finally:
                    transformers.AutoTokenizer.from_pretrained = orig_auto_tok
                    upstream_eval.datetime = orig_datetime
                    upstream_eval.make_teacher = orig_make_teacher
                    upstream_eval.make_judge_client = orig_make_judge
                    upstream_eval.get_async_openai_client = orig_get_client
                    upstream_middleturn.get_async_openai_client = orig_mid_get_client

                return {
                    "teacher_inits": teacher_inits,
                    "teacher_calls": teacher_calls,
                    "student_calls": student_calls,
                    "agnes_calls": agnes_calls,
                    "replay_stats": json.loads((out_dir / "replay_stats.json").read_text(encoding="utf-8")),
                }

            # 1) Uninterrupted baseline (using fixed eval_dir path so args.output_dir matches byte-for-byte)
            import shutil
            eval_dir = root / "eval_run_dir"
            run_eval_with_hooks(eval_dir, resume=False)
            base_json = eval_dir / "transfer_eval_synth_test_20261123_120000.json"
            self.assertTrue(base_json.is_file())
            base_bytes = base_json.read_bytes()
            manifest_txt = (eval_dir / "replay_cache" / "replay_manifest.json").read_text(encoding="utf-8")
            stats_txt = (eval_dir / "replay_stats.json").read_text(encoding="utf-8")
            shutil.rmtree(eval_dir)

            # 2) Crash during Scoring (on 5th Agnes call), then --resume
            with self.assertRaises(_CrashNow):
                run_eval_with_hooks(eval_dir, resume=False, crash_config={"crash_on_agnes_call": 5})
            self.assertFalse((eval_dir / "transfer_eval_synth_test_20261123_120000.json").exists())

            res_score_info = run_eval_with_hooks(eval_dir, resume=True)
            res_score_bytes = (eval_dir / "transfer_eval_synth_test_20261123_120000.json").read_bytes()
            self.assertEqual(base_bytes, res_score_bytes)
            # Because Rollout already completed before crash, Tutor GPU engine did NOT even need to initialize!
            self.assertEqual(res_score_info["teacher_inits"], 0)
            self.assertGreater(res_score_info["replay_stats"]["hits_by_role"]["tutor"], 0)
            self.assertGreater(res_score_info["replay_stats"]["hits_by_role"]["student"], 0)
            self.assertGreater(res_score_info["replay_stats"]["hits_by_role"]["judge"], 0)
            shutil.rmtree(eval_dir)

            # 3) Crash during Rollout (on 3rd Student dialogue call), then --resume
            with self.assertRaises(_CrashNow):
                run_eval_with_hooks(eval_dir, resume=False, crash_config={"crash_on_student_call": 3})

            res_rollout_info = run_eval_with_hooks(eval_dir, resume=True)
            res_rollout_bytes = (eval_dir / "transfer_eval_synth_test_20261123_120000.json").read_bytes()
            self.assertEqual(base_bytes, res_rollout_bytes)
            self.assertEqual(res_rollout_info["teacher_inits"], 1)  # lazily loaded for remaining rounds!
            self.assertGreater(res_rollout_info["replay_stats"]["hits_by_role"]["tutor"], 0)
            self.assertGreater(res_rollout_info["replay_stats"]["hits_by_role"]["student"], 0)

            # Ensure manifest and replay stats never leak raw problem/dialogue text
            for secret in ("SYNTHETIC_PROB_0", "SYNTHETIC_TRANS_1", "SYNTHETIC_EVAL_REASON", "SYNTHETIC_GATE_REASON"):
                self.assertNotIn(secret, manifest_txt)
                self.assertNotIn(secret, stats_txt)


if __name__ == "__main__":
    unittest.main()
