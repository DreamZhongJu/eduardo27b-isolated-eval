#!/usr/bin/env python3
"""Unit tests for Agnes congestion window, 27B wrapper, and sanitization (synthetic data only)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai

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

from eduardo.metrics import judge_eval, judge_quality  # noqa: E402
import eval.run_env_test_eval as upstream_eval  # noqa: E402

import agnes_window as aw  # noqa: E402
import probe_27b_load as probe  # noqa: E402
import run_eval_27b_author494 as runner27b  # noqa: E402
import summarize_author494 as s  # noqa: E402


def _make_http_response(status_code: int) -> httpx.Response:
    req = httpx.Request("POST", "https://api.agnes-ai.cn/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=req)


def _fake_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="agnes-3.0-flash",
        id="resp_synth",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


class TestAgnesWindow27B(unittest.TestCase):
    def test_window_halves_on_429_grows_on_success_and_caps_at_3(self):
        now = [100.0]
        sleeps: list[float] = []

        async def fake_sleep(dt: float) -> None:
            sleeps.append(dt)
            now[0] += dt

        win = aw.AsyncCongestionWindow(
            monotonic_fn=lambda: now[0],
            sleep_fn=fake_sleep,
            rand_fn=lambda _lo, _hi: 1.0,
        )
        self.assertEqual(win.limit, 1)

        async def scenario():
            # Consecutive successes grow 1 -> 2 -> ... -> 16 and stay capped at 16
            for _ in range(25):
                lease = await win.acquire()
                await win.report_success(lease)
                await win.release()
            self.assertEqual(win.limit, 16)
            self.assertEqual(win.max_limit_reached, 16)

            # 429 congestion halves 16 -> 8 and blocks for backoff_base=2.0s
            lease = await win.acquire()
            delay = await win.report_congestion()
            await win.release()
            self.assertEqual(win.limit, 8)
            self.assertEqual(win.window_drops, 1)
            self.assertAlmostEqual(delay, 2.0)

            # Next acquire waits out cooldown, then succeeds 8 -> 9 -> 10
            lease = await win.acquire()
            await win.report_success(lease)
            await win.release()
            self.assertEqual(win.limit, 9)

            lease = await win.acquire()
            await win.report_success(lease)
            await win.release()
            self.assertEqual(win.limit, 10)

            # Consecutive congestions halve 10 -> 5 -> 2 -> 1 -> 1; backoff grows 2 -> 4 -> 8 -> 16
            expected_limits = [5, 2, 1, 1]
            expected_delays = [2.0, 4.0, 8.0, 16.0]
            for exp_lim, exp_del in zip(expected_limits, expected_delays):
                lease = await win.acquire()
                d = await win.report_congestion()
                await win.release()
                self.assertEqual(win.limit, exp_lim)
                self.assertAlmostEqual(d, exp_del)

            # After consecutive 429s drove window down to 1, successes grow it back to 16
            for _ in range(20):
                lease = await win.acquire()
                await win.report_success(lease)
                await win.release()
            self.assertEqual(win.limit, 16)

        asyncio.run(scenario())
        self.assertTrue(all(dt <= aw.BACKOFF_CAP_SECONDS for dt in sleeps))

    def test_consecutive_429_collapses_to_1_and_recovers_to_16(self):
        now = [0.0]

        async def fake_sleep(dt: float) -> None:
            now[0] += dt

        win = aw.AsyncCongestionWindow(
            monotonic_fn=lambda: now[0],
            sleep_fn=fake_sleep,
            rand_fn=lambda _lo, _hi: 1.0,
        )
        # Grow up to 16 first
        async def grow_to_16():
            for _ in range(15):
                lease = await win.acquire()
                await win.report_success(lease)
                await win.release()

        asyncio.run(grow_to_16())
        self.assertEqual(win.limit, 16)

        # Feed 429 errors via client calls until window hits 1, then recover via 200 OKs
        phase = {"fail_429": 4}

        class _Burst429Completions:
            async def create(self, **_kwargs):
                if phase["fail_429"] > 0:
                    phase["fail_429"] -= 1
                    raise openai.RateLimitError(
                        "rate limited",
                        response=_make_http_response(429),
                        body=None,
                    )
                return _fake_completion('{"reasoning": "ok", "decision": "OK"}')

        client = aw.AgnesWindowClient(
            SimpleNamespace(chat=SimpleNamespace(completions=_Burst429Completions())),
            window=win,
            inner_max_retries=5,
        )
        # First call fails 4 times with 429 (16 -> 8 -> 4 -> 2 -> 1), then succeeds (1 -> 2)
        resp = asyncio.run(client.chat.completions.create(model="m", messages=[]))
        self.assertIn("OK", resp.choices[0].message.content)
        self.assertEqual(win.http_429_events, 4)
        self.assertEqual(win.window_drops, 4)
        self.assertEqual(win.limit, 2)

        # 14 more successes bring it back up to 16 and cap there
        async def recover():
            for _ in range(16):
                await client.chat.completions.create(model="m", messages=[])

        asyncio.run(recover())
        self.assertEqual(win.limit, 16)
        self.assertEqual(win.max_limit_reached, 16)

    def test_concurrency_respects_window_bounds(self):
        now = [0.0]

        async def fake_sleep(dt: float) -> None:
            now[0] += dt
            await asyncio.sleep(0)

        win = aw.AsyncCongestionWindow(
            monotonic_fn=lambda: now[0],
            sleep_fn=fake_sleep,
            rand_fn=lambda _lo, _hi: 1.0,
        )
        active = 0
        observed_max = 0

        class _FakeCompletions:
            async def create(self, **_kwargs):
                nonlocal active, observed_max
                active += 1
                observed_max = max(observed_max, active)
                await asyncio.sleep(0.01)
                active -= 1
                return _fake_completion('{"reasoning": "ok", "decision": "OK"}')

        client = aw.AgnesWindowClient(
            SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions())),
            window=win,
        )

        async def run_batch():
            await asyncio.gather(*(client.chat.completions.create(model="m", messages=[]) for _ in range(40)))

        asyncio.run(run_batch())
        self.assertLessEqual(observed_max, 16)
        self.assertEqual(win.max_limit_reached, 16)
        self.assertEqual(win.limit, 16)

    def test_technical_vs_semantic_and_ambiguous_failures(self):
        now = [0.0]

        async def fake_sleep(dt: float) -> None:
            now[0] += dt

        # 1) HTTP 200 non-JSON/semantic text is NEVER retried in wrapper
        calls_200 = 0

        class _BadJson200:
            async def create(self, **_kwargs):
                nonlocal calls_200
                calls_200 += 1
                return _fake_completion("Not valid JSON SYNTHETIC_RAW_TEXT")

        c200 = aw.AgnesWindowClient(
            SimpleNamespace(chat=SimpleNamespace(completions=_BadJson200())),
            window=aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0),
        )
        resp = asyncio.run(c200.chat.completions.create(model="m", messages=[]))
        self.assertEqual(calls_200, 1)
        self.assertEqual(c200.stats()["http_attempts"], 1)
        self.assertEqual(c200.stats()["transient_retries"], 0)
        self.assertEqual(resp.choices[0].message.content, "Not valid JSON SYNTHETIC_RAW_TEXT")

        # 2) 429 recovers within inner retries (3 -> 1, then succeeds)
        attempts_429 = 0

        class _Flaky429:
            async def create(self, **_kwargs):
                nonlocal attempts_429
                attempts_429 += 1
                if attempts_429 == 1:
                    raise openai.RateLimitError(
                        "rate limited",
                        response=_make_http_response(429),
                        body=None,
                    )
                return _fake_completion('{"reasoning": "ok", "decision": "OK"}')

        c429 = aw.AgnesWindowClient(
            SimpleNamespace(chat=SimpleNamespace(completions=_Flaky429())),
            window=aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0),
        )
        resp_429 = asyncio.run(c429.chat.completions.create(model="m", messages=[]))
        self.assertEqual(attempts_429, 2)
        self.assertEqual(c429.stats()["transient_retries"], 1)
        self.assertEqual(c429.stats()["congestion_events"], 1)
        self.assertIn("OK", resp_429.choices[0].message.content)

        # 3) Ambiguous outcome (empty choices envelope) counts as technical failure,
        # retries inner_max_retries=2 (3 attempts), and under author's outer retry_kwargs=3
        # makes 4 * 3 = 12 HTTP requests and becomes "Judge call failed after retries"
        ambig_attempts = 0

        class _AmbiguousEnvelope:
            async def create(self, **_kwargs):
                nonlocal ambig_attempts
                ambig_attempts += 1
                return SimpleNamespace(choices=[])

        c_ambig = aw.AgnesWindowClient(
            SimpleNamespace(chat=SimpleNamespace(completions=_AmbiguousEnvelope())),
            window=aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0),
        )
        no_leak, _teaching, reasoning = asyncio.run(
            judge_quality.majority_vote(
                messages=[{"role": "user", "content": "SYNTHETIC_Q"}, {"role": "assistant", "content": "SYNTHETIC_A"}],
                problem="SYNTHETIC_P",
                answer="42",
                client=c_ambig,
                model="agnes-3.0-flash",
                num_votes=1,
                retry_kwargs={"num_retries": 3, "backoff_base": 0.0, "backoff_cap": 0.0, "jitter": False},
            )
        )
        self.assertEqual(no_leak, "REJECT")
        self.assertIn("Judge call failed after retries", reasoning["no_leak"])
        # 2 criteria (no_leak + teaching) * 1 vote * (1+3 outer) * (1+2 inner) = 24
        self.assertEqual(ambig_attempts, 24)
        st = c_ambig.stats(outer_num_retries=3)
        self.assertEqual(st["max_http_requests_per_vote"], 12)
        self.assertEqual(st["ambiguous_failures"], 24)

    def test_wrapper_does_not_alter_verdict_parsing(self):
        now = [0.0]

        async def fake_sleep(dt: float) -> None:
            now[0] += dt

        canned_replies = [
            '{"reasoning": "good guidance", "decision": "OK"}',
            '{"reasoning": "broken latex \\frac{1}{2}", "decision": "REJECT"}',
            "completely unparseable output",
            '{"reasoning": "solid", "score": 4}',
            'malformed "score": 5 trailing',
        ]

        def make_fake():
            idx = 0

            class _FakeComp:
                async def create(self, **_kwargs):
                    nonlocal idx
                    text = canned_replies[idx % len(canned_replies)]
                    idx += 1
                    return _fake_completion(text)

            return SimpleNamespace(chat=SimpleNamespace(completions=_FakeComp()))

        raw_client = make_fake()
        wrapped_client = aw.AgnesWindowClient(
            make_fake(),
            window=aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0),
        )

        msgs = [{"role": "user", "content": "help"}, {"role": "assistant", "content": "try factoring"}]
        raw_qual = asyncio.run(
            judge_quality.majority_vote(
                messages=msgs, problem="p", answer="a", client=raw_client, model="m", num_votes=3
            )
        )
        wrap_qual = asyncio.run(
            judge_quality.majority_vote(
                messages=msgs, problem="p", answer="a", client=wrapped_client, model="m", num_votes=3
            )
        )
        self.assertEqual(raw_qual, wrap_qual)

        raw_eval = asyncio.run(
            judge_eval.score_conversation(
                messages=msgs, problem="p", answer="a", client=raw_client, model="m"
            )
        )
        wrap_eval = asyncio.run(
            judge_eval.score_conversation(
                messages=msgs, problem="p", answer="a", client=wrapped_client, model="m"
            )
        )
        self.assertEqual(raw_eval, wrap_eval)

    def test_logs_and_failure_excerpts_omit_synthetic_raw_text(self):
        now = [0.0]

        async def fake_sleep(dt: float) -> None:
            now[0] += dt

        secret = "SYNTHETIC_SECRET_TEXT_DO_NOT_LEAK_XYZ987"

        class _ErrComp:
            async def create(self, **_kwargs):
                raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.agnes-ai.cn/v1"))

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        aw_logger = logging.getLogger("eduardo27b.agnes_window")
        aw_logger.addHandler(handler)
        aw_logger.setLevel(logging.INFO)
        try:
            client = aw.AgnesWindowClient(
                SimpleNamespace(chat=SimpleNamespace(completions=_ErrComp())),
                window=aw.AsyncCongestionWindow(monotonic_fn=lambda: now[0], sleep_fn=fake_sleep, rand_fn=lambda _a, _b: 1.0),
            )
            with self.assertRaises(openai.APITimeoutError):
                asyncio.run(client.chat.completions.create(model="m", messages=[{"role": "user", "content": secret}]))
        finally:
            aw_logger.removeHandler(handler)

        fake_log = "\n".join([
            buf.getvalue(),
            f"2026-11-23 16:00:00,000 [INFO] eduardo27b.agnes_window: [27b_agnes_window] STARTUP {client.startup_summary()}",
            f"2026-11-23 16:00:01,000 [WARNING] eduardo.metrics.judge_quality: Parse failure {secret}",
            "Traceback (most recent call last):",
            '  File "/workspace/eval/run_env_test_eval.py", line 525, in score_records',
            f"    raise ValueError('{secret}')",
            f"ValueError: {secret}",
            f"2026-11-23 16:05:00,000 [INFO] eduardo27b.agnes_window: [27b_agnes_window] SUMMARY logical_calls=1 http_attempts=3",
        ])
        excerpt = s.format_eval_log_excerpt(fake_log)
        fail_sum = s.extract_safe_failure_summary(fake_log)
        self.assertNotIn(secret, buf.getvalue())
        self.assertNotIn(secret, excerpt)
        self.assertNotIn(secret, fail_sum)
        self.assertIn("[27b_agnes_window] STARTUP", excerpt)
        self.assertIn("[27b_agnes_window] SUMMARY", excerpt)
        self.assertEqual(
            fail_sum,
            'File "/workspace/eval/run_env_test_eval.py", line 525, in score_records\nValueError',
        )

    def test_eduardo27b_paper_baseline_and_probe_classifier(self):
        v_ok, _ = s.judge_alignment_verdict(
            0.28, (0.24, 0.31), 0.45, (0.41, 0.49), baseline=s.PAPER_EDUARDO_27B
        )
        self.assertEqual(v_ok, "对上")
        v_mid, _ = s.judge_alignment_verdict(
            0.23, (0.19, 0.27), 0.46, (0.42, 0.50), baseline=s.PAPER_EDUARDO_27B
        )
        self.assertEqual(v_mid, "部分对上")
        v_bad, _ = s.judge_alignment_verdict(
            0.18, (0.14, 0.22), 0.46, (0.42, 0.50), baseline=s.PAPER_EDUARDO_27B
        )
        self.assertEqual(v_bad, "对不上")

        self.assertEqual(
            probe.classify_probe_failure("ValueError", "No available memory for the cache blocks"),
            "OOM_OR_KV_CAPACITY",
        )
        self.assertEqual(
            probe.classify_probe_failure("KeyError", "'qwen3_next'"),
            "ENV_INCOMPATIBLE",
        )

    def test_hooks_share_single_window_between_judge_and_eval_judge(self):
        orig_make = upstream_eval.make_judge_client
        orig_get = upstream_eval.get_async_openai_client
        try:
            fake_agnes = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: None)))
            fake_student = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: None)))

            upstream_eval.make_judge_client = lambda args: (fake_agnes, args.gemini_model, {})
            upstream_eval.get_async_openai_client = (
                lambda base, key, timeout=300.0: fake_agnes if key == "AGNES_API_KEY" else fake_student
            )

            runner27b.install_agnes_window_hooks(
                upstream_eval,
                agnes_api_base="https://api.agnes-ai.cn/v1",
                agnes_api_key_env="AGNES_API_KEY",
                outer_num_retries=3,
            )
            fake_args = SimpleNamespace(gemini=True, gemini_model="agnes-3.0-flash")
            jc, _, _ = upstream_eval.make_judge_client(fake_args)
            ejc = upstream_eval.get_async_openai_client("https://api.agnes-ai.cn/v1", "AGNES_API_KEY")
            sc = upstream_eval.get_async_openai_client("http://127.0.0.1:18961/v1", "SERVING_API_KEY")

            self.assertIsInstance(jc, aw.AgnesWindowClient)
            self.assertIs(jc, ejc)
            self.assertIs(sc, fake_student)
        finally:
            upstream_eval.make_judge_client = orig_make
            upstream_eval.get_async_openai_client = orig_get


if __name__ == "__main__":
    unittest.main()
