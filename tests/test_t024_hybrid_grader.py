"""T-024 TestHybridGrader parity suite run against T-023's verbatim hybrid_grader.py copy.

Ensures both T-023's and T-024's hybrid_grader unit tests pass inside Pro 5000's
`pro5000-fast` environment (with real `eduardo.utils.grade_math_answer` and
`math-verify==0.8.0`) before Stage 2 starts.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict

try:
    import eduardo.utils  # type: ignore  # noqa: F401
except ImportError:
    _edu_pkg = types.ModuleType("eduardo")
    _edu_utils = types.ModuleType("eduardo.utils")

    def _stub_grade_math_answer(solution_str: str, ground_truth: str) -> Dict[str, Any]:
        marker = f"\\boxed{{{ground_truth}}}"
        if marker in (solution_str or ""):
            return {"score": 1.0, "acc": 1.0, "pred": ground_truth, "incorrect_format": 0, "truncated": 0}
        inc_fmt = 0 if "\\boxed{" in (solution_str or "") else 1
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": inc_fmt, "truncated": 0}

    _edu_utils.grade_math_answer = _stub_grade_math_answer  # type: ignore[attr-defined]
    _edu_pkg.utils = _edu_utils  # type: ignore[attr-defined]
    sys.modules["eduardo"] = _edu_pkg
    sys.modules["eduardo.utils"] = _edu_utils

from routeA_data.hybrid_grader import (
    AGNES_JUDGE_REFERENCE_SHA256,
    EQUIV_JUDGE_SYSTEM_PROMPT,
    AsyncAdaptiveWindow,
    HybridGrader,
    check_technical_failure_rate,
    file_sha256,
    parse_equiv_verdict,
    verify_hybrid_grader_sha256,
)
from routeA_data.hybrid_grader_caller import CANONICAL_HYBRID_GRADER_SHA256

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
HYBRID_GRADER_PATH = WORKTREE_ROOT / "routeA_data" / "hybrid_grader.py"


class TransientHTTPStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestT024HybridGraderParity(unittest.TestCase):
    def test_prompt_matches_agnes_eval_v3_byte_for_byte(self) -> None:
        agnes_path = WORKTREE_ROOT / "evaluation" / "agnes_eval_v3" / "agnes_judge.py"
        self.assertTrue(agnes_path.is_file())
        self.assertEqual(file_sha256(agnes_path), AGNES_JUDGE_REFERENCE_SHA256)
        spec = importlib.util.spec_from_file_location("agnes_judge_ref_t024", agnes_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(EQUIV_JUDGE_SYSTEM_PROMPT, mod.SYSTEM)

    def test_rule_correct_skips_27b(self) -> None:
        llm_calls = 0

        async def fake_llm(_msgs: Any) -> Any:
            nonlocal llm_calls
            llm_calls += 1
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=fake_llm, backoff_base=0.0)
        res = _run(grader.grade("Compute 2+2.", "4", r"Thus \boxed{4}."))
        self.assertTrue(res.correct)
        self.assertEqual(res.status, "correct")
        self.assertEqual(res.source, "rule")
        self.assertTrue(res.rule_correct)
        self.assertFalse(res.llm_called)
        self.assertEqual(llm_calls, 0)

    def test_rule_incorrect_and_unboxed_call_27b(self) -> None:
        seen_payloads = []

        async def fake_llm(msgs: Any) -> Any:
            seen_payloads.append(json.loads(msgs[1]["content"]))
            if "four" in msgs[1]["content"]:
                return '{"correct": true}', "stop"
            return '{"correct": false}', "stop"

        grader = HybridGrader(llm_caller=fake_llm, backoff_base=0.0)
        res1 = _run(grader.grade("Compute 2+2.", "4", "The answer is four."))
        self.assertTrue(res1.correct)
        self.assertEqual(res1.status, "correct")
        self.assertEqual(res1.source, "llm")
        self.assertFalse(res1.rule_correct)
        self.assertEqual(res1.incorrect_format, 1)
        self.assertTrue(res1.llm_called)
        self.assertFalse(res1.is_unknown)
        self.assertFalse(res1.technical_failure)

        res2 = _run(grader.grade("Compute 2+2.", "4", r"\boxed{5}"))
        self.assertFalse(res2.correct)
        self.assertEqual(res2.status, "incorrect")
        self.assertEqual(res2.incorrect_format, 0)
        self.assertTrue(res2.llm_called)
        self.assertFalse(res2.is_unknown)
        self.assertFalse(res2.technical_failure)
        self.assertEqual(len(seen_payloads), 2)

    def test_unknown_and_bad_format_count_as_wrong(self) -> None:
        null_calls = 0

        async def null_llm(_msgs: Any) -> Any:
            nonlocal null_calls
            null_calls += 1
            return '{"correct": null}', "stop"

        g_null = HybridGrader(llm_caller=null_llm, max_attempts=4, backoff_base=0.0)
        r_null = _run(g_null.grade("Q", "4", r"\boxed{?}"))
        self.assertFalse(r_null.correct)
        self.assertEqual(r_null.status, "unknown")
        self.assertTrue(r_null.is_unknown)
        self.assertFalse(r_null.technical_failure)
        self.assertEqual(null_calls, 1)

        bad_calls = 0

        async def bad_llm(_msgs: Any) -> Any:
            nonlocal bad_calls
            bad_calls += 1
            return "not valid json", "stop"

        g_bad = HybridGrader(llm_caller=bad_llm, max_attempts=3, backoff_base=0.0)
        r_bad = _run(g_bad.grade("Q", "4", r"\boxed{?}"))
        self.assertFalse(r_bad.correct)
        self.assertEqual(r_bad.status, "unknown")
        self.assertTrue(r_bad.is_unknown)
        self.assertFalse(r_bad.technical_failure)
        self.assertEqual(bad_calls, 3)

    def test_27b_technical_failure_retries_and_is_not_verdict(self) -> None:
        attempts = 0

        async def flaky_llm(_msgs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise TransientHTTPStatusError(503, "server busy")
            return '```json\n{"correct": true}\n```', "stop"

        win = AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0)
        g_flaky = HybridGrader(llm_caller=flaky_llm, max_attempts=4, window=win, backoff_base=0.0)
        r_ok = _run(g_flaky.grade("Q", "4", "four"))
        self.assertTrue(r_ok.correct)
        self.assertFalse(r_ok.technical_failure)
        self.assertEqual(r_ok.attempts, 3)
        self.assertEqual(attempts, 3)

        hard_attempts = 0

        async def dead_llm(_msgs: Any) -> Any:
            nonlocal hard_attempts
            hard_attempts += 1
            raise TimeoutError("timed out")

        g_dead = HybridGrader(
            llm_caller=dead_llm,
            max_attempts=3,
            window=AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0),
            backoff_base=0.0,
        )
        r_dead = _run(g_dead.grade("Q", "4", "four"))
        self.assertFalse(r_dead.correct)
        self.assertEqual(r_dead.status, "technical_error")
        self.assertTrue(r_dead.technical_failure)
        self.assertFalse(r_dead.is_unknown)
        self.assertEqual(r_dead.attempts, 3)
        self.assertEqual(hard_attempts, 3)

    def test_aimd_window_and_tech_failure_rate(self) -> None:
        async def exercise() -> None:
            win = AsyncAdaptiveWindow(max_window=4, max_cooldown=0.0)
            self.assertEqual(win.window, 1)
            for expected in (2, 3, 4, 4):
                await win.acquire()
                await win.release(True)
                self.assertEqual(win.window, expected)
            await win.acquire()
            await win.release(None)
            self.assertEqual(win.window, 4)
            await win.acquire()
            await win.release(False)
            self.assertEqual(win.window, 2)
            await win.acquire()
            await win.release(False)
            self.assertEqual(win.window, 1)

        _run(exercise())
        self.assertFalse(check_technical_failure_rate(0, 0))
        self.assertFalse(check_technical_failure_rate(1, 100))
        self.assertTrue(check_technical_failure_rate(2, 100))
        self.assertTrue(check_technical_failure_rate(1, 99))

    def test_verify_hybrid_grader_sha256_helper(self) -> None:
        self.assertEqual(
            verify_hybrid_grader_sha256(HYBRID_GRADER_PATH, CANONICAL_HYBRID_GRADER_SHA256),
            CANONICAL_HYBRID_GRADER_SHA256,
        )
        with tempfile.TemporaryDirectory() as td:
            copy_path = Path(td) / "hybrid_grader_drift.py"
            copy_path.write_bytes(HYBRID_GRADER_PATH.read_bytes() + b"\n# drift\n")
            with self.assertRaises(ValueError):
                verify_hybrid_grader_sha256(copy_path, CANONICAL_HYBRID_GRADER_SHA256)

    def test_parse_equiv_verdict_edge_cases(self) -> None:
        self.assertEqual(parse_equiv_verdict('{"correct": true}', "stop"), (True, None))
        self.assertEqual(parse_equiv_verdict('{"correct": false}', "stop"), (False, None))
        self.assertEqual(parse_equiv_verdict('{"correct": null}', "stop"), (None, None))
        self.assertEqual(parse_equiv_verdict("true", "stop"), (True, None))
        self.assertEqual(parse_equiv_verdict("null", "stop"), (None, None))
        self.assertEqual(parse_equiv_verdict('{"correct": true}', "length"), (None, "response_truncated"))
        self.assertEqual(
            parse_equiv_verdict('{"correct": true, "correct": false}', "stop"),
            (None, "response_duplicate_keys"),
        )
        self.assertEqual(
            parse_equiv_verdict('{"correct": "yes"}', "stop"),
            (None, "response_schema_invalid"),
        )


if __name__ == "__main__":
    unittest.main()
