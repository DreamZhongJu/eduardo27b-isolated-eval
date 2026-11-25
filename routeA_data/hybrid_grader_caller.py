"""Thin caller/wrapper around T-024's canonical hybrid_grader.py (SHA-256 locked).

Per T-023 Card v3 & Inbox #3:
- `routeA_data/hybrid_grader.py` is a verbatim byte-for-byte copy of T-024's
  `training/routeA_reward/hybrid_grader.py` (sha256 7d2a5f46...) and MUST NOT be edited.
- This wrapper adds:
  1. Startup SHA-256 verification of `hybrid_grader.py`.
  2. Caller-level >1% technical failure abort gate (`TechnicalFailureTracker`).
  3. Thin synchronous call helper (`grade_sync`) for callers outside an event loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from routeA_data.hybrid_grader import (
    AGNES_JUDGE_REFERENCE_SHA256,
    DEFAULT_MAX_WINDOW,
    EQUIV_JUDGE_SYSTEM_PROMPT,
    AsyncAdaptiveWindow,
    GradeResult,
    HybridGrader,
    check_technical_failure_rate,
    verify_hybrid_grader_sha256,
)

CANONICAL_HYBRID_GRADER_SHA256 = (
    "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
)


class TechnicalFailureGateError(RuntimeError):
    """Raised by the caller layer when technical failures exceed 1%."""


def assert_canonical_hybrid_grader(
    path: Optional[Path] = None,
    expected_sha256: str = CANONICAL_HYBRID_GRADER_SHA256,
) -> str:
    """Verify byte-for-byte SHA-256 of the local copy of hybrid_grader.py."""
    target = path if path is not None else (Path(__file__).resolve().parent / "hybrid_grader.py")
    return verify_hybrid_grader_sha256(target, expected_sha256)


@dataclass
class TechnicalFailureTracker:
    """Caller-layer >1% technical failure gate.

    Uses exact integer comparison from `hybrid_grader.check_technical_failure_rate`
    (`100 * n_tech_failed > max_percent * n_total_calls`) once `n_total_calls >= min_samples`
    (and also supports explicit end-of-batch `check_final()` without `min_samples`).
    """

    max_percent: int = 1
    min_samples: int = 100
    n_calls: int = 0
    n_tech_failed: int = 0

    def record(self, *, technical_failure: bool) -> None:
        self.n_calls += 1
        if technical_failure:
            self.n_tech_failed += 1
        if self.n_calls >= self.min_samples and check_technical_failure_rate(
            self.n_tech_failed, self.n_calls, max_percent=self.max_percent
        ):
            raise TechnicalFailureGateError(
                f"Technical failure rate {self.n_tech_failed}/{self.n_calls} "
                f"({100.0 * self.n_tech_failed / self.n_calls:.2f}%) exceeded >{self.max_percent}%"
            )

    def check_final(self) -> None:
        if check_technical_failure_rate(
            self.n_tech_failed, self.n_calls, max_percent=self.max_percent
        ):
            raise TechnicalFailureGateError(
                f"Final technical failure rate {self.n_tech_failed}/{self.n_calls} "
                f"({100.0 * self.n_tech_failed / self.n_calls:.2f}%) exceeded >{self.max_percent}%"
            )


def grade_sync(
    grader: HybridGrader,
    *,
    problem: str,
    reference_answer: str,
    student_response: str,
) -> GradeResult:
    """Thin synchronous wrapper around `HybridGrader.grade`."""
    return asyncio.run(grader.grade(problem, reference_answer, student_response))


__all__ = [
    "AGNES_JUDGE_REFERENCE_SHA256",
    "CANONICAL_HYBRID_GRADER_SHA256",
    "DEFAULT_MAX_WINDOW",
    "EQUIV_JUDGE_SYSTEM_PROMPT",
    "AsyncAdaptiveWindow",
    "GradeResult",
    "HybridGrader",
    "TechnicalFailureGateError",
    "TechnicalFailureTracker",
    "assert_canonical_hybrid_grader",
    "check_technical_failure_rate",
    "grade_sync",
]
