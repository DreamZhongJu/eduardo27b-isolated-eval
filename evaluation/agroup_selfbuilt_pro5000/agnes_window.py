#!/usr/bin/env python3
"""Synchronous AIMD congestion window wrapper for Pro 5000 Agnes grading (T-021).

Adapted from T-022's ``agnes_window.py`` for ThreadPoolExecutor callers around
unmodified ``evaluation/agnes_eval_v3/agnes_judge.py``.

Bounds & behavior:
- Starts at window=1, grows +1 per aligned valid verdict, capped at 16.
- Halves window (floor 1) and sets jittered exponential backoff (capped at 60s)
  on HTTP 429, 408/5xx, transport/timeout exceptions, or HTTP 200 unclear/invalid
  verdict envelopes.
- HTTP 200 semantic null (``{"correct": null}`` / ``null``) is NEVER treated as a
  technical failure and does not trigger backoff or retry.
"""

from __future__ import annotations

import threading
import time
import random
from typing import Any, Callable

WINDOW_INIT = 1
WINDOW_MAX = 16
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0
JITTER_LOW = 0.5
JITTER_HIGH = 1.5


def is_semantic_null_response(judge_mod: Any, raw: Any) -> bool:
    """Return True iff raw is a clean HTTP 200 Agnes completion resolving to semantic null."""
    if not isinstance(raw, dict):
        return False
    if raw.get("model") != judge_mod.MODEL:
        return False
    try:
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            return False
    except (AttributeError, TypeError, KeyError, IndexError):
        return False
    if judge_mod.verdict(raw) is not None:
        return False
    return judge_mod.retryable_content_issue(raw) is None


class SyncCongestionWindow:
    """Thread-safe AIMD concurrency window with epoch-guarded additive growth and backoff."""

    def __init__(
        self,
        *,
        initial: int = WINDOW_INIT,
        maximum: int = WINDOW_MAX,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        backoff_cap: float = BACKOFF_CAP_SECONDS,
        jitter_low: float = JITTER_LOW,
        jitter_high: float = JITTER_HIGH,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], Any] = time.sleep,
        rand_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not (1 <= initial <= maximum <= WINDOW_MAX):
            raise ValueError(f"Invalid Agnes window bounds: initial={initial}, maximum={maximum}")
        if backoff_base < 0 or backoff_cap < backoff_base:
            raise ValueError("Invalid backoff bounds")
        self.initial = initial
        self.maximum = maximum
        self.limit = initial
        self.inflight = 0
        self.epoch = 0
        self.failures = 0
        self.blocked_until = 0.0
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.jitter_low = jitter_low
        self.jitter_high = jitter_high
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self._rand = rand_fn
        self._cond = threading.Condition()

        # Sanitized aggregate telemetry
        self.max_limit_reached = initial
        self.max_inflight_reached = 0
        self.window_drops = 0
        self.congestion_events = 0
        self.http_429_events = 0
        self.http_5xx_events = 0
        self.ambiguous_events = 0
        self.transport_error_events = 0
        self.http_attempts = 0
        self.success_calls = 0
        self.semantic_neutral_calls = 0

    def acquire(self) -> int:
        with self._cond:
            while True:
                now = self._monotonic()
                delay = self.blocked_until - now
                if delay > 0:
                    if self._sleep is time.sleep:
                        self._cond.wait(timeout=delay)
                    else:
                        self._cond.release()
                        try:
                            self._sleep(delay)
                        finally:
                            self._cond.acquire()
                    continue
                if self.inflight < self.limit:
                    self.inflight += 1
                    if self.inflight > self.max_inflight_reached:
                        self.max_inflight_reached = self.inflight
                    return self.epoch
                self._cond.wait(timeout=0.25)

    def record_attempt(self) -> None:
        with self._cond:
            self.http_attempts += 1

    def report_success(self, lease: int) -> None:
        with self._cond:
            if lease == self.epoch and self._monotonic() >= self.blocked_until:
                self.limit = min(self.maximum, self.limit + 1)
                if self.limit > self.max_limit_reached:
                    self.max_limit_reached = self.limit
                self.failures = 0
            self.success_calls += 1
            self._cond.notify_all()

    def report_neutral(self) -> None:
        with self._cond:
            self.semantic_neutral_calls += 1
            self._cond.notify_all()

    def report_congestion(self, *, event: str) -> float:
        with self._cond:
            prev = self.limit
            self.limit = max(1, self.limit // 2)
            if self.limit < prev:
                self.window_drops += 1
            self.congestion_events += 1
            if event == "http_429":
                self.http_429_events += 1
            elif event == "http_5xx":
                self.http_5xx_events += 1
            elif event == "ambiguous":
                self.ambiguous_events += 1
            elif event == "exception":
                self.transport_error_events += 1
            self.failures += 1
            self.epoch += 1
            exp_idx = min(self.failures - 1, 16)
            raw_delay = self.backoff_base * (2 ** exp_idx)
            jitter = self._rand(self.jitter_low, self.jitter_high)
            delay = min(self.backoff_cap, raw_delay * jitter)
            self.blocked_until = max(self.blocked_until, self._monotonic() + delay)
            self._cond.notify_all()
            return delay

    def release(self) -> None:
        with self._cond:
            if self.inflight < 1:
                raise RuntimeError("Agnes window permit underflow")
            self.inflight -= 1
            self._cond.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {
                "window_init": self.initial,
                "window_max": self.maximum,
                "final_limit": self.limit,
                "max_limit_reached": self.max_limit_reached,
                "max_inflight_reached": self.max_inflight_reached,
                "window_drops": self.window_drops,
                "congestion_events": self.congestion_events,
                "http_429_events": self.http_429_events,
                "http_5xx_events": self.http_5xx_events,
                "ambiguous_events": self.ambiguous_events,
                "transport_error_events": self.transport_error_events,
                "http_attempts": self.http_attempts,
                "success_calls": self.success_calls,
                "semantic_neutral_calls": self.semantic_neutral_calls,
            }


class AgnesWindowCaller:
    """Caller proxy passed to unmodified agnes_judge.grade_one(..., caller=proxy)."""

    def __init__(
        self,
        *,
        judge_mod: Any,
        inner_caller: Callable[[dict[str, Any], str], tuple[Any, Any]] | None = None,
        window: SyncCongestionWindow | None = None,
    ) -> None:
        self._judge = judge_mod
        self._inner = inner_caller if inner_caller is not None else judge_mod.api_call
        self.window = window if window is not None else SyncCongestionWindow()

    def __call__(self, body: dict[str, Any], key: str) -> tuple[Any, Any]:
        self.window.record_attempt()
        lease = self.window.acquire()
        try:
            code, raw = self._inner(body, key)
        except Exception:
            self.window.report_congestion(event="exception")
            raise
        else:
            if code == 429:
                self.window.report_congestion(event="http_429")
            elif isinstance(code, int) and (code == 408 or 500 <= code <= 599):
                self.window.report_congestion(event="http_5xx")
            elif code == 200:
                if isinstance(self._judge.verdict(raw), bool):
                    self.window.report_success(lease)
                elif is_semantic_null_response(self._judge, raw):
                    self.window.report_neutral()
                else:
                    self.window.report_congestion(event="ambiguous")
            else:
                self.window.report_congestion(event="other_status")
            return code, raw
        finally:
            self.window.release()
