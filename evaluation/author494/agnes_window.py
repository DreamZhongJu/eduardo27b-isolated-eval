#!/usr/bin/env python3
"""Shared AIMD congestion window and transport retry wrapper for Agnes judge calls.

Wraps the async OpenAI client returned by author's ``make_judge_client`` /
``get_async_openai_client`` without modifying any author files, prompts,
sampling parameters, or verdict parsing logic.

Hardcoded bounds:
- Window: starts at 1, grows +1 per aligned success, capped at 16.
- Congestion (408/429/5xx/network/timeout/ambiguous response): halves window
  (floor 1) and applies jittered exponential backoff capped at 60.0s.
- Inner retries: up to 2 retries (3 HTTP attempts) per create() call.
  Combined with author's outer num_retries=3 (4 outer attempts), each vote
  triggers at most 4 * 3 = 12 HTTP requests.
- HTTP 200 with a valid completion envelope is NEVER retried even if the
  model text fails JSON parsing or is semantically unresolved.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable

import openai

logger = logging.getLogger("eduardo27b.agnes_window")

WINDOW_INIT = 1
WINDOW_MAX = 16
INNER_MAX_RETRIES = 2
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0
JITTER_LOW = 0.5
JITTER_HIGH = 1.5

_RETRIABLE_STATUS_CODES = frozenset({408, 429})


def is_valid_completion_envelope(resp: Any) -> bool:
    """Return True iff HTTP 200 returned a well-formed completion choice envelope."""
    if resp is None:
        return False
    choices = getattr(resp, "choices", None)
    if not isinstance(choices, (list, tuple)) or len(choices) == 0:
        return False
    msg = getattr(choices[0], "message", None)
    if msg is None or not hasattr(msg, "content"):
        return False
    content = getattr(msg, "content", None)
    return content is None or isinstance(content, str)


def classify_exception(exc: BaseException) -> tuple[str, bool]:
    """Classify an exception into (fixed_category, is_retriable_technical).

    Never inspects or returns exception message strings.
    """
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in _RETRIABLE_STATUS_CODES or (status is not None and 500 <= status < 600):
            return f"http_{status}", True
        return f"http_{status}_nonretriable", False
    if isinstance(exc, (openai.APITimeoutError, asyncio.TimeoutError, TimeoutError)):
        return "timeout", True
    if isinstance(exc, openai.APIConnectionError):
        return "connection_error", True
    if isinstance(exc, OSError):
        return "network_os_error", True
    return "ambiguous_transport_error", True


class AsyncCongestionWindow:
    """In-process async AIMD concurrency window with cooldown backoff."""

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
        sleep_fn: Callable[[float], Any] = asyncio.sleep,
        rand_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not (1 <= initial <= maximum <= 16):
            raise ValueError(f"Invalid window bounds: initial={initial}, maximum={maximum}")
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
        self._cond = asyncio.Condition()

        # Sanitized aggregate telemetry
        self.max_limit_reached = initial
        self.window_drops = 0
        self.congestion_events = 0
        self.http_429_events = 0
        self.logical_calls = 0
        self.http_attempts = 0
        self.success_calls = 0
        self.transient_retries = 0
        self.ambiguous_failures = 0
        self.non_retriable_failures = 0

    async def acquire(self) -> int:
        async with self._cond:
            while True:
                now = self._monotonic()
                delay = self.blocked_until - now
                if delay > 0:
                    self._cond.release()
                    try:
                        await self._sleep(delay)
                    finally:
                        await self._cond.acquire()
                    continue
                if self.inflight < self.limit:
                    self.inflight += 1
                    return self.epoch
                await self._cond.wait()

    async def report_success(self, lease: int) -> None:
        async with self._cond:
            if lease == self.epoch and self._monotonic() >= self.blocked_until:
                self.limit = min(self.maximum, self.limit + 1)
                if self.limit > self.max_limit_reached:
                    self.max_limit_reached = self.limit
                self.failures = 0
            self.success_calls += 1
            self._cond.notify_all()

    async def report_congestion(self) -> float:
        async with self._cond:
            prev = self.limit
            self.limit = max(1, self.limit // 2)
            if self.limit < prev:
                self.window_drops += 1
            self.congestion_events += 1
            self.failures += 1
            self.epoch += 1
            exp_idx = min(self.failures - 1, 16)
            raw_delay = self.backoff_base * (2 ** exp_idx)
            jitter = self._rand(self.jitter_low, self.jitter_high)
            delay = min(self.backoff_cap, raw_delay * jitter)
            self.blocked_until = max(self.blocked_until, self._monotonic() + delay)
            self._cond.notify_all()
            return delay

    async def release(self) -> None:
        async with self._cond:
            if self.inflight < 1:
                raise RuntimeError("API window permit underflow")
            self.inflight -= 1
            self._cond.notify_all()

    def stats(self) -> dict[str, Any]:
        return {
            "window_init": self.initial,
            "window_max": self.maximum,
            "backoff_base_s": self.backoff_base,
            "backoff_cap_s": self.backoff_cap,
            "final_limit": self.limit,
            "max_limit_reached": self.max_limit_reached,
            "window_drops": self.window_drops,
            "congestion_events": self.congestion_events,
            "http_429_events": self.http_429_events,
            "logical_calls": self.logical_calls,
            "http_attempts": self.http_attempts,
            "success_calls": self.success_calls,
            "transient_retries": self.transient_retries,
            "ambiguous_failures": self.ambiguous_failures,
            "non_retriable_failures": self.non_retriable_failures,
        }


class _RawCompletionsProxy:
    def __init__(self, wrapper: "AgnesWindowClient") -> None:
        self._wrapper = wrapper

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await self._wrapper._call_with_window(raw=True, *args, **kwargs)


class _CompletionsProxy:
    def __init__(self, wrapper: "AgnesWindowClient") -> None:
        self._wrapper = wrapper
        self.with_raw_response = _RawCompletionsProxy(wrapper)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await self._wrapper._call_with_window(raw=False, *args, **kwargs)


class _ChatProxy:
    def __init__(self, wrapper: "AgnesWindowClient") -> None:
        self.completions = _CompletionsProxy(wrapper)


class AgnesWindowClient:
    """AsyncOpenAI-compatible wrapper adding AIMD congestion window and transport retries."""

    def __init__(
        self,
        inner_client: Any,
        *,
        window: AsyncCongestionWindow | None = None,
        inner_max_retries: int = INNER_MAX_RETRIES,
    ) -> None:
        if inner_max_retries < 0:
            raise ValueError("inner_max_retries must be non-negative")
        self._inner = inner_client
        self.window = window if window is not None else AsyncCongestionWindow()
        self.inner_max_retries = inner_max_retries
        self.chat = _ChatProxy(self)

    def startup_summary(self, *, outer_num_retries: int = 3) -> str:
        max_reqs_per_vote = (1 + outer_num_retries) * (1 + self.inner_max_retries)
        return (
            f"AgnesWindowClient config: window_init={self.window.initial} "
            f"window_max={self.window.maximum} inner_max_retries={self.inner_max_retries} "
            f"outer_num_retries={outer_num_retries} max_http_requests_per_vote={max_reqs_per_vote} "
            f"backoff_base_s={self.window.backoff_base} backoff_cap_s={self.window.backoff_cap}"
        )

    def stats(self, *, outer_num_retries: int = 3) -> dict[str, Any]:
        out = self.window.stats()
        out["inner_max_retries"] = self.inner_max_retries
        out["outer_num_retries"] = outer_num_retries
        out["max_http_requests_per_vote"] = (1 + outer_num_retries) * (1 + self.inner_max_retries)
        return out

    async def _invoke_inner(self, *, raw: bool, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        completions = self._inner.chat.completions
        if raw:
            return await completions.with_raw_response.create(*args, **kwargs)
        return await completions.create(*args, **kwargs)

    async def _call_with_window(self, *args: Any, raw: bool = False, **kwargs: Any) -> Any:
        self.window.logical_calls += 1
        for attempt in range(self.inner_max_retries + 1):
            lease = await self.window.acquire()
            self.window.http_attempts += 1
            try:
                resp = await self._invoke_inner(raw=raw, args=args, kwargs=kwargs)
                if raw:
                    # Raw response path (used by agnes_ping): caller parses envelope.
                    if getattr(resp, "status_code", 200) != 200:
                        self.window.ambiguous_failures += 1
                        await self.window.report_congestion()
                        if attempt < self.inner_max_retries:
                            self.window.transient_retries += 1
                            continue
                        raise openai.APIConnectionError(request=None)
                    await self.window.report_success(lease)
                    return resp

                if is_valid_completion_envelope(resp):
                    # HTTP 200 with valid choice/message envelope: do NOT retry parse/semantic issues.
                    await self.window.report_success(lease)
                    return resp

                # Request sent, response envelope incomplete/ambiguous -> technical failure.
                self.window.ambiguous_failures += 1
                delay = await self.window.report_congestion()
                logger.warning(
                    "Agnes ambiguous response envelope: attempt=%d/%d next_delay=%.1fs window=%d",
                    attempt + 1,
                    self.inner_max_retries + 1,
                    delay,
                    self.window.limit,
                )
                if attempt < self.inner_max_retries:
                    self.window.transient_retries += 1
                    continue
                raise openai.APIConnectionError(request=None)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except openai.APIStatusError as exc:
                cat, retriable = classify_exception(exc)
                if not retriable:
                    self.window.non_retriable_failures += 1
                    raise
                if getattr(exc, "status_code", None) == 429:
                    self.window.http_429_events += 1
                delay = await self.window.report_congestion()
                logger.warning(
                    "Agnes transient status: category=%s attempt=%d/%d next_delay=%.1fs window=%d",
                    cat,
                    attempt + 1,
                    self.inner_max_retries + 1,
                    delay,
                    self.window.limit,
                )
                if attempt < self.inner_max_retries:
                    self.window.transient_retries += 1
                    continue
                raise
            except Exception as exc:
                cat, _ = classify_exception(exc)
                if cat == "ambiguous_transport_error":
                    self.window.ambiguous_failures += 1
                delay = await self.window.report_congestion()
                logger.warning(
                    "Agnes transport error: category=%s exc=%s attempt=%d/%d next_delay=%.1fs window=%d",
                    cat,
                    type(exc).__name__,
                    attempt + 1,
                    self.inner_max_retries + 1,
                    delay,
                    self.window.limit,
                )
                if attempt < self.inner_max_retries:
                    self.window.transient_retries += 1
                    continue
                if isinstance(
                    exc,
                    (
                        openai.APITimeoutError,
                        openai.APIConnectionError,
                        openai.RateLimitError,
                        openai.InternalServerError,
                        asyncio.TimeoutError,
                    ),
                ):
                    raise
                raise openai.APIConnectionError(request=getattr(exc, "request", None)) from None
            finally:
                await self.window.release()

    async def close(self) -> None:
        close_fn = getattr(self._inner, "close", None)
        if callable(close_fn):
            res = close_fn()
            if asyncio.iscoroutine(res):
                await res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
