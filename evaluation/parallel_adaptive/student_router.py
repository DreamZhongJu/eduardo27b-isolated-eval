"""Experimental student-client router. Not connected to a running evaluator.

Dispatch to an available slot on either *literal loopback* vLLM endpoint.
Requests, sampling parameters and responses pass through
unchanged. Never retry or fall back: a timeout can mean that a remote request
was executed even if its response was lost. The frozen evaluator remains
responsible for request identities, intents, receipts, and any replay policy.

The class is not a supervisor, HTTP proxy, GPU allocator, or a hot-add path.
It needs a deliberately integrated evaluator started at a source boundary.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import urlsplit


def _endpoint(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("endpoint must be a string")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid endpoint") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or port is None or not 1 <= port <= 65535
            or parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment):
        raise ValueError("only explicit loopback /v1 endpoints are allowed")
    return f"http://127.0.0.1:{port}/v1"


class StudentRouter:
    """Minimal OpenAI-compatible async interface for offline_eval.student_sample."""

    def __init__(self, endpoints, *, slots_per_student=2, client_factory=None):
        endpoints = tuple(_endpoint(e) for e in endpoints)
        if len(endpoints) != 2 or len(set(endpoints)) != 2:
            raise ValueError("two distinct student endpoints required")
        if type(slots_per_student) is not int or not 1 <= slots_per_student <= 16:
            raise ValueError("invalid student capacity")
        if client_factory is None:
            import openai
            import httpx

            def client_factory(**settings):
                # Do not route local data through user-configured HTTP proxies
                # and do not follow any redirect to a nonlocal destination.
                transport = httpx.AsyncClient(trust_env=False, follow_redirects=False)
                return openai.AsyncClient(http_client=transport, **settings)
        self._clients = tuple(client_factory(api_key="loopback-only", base_url=e,
                                             timeout=300, max_retries=0)
                              for e in endpoints)
        self._capacity = slots_per_student
        self._active = [0, 0]
        self._disabled = [False, False]
        self._aborting = False
        self._condition = asyncio.Condition()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def _acquire(self):
        async with self._condition:
            while True:
                if self._aborting:
                    raise RuntimeError("student router stopped after uncertain request")
                available = [i for i in (0, 1) if not self._disabled[i]
                             and self._active[i] < self._capacity]
                if available:
                    # Spread equal demand, while allowing the faster worker to
                    # take more work as its slots become free.
                    index = min(available, key=lambda i: (self._active[i], i))
                    self._active[index] += 1
                    return index
                if all(self._disabled):
                    raise RuntimeError("all student workers unavailable")
                await self._condition.wait()

    async def _release(self, index, failed):
        async with self._condition:
            self._active[index] -= 1
            if failed:
                # A timeout or cancellation may leave server-side work running.
                # Do not give this worker new work until an operator-controlled
                # restart or another independently verified recovery procedure.
                self._disabled[index] = True
            self._condition.notify_all()

    async def create(self, **params):
        seed = params.get("seed")
        if type(seed) is not int or not 0 <= seed < 2 ** 31:
            raise ValueError("original student sample seed required")
        index = await self._acquire()
        failed = True
        try:
            # No fallback or retry: the caller's existing intent/receipt
            # semantics govern uncertain execution.
            result = await self._clients[index].chat.completions.create(**params)
            failed = False
            return result
        finally:
            # Shield cleanup so cancellation does not lose capacity accounting;
            # cancellation propagates to the caller without a retry.
            if failed:
                # Mark synchronously before yielding: no queued producer may
                # dispatch onto this worker before cleanup gets the condition.
                self._disabled[index] = True
                self._aborting = True
            await asyncio.shield(self._release(index, failed))

    async def close(self):
        # gather() in the frozen evaluator can raise while sibling requests
        # still run. Let those settle before closing transports or returning.
        async with self._condition:
            self._aborting = True
            self._condition.notify_all()
            while any(self._active):
                await self._condition.wait()
        for client in self._clients:
            await client.close()
