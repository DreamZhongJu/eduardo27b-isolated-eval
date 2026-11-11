import asyncio
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from student_router import StudentRouter


class FakeClient:
    def __init__(self, **settings):
        self.settings = settings
        self.calls = []
        self.failure = False
        self.closed = False
        self.gate = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **params):
        self.calls.append(copy.deepcopy(params))
        if self.failure:
            raise TimeoutError("uncertain request")
        if self.gate is not None:
            await self.gate.wait()
        return {"student": self.settings["base_url"]}

    async def close(self):
        self.closed = True


class RouterTests(unittest.IsolatedAsyncioTestCase):
    def make_router(self):
        clients = []

        def factory(**settings):
            client = FakeClient(**settings)
            clients.append(client)
            return client

        return StudentRouter(("http://127.0.0.1:18004/v1", "http://127.0.0.1:18005/v1"),
                             client_factory=factory), clients

    async def test_dispatch_preserves_entire_body(self):
        router, clients = self.make_router()
        clients[0].gate = asyncio.Event()
        body = {"model": "Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "synthetic"}],
                "temperature": 0.6, "max_tokens": 1024, "seed": 10,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        original = copy.deepcopy(body)
        first = asyncio.create_task(router.chat.completions.create(**body))
        await asyncio.sleep(0)
        second = asyncio.create_task(router.chat.completions.create(**body))
        await asyncio.sleep(0)
        clients[0].gate.set()
        await asyncio.gather(first, second)
        body2 = {**body, "seed": 11}
        await router.chat.completions.create(**body2)
        self.assertEqual(body, original)
        self.assertEqual(clients[0].calls, [original, body2])
        self.assertEqual(clients[1].calls, [original])
        self.assertTrue(all(x.settings['max_retries'] == 0 for x in clients))
        await router.close()
        self.assertTrue(all(x.closed for x in clients))

    async def test_uncertain_failure_is_never_retried_or_sent_elsewhere(self):
        router, clients = self.make_router()
        clients[0].failure = True
        with self.assertRaises(TimeoutError):
            await router.chat.completions.create(seed=2, messages=[])
        self.assertEqual([len(x.calls) for x in clients], [1, 0])
        with self.assertRaises(RuntimeError):
            await router.chat.completions.create(seed=3, messages=[])
        self.assertEqual([len(x.calls) for x in clients], [1, 0])

    async def test_idle_worker_takes_work_and_per_worker_capacity_is_enforced(self):
        router, clients = self.make_router()
        clients[0].gate, clients[1].gate = asyncio.Event(), asyncio.Event()
        tasks = [asyncio.create_task(router.chat.completions.create(seed=i)) for i in range(5)]
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertEqual([len(x.calls) for x in clients], [2, 2])
        self.assertFalse(tasks[4].done())
        clients[1].gate.set()
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertEqual([len(x.calls) for x in clients], [2, 3])
        clients[0].gate.set()
        await asyncio.gather(*tasks)

    async def test_cancel_during_request_is_not_retried_and_disables_worker(self):
        router, clients = self.make_router()
        clients[0].gate = asyncio.Event()
        task = asyncio.create_task(router.chat.completions.create(seed=2))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        for _ in range(5):
            await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await router.chat.completions.create(seed=3)
        self.assertEqual([len(x.calls) for x in clients], [1, 0])

    async def test_close_waits_for_inflight_request(self):
        router, clients = self.make_router()
        clients[0].gate = asyncio.Event()
        work = asyncio.create_task(router.chat.completions.create(seed=5))
        await asyncio.sleep(0)
        closing = asyncio.create_task(router.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        with self.assertRaises(RuntimeError):
            await router.chat.completions.create(seed=6)
        clients[0].gate.set()
        await asyncio.gather(work, closing)
        self.assertTrue(all(x.closed for x in clients))

    async def test_invalid_seed_rejected_without_any_calls(self):
        router, clients = self.make_router()
        for seed in (None, True, -1, 2 ** 31, "1"):
            with self.assertRaises(ValueError):
                await router.chat.completions.create(seed=seed)
        self.assertEqual([len(x.calls) for x in clients], [0, 0])

    async def test_default_clients_disable_proxy_and_redirect(self):
        transports = []
        clients = []

        def http_client(**settings):
            transports.append(settings)
            return object()

        def openai_client(**settings):
            client = FakeClient(**settings)
            clients.append(client)
            return client

        with patch.dict(sys.modules, {
            'httpx': SimpleNamespace(AsyncClient=http_client),
            'openai': SimpleNamespace(AsyncClient=openai_client),
        }):
            router = StudentRouter(('http://127.0.0.1:18004/v1', 'http://127.0.0.1:18005/v1'))
        self.assertEqual(transports, [dict(trust_env=False, follow_redirects=False)] * 2)
        self.assertTrue(all(c.settings['max_retries'] == 0 for c in clients))
        await router.close()

    async def test_forbids_external_and_invalid_endpoints(self):
        bad = ("https://127.0.0.1:18005/v1", "http://example.com:18005/v1",
               "http://127.0.0.1:18005/v1?q=secret", "http://127.0.0.1:18005/other",
               "http://x@127.0.0.1:18005/v1", "http://127.0.0.1/v1")
        for endpoint in bad:
            with self.assertRaises(ValueError):
                StudentRouter(("http://127.0.0.1:18004/v1", endpoint), client_factory=FakeClient)
        with self.assertRaises(ValueError):
            StudentRouter(("http://127.0.0.1:18004/v1", "http://127.0.0.1:18004/v1/"),
                          client_factory=FakeClient)


if __name__ == '__main__':
    unittest.main()
