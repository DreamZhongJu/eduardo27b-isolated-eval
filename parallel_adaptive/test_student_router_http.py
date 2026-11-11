"""SDK integration checks with synthetic payloads and loopback-only servers."""
import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from student_router import StudentRouter

try:
    import httpx  # noqa: F401
    import openai  # noqa: F401
except ImportError:
    HAVE_SDK = False
else:
    HAVE_SDK = True


class FakeVllm:
    def __init__(self, *, status=200, redirect=None):
        self.calls = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get('Content-Length', '0'))
                owner.calls.append(json.loads(self.rfile.read(size)))
                self.send_response(status)
                if redirect:
                    self.send_header('Location', redirect)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                if status == 200:
                    self.wfile.write(json.dumps({
                        'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
                        'model': 'student', 'choices': [{'index': 0, 'finish_reason': 'stop',
                            'message': {'role': 'assistant', 'content': '5'}}],
                    }).encode())

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


@unittest.skipUnless(HAVE_SDK, 'openai and httpx not installed')
class StudentSdkHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_sdk_preserves_seed_and_body(self):
        one, two = FakeVllm(), FakeVllm()
        router = StudentRouter((one.endpoint, two.endpoint))
        try:
            bodies = [dict(model='student', messages=[{'role': 'user', 'content': 'synthetic'}],
                           temperature=0.6, max_tokens=16, seed=i,
                           extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                      for i in (1, 2, 3, 4)]
            responses = await asyncio.gather(*(router.chat.completions.create(**b) for b in bodies))
            sent = one.calls + two.calls
            self.assertEqual(sorted(x['seed'] for x in sent), [1, 2, 3, 4])
            for b in bodies:
                # The SDK merges extra_body into the HTTP JSON; the router must
                # hand the original argument through unchanged to that SDK.
                expected = {k: v for k, v in b.items() if k != 'extra_body'}
                expected.update(b['extra_body'])
                self.assertIn(expected, sent)
            self.assertTrue(one.calls and two.calls)
            self.assertTrue(all(x.choices[0].message.content == '5' for x in responses))
        finally:
            await router.close()
            one.close()
            two.close()

    async def test_sdk_does_not_follow_redirect(self):
        target = FakeVllm()
        redirect = FakeVllm(status=307, redirect=target.endpoint + '/chat/completions')
        other = FakeVllm()
        router = StudentRouter((redirect.endpoint, other.endpoint))
        try:
            with self.assertRaises(Exception):
                await router.chat.completions.create(model='student', seed=1,
                    messages=[{'role': 'user', 'content': 'synthetic'}])
            self.assertEqual(len(redirect.calls), 1)
            self.assertEqual(len(target.calls), 0)
            self.assertEqual(len(other.calls), 0)
        finally:
            await router.close()
            redirect.close()
            target.close()
            other.close()


if __name__ == '__main__':
    unittest.main()
