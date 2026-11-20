"""Lightweight async loopback OpenAI proxy + least-inflight balancer with numeric-only telemetry."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Iterable


def merge_busy_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Compute wall-clock union duration of overlapping [start, end] intervals."""
    valid = sorted((s, e) for s, e in intervals if e > s)
    if not valid:
        return 0.0
    total = 0.0
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def summarize_proxy_window(jsonl_path: Path, spans: list[tuple[float, float]], num_backends: int) -> dict:
    """Aggregate per-backend numeric proxy stats clipped to phase wall-clock spans."""
    if not jsonl_path.is_file():
        return {'missing': True, 'total_requests': 0}
    intervals_all: list[tuple[float, float]] = []
    by_backend: dict[int, dict] = {
        idx: {
            'requests': 0,
            'chat_requests': 0,
            'non_2xx': 0,
            'errors': 0,
            'intervals': [],
            'duration_sum_s': 0.0,
            'completion_tokens_all': [],
            'completion_tokens_max500': [],
            'completion_tokens_max800': [],
        }
        for idx in range(num_backends)
    }

    with jsonl_path.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, list) or len(row) < 7:
                continue
            ts_s, ts_e, idx, status, is_chat, comp_tok, max_tok = row[:7]
            if idx not in by_backend or not (math.isfinite(ts_s) and math.isfinite(ts_e)):
                continue
            # Check overlap with any span (or include all if spans is empty).
            if spans:
                matched = False
                for sp_s, sp_e in spans:
                    ov_s, ov_e = max(ts_s, sp_s), min(ts_e, sp_e)
                    if ov_e > ov_s:
                        matched = True
                        by_backend[idx]['intervals'].append((ov_s, ov_e))
                        intervals_all.append((ov_s, ov_e))
                if not matched:
                    continue
            else:
                if ts_e > ts_s:
                    by_backend[idx]['intervals'].append((ts_s, ts_e))
                    intervals_all.append((ts_s, ts_e))

            b = by_backend[idx]
            b['requests'] += 1
            if is_chat:
                b['chat_requests'] += 1
            if status <= 0:
                b['errors'] += 1
                b['non_2xx'] += 1
            elif not (200 <= status < 300):
                b['non_2xx'] += 1
            b['duration_sum_s'] += max(0.0, ts_e - ts_s)
            if isinstance(comp_tok, int) and comp_tok >= 0:
                b['completion_tokens_all'].append(comp_tok)
                if max_tok == 500:
                    b['completion_tokens_max500'].append(comp_tok)
                elif max_tok == 800:
                    b['completion_tokens_max800'].append(comp_tok)

    def _avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else None

    backends_out = {}
    all_toks, all_500, all_800 = [], [], []
    total_req = total_chat = total_non2xx = total_err = 0
    for idx, b in by_backend.items():
        all_toks.extend(b['completion_tokens_all'])
        all_500.extend(b['completion_tokens_max500'])
        all_800.extend(b['completion_tokens_max800'])
        total_req += b['requests']
        total_chat += b['chat_requests']
        total_non2xx += b['non_2xx']
        total_err += b['errors']
        backends_out[str(idx)] = {
            'requests': b['requests'],
            'chat_requests': b['chat_requests'],
            'non_2xx': b['non_2xx'],
            'errors': b['errors'],
            'busy_wall_s': round(merge_busy_seconds(b['intervals']), 3),
            'active_sum_s': round(b['duration_sum_s'], 3),
            'avg_completion_tokens': _avg(b['completion_tokens_all']),
            'avg_interaction_tokens': _avg(b['completion_tokens_max500']),
            'avg_final_tokens': _avg(b['completion_tokens_max800']),
        }

    return {
        'total_requests': total_req,
        'total_chat_requests': total_chat,
        'non_2xx': total_non2xx,
        'errors': total_err,
        'combined_busy_wall_s': round(merge_busy_seconds(intervals_all), 3),
        'avg_completion_tokens': _avg(all_toks),
        'avg_interaction_tokens': _avg(all_500),
        'avg_final_tokens': _avg(all_800),
        'backends': backends_out,
    }


class ProxyGroup:
    """Async HTTP reverse proxy with least-inflight load balancing."""

    def __init__(self, backends: list[str], log_path: Path):
        self.backends = [b.rstrip('/') for b in backends]
        self.inflight = [0] * len(self.backends)
        self.rr_cursor = 0
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open('a', encoding='ascii', buffering=1)
        self.session = None

    def pick_backend(self) -> int:
        n = len(self.backends)
        best_idx = self.rr_cursor % n
        best_load = self.inflight[best_idx]
        for offset in range(1, n):
            cand = (self.rr_cursor + offset) % n
            if self.inflight[cand] < best_load:
                best_idx = cand
                best_load = self.inflight[cand]
        self.rr_cursor = (best_idx + 1) % n
        return best_idx

    async def handle(self, request):
        from aiohttp import web

        path = request.raw_path
        is_chat = int(request.method == 'POST' and path.startswith('/v1/chat/completions'))
        body = await request.read()
        max_tok = None
        if is_chat and body:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict) and isinstance(payload.get('max_tokens'), int):
                    max_tok = payload['max_tokens']
            except ValueError:
                pass

        idx = self.pick_backend()
        target = self.backends[idx] + path
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ('host', 'content-length', 'transfer-encoding', 'connection')
        }

        self.inflight[idx] += 1
        t_start = time.time()
        status = 0
        comp_tok = None
        try:
            async with self.session.request(
                request.method, target, headers=headers, data=body, allow_redirects=False
            ) as resp:
                status = resp.status
                resp_body = await resp.read()
                if is_chat and 200 <= status < 300 and resp_body:
                    try:
                        parsed = json.loads(resp_body)
                        usage = parsed.get('usage') if isinstance(parsed, dict) else None
                        if isinstance(usage, dict) and isinstance(usage.get('completion_tokens'), int):
                            comp_tok = usage['completion_tokens']
                    except ValueError:
                        pass
                resp_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ('content-length', 'transfer-encoding', 'connection')
                }
                return web.Response(status=status, body=resp_body, headers=resp_headers)
        except Exception:
            return web.Response(status=502, text='upstream proxy error')
        finally:
            t_end = time.time()
            self.inflight[idx] -= 1
            if is_chat or path.startswith('/v1/'):
                row = [round(t_start, 6), round(t_end, 6), idx, status, is_chat, comp_tok, max_tok]
                self.log_file.write(json.dumps(row, separators=(',', ':')) + '\n')


async def serve(student_port: int, student_backends: list[str], judge_port: int, judge_backends: list[str], work: Path):
    import aiohttp
    from aiohttp import web

    connector = aiohttp.TCPConnector(limit=1024, limit_per_host=512)
    timeout = aiohttp.ClientTimeout(total=600)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False)

    student_group = ProxyGroup(student_backends, work / 'proxy-student.jsonl')
    judge_group = ProxyGroup(judge_backends, work / 'proxy-judge.jsonl')
    student_group.session = session
    judge_group.session = session

    async def start_site(port, handler):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route('*', '/{tail:.*}', handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', port, backlog=512)
        await site.start()
        return runner

    runners = [
        await start_site(student_port, student_group.handle),
        await start_site(judge_port, judge_group.handle),
    ]
    print(f'SERVING_PROXY_READY student={student_port}->{student_backends} judge={judge_port}->{judge_backends}', flush=True)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)
    try:
        await stop_event.wait()
    finally:
        for r in runners:
            await r.cleanup()
        await session.close()
        student_group.log_file.close()
        judge_group.log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--student-port', type=int, required=True)
    parser.add_argument('--student-backends', nargs='+', required=True)
    parser.add_argument('--judge-port', type=int, required=True)
    parser.add_argument('--judge-backends', nargs='+', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    asyncio.run(serve(args.student_port, args.student_backends, args.judge_port, args.judge_backends, args.work))


if __name__ == '__main__':
    main()
