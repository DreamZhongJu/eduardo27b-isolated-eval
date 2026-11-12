"""Measure a local vLLM student with synthetic prompts; never submit experiment data.

This is an opt-in diagnostic. It does not use the durable queue or Agnes, launch
models, allocate GPUs, print model output, or change the 16-sample protocol.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
import time
import unicodedata
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .pipeline import PARAMS, STUDENT_PROMPT


def endpoint(base):
    """Allow only direct HTTP requests to a loopback service, with no redirects."""
    parts = urlsplit(base)
    if (parts.scheme != 'http' or parts.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError('学生服务必须使用本机 HTTP 地址')
    return base.rstrip('/') + '/chat/completions'


def payload(model, index, max_tokens):
    # Synthetic questions are each repeated 16 times, mimicking the real
    # request grouping. Levels use the same prompt and seed set.
    group = index // 16
    problem = ('A rectangle has side lengths %s and %s. Find its perimeter '
               'and area; show the arithmetic.' % (group + 7, group + 11))
    prompt = STUDENT_PROMPT.format('{}', problem)
    return {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            **PARAMS, 'max_tokens': max_tokens, 'seed': 20261113 + index, 'n': 1}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def one_request(url, model, index, max_tokens, timeout, key):
    start = time.monotonic()
    try:
        request = urllib.request.Request(
            url, data=json.dumps(payload(model, index, max_tokens)).encode('utf8'),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
            method='POST')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            data = json.load(response)
        if data.get('model') != model:
            return {'ok': False, 'reason': 'model_mismatch'}
        choices = data.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            return {'ok': False, 'reason': 'invalid_choice'}
        choice = choices[0]
        message = choice.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        finish = choice.get('finish_reason')
        if not isinstance(content, str) or not content.strip() or finish not in ('stop', 'length'):
            return {'ok': False, 'reason': 'invalid_response'}
        usage = data.get('usage')
        tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens < 0:
            tokens = None
        return {'ok': True, 'seconds': time.monotonic() - start,
                'tokens': tokens, 'truncated': finish == 'length'}
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'reason': 'http_%d' % exc.code}
    except Exception:
        # Never output exceptions, bodies, keys, local paths, or responses.
        return {'ok': False, 'reason': 'connection_or_parse_error'}


def run_level(url, model, workers, requests, max_tokens, timeout, key):
    if not 1 <= workers <= 64 or requests < workers:
        raise ValueError('每档请求数必须至少等于并发数，且并发不超过 64')
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        results = []
        submitted = 0
        failed = False
        while pending or (submitted < requests and not failed):
            while len(pending) < workers and submitted < requests and not failed:
                pending.add(executor.submit(one_request, url, model, submitted,
                                            max_tokens, timeout, key))
                submitted += 1
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                failed = failed or not result['ok']
            # On the first failure, no further requests are submitted. Allow
            # already-submitted calls to finish; server-side work may outlive
            # a client timeout and must be drained before another benchmark.
    elapsed = time.monotonic() - start
    good = [r for r in results if r['ok']]
    durations = sorted(r['seconds'] for r in good)
    def percentile(q):
        if not durations:
            return None
        position = q * (len(durations) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return durations[lo] + (durations[hi] - durations[lo]) * (position - lo)
    token_values = [r['tokens'] for r in good]
    tokens = sum(token_values) if good and all(t is not None for t in token_values) else None
    return {'workers': workers, 'requested': requests, 'submitted': submitted,
            'unsubmitted': requests - submitted, 'succeeded': len(good),
            'failed': len(results) - len(good), 'truncated': sum(r['truncated'] for r in good),
            'failure_reasons': dict(Counter(r['reason'] for r in results if not r['ok'])),
            'seconds': elapsed, 'samples_s': len(good) / elapsed if elapsed else 0,
            'tokens_s': tokens / elapsed if tokens is not None and elapsed else None,
            'mean_output_tokens': tokens / len(good) if tokens is not None and good else None,
            'p50_s': percentile(.50), 'p95_s': percentile(.95)}


def table(rows):
    headers = ('并发', '成功/提交/计划', '失败', '未提交', '失败类型', '截断',
               '平均输出token', '作答/秒', '输出token/秒', 'P50秒', 'P95秒')
    def number(value):
        return '不可用' if value is None else '%.2f' % value
    data = [headers] + [(str(r['workers']), '%d/%d/%d' % (r['succeeded'], r['submitted'], r['requested']),
                         str(r['failed']), str(r['unsubmitted']),
                         ', '.join('%s:%s' % item for item in sorted(r['failure_reasons'].items())) or '-',
                         str(r['truncated']), number(r['mean_output_tokens']), number(r['samples_s']),
                         number(r['tokens_s']), number(r['p50_s']), number(r['p95_s']))
                        for r in rows]
    def width(value):
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in value)
    widths = [max(width(row[i]) for row in data) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    def line(row):
        return '| ' + ' | '.join(v + ' ' * (widths[i] - width(v)) for i, v in enumerate(row)) + ' |'
    return '\n'.join([border, line(data[0]), border] + [line(row) for row in data[1:]] + [border])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--student-model', required=True)
    parser.add_argument('--levels', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--requests', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if (not args.levels or len(set(args.levels)) != len(args.levels)
            or args.requests > 1024 or not 1 <= args.max_tokens <= PARAMS['max_tokens']
            or args.timeout <= 0):
        parser.error('无效的并发、请求数、输出上限或超时')
    try:
        url = endpoint(args.student_base)
        if any(not 1 <= level <= 64 or args.requests < level for level in args.levels):
            raise ValueError('每档请求数必须至少等于并发数，且并发不超过 64')
    except ValueError as exc:
        parser.error(str(exc))
    key = os.environ.get('STUDENT_API_KEY', 'local')
    print('本机虚构题吞吐测试：不使用实验题目、数据库或 Agnes。', flush=True)
    print('测试会占用学生服务，请不要与正式测量同时运行。', flush=True)
    print('每档计划 %d 个请求，共 %d 档，另有最多 4 次预热；可能耗时。' %
          (args.requests, len(args.levels)), flush=True)
    print('输出上限 %d；正式配置为 %d。合成短题可能提前结束，即使上限一致也不能代表正式吞吐。' %
          (args.max_tokens, PARAMS['max_tokens']), flush=True)
    # Warm four independent question prefixes before measurement. Warmup is
    # reported only as a success count and never included in measured timing.
    warm = []
    for i in (0, 16, 32, 48):
        result = one_request(url, args.student_model, i, args.max_tokens, args.timeout, key)
        warm.append(result)
        if not result['ok']:
            print('预热失败：%s；已停止，请确认学生服务的在途请求排空后再重试。' % result['reason'])
            return 1
    rows = []
    for level in args.levels:
        print('正在测试并发 %d ...' % level, flush=True)
        row = run_level(url, args.student_model, level, args.requests,
                        args.max_tokens, args.timeout, key)
        rows.append(row)
        if row['failed']:
            print('该并发档出现失败，已停止提交及后续档位；请确认学生服务的在途请求排空后再重试。', flush=True)
            break
    print(table(rows))
    print('这是合成短题的单次测量；P50/P95 为已提交请求耗时，不含业务队列等待。')
    print('请结合平均输出 token、正式任务输出长度、完成率、显存和排队情况选择并发。')
    return 1 if any(r['failed'] for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
