"""Measure author client latency inside Ray without changing request policy."""
from __future__ import annotations

from collections import defaultdict
from functools import wraps
import importlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import weakref


LABELS = ('student_interaction', 'student_final', 'judge', 'hybrid_grader')
ENV = 'EDU_QUEUE_METRICS_DIR'
MAX_CALLS_PER_LABEL_PER_WORKER = 262144
AUTHOR_ALIASES = ('eduardo.eduardo_interaction', 'eduardo.metrics.correctness',
                  'eduardo.metrics.judge_quality')
_CUSTOM_SINK_LOCK = threading.Lock()
_CUSTOM_SINKS: dict[str, '_WorkerSink'] = {}


class _Timing:
    def __init__(self, clock):
        self.clock = clock
        self.queued_ns = 0
        self.held_ns = 0
        self.service_ns = 0
        self.attempts = 0
        self.attempt_errors = 0
        self.fallback = 0
        self.call_error = 0


class _TimedSemaphore:
    """Delegate to the same semaphore so ordering and slot ownership stay intact."""

    def __init__(self, inner, timing):
        self.inner = inner
        self.timing = timing

    async def __aenter__(self):
        start = self.timing.clock()
        try:
            result = await self.inner.__aenter__()
        finally:
            self.timing.queued_ns = self.timing.clock() - start
        self.acquired = self.timing.clock()
        return result

    async def __aexit__(self, *exc):
        try:
            return await self.inner.__aexit__(*exc)
        finally:
            self.timing.held_ns = self.timing.clock() - self.acquired


class _WorkerSink:
    """Bound local numeric samples. Never add an exception to the request path."""

    def __init__(self, directory, limit):
        self.path = directory / f'queue-{os.getpid()}-{time.monotonic_ns()}.jsonl'
        self.limit = limit
        self.counts = defaultdict(int)
        self.file = None
        self.disabled = False

    def admit(self, label):
        if self.disabled or self.counts[label] >= self.limit:
            return False
        self.counts[label] += 1
        return True

    def write(self, label, start, end, timing):
        if self.disabled:
            return
        row = [label, start, end, timing.queued_ns, timing.service_ns, timing.held_ns,
               timing.attempts, timing.attempt_errors, timing.fallback, timing.call_error]
        try:
            if self.file is None:
                self.file = self.path.open('x', encoding='ascii', buffering=1)
                weakref.finalize(self, self.file.close)
            self.file.write(json.dumps(row, separators=(',', ':')) + '\n')
        except OSError:
            self.disabled = True


def instrument(original, directory, *, clock=time.monotonic_ns, limit=MAX_CALLS_PER_LABEL_PER_WORKER):
    sink = _WorkerSink(Path(directory), limit)

    @wraps(original)
    async def timed(fn, *args, **kwargs):
        label = kwargs.get('label')
        if label not in LABELS or not sink.admit(label):
            return await original(fn, *args, **kwargs)

        timing = _Timing(clock)
        start = clock()
        semaphore = kwargs.get('semaphore')
        if semaphore is not None:
            kwargs = dict(kwargs, semaphore=_TimedSemaphore(semaphore, timing))

        @wraps(fn)
        async def attempt(*fn_args, **fn_kwargs):
            attempt_start = clock()
            timing.attempts += 1
            try:
                return await fn(*fn_args, **fn_kwargs)
            except BaseException:
                timing.attempt_errors += 1
                raise
            finally:
                timing.service_ns += clock() - attempt_start

        try:
            result = await original(attempt, *args, **kwargs)
            if kwargs.get('use_fallback') and result is kwargs.get('fallback'):
                timing.fallback = 1
            return result
        except BaseException:
            timing.call_error = 1
            raise
        finally:
            sink.write(label, start, clock(), timing)

    timed._eduardo_queue_timing = True
    return timed


def install():
    """Ray worker-process setup hook for the isolated candidate only."""
    directory = Path(os.environ[ENV])
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise RuntimeError('Invalid private EDU queue metrics directory')
    author = importlib.import_module('eduardo.utils')
    original = author.call_openai_with_retries
    if getattr(original, '_eduardo_queue_timing', False):
        return
    wrapped = instrument(original, directory)
    author.call_openai_with_retries = wrapped
    # Cover exact aliases if they were loaded before the Ray setup hook.
    for name in AUTHOR_ALIASES:
        caller = sys.modules.get(name)
        if caller is not None and getattr(caller, 'call_openai_with_retries', None) is original:
            caller.call_openai_with_retries = wrapped


def record_custom_timing(
    label: str,
    *,
    start_ns: int,
    end_ns: int,
    queued_ns: int = 0,
    service_ns: int = 0,
    held_ns: int = 0,
    attempts: int = 1,
    attempt_errors: int = 0,
    fallback: int = 0,
    call_error: int = 0,
) -> None:
    """Record a single non-author request timing sample when EDU_QUEUE_METRICS_DIR is set."""
    if label not in LABELS:
        return
    raw_dir = os.environ.get(ENV, '').strip()
    if not raw_dir:
        return
    try:
        directory = Path(raw_dir)
        if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
            return
        with _CUSTOM_SINK_LOCK:
            sink = _CUSTOM_SINKS.get(raw_dir)
            if sink is None:
                sink = _WorkerSink(directory, MAX_CALLS_PER_LABEL_PER_WORKER)
                _CUSTOM_SINKS[raw_dir] = sink
            if not sink.admit(label):
                return
            timing = _Timing(time.monotonic_ns)
            timing.queued_ns = max(0, int(queued_ns))
            timing.service_ns = max(0, int(service_ns))
            timing.held_ns = max(0, int(held_ns))
            timing.attempts = max(0, int(attempts))
            timing.attempt_errors = max(0, int(attempt_errors))
            timing.fallback = 1 if fallback else 0
            timing.call_error = 1 if call_error else 0
            sink.write(label, max(0, int(start_ns)), max(int(start_ns), int(end_ns)), timing)
    except Exception:
        return


def _percentile_ms(samples, rank):
    if not samples:
        return None
    ix = math.ceil(len(samples) * rank) - 1
    return round(sorted(samples)[ix] / 1e6, 3)


def _valid(row):
    return (isinstance(row, list) and len(row) == 10 and row[0] in LABELS
            and all(type(v) is int and v >= 0 for v in row[1:])
            and row[1] <= row[2] and row[7] <= row[6] and row[8] <= 1 and row[9] <= 1)


def summarize(directory, start_ns, end_ns):
    """Aggregate one local rollout window. Time values are milliseconds.

    Wait is time before the author semaphore acquires. Service totals the time
    spent awaiting OpenAI calls for each logical request, excluding explicit
    retry backoff. Held time includes backoff and applies when a slot is used.
    """
    if start_ns < 0 or end_ns < start_ns:
        raise ValueError('Invalid queue timing window')
    grouped = {label: [] for label in LABELS}
    bad = missing = crossing = 0
    for path in sorted(Path(directory).glob('queue-*.jsonl')):
        if path.is_symlink():
            missing += 1
            continue
        try:
            with path.open('r', encoding='ascii') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    if not _valid(row):
                        bad += 1
                        continue
                    if start_ns <= row[1] < end_ns:
                        if row[2] <= end_ns:
                            grouped[row[0]].append(row[3:])
                        else:
                            crossing += 1
        except (OSError, UnicodeError):
            missing += 1
    metrics = {}
    for label, rows in grouped.items():
        data = {
            'calls': len(rows), 'attempts': sum(r[3] for r in rows),
            'attempt_errors': sum(r[4] for r in rows), 'fallbacks': sum(r[5] for r in rows),
            'call_errors': sum(r[6] for r in rows),
        }
        for i, name in enumerate(('wait', 'service', 'held')):
            samples = [r[i] for r in rows if i == 0 or r[3] > 0]
            data[name + '_p50_ms'] = _percentile_ms(samples, 0.5)
            data[name + '_p95_ms'] = _percentile_ms(samples, 0.95)
        metrics[label] = data
    return {'groups': metrics, 'malformed_rows': bad, 'unreadable_files': missing,
            'crossing_window': crossing}
