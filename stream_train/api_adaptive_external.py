"""API-only producer with adjustable AIMD, retaining the frozen run protocol.

Run only after the prior producer has safely drained and released api.lock.
The original pipeline, database protocol and started/unknown records are kept.
"""
import argparse
from collections import Counter
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time
from urllib.parse import quote
import uuid

from stream_train import __main__ as app
from stream_train.store import Store, encode


def event(name, **fields):
    # Values must be fixed categories and aggregate numbers only.
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'event': name, **fields}, ensure_ascii=False), flush=True)


class Window(app.AdaptiveAPIWindow):
    def __init__(self, maximum):
        if not 1 <= maximum <= 32:
            raise ValueError('maximum must be between 1 and 32')
        super().__init__(maximum=min(5, maximum))
        self.maximum = maximum
        self.closed = False

    def acquire(self):
        with self.condition:
            while not self.closed and self.stop_reason is None:
                delay = self.blocked_until - time.monotonic()
                if delay > 0:
                    self.condition.wait(timeout=delay)
                elif self.inflight < self.limit:
                    self.inflight += 1
                    return self.epoch
                else:
                    self.condition.wait()
            return None

    def report(self, lease, outcome):
        with self.condition:
            before = self.limit
            super().report(lease, outcome)
            if before != self.limit or outcome in ('congestion', 'protocol', 'auth', 'model'):
                event('window', reason=outcome, before=before, current=self.limit,
                      maximum=self.maximum, inflight=self.inflight,
                      circuit=self.stop_reason or 'none')

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def read_protocol(db_path):
    """Refuse missing databases without creating files; see the live WAL."""
    if not db_path.is_absolute() or db_path.is_symlink() or not db_path.is_file():
        raise ValueError('existing absolute database is required')
    uri = 'file:' + quote(str(db_path), safe='/') + '?mode=ro'
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
        if not row:
            raise ValueError('protocol missing')
        return json.loads(row[0])
    finally:
        conn.close()


def active_counts(store):
    counts = Counter()
    for row in store.counts():
        kind, state, count = row['kind'], row['state'], row['count']
        if kind not in ('generate', 'review', 'equivalence', 'student', 'grade'):
            raise ValueError('unknown job kind')
        if state not in ('pending', 'started', 'done', 'rejected', 'unknown'):
            raise ValueError('unknown job state')
        counts[state] += count
    if not sum(counts.values()):
        raise ValueError('empty queue is not complete')
    return counts


def announce_ready():
    """Notify an optional handoff parent after binding, lock and durable receipt."""
    raw = os.environ.pop('STREAM_PRODUCER_READY_FD', None)
    if raw is None:
        return
    try:
        fd = int(raw)
    except ValueError:
        raise ValueError('invalid ready descriptor') from None
    if fd <= 2:
        raise ValueError('invalid ready descriptor')
    try:
        if os.write(fd, b'R') != 1:
            raise OSError('ready notification failed')
    finally:
        os.close(fd)


def run(db_path, config_path, maximum, policy_path, idle_seconds=120, poll_seconds=15):
    if not 1 <= maximum <= 32 or idle_seconds < 60 or poll_seconds <= 0:
        raise ValueError('invalid limits')
    if not config_path.is_absolute() or not policy_path.is_absolute():
        raise ValueError('absolute configuration and policy paths required')
    cfg = app.config(config_path)
    if read_protocol(db_path) != cfg:
        raise ValueError('frozen protocol mismatch')
    key = os.environ.get('AGNES_API_KEY', '').strip()
    if not key:
        raise ValueError('missing API key')

    lock_path = Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    window = Window(maximum)
    failure = []
    stop = threading.Event()
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Bind through original code, without changing the immutable metadata.
        store = Store(db_path)
        store.bind('protocol', cfg)
        stat = db_path.stat()
        policy = {'policy': 'adaptive_external_v1', 'maximum': maximum, 'initial': 1,
                  'queue_device': stat.st_dev, 'queue_inode': stat.st_ino,
                  'protocol_sha256': hashlib.sha256(encode(cfg).encode()).hexdigest(),
                  'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'utc_started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        # Never overwrite an existing policy or queue file.
        with policy_path.open('x', encoding='utf-8') as f:
            os.chmod(policy_path, 0o600)
            json.dump(policy, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())

        def worker():
            while not stop.is_set():
                lease = window.acquire()
                if lease is None:
                    return
                try:
                    found = app.run_one(store, cfg, key,
                                        ['generate', 'review', 'equivalence', 'grade'],
                                        observer=lambda outcome: window.report(lease, outcome))
                except Exception as exc:
                    failure.append(type(exc).__name__)
                    event('worker_error', category='internal_error')
                    stop.set()
                    window.close()
                    return
                finally:
                    window.release()
                if not found:
                    stop.wait(2)

        def request_stop(signum, frame):
            event('stop_requested', category='signal')
            stop.set()
            window.close()

        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, request_stop)
        event('start', maximum=maximum, current=window.limit)
        stable_since = None
        try:
            # The parent only sees R after protocol validation, obtaining the
            # exclusive API lock and fsync of the independent policy receipt.
            # Notify before workers claim any new tasks.
            announce_ready()
            with concurrent.futures.ThreadPoolExecutor(max_workers=maximum) as pool:
                futures = [pool.submit(worker) for _ in range(maximum)]
                try:
                    while not stop.wait(poll_seconds):
                        if window.stop_reason:
                            event('circuit', reason=window.stop_reason)
                            stop.set()
                            window.close()
                            break
                        counts = active_counts(store)
                        with window.condition:
                            inflight, limit = window.inflight, window.limit
                        event('progress', pending=counts['pending'], started=counts['started'],
                              unknown=counts['unknown'], rejected=counts['rejected'],
                              inflight=inflight, current=limit, maximum=window.maximum)
                        if not counts['pending'] and not counts['started'] and not inflight:
                            if stable_since is None:
                                stable_since = time.monotonic()
                            elif time.monotonic() - stable_since >= idle_seconds:
                                event('queue_drained', unknown=counts['unknown'])
                                stop.set()
                                window.close()
                                break
                        else:
                            stable_since = None
                except Exception:
                    # Close before the executor waits, or idle workers never exit.
                    failure.append('monitor_error')
                    event('monitor_error', category='internal_error')
                    stop.set()
                    window.close()
                for future in futures:
                    future.result()
        finally:
            stop.set()
            window.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if failure or window.stop_reason:
            return 1
        return 0


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-concurrency', '--maximum', dest='maximum', type=int, default=20)
    parser.add_argument('--policy-output', help='optional unique policy receipt path')
    args = parser.parse_args()
    try:
        policy_path = (Path(args.policy_output) if args.policy_output else
                       Path(args.db).parent / ('api-policy-' + uuid.uuid4().hex + '.json'))
        return run(Path(args.db), Path(args.config), args.maximum, policy_path)
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        event('startup_or_monitor_error', category=type(exc).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
