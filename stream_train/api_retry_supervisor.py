"""Supervise the existing API producer with bounded backoff and synthetic probes.

Only synthetic problems are sent during probes. Existing started/unknown work
is never requeued. The original code and formal queue binding remain intact.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import quote

from stream_train import __main__ as app
from stream_train import api_adaptive_external as producer

API_KINDS = ('generate', 'review', 'equivalence', 'grade')
STATES = ('pending', 'started', 'done', 'rejected', 'unknown')
KINDS = API_KINDS + ('student',)


def emit(event, **values):
    # All values are fixed labels or aggregate numbers, never API bodies.
    print(json.dumps({'event': event, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **values}, ensure_ascii=False), flush=True)


def read_counts(db_path):
    db_path = Path(db_path)
    if not db_path.is_absolute() or db_path.is_symlink() or not db_path.is_file():
        raise ValueError('invalid database')
    connection = sqlite3.connect('file:' + quote(str(db_path), safe='/') + '?mode=ro', uri=True, timeout=5)
    try:
        rows = connection.execute('SELECT kind,state,COUNT(*) FROM jobs GROUP BY kind,state').fetchall()
    finally:
        connection.close()
    if not rows or any(k not in KINDS or s not in STATES or n < 0 for k, s, n in rows):
        raise ValueError('invalid queue state')
    counts = Counter()
    for kind, state, n in rows:
        counts[state] += n
    return counts


def is_drained(counts):
    return not counts['pending'] and not counts['started']


def probe(config_path, key, stop, lock_fd, timeout=600):
    """Run the independent synthetic-only contract probe in a child process."""
    child_env = os.environ.copy()
    child_env['AGNES_API_KEY'] = key
    proc = subprocess.Popen([sys.executable, '-m', 'stream_train.agnes_protocol_probe',
                             '--config', str(config_path), '--timeout', '120'],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=child_env, pass_fds=(lock_fd,))
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if stop.wait(1) or time.monotonic() >= deadline:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return False, 'probe_timeout_or_stop', 'other'
    raw = proc.stdout.read(65536) if proc.stdout else b''
    if proc.stdout:
        proc.stdout.close()
    stage = 'unknown'
    category = 'other'
    for line in raw.decode('utf-8', 'replace').splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if (isinstance(item, dict) and item.get('status') == 'failed' and
                item.get('stage') in ('generate', 'review', 'equivalence', 'grade', 'configuration')):
            stage = item['stage']
            value = item.get('category')
            if value in ('auth', 'model', 'protocol', 'congestion', 'neutral', 'configuration'):
                category = value
    return (True, 'ok', 'none') if proc.returncode == 0 else (False, stage, category)


def api_lock_path():
    return Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'


def acquire_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) |
                 getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        stat = os.fstat(fd)
        import stat as stat_module
        if not stat_module.S_ISREG(stat.st_mode) or stat.st_uid != os.geteuid():
            raise ValueError('unsafe lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def try_probe(config_path, key, stop):
    """Use the normal single-producer lock during probes, not real requests in parallel."""
    try:
        fd = acquire_lock(api_lock_path())
    except BlockingIOError:
        return False, 'producer_active', 'none'
    try:
        return probe(config_path, key, stop, fd)
    finally:
        os.close(fd)


def retry_wait(delay, maximum_wait, elapsed):
    return max(0, min(delay, maximum_wait) - max(0, elapsed))


def supervise(db, config, maximum, adaptive_log, stop, minimum_wait=60, max_wait=3600,
              poll_seconds=15, stable_seconds=120, key_file=None):
    db, config, adaptive_log = Path(db), Path(config), Path(adaptive_log)
    if not all(p.is_absolute() for p in (db, config, adaptive_log)):
        raise ValueError('absolute paths required')
    if not 1 <= maximum <= 32 or not 1 <= minimum_wait <= max_wait <= 3600:
        raise ValueError('invalid timing or concurrency')
    cfg = app.config(config)
    if producer.read_protocol(db) != cfg:
        raise ValueError('frozen protocol mismatch')
    if key_file is not None and not Path(key_file).is_absolute():
        raise ValueError('absolute key file required')
    st = db.stat()
    identity = (st.st_dev, st.st_ino)
    lock = acquire_lock(db.parent / ('.' + db.name + '.api-retry-supervisor.lock'))
    child = None
    delay = minimum_wait
    stable = None
    try:
        emit('supervisor_started', maximum=maximum)
        while not stop.is_set():
            try:
                now_st = db.stat()
                if (now_st.st_dev, now_st.st_ino) != identity or producer.read_protocol(db) != cfg:
                    # Configuration or DB replacement is not safe to operate on.
                    emit('binding_changed')
                    return 1
                counts = read_counts(db)
                if is_drained(counts):
                    if stable is None:
                        stable = time.monotonic()
                    elif time.monotonic() - stable >= stable_seconds:
                        emit('queue_drained', unknown=counts['unknown'])
                        return 0
                    stop.wait(poll_seconds)
                    continue
                stable = None
                key = (Path(key_file).read_text(encoding='utf-8').strip() if key_file is not None
                       else os.environ.get('AGNES_API_KEY', '').strip())
                if not key:
                    raise ValueError('missing key')
                probe_started = time.monotonic()
                ok, stage, category = try_probe(config, key, stop)
                if not ok:
                    if stage == 'producer_active':
                        emit('producer_already_active', next_wait_seconds=poll_seconds)
                        stop.wait(poll_seconds)
                    else:
                        elapsed = max(0, time.monotonic() - probe_started)
                        remaining = retry_wait(delay, max_wait, elapsed)
                        emit('probe_failed', stage=stage, category=category, next_wait_seconds=int(remaining))
                        stop.wait(remaining)
                        delay = min(max_wait, delay * 2)
                    continue
                if stop.is_set():
                    break
                started_at = time.monotonic()
                with adaptive_log.open('ab', buffering=0) as out:
                    child_env = os.environ.copy()
                    child_env['AGNES_API_KEY'] = key
                    child = subprocess.Popen([sys.executable, '-m', 'stream_train.api_adaptive_external',
                        '--db', str(db), '--config', str(config), '--max-concurrency', str(maximum)],
                        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=child_env)
                emit('producer_spawned', maximum=maximum)
                while child.poll() is None and not stop.wait(poll_seconds):
                    pass
                if stop.is_set() and child.poll() is None:
                    child.terminate()
                    # Producer drains existing requests. No SIGKILL or automatic resubmission.
                    child.wait()
                code = child.wait()
                child = None
                if stop.is_set():
                    break
                counts = read_counts(db)
                if code == 0 and is_drained(counts):
                    stable = time.monotonic()
                    continue
                if time.monotonic() - started_at >= 900:
                    delay = minimum_wait
                emit('producer_exited', exit_code=code, next_wait_seconds=delay,
                     pending=counts['pending'], started=counts['started'], unknown=counts['unknown'])
                stop.wait(delay)
                delay = min(max_wait, delay * 2)
            except (OSError, sqlite3.Error, ValueError, TypeError):
                emit('temporary_failure', next_wait_seconds=delay)
                stop.wait(delay)
                delay = min(max_wait, delay * 2)
        emit('supervisor_stopped')
        return 0
    finally:
        # If interrupted during monitor logic, do not kill an in-flight child.
        # A remaining child still holds the producer lock, preventing duplicates.
        os.close(lock)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--adaptive-log', required=True)
    parser.add_argument('--max-concurrency', type=int, default=8)
    parser.add_argument('--key-file', help='reread the key file before each retry')
    args = parser.parse_args()
    stop = threading.Event()
    def on_signal(signum, frame):
        stop.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    try:
        return supervise(args.db, args.config, args.max_concurrency, args.adaptive_log, stop, key_file=args.key_file)
    except (OSError, sqlite3.Error, ValueError, TypeError):
        emit('startup_failure')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
