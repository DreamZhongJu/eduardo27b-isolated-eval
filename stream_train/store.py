"""Durable local queue. One local filesystem and SQLite database per run."""
from contextlib import contextmanager
import json
import sqlite3
import time
from pathlib import Path

# One candidate normally produces 16 sample judgments.  The saturated-cycle
# ratio below is a theoretical starting point, not a measured optimum; real
# request latencies and rejection rates must be benchmarked at the deployment.
# Empty kinds are skipped, so no API worker waits while other work is pending.
_API_CYCLE = ('grade',) * 16 + ('review', 'equivalence', 'generate')
_API_KINDS = frozenset(_API_CYCLE)
_API_CURSOR_KEY = '__api_fair_cursor_v1__'


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class Store:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL, result TEXT, created REAL NOT NULL, updated REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS by_state ON jobs(kind,state,created);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute('PRAGMA busy_timeout=30000')
            db.execute('PRAGMA synchronous=FULL')
            # sqlite's connection context manager commits or rolls back but
            # does not close. Preserve that transaction behavior and close it
            # even if a pragma, caller operation, or context exit fails.
            with db:
                yield db
        finally:
            db.close()

    def bind(self, key, value):
        value = encode(value)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key, value))
            row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
            if row[0] != value:
                db.rollback()
                raise ValueError('run binding changed: ' + key)
            db.commit()

    @staticmethod
    def enqueue_in(db, job_id, kind, payload):
        payload = encode(payload); now = time.time()
        row = db.execute('SELECT kind,payload FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row:
            if row != (kind, payload):
                raise ValueError('job identity collision: ' + job_id)
            return
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)',
                   (job_id, kind, payload, 'pending', None, now, now))

    def enqueue(self, job_id, kind, payload):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE'); self.enqueue_in(db, job_id, kind, payload); db.commit()

    def claim(self, kinds):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if frozenset(kinds) == _API_KINDS:
                # The cursor lives in the same database and transaction as
                # the claim. Multiple worker processes therefore share one
                # rotation, even when they happen to claim simultaneously.
                db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                           (_API_CURSOR_KEY, '0'))
                cursor = int(db.execute('SELECT value FROM metadata WHERE key=?',
                                        (_API_CURSOR_KEY,)).fetchone()[0])
                if not 0 <= cursor < len(_API_CYCLE):
                    raise ValueError('invalid API scheduler cursor')
                row = None
                tried = set()
                for offset in range(len(_API_CYCLE)):
                    index = (cursor + offset) % len(_API_CYCLE)
                    kind = _API_CYCLE[index]
                    if kind in tried:
                        continue
                    tried.add(kind)
                    row = db.execute(
                        "SELECT id,kind,payload FROM jobs WHERE state='pending' AND kind=? "
                        'ORDER BY created,id LIMIT 1', (kind,)).fetchone()
                    if row is not None:
                        db.execute('UPDATE metadata SET value=? WHERE key=?',
                                   (str((index + 1) % len(_API_CYCLE)), _API_CURSOR_KEY))
                        break
            else:
                marks = ','.join('?' for _ in kinds)
                row = db.execute('SELECT id,kind,payload FROM jobs WHERE state=? AND kind IN (' + marks + ') ORDER BY created,id LIMIT 1',
                                 ('pending', *kinds)).fetchone()
            if row:
                db.execute("UPDATE jobs SET state='started',updated=? WHERE id=?", (time.time(), row[0]))
            db.commit()
        return (row[0], row[1], json.loads(row[2])) if row else None

    def finish(self, job_id, result, children=(), state='done'):
        if state not in ('done', 'unknown', 'rejected'):
            raise ValueError('invalid terminal state')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if old != ('started',):
                raise ValueError('job is not started')
            for child in children:
                self.enqueue_in(db, *child)
            db.execute('UPDATE jobs SET state=?,result=?,updated=? WHERE id=?',
                       (state, encode(result), time.time(), job_id))
            db.commit()

    def counts(self):
        with self.connect() as db:
            rows = db.execute('SELECT kind,state,COUNT(*) FROM jobs GROUP BY kind,state ORDER BY kind,state').fetchall()
        return [{'kind': k, 'state': s, 'count': n} for k, s, n in rows]
