"""Durable local queue. One local filesystem and SQLite database per run."""
import json
import sqlite3
import time
from pathlib import Path


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

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute('PRAGMA busy_timeout=30000')
        db.execute('PRAGMA synchronous=FULL')
        return db

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
