"""Tests for API queue fairness, durability and atomic claims."""
import concurrent.futures
import os
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path

from stream_train.store import Store


KINDS = ['generate', 'review', 'equivalence', 'grade']


def claim_batch(path, count):
    store = Store(path)
    return [store.claim(KINDS) for _ in range(count)]


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        # The test exercises SQLite transactions, not disk throughput. A
        # RAM-backed local directory avoids unnecessary fsync delays on shared
        # development hosts; retain the standard temp directory as fallback.
        ram = Path('/dev/shm')
        temp_dir = str(ram) if ram.is_dir() and os.access(ram, os.W_OK) else None
        self.tmp = tempfile.TemporaryDirectory(dir=temp_dir)
        self.path = Path(self.tmp.name) / 'queue.sqlite'
        self.store = Store(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def enqueue(self, kind, count):
        for i in range(count):
            self.store.enqueue(f'{kind}-{i:03}', kind, {'index': i})

    def test_later_review_and_grade_are_not_blocked_by_generation_backlog(self):
        self.enqueue('generate', 30)
        self.enqueue('review', 1)
        self.enqueue('grade', 1)
        got = [self.store.claim(KINDS)[1] for _ in range(3)]
        self.assertEqual(got, ['grade', 'review', 'generate'])

    def test_each_saturated_kind_receives_bounded_opportunities(self):
        for kind in KINDS:
            self.enqueue(kind, 20)
        got = [self.store.claim(KINDS)[1] for _ in range(19)]
        self.assertEqual(Counter(got), {'grade': 16, 'review': 1,
                                       'equivalence': 1, 'generate': 1})

    def test_empty_kinds_do_not_leave_workers_idle(self):
        self.enqueue('generate', 8)
        self.assertEqual([self.store.claim(KINDS)[1] for _ in range(8)],
                         ['generate'] * 8)
        self.assertIsNone(self.store.claim(KINDS))

    def test_cursor_survives_reopening_database(self):
        for kind in KINDS:
            self.enqueue(kind, 20)
        self.assertEqual(self.store.claim(KINDS)[1], 'grade')
        reopened = Store(self.path)
        got = [reopened.claim(KINDS)[1] for _ in range(18)]
        self.assertEqual(Counter(got), {'grade': 15, 'review': 1,
                                       'equivalence': 1, 'generate': 1})

    def test_started_and_unknown_are_not_reclaimed(self):
        self.enqueue('review', 2)
        first = self.store.claim(KINDS)
        self.store.finish(first[0], {'reason': 'uncertain'}, state='unknown')
        second = self.store.claim(KINDS)
        self.assertNotEqual(first[0], second[0])
        self.assertIsNone(Store(self.path).claim(KINDS))
        self.assertEqual(self.store.counts(), [
            {'kind': 'review', 'state': 'started', 'count': 1},
            {'kind': 'review', 'state': 'unknown', 'count': 1}])

    def test_concurrent_claims_are_unique_and_share_fair_cursor(self):
        self.enqueue('grade', 36)
        for kind in ('generate', 'review', 'equivalence'):
            self.enqueue(kind, 3)
        barrier = threading.Barrier(5, timeout=20)

        def worker():
            store = Store(self.path)
            barrier.wait()
            return [store.claim(KINDS) for _ in range(8)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            batches = list(pool.map(lambda _: worker(), range(5)))
        claimed = [job for batch in batches for job in batch]
        self.assertTrue(all(job is not None for job in claimed))
        self.assertEqual(len({job[0] for job in claimed}), 40)
        self.assertEqual(Counter(job[1] for job in claimed),
                         {'grade': 34, 'review': 2, 'equivalence': 2,
                          'generate': 2})

    def test_single_kind_student_queue_remains_fifo(self):
        self.enqueue('student', 3)
        self.assertEqual([self.store.claim(['student'])[0] for _ in range(3)],
                         ['student-000', 'student-001', 'student-002'])

    def test_multiple_processes_use_same_durable_rotation(self):
        self.enqueue('grade', 20)
        for kind in ('generate', 'review', 'equivalence'):
            self.enqueue(kind, 2)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(claim_batch, [str(self.path)] * 4, [5] * 4))
        claimed = [job for batch in batches for job in batch]
        self.assertTrue(all(job is not None for job in claimed))
        self.assertEqual(len({job[0] for job in claimed}), 20)
        self.assertEqual(Counter(job[1] for job in claimed),
                         {'grade': 17, 'review': 1, 'equivalence': 1,
                          'generate': 1})


if __name__ == '__main__':
    unittest.main()
