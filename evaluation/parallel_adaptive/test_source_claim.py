import multiprocessing
import tempfile
import unittest
from pathlib import Path

from source_claim import claim_first


def child_claim(root, ready, release, output):
    rows = [{'source_id': 'a'}, {'source_id': 'b'}]
    with claim_first(Path(root), 'EASY', rows, lambda row: True) as claim:
        output.put(claim.row['source_id'])
        ready.set()
        release.wait(5)


class SourceClaimTests(unittest.TestCase):
    def test_competing_processes_skip_locked_and_preserve_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context('spawn')
            ready, release, output = ctx.Event(), ctx.Event(), ctx.Queue()
            worker = ctx.Process(target=child_claim, args=(tmp, ready, release, output))
            worker.start()
            try:
                self.assertTrue(ready.wait(5))
                self.assertEqual(output.get(timeout=2), 'a')
                rows = [{'source_id': 'a'}, {'source_id': 'b'}]
                with claim_first(Path(tmp), 'EASY', rows, lambda row: True) as claim:
                    self.assertEqual(claim.row['source_id'], 'b')
                    self.assertIsNone(claim_first(Path(tmp), 'EASY', rows, lambda row: True))
                with claim_first(Path(tmp), 'HARD', rows, lambda row: True) as claim:
                    self.assertEqual(claim.row['source_id'], 'a')
            finally:
                release.set()
                worker.join(5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            self.assertEqual(worker.exitcode, 0)

    def test_source_state_rechecked_after_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{'source_id': 'a'}, {'source_id': 'b'}]
            completed = set()
            with claim_first(Path(tmp), 'MEDIUM', rows, lambda row: row['source_id'] not in completed) as claim:
                completed.add(claim.row['source_id'])
            with claim_first(Path(tmp), 'MEDIUM', rows, lambda row: row['source_id'] not in completed) as claim:
                self.assertEqual(claim.row['source_id'], 'b')

    def test_failed_inspection_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            def broken(_row):
                raise ValueError('invalid source')
            with self.assertRaises(ValueError):
                claim_first(Path(tmp), 'HARD', [{'source_id': 'a'}], broken)
            with claim_first(Path(tmp), 'HARD', [{'source_id': 'a'}], lambda _row: True) as claim:
                self.assertEqual(claim.row['source_id'], 'a')

    def test_crashed_worker_does_not_make_partial_source_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context('spawn')
            ready, release, output = ctx.Event(), ctx.Event(), ctx.Queue()
            worker = ctx.Process(target=child_claim, args=(tmp, ready, release, output))
            worker.start()
            try:
                self.assertTrue(ready.wait(5))
                self.assertEqual(output.get(timeout=2), 'a')
                worker.terminate()
                worker.join(5)
                self.assertFalse(worker.is_alive())
                rows = [{'source_id': 'a'}, {'source_id': 'b'}]
                # The caller's durable-state inspection rejects 'a' as partial.
                with claim_first(Path(tmp), 'EASY', rows, lambda row: row['source_id'] != 'a') as claim:
                    self.assertEqual(claim.row['source_id'], 'b')
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()


if __name__ == '__main__':
    unittest.main()
