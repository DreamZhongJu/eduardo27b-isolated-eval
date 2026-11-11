import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import handoff_boundary as boundary


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.base = self.root / 'HARD' / 'sources'
        self.base.mkdir(parents=True)
        self.sid = 'synthetic-source'
        self.sha = 'a' * 64
        self.src = self.base / hashlib.sha256(self.sid.encode()).hexdigest()[:24]
        self.src.mkdir()
        self.write(self.src / 'SOURCE.json', {'source_id': self.sid, 'sha256': self.sha})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_source_only_is_quiescent(self):
        self.assertTrue(boundary.stage_quiescent(self.root, 'HARD'))

    def test_any_started_source_is_not_quiescent(self):
        (self.src / 'samples').mkdir()
        self.assertFalse(boundary.stage_quiescent(self.root, 'HARD'))

    def test_held_is_not_assumed_safe(self):
        self.write(self.src / 'SOURCE_HELD.json', {'status': 'held'})
        self.assertFalse(boundary.stage_quiescent(self.root, 'HARD'))

    def test_valid_completion_is_quiescent(self):
        self.write(self.src / 'SCORE_INPUT.json', [])
        self.write(self.src / 'RAW_COMPLETE.json',
                   {'source_id': self.sid, 'source_sha256': self.sha, 'samples': 96})
        self.assertTrue(boundary.stage_quiescent(self.root, 'HARD'))

    def test_bad_completion_refused(self):
        self.write(self.src / 'SCORE_INPUT.json', [])
        self.write(self.src / 'RAW_COMPLETE.json', {'source_id': self.sid, 'samples': 96})
        with self.assertRaises(ValueError):
            boundary.stage_quiescent(self.root, 'HARD')

    def test_grader_in_flight_blocks(self):
        r = self.root / 'receipts'
        r.mkdir()
        self.write(r / 'ATTEMPT-0001-INTENT.json', {'attempt': 1})
        self.assertFalse(boundary.grader_quiescent(r))
        self.write(r / 'ATTEMPT-0001-RESPONSE.json', {'attempt': 1})
        self.assertTrue(boundary.grader_quiescent(r))


if __name__ == '__main__':
    unittest.main()
