import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import status


def source(root, stage, sid, state='complete', grade='missing'):
    key = hashlib.sha256(sid.encode()).hexdigest()[:24]
    d = root/stage/'sources'/key
    d.mkdir(parents=True)
    if state == 'complete':
        (d/'RAW_COMPLETE.json').write_text(json.dumps({'source_id': sid, 'samples': 96}))
    elif state == 'invalid':
        (d/'RAW_COMPLETE.json').write_text('not json')
    elif state == 'held':
        (d/'SOURCE_HELD.json').write_text('{}')
    if grade != 'missing':
        dest = root/'AGNES'/stage/'sources'/key
        dest.mkdir(parents=True)
        if grade == 'held':
            (dest/'HELD.json').write_text('{}')
        elif grade == 'invalid':
            (dest/'RESULTS.json').write_text('not json')
        else:
            rows = []
            for target in ('source', 'ours_easy'):
                for phase in ('pre', 'control', 'post'):
                    for i in range(16):
                        rows.append({'identity': {'source_id': sid, 'target': target, 'phase': phase, 'index': i},
                                     'status': 'correct' if i < 8 else ('incorrect' if i < 12 else 'unknown')})
            (dest/'RESULTS.json').write_text(json.dumps({'source_id': sid, 'input_sha256': 'a'*64, 'results': rows}))
    return d


class TestStatus(unittest.TestCase):
    def test_aggregate_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source(root, 'PILOT', 'SYNTHETIC_PRIVATE_1', grade='complete')
            source(root, 'EASY', 'SYNTHETIC_PRIVATE_2', grade='missing')
            source(root, 'EASY', 'SYNTHETIC_PRIVATE_3', state='partial')
            source(root, 'EASY', 'SYNTHETIC_PRIVATE_4', state='held')
            source(root, 'EASY', 'SYNTHETIC_PRIVATE_5', grade='invalid')
            source(root, 'EASY', 'SYNTHETIC_PRIVATE_6', state='invalid')
            source(root, 'MEDIUM', 'SYNTHETIC_PRIVATE_7', grade='held')
            orphan = root/'AGNES'/'MEDIUM'/'sources'/('a'*24)
            orphan.mkdir(parents=True)
            result = status.snapshot(root)
            a = result['easy']
            self.assertEqual(a['sources_started'], 6)
            self.assertEqual((a['generation_complete'], a['generation_held'], a['generation_partial'],
                              a['generation_invalid']), (3, 1, 1, 1))
            self.assertEqual((a['grading_complete'], a['grading_pending'], a['grading_invalid']), (1, 1, 1))
            self.assertEqual((a['correct'], a['incorrect'], a['unknown']), (48, 24, 24))
            self.assertEqual(result['medium']['grading_held'], 1)
            self.assertEqual(result['medium']['grading_orphan'], 1)
            self.assertEqual(result['hard']['sources_started'], 0)
            self.assertNotIn('SYNTHETIC_PRIVATE', json.dumps(result))

    def test_process_running_missing_and_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            entry = proc/'123456'
            entry.mkdir()
            script = Path('/isolated/supervisor.py')
            (entry/'cmdline').write_bytes(b'python\0/isolated/supervisor.py\0--run\0')
            self.assertEqual(status.process_status(script, proc)['state'], 'running')
            (entry/'cmdline').write_bytes(b'python\0/isolated/supervisor.py\0--status\0')
            self.assertEqual(status.process_status(script, proc)['state'], 'not_running')
            with patch.object(Path, 'read_bytes', side_effect=PermissionError):
                self.assertEqual(status.process_status(script, proc)['state'], 'unknown')

    def test_reject_duplicate_identity_and_missing_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = source(root, 'EASY', 'secret', grade='complete')
            p = root/'AGNES'/'EASY'/'sources'/folder.name/'RESULTS.json'
            doc = json.loads(p.read_text())
            doc['results'][1]['identity'] = dict(doc['results'][0]['identity'])
            p.write_text(json.dumps(doc))
            self.assertEqual(status.snapshot(root)['easy']['grading_invalid'], 1)
            with self.assertRaises(ValueError): status.snapshot(root/'no-such-results')

    def test_unknown_status_not_false(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(status.process_status(Path('/x'), Path(temp)/'missing')['state'], 'unknown')


if __name__ == '__main__':
    unittest.main()
