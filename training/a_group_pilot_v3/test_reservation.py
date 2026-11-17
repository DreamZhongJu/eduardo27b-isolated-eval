import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).with_name('pilot.py')
spec = importlib.util.spec_from_file_location('pilot3', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class FakeProc:
    def __init__(self): self.returncode = None; self.stdout = None
    def poll(self): return self.returncode
    def terminate(self): self.returncode = 0
    def wait(self, timeout): return self.returncode
    def kill(self): self.returncode = -9

class Tests(unittest.TestCase):
    def tearDown(self): m.release_holders()

    def test_eligible_never_accepts_occupied_gpu(self):
        self.assertTrue(m.eligible((81920, 81920, 0, False)))
        for value in [(81920, 75500, 0, False), (81920, 81920, 1, False),
                      (81920, 81920, 0, True), (32000, 32000, 0, False)]:
            self.assertFalse(m.eligible(value))

    def test_handoff_only_after_four_claims(self):
        snapshots = {i: (81920, 81920, 0, False) for i in range(6)}
        states = []
        def claim(i, work):
            m.HOLDERS[i] = FakeProc()
            return True
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'gpu_snapshot', return_value=snapshots), \
             patch.object(m.time, 'sleep'), patch.object(m, 'try_reserve', side_effect=claim), \
             patch.object(m, 'check_cards') as check, \
             patch.object(m, 'state', side_effect=lambda *v: states.append(v)):
            ids = m.wait_cards(Path(d))
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(m.HOLDERS, {})
        check.assert_called_once_with(ids)
        self.assertTrue(any('凑够 4' in s[2] for s in states))

    def test_release_only_owned_workers(self):
        p1, p2, stranger = FakeProc(), FakeProc(), FakeProc()
        m.HOLDERS.update({0: p1, 2: p2})
        m.release_holders()
        self.assertEqual((p1.returncode, p2.returncode), (0, 0))
        self.assertIsNone(stranger.returncode)

if __name__ == '__main__': unittest.main()
