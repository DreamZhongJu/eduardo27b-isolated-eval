import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import supervisor as sup

class Dummy:
    def __init__(self, code=None): self.returncode = code
    def poll(self): return self.returncode

class Tests(unittest.TestCase):
    def test_five_cards_required(self):
        cards = {i: {'name':'A100', 'total':81920, 'free':81920, 'util':0,
                     'uuid':f'GPU-{i}'} for i in range(5)}
        self.assertIsNone(sup.choose_free_elastic(dict(list(cards.items())[:4])))
        self.assertEqual(sup.choose_free_elastic(cards), [0,1,2,3,4])

    def test_all_stages_share_both_students(self):
        commands = sup.evaluator_plans('py', 'guard', Path('/root'), Path('/repo'),
                                        Path('/data'), Path('/teacher'), list(range(18001,18006)))
        self.assertEqual(len(commands), 3)
        for c in commands:
            self.assertEqual(c[c.index('--student-url')+1], 'http://127.0.0.1:18004/v1')
            self.assertEqual(c[c.index('--student-url-secondary')+1], 'http://127.0.0.1:18005/v1')

    def test_second_student_failure_stops_all_affected(self):
        evaluators = [Dummy(), Dummy(), Dummy()]
        services = [Dummy(), Dummy(), Dummy(), Dummy(), Dummy(1)]
        class Watcher:
            def tick(self, _): pass
        def stop(p, _): p.returncode = 1
        with patch.object(sup, 'terminate_own', stop):
            with self.assertRaises(RuntimeError):
                sup.wait_evaluators(evaluators, services, Watcher(), lambda: False, 2, 'run', sleep=lambda _:None)
        self.assertTrue(all(p.returncode == 1 for p in evaluators))

if __name__ == '__main__': unittest.main()
