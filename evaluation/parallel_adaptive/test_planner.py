import unittest

from planner import Teacher, plan


class PlannerTest(unittest.TestCase):
    def test_two_gpus_start_one_teacher(self):
        self.assertEqual(plan([], [1], [0], dict(easy=20, medium=10, hard=5)), {0: 'easy'})

    def test_four_gpus_can_start_three_lanes(self):
        result = plan([], [3], [0, 1, 2], dict(easy=100, medium=100, hard=100))
        self.assertEqual(set(result.values()), {'easy', 'medium', 'hard'})

    def test_finished_lane_is_reassigned_to_hard(self):
        workers = [Teacher(0, 'easy', False), Teacher(1, 'medium', True), Teacher(2, 'hard', True)]
        result = plan(workers, [3], [], dict(easy=0, medium=0, hard=100))
        self.assertEqual(result, {1: 'medium', 2: 'hard', 0: 'hard'})

    def test_in_flight_source_is_never_moved(self):
        result = plan([Teacher(0, 'easy', True)], [1], [2], dict(easy=0, medium=0, hard=5))
        self.assertEqual(result, {0: 'easy', 2: 'hard'})

    def test_no_work_no_new_service(self):
        result = plan([Teacher(0, 'easy', False)], [1], [2, 3], dict(easy=0, medium=0, hard=0))
        self.assertEqual(result, {0: None})

    def test_one_source_one_new_worker(self):
        result = plan([], [3], [0, 1, 2], dict(easy=0, medium=0, hard=1))
        self.assertEqual(result, {0: 'hard'})

    def test_duplicate_gpu_rejected(self):
        with self.assertRaises(ValueError):
            plan([Teacher(0, 'easy', True)], [0], [], dict(easy=1, medium=1, hard=1))

    def test_invalid_pending_rejected(self):
        with self.assertRaises(ValueError):
            plan([], [0], [1], dict(easy=-1, medium=0, hard=1))

    def test_many_free_cards_cap_total_including_student(self):
        result = plan([], [7], list(range(7)), dict(easy=20, medium=20, hard=20))
        self.assertEqual(len(result), 5)
        self.assertEqual(len(result) + 1, 6)

    def test_existing_students_and_teachers_count_against_cap(self):
        result = plan([Teacher(0, 'hard', True), Teacher(1, 'hard', True)],
                      [2, 3], [4, 5, 6, 7], dict(easy=0, medium=0, hard=20))
        self.assertEqual(set(result), {0, 1, 4, 5})
        self.assertEqual(len(result) + 2, 6)

    def test_over_cap_existing_run_refused(self):
        with self.assertRaises(ValueError):
            plan([Teacher(i, 'hard', True) for i in range(5)], [5, 6], [],
                 dict(easy=0, medium=0, hard=20))

    def test_teacher_limit_above_global_cap_refused(self):
        with self.assertRaises(ValueError):
            plan([], [0], [1], dict(easy=1, medium=0, hard=0), max_teachers=6)


if __name__ == '__main__':
    unittest.main()
