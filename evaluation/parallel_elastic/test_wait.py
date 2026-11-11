import contextlib
import io
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import supervisor as s
import status


def card(i, free=True):
    return {'name': 'NVIDIA A100 80GB', 'total': 81920,
            'free': 81000 if free else 20000, 'util': 0,
            'uuid': f'GPU-{i}'}


def fixture(snapshots):
    snapshots = iter(snapshots)
    old = types.SimpleNamespace(gpu_snapshot=lambda: next(snapshots),
        old_process_running=lambda *a: False,
        verify_old_binding=lambda *a: None, verify_inputs=lambda *a: None,
        verify_pilot=lambda *a: None, verify_versions=lambda *a: None)
    args = types.SimpleNamespace(repo=Path('/repo'), root=Path('/root'),
        data=Path('/data'), teacher=Path('/teacher'), student=Path('/student'),
        gpus=[0, 1, 2, 3])
    return old, args


class WaitTests(unittest.TestCase):
    def test_select_free_only_and_distinct(self):
        snapshot = {i: card(i, i >= 3) for i in range(8)}
        self.assertEqual(s.choose_free_four(snapshot), [3, 4, 5, 6])
        snapshot[4]['uuid'] = snapshot[3]['uuid']
        self.assertEqual(s.choose_free_four(snapshot), [3, 5, 6, 7])
        snapshot[7]['util'] = 10
        self.assertIsNone(s.choose_free_four(snapshot))

    def test_changed_uuid_fails_without_starting(self):
        snapshot = {i: card(i) for i in range(4)}
        ids = [f'GPU-{i}' for i in range(4)]
        self.assertEqual(s.verify_same_devices([0, 1, 2, 3], ids, snapshot), ids)
        snapshot[2]['uuid'] = 'GPU-CHANGED'
        with self.assertRaises(s.PreflightFailure) as caught:
            s.verify_same_devices([0, 1, 2, 3], ids, snapshot)
        self.assertEqual(caught.exception.step, 'GPU_VALIDATION')

    def test_wait_then_select_any_four_without_rehash(self):
        busy = {i: card(i, i >= 5) for i in range(8)}
        free = {i: card(i, i in (0, 2, 5, 7)) for i in range(8)}
        old, args = fixture([busy, free, free])
        with mock.patch.object(s, 'watcher_conflict', return_value=False), \
             mock.patch.object(s.process_guard, 'prior_conflict', return_value=False), \
             mock.patch.object(s, 'model_signature', return_value=('same',)), \
             mock.patch.object(s, 'verify_models_stable', side_effect=AssertionError('unnecessary hash')), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            ids, signature = s.wait_for_four(old, args, ('same',), lambda: False, sleep=lambda _: None)
        self.assertEqual(args.gpus, [0, 2, 5, 7])
        self.assertEqual(ids, ['GPU-0', 'GPU-2', 'GPU-5', 'GPU-7'])
        self.assertEqual(signature, ('same',))
        self.assertIn('WAITING_FOR_FREE_GPUS', out.getvalue())
        self.assertIn('GPU_READY', out.getvalue())

    def test_model_change_rehashes_once(self):
        free = {i: card(i) for i in range(4)}
        old, args = fixture([free, free])
        with mock.patch.object(s, 'watcher_conflict', return_value=False), \
             mock.patch.object(s.process_guard, 'prior_conflict', return_value=False), \
             mock.patch.object(s, 'model_signature', return_value=('changed',)), \
             mock.patch.object(s, 'verify_models_stable', return_value=('changed',)) as check, \
             contextlib.redirect_stdout(io.StringIO()):
            _, signature = s.wait_for_four(old, args, ('old',), lambda: False, sleep=lambda _: None)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(signature, ('changed',))

    def test_card_taken_before_recheck_waits_and_reselects(self):
        free_a = {i: card(i, i < 4) for i in range(8)}
        lost = {i: card(i, i in (1, 2, 3)) for i in range(8)}
        free_b = {i: card(i, i >= 4) for i in range(8)}
        old, args = fixture([free_a, lost, lost, free_b, free_b])
        with mock.patch.object(s, 'watcher_conflict', return_value=False), \
             mock.patch.object(s.process_guard, 'prior_conflict', return_value=False), \
             mock.patch.object(s, 'model_signature', return_value=('same',)), \
             mock.patch.object(s, 'verify_models_stable', side_effect=AssertionError('unnecessary hash')), \
             contextlib.redirect_stdout(io.StringIO()):
            ids, _ = s.wait_for_four(old, args, ('same',), lambda: False, sleep=lambda _: None)
        self.assertEqual(args.gpus, [4, 5, 6, 7])
        self.assertEqual(ids, ['GPU-4', 'GPU-5', 'GPU-6', 'GPU-7'])

    def test_non_gpu_error_stops_without_sleep(self):
        old, args = fixture([])
        old.gpu_snapshot = mock.Mock(side_effect=OSError('PRIVATE'))
        sleep = mock.Mock()
        with self.assertRaises(s.PreflightFailure) as caught:
            s.wait_for_four(old, args, ('same',), lambda: False, sleep=sleep)
        self.assertEqual(caught.exception.step, 'GPU_QUERY')
        self.assertEqual(caught.exception.error_type, 'OSError')
        sleep.assert_not_called()

    def test_input_error_stops_instead_of_waiting(self):
        free = {i: card(i) for i in range(4)}
        old, args = fixture([free])
        old.verify_inputs = mock.Mock(side_effect=ValueError('PRIVATE'))
        sleep = mock.Mock()
        with mock.patch.object(s, 'watcher_conflict', return_value=False), \
             mock.patch.object(s.process_guard, 'prior_conflict', return_value=False):
            with self.assertRaises(s.PreflightFailure) as caught:
                s.wait_for_four(old, args, ('same',), lambda: False, sleep=sleep)
        self.assertEqual(caught.exception.step, 'DATA_BINDING')
        self.assertEqual(caught.exception.error_type, 'ValueError')
        sleep.assert_not_called()

    def test_change_during_hash_fails_closed(self):
        old = types.SimpleNamespace(verify_models=lambda *a: None)
        with mock.patch.object(s, 'model_signature', side_effect=[('one',), ('two',)]):
            with self.assertRaises(ValueError):
                s.verify_models_stable(old, Path('/t'), Path('/s'))

    def test_status_reports_only_fixed_phase(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / 'shared-supervisor-elastic.log'
            path.write_text('{"state":"WAITING_FOR_FREE_GPUS", "private":"SECRET"}\n')
            self.assertEqual(status.wait_phase(root), {'state': 'WAITING_FOR_FREE_GPUS'})
            path.write_text('{"state":"ERROR", "step":"DATA_BINDING", "secret":"SECRET"}\n')
            self.assertEqual(status.wait_phase(root), {'state':'ERROR', 'step':'DATA_BINDING'})
            path.write_text('{"state":"ERROR", "step":"SECRET"}\n')
            self.assertEqual(status.wait_phase(root), {'state':'ERROR'})

    def test_wait_respects_stop(self):
        busy = {i: card(i, False) for i in range(8)}
        old, args = fixture([busy])
        calls = iter([False, True])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(InterruptedError):
                s.wait_for_four(old, args, ('same',), lambda: next(calls), sleep=lambda _: None)


if __name__ == '__main__':
    unittest.main()
