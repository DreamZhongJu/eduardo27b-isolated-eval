"""CPU-only mock tests of shared two-GPU layout and elastic selection."""
import contextlib
import hashlib
import io
import json
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import supervisor as s
import status
from test_supervisor import ModelHandler, Server
from test_wait import card, fixture


class ElasticTests(unittest.TestCase):
    def test_status_reports_only_current_selected_count(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root/'shared-supervisor-elastic.log'
            events = [{'state':'STARTING'}, {'state':'GPU_READY','available_gpus':4},
                      {'state':'WAITING_FOR_FREE_GPUS'}, {'state':'GPU_READY','available_gpus':2},
                      {'state':'PILOT_RUNNING'}]
            log.write_text(''.join(json.dumps(e)+'\n' for e in events))
            self.assertEqual(status.wait_phase(root), {'state':'PILOT_RUNNING','selected_gpus':2})
            with log.open('a') as f: f.write('{"state":"STARTING"}\n')
            self.assertEqual(status.wait_phase(root), {'state':'STARTING'})
            with log.open('a') as f: f.write('{"state":"GPU_READY","available_gpus":99}\n')
            self.assertEqual(status.wait_phase(root), {'state':'GPU_READY'})
            with log.open('a') as f:
                f.write('{"state":"GPU_READY","available_gpus":5}\n{"state":"STARTING_SERVICES"}\n')
            self.assertEqual(status.wait_phase(root), {'state':'STARTING_SERVICES','selected_gpus':5})

    def test_choose_two_three_and_four(self):
        all_busy = {i: card(i, False) for i in range(8)}
        self.assertIsNone(s.choose_free_elastic(all_busy))
        snap = {i: card(i, i in (2, 5)) for i in range(8)}
        self.assertEqual(s.choose_free_elastic(snap), [2, 5])
        snap[7] = card(7)
        self.assertEqual(s.choose_free_elastic(snap), [2, 5, 7])
        snap[0] = card(0)
        self.assertEqual(s.choose_free_elastic(snap), [0, 2, 5, 7])
        snap[2]['uuid'] = snap[0]['uuid']
        self.assertEqual(s.choose_free_elastic(snap), [0, 5, 7])
        self.assertEqual(s.choose_free_elastic({i: card(i) for i in range(8)}), [0, 1, 2, 3, 4])

    def test_wait_uses_three_and_adjusts_ports(self):
        busy = {i: card(i, i == 2) for i in range(8)}
        free = {i: card(i, i in (2, 5, 7)) for i in range(8)}
        old, args = fixture([busy, free, free])
        args.ports = [18001, 18002, 18003, 18004]
        with mock.patch.object(s, 'watcher_conflict', return_value=False), \
             mock.patch.object(s.process_guard, 'prior_conflict', return_value=False), \
             mock.patch.object(s, 'model_signature', return_value=('same',)), \
             mock.patch.object(s, 'verify_models_stable', side_effect=AssertionError('unnecessary hash')), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            ids, sig = s.wait_for_four(old, args, ('same',), lambda: False, sleep=lambda _: None,
                                       elastic=True, port_pool=(18001, 18002, 18003, 18004))
        self.assertEqual(args.gpus, [2, 5, 7])
        self.assertEqual(args.ports, [18001, 18002, 18003])
        self.assertEqual(ids, ['GPU-2', 'GPU-5', 'GPU-7'])
        self.assertEqual(sig, ('same',))
        self.assertIn('WAITING_FOR_FREE_GPUS', out.getvalue())

    def test_reselection_restores_entire_port_pool(self):
        three = {i: card(i, i in (0, 2, 4)) for i in range(8)}
        five = {i: card(i, i in (0, 1, 3, 5, 7)) for i in range(8)}
        old, args = fixture([three, three, five, five])
        args.ports = [18001, 18002, 18003, 18004, 18005]
        pool = tuple(args.ports)
        with mock.patch.object(s, 'watcher_conflict', return_value=False), \
             mock.patch.object(s.process_guard, 'prior_conflict', return_value=False), \
             mock.patch.object(s, 'model_signature', return_value=('same',)), \
             contextlib.redirect_stdout(io.StringIO()):
            s.wait_for_four(old, args, ('same',), lambda: False, sleep=lambda _: None,
                            elastic=True, port_pool=pool)
            self.assertEqual(args.ports, [18001, 18002, 18003])
            s.wait_for_four(old, args, ('same',), lambda: False, sleep=lambda _: None,
                            elastic=True, port_pool=pool)
        self.assertEqual(args.gpus, [0, 1, 3, 5, 7])
        self.assertEqual(args.ports, [18001, 18002, 18003, 18004, 18005])

    def test_two_gpu_services_and_three_isolated_evaluators(self):
        gpu, ports = [2, 5], [18001, 18002]
        plans = s.service_plans('/python', '/teacher', '/student', gpu, ports)
        self.assertEqual(len(plans), 2)
        teacher_cmd, student_cmd = plans[0][0], plans[1][0]
        self.assertEqual(plans[0][1], 2)
        self.assertEqual(plans[1][1], 5)
        self.assertEqual(teacher_cmd[teacher_cmd.index('--max-num-seqs') + 1], '1')
        self.assertIn('Eduardo-27b', teacher_cmd)
        self.assertIn('Llama-3.1-8B-Instruct', student_cmd)
        evals = s.evaluator_plans('/python', '/guard', Path('/results'), Path('/repo'),
                                  Path('/data'), Path('/teacher'), ports)
        self.assertEqual(len(evals), 3)
        self.assertEqual([c[c.index('--teacher-url') + 1] for c in evals], ['http://127.0.0.1:18001/v1'] * 3)
        self.assertEqual([c[c.index('--student-url') + 1] for c in evals], ['http://127.0.0.1:18002/v1'] * 3)
        self.assertEqual(len({c[c.index('--results') + 1] for c in evals}), 3)
        self.assertEqual([c[c.index('--difficulty') + 1] for c in evals], ['easy', 'medium', 'hard'])

    def test_three_gpu_teacher_routes_and_isolated_evaluators(self):
        gpu, ports = [0, 3, 5], [18001, 18002, 18003]
        plans = s.service_plans('/python', '/teacher', '/student', gpu, ports)
        self.assertEqual([pair[1] for pair in plans], gpu)
        self.assertEqual(len(plans), 3)
        for cmd, _ in plans[:2]:
            self.assertEqual(cmd[cmd.index('--max-num-seqs') + 1], '1')
            self.assertIn('Eduardo-27b', cmd)
        self.assertIn('Llama-3.1-8B-Instruct', plans[2][0])
        evals = s.evaluator_plans('/python', '/guard', Path('/results'), Path('/repo'),
                                  Path('/data'), Path('/teacher'), ports)
        self.assertEqual([c[c.index('--teacher-url') + 1] for c in evals],
                         ['http://127.0.0.1:18001/v1', 'http://127.0.0.1:18002/v1', 'http://127.0.0.1:18001/v1'])
        self.assertEqual([c[c.index('--student-url') + 1] for c in evals], ['http://127.0.0.1:18003/v1'] * 3)
        self.assertEqual(len({c[c.index('--results') + 1] for c in evals}), 3)

    def test_five_gpu_original_three_teacher_two_student_routes(self):
        gpu, ports = [0, 1, 2, 3, 4], [18001, 18002, 18003, 18004, 18005]
        plans = s.service_plans('/python', '/teacher', '/student', gpu, ports)
        self.assertEqual([pair[1] for pair in plans], gpu)
        self.assertEqual(len(plans), 5)
        for cmd, _ in plans[:3]:
            self.assertEqual(cmd[cmd.index('--max-num-seqs') + 1], '1')
            self.assertIn('Eduardo-27b', cmd)
        for cmd, _ in plans[3:]:
            self.assertIn('Llama-3.1-8B-Instruct', cmd)
        evals = s.evaluator_plans('/python', '/guard', Path('/results'), Path('/repo'),
                                  Path('/data'), Path('/teacher'), ports)
        self.assertEqual([c[c.index('--teacher-url') + 1] for c in evals],
                         ['http://127.0.0.1:18001/v1', 'http://127.0.0.1:18002/v1', 'http://127.0.0.1:18003/v1'])
        self.assertEqual([c[c.index('--student-url') + 1] for c in evals],
                         ['http://127.0.0.1:18004/v1', 'http://127.0.0.1:18004/v1', 'http://127.0.0.1:18005/v1'])

    def test_three_mock_services_share_student(self):
        services, threads = [], []
        try:
            for model in ('Eduardo-27b', 'Eduardo-27b', 'Llama-3.1-8B-Instruct'):
                service = Server(('127.0.0.1', 0), ModelHandler)
                service.models = []
                service.model = model
                thread = threading.Thread(target=service.serve_forever, daemon=True)
                thread.start()
                services.append(service)
                threads.append(thread)
            for service in services:
                s.ready_local(types.SimpleNamespace(poll=lambda: None), service.server_address[1],
                              service.model, lambda: False, timeout=2)
            s.smoke_services([service.server_address[1] for service in services])
            self.assertEqual([len(service.models) for service in services], [1, 1, 12])
        finally:
            for service in services:
                service.shutdown()
                service.server_close()
            for thread in threads:
                thread.join()

    def test_two_mock_services_serve_twelve_parallel_student_calls(self):
        services, threads = [], []
        try:
            for model in ('Eduardo-27b', 'Llama-3.1-8B-Instruct'):
                service = Server(('127.0.0.1', 0), ModelHandler)
                service.models = []
                service.model = model
                thread = threading.Thread(target=service.serve_forever, daemon=True)
                thread.start()
                services.append(service)
                threads.append(thread)
            for service in services:
                s.ready_local(types.SimpleNamespace(poll=lambda: None), service.server_address[1],
                              service.model, lambda: False, timeout=2)
            s.smoke_services([service.server_address[1] for service in services])
            self.assertEqual([len(service.models) for service in services], [1, 12])
        finally:
            for service in services:
                service.shutdown()
                service.server_close()
            for thread in threads:
                thread.join()

    def test_shared_teacher_failure_stops_all_dependent_evaluators(self):
        class Process:
            def __init__(self, code=None): self.code = code
            def poll(self): return self.code
            @property
            def returncode(self): return self.code
        evals = [Process(), Process(), Process()]
        services = [Process(1), Process(None)]
        watcher = types.SimpleNamespace(tick=lambda now: True)
        stopped = []
        def terminate(proc, tag):
            stopped.append(proc)
            proc.code = -15
        with mock.patch.object(s, 'terminate_own', side_effect=terminate):
            with self.assertRaises(RuntimeError):
                s.wait_evaluators(evals, services, watcher, lambda: False, 1, 'tag', sleep=lambda _: None)
        self.assertEqual(stopped, evals)

    def test_three_gpu_one_teacher_failure_isolates_affected_lanes(self):
        class Process:
            def __init__(self, code=None): self.code = code
            def poll(self): return self.code
            @property
            def returncode(self): return self.code
        evals = [Process(), Process(), Process()]
        services = [Process(1), Process(None), Process(None)]
        watcher = types.SimpleNamespace(tick=lambda now: True)
        stopped = []
        def terminate(proc, tag):
            stopped.append(proc)
            proc.code = -15
        def sleep(_): evals[1].code = 0
        with mock.patch.object(s, 'terminate_own', side_effect=terminate):
            with self.assertRaises(RuntimeError):
                s.wait_evaluators(evals, services, watcher, lambda: False, 1, 'tag', sleep=sleep)
        self.assertEqual(stopped, [evals[0], evals[2]])
        self.assertEqual(evals[1].returncode, 0)

    def test_full_entrypoint_reaches_cuda_for_two_selected_cards(self):
        class ReachedCuda(Exception): pass
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'root'
            root.mkdir()
            watcher, judge, key = Path(tmp)/'watch_sources.py', Path(tmp)/'agnes_judge.py', Path(tmp)/'key'
            watcher.write_text('# synthetic')
            judge.write_text('# synthetic')
            key.write_text('synthetic-key')
            wh = hashlib.sha256(watcher.read_bytes()).hexdigest()
            jh = hashlib.sha256(judge.read_bytes()).hexdigest()
            args = types.SimpleNamespace(run=True, wait_gpus=True, elastic=True, root=root,
                repo=Path('/repo'), v4_dir=Path('/v4'), data=Path('/data'), teacher=Path('/teacher'),
                student=Path('/student'), key=key, watcher=watcher, judge=judge,
                watcher_sha256=wh, judge_sha256=jh, gpus=[0,1,2,3,4], ports=[18001,18002,18003,18004,18005])
            old = mock.Mock()
            old.old_process_running.return_value = False
            lane = mock.Mock()
            lane.clean_env.return_value = {}
            lane.verify_numeric_cuda.side_effect = ReachedCuda
            def select(old_value, args_value, signature, cancelled, **kwargs):
                self.assertTrue(kwargs['elastic'])
                args_value.gpus[:] = [2, 5]
                args_value.ports = [18001, 18002]
                return ['GPU-2', 'GPU-5'], signature
            real_path = Path
            def translated(value):
                if value == '/workspace/results/own-test-499-v1': return root
                if value == '/workspace/.own-test-499.lock': return Path(tmp)/'lock'
                return real_path(value)
            with mock.patch.object(s, 'APPROVED_GRADER_PAIRS', {(wh, jh)}), \
                 mock.patch.object(s, 'Path', side_effect=translated), \
                 mock.patch.object(s, 'import_checked', side_effect=[old, lane]), \
                 mock.patch.object(s, 'watcher_conflict', return_value=False), \
                 mock.patch.object(s.process_guard, 'prior_conflict', return_value=False), \
                 mock.patch.object(s, 'assert_ports_free'), \
                 mock.patch.object(s, 'verify_models_stable', return_value=('same',)), \
                 mock.patch.object(s, 'wait_for_four', side_effect=select), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(s.PreflightFailure) as caught:
                    s.run(args)
            self.assertEqual(caught.exception.step, 'CUDA_DEVICE')
            self.assertEqual(lane.verify_numeric_cuda.call_args.args[0:2], (2, 'GPU-2'))
            self.assertEqual(args.ports, [18001, 18002])


if __name__ == '__main__':
    unittest.main()
