"""Pure CPU tests: fake devices, loopback mock services, synthetic receipts."""
import asyncio
import hashlib
import http.server
import json
import socket
import socketserver
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import evaluator_guard as guard
import process_guard
import supervisor as sup


def item(source='TEST_ID'):
    return {'source_id': source, 'problem': 'synthetic q', 'answer': '1',
            'transfer_problem': 'synthetic transfer', 'transfer_answer': '2', 'transfer_target': 'ours_easy',
            'transfer_candidate_id': 'synthetic-candidate', 'transfer_position': 3,
            'student_correct': 12, 'student_n_scored': 16, 'score_tie': False}


def synthetic_complete(root, row):
    source_hash = guard.digest(guard.stable(row))
    folder = root/'sources'/guard.digest(row['source_id'].encode())[:24]
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'SOURCE.json').write_text(json.dumps({'source_id': row['source_id'], 'sha256': source_hash}))
    (folder/'RAW_COMPLETE.json').write_text(json.dumps({'source_id': row['source_id'], 'source_sha256': source_hash,
                                                        'samples': 96, 'grading_status': 'ungraded'}))
    entries = []
    for target in ('source', 'ours_easy'):
        for phase in ('pre', 'control', 'post'):
            for i in range(16):
                p = folder/'samples'/target/phase/f'{i:02d}'/'RESPONSE.json'
                p.parent.mkdir(parents=True)
                p.write_text('{}')
                transfer = target != 'source'
                entry = {'source_id': row['source_id'], 'target': target, 'phase': phase, 'index': i,
                         'problem': row['transfer_problem'] if transfer else row['problem'],
                         'reference_answer': row['transfer_answer'] if transfer else row['answer'],
                         'student_response': 'synthetic response'}
                if transfer:
                    entry.update(candidate_id=row['transfer_candidate_id'], difficulty='easy',
                                 position_by_student_correct=row['transfer_position'],
                                 historical_student_correct=row['student_correct'],
                                 historical_student_n_scored=row['student_n_scored'],
                                 historical_group_score_tie=row['score_tie'])
                entries.append(entry)
    (folder/'SCORE_INPUT.json').write_text(json.dumps(entries))
    return folder


class ModelHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            data = b'{}'
        elif self.path == '/v1/models':
            data = json.dumps({'data': [{'id': self.server.model}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        req = json.loads(self.rfile.read(length))
        self.server.models.append(req['model'])
        data = b'{"choices":[{"message":{"content":"5"}}]}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *_):
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    request_queue_size = 32


class TestTopology(unittest.TestCase):
    def test_routing_and_preserved_parameters(self):
        for count in (4, 5):
            gpus = list(range(count))
            ports = list(range(18001, 18001 + count))
            plans = sup.service_plans('/python', '/teacher', '/student', gpus, ports)
            self.assertEqual(len(plans), count)
            self.assertEqual([p[1] for p in plans], gpus)
            for cmd, _ in plans[:3]:
                self.assertEqual(cmd[cmd.index('--dtype')+1], 'bfloat16')
                self.assertIn('--enforce-eager', cmd)
                self.assertEqual(cmd[cmd.index('--max-num-seqs')+1], '1')
            for cmd, _ in plans[3:]:
                self.assertEqual(cmd[cmd.index('--dtype')+1], 'float16')
                self.assertEqual(cmd[cmd.index('--max-model-len')+1], '20000')
            evals = sup.evaluator_plans('/python', '/guard', Path('/results'), Path('/repo'),
                                        Path('/data'), Path('/teacher'), ports)
            student_urls = [c[c.index('--student-url')+1] for c in evals]
            self.assertEqual(student_urls[:2], [f'http://127.0.0.1:{ports[3]}/v1']*2)
            self.assertEqual(student_urls[2], f'http://127.0.0.1:{ports[-1]}/v1')
            self.assertEqual(len({c[c.index('--teacher-url')+1] for c in evals}), 3)

    def test_gpu_device_and_port_checks(self):
        def device(i):
            return {'name': 'NVIDIA A100-SXM4-80GB', 'total': 81050, 'free': 80000,
                    'util': 0, 'uuid': 'GPU-' + str(i)}
        snapshot = {i: device(i) for i in range(5)}
        self.assertEqual(len(sup.verify_gpu_snapshot([0, 1, 2, 3], snapshot)), 4)
        self.assertEqual(len(sup.verify_gpu_snapshot([0, 1, 2, 3, 4], snapshot)), 5)
        for gpus in ([0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3, 4, 5]):
            with self.assertRaises(ValueError): sup.verify_gpu_snapshot(gpus, snapshot)
        snapshot[3]['free'] = 40000
        with self.assertRaises(ValueError): sup.verify_gpu_snapshot([0, 1, 2, 3], snapshot)
        snapshot[3] = device(2)
        with self.assertRaises(ValueError): sup.verify_gpu_snapshot([0, 1, 2, 3], snapshot)
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            port = occupied.getsockname()[1]
            others = [p for p in (18001, 18002, 18003, 18004) if p != port][:3]
            with self.assertRaises(OSError): sup.assert_ports_free([port, *others])

    def test_exclusive_root_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp)/'lock'
            fd = sup.take_lock(lock)
            try:
                with self.assertRaises(BlockingIOError): sup.take_lock(lock)
            finally:
                import os
                os.close(fd)
            fd2 = sup.take_lock(lock)
            import os
            os.close(fd2)

    def test_mock_http_services_4_and_5_devices(self):
        for size in (4, 5):
            services = []
            threads = []
            try:
                for _ in range(size):
                    s = Server(('127.0.0.1', 0), ModelHandler)
                    s.models = []
                    s.model = 'Eduardo-27b' if len(services) < 3 else 'Llama-3.1-8B-Instruct'
                    t = threading.Thread(target=s.serve_forever, daemon=True)
                    t.start()
                    services.append(s)
                    threads.append(t)
                for s in services:
                    sup.ready_local(types.SimpleNamespace(poll=lambda: None), s.server_address[1],
                                    s.model, lambda: False, timeout=2)
                sup.smoke_services([s.server_address[1] for s in services])
                self.assertEqual([len(s.models) for s in services[:3]], [1, 1, 1])
                self.assertEqual([len(s.models) for s in services[3:]], [12] if size == 4 else [8, 8])
            finally:
                for s in services:
                    s.shutdown()
                    s.server_close()
                for t in threads:
                    t.join()

    def test_watcher_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            p = proc/'12345'
            p.mkdir()
            root = Path('/workspace/results/own-test-499-v1')
            (p/'cmdline').write_bytes(b'python\0watch.py\0--receipts-dir\0' + str(root/'AGNES'/'receipts').encode() + b'\0')
            self.assertTrue(sup.watcher_conflict(root, proc))
            (p/'cmdline').write_bytes(b'python\0other.py\0')
            self.assertFalse(sup.watcher_conflict(root, proc))

    def test_watcher_requires_approved_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            watcher = Path(tmp)/'watcher.py'
            watcher.write_text('# synthetic watcher')
            args = types.SimpleNamespace(run=True, root=Path('/workspace/results/own-test-499-v1'),
                    repo=Path('/repo'), v4_dir=Path('/v4'), data=Path('/data'), teacher=Path('/teacher'),
                    student=Path('/student'), key=Path('/secret'), watcher=watcher,
                    watcher_sha256=hashlib.sha256(watcher.read_bytes()).hexdigest(),
                    judge=Path(tmp)/'agnes_judge.py', judge_sha256='0'*64)
            with mock.patch.object(sup, 'take_lock', side_effect=AssertionError('must reject before locks')):
                with self.assertRaises(ValueError): sup.run(args)

    def test_run_reaches_unique_tag_and_cuda_validation_without_gpu(self):
        class ReachedCudaValidation(Exception): pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'root'
            root.mkdir()
            watcher, judge, key = Path(tmp)/'watch_sources.py', Path(tmp)/'agnes_judge.py', Path(tmp)/'key'
            watcher.write_text('# synthetic')
            judge.write_text('# synthetic')
            key.write_text('synthetic-key')
            wh, jh = hashlib.sha256(watcher.read_bytes()).hexdigest(), hashlib.sha256(judge.read_bytes()).hexdigest()
            args = types.SimpleNamespace(run=True, root=root, repo=Path('/repo'), v4_dir=Path('/v4'),
                    data=Path('/data'), teacher=Path('/teacher'), student=Path('/student'), key=key,
                    watcher=watcher, judge=judge, watcher_sha256=wh, judge_sha256=jh,
                    gpus=[0, 1, 2, 3], ports=[18001, 18002, 18003, 18004])
            old = mock.Mock()
            old.old_process_running.return_value = False
            old.gpu_snapshot.return_value = {i: {'name': 'A100', 'total': 81050, 'free': 81050,
                                                 'util': 0, 'uuid': 'GPU-' + str(i)} for i in range(4)}
            lane = mock.Mock()
            lane.clean_env.return_value = {}
            lane.verify_numeric_cuda.side_effect = ReachedCudaValidation
            real_path = Path
            def translated_path(value):
                if value == '/workspace/results/own-test-499-v1': return root
                if value == '/workspace/.own-test-499.lock': return Path(tmp)/'lock'
                return real_path(value)
            with mock.patch.object(sup, 'APPROVED_GRADER_PAIRS', {(wh, jh)}), \
                 mock.patch.object(sup, 'Path', side_effect=translated_path), \
                 mock.patch.object(sup, 'import_checked', side_effect=[old, lane]), \
                 mock.patch.object(sup, 'watcher_conflict', return_value=False), \
                 mock.patch.object(process_guard, 'prior_conflict', return_value=False), \
                 mock.patch.object(sup, 'assert_ports_free'):
                with self.assertRaises(sup.PreflightFailure) as caught: sup.run(args)
                self.assertEqual(caught.exception.step, 'CUDA_DEVICE')
                self.assertEqual(caught.exception.error_type, 'ReachedCudaValidation')
            env = lane.verify_numeric_cuda.call_args.args[2]
            self.assertEqual(len(env['FIVEGPU_RUN_ID']), 32)
            self.assertEqual(env['FIVEGPU_RUN_ROOT'], str(root))

    def test_watcher_exits_and_restarts_without_concurrent_writer(self):
        class Proc:
            code = None
            def poll(self): return self.code
        launched = []
        def start():
            self.assertTrue(all(x.code is not None for x in launched))
            p = Proc()
            launched.append(p)
            return p
        watcher = sup.WatcherManager(start)
        self.assertTrue(watcher.tick(0))
        self.assertTrue(watcher.tick(1))
        self.assertEqual(len(launched), 1)
        launched[0].code = 1
        self.assertFalse(watcher.tick(2))
        self.assertFalse(watcher.tick(2.5))
        self.assertTrue(watcher.tick(3))
        self.assertEqual(len(launched), 2)
        self.assertEqual(watcher.failures, 1)
        self.assertTrue(watcher.tick(34))
        self.assertEqual(watcher.failures, 0)

    def test_teacher_failure_does_not_kill_other_lanes(self):
        class Proc:
            def __init__(self, code=None): self.returncode = code
            def poll(self): return self.returncode
        evaluators = [Proc(), Proc(), Proc()]
        services = [Proc(1), Proc(), Proc(), Proc()]
        stopped = []
        def terminate(proc, tag):
            stopped.append(proc)
            proc.returncode = -15
        def sleep(_):
            evaluators[1].returncode = 0
            evaluators[2].returncode = 0
        with mock.patch.object(sup, 'terminate_own', side_effect=terminate):
            with self.assertRaises(RuntimeError):
                sup.wait_evaluators(evaluators, services, types.SimpleNamespace(tick=lambda _: None),
                                    lambda: False, 1, 'synthetic', sleep=sleep)
        self.assertEqual(stopped, [evaluators[0]])
        self.assertEqual(evaluators[1].returncode, 0)
        self.assertEqual(evaluators[2].returncode, 0)

    def test_only_fixed_evaluator_summary_is_consumed(self):
        counts = {'complete': 2, 'new_complete': 1, 'held': 1, 'partial': 2, 'invalid': 0, 'failed': 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'eval.log'
            path.write_text('SYNTHETIC_PRIVATE_LINE\n' + json.dumps({'state': 'DONE', 'counts': counts}) + '\n')
            self.assertEqual(sup.read_evaluator_result(path)['counts'], counts)
            counts['partial'] = '2'
            path.write_text(json.dumps({'state': 'DONE', 'counts': counts}))
            with self.assertRaises(ValueError): sup.read_evaluator_result(path)

    def test_old_guard_and_tagged_service_block_restart(self):
        root = Path('/workspace/results/own-test-499-v1')
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            p = proc/'12345'
            p.mkdir()
            (p/'cmdline').write_bytes(b'/python\0/work/evaluator_guard.py\0--results\0' + str(root/'EASY').encode() + b'\0')
            self.assertTrue(process_guard.prior_conflict(root, proc))
            (p/'cmdline').write_bytes(b'/python\0-m\0vllm.entrypoints.openai.api_server\0')
            (p/'environ').write_bytes(b'FIVEGPU_RUN_ROOT=' + str(root).encode() + b'\0')
            self.assertTrue(process_guard.prior_conflict(root, proc))
            (p/'environ').write_bytes(b'FIVEGPU_RUN_ROOT=/other\0')
            self.assertFalse(process_guard.prior_conflict(root, proc))

    def test_tagged_orphan_child_killed_without_pgid_reuse(self):
        tag = 'synthetic-' + os.urandom(8).hex()
        env = dict(os.environ, FIVEGPU_RUN_ID=tag)
        code = ('import os,time\n'
                'p=os.fork()\n'
                'if p == 0:\n time.sleep(60)\n os._exit(0)\n'
                'print(p,flush=True)\ntime.sleep(0.2)\nos._exit(0)\n')
        # Parent exits, leaving a child in its original session.
        proc = subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        child_fd = None
        try:
            child_pid = int(proc.stdout.readline().strip())
            child_fd = process_guard.open_pidfd(child_pid)
            proc.wait(timeout=5)
            self.assertFalse(select.select([child_fd], [], [], 0)[0])
            process_guard.terminate_tagged(proc, tag, grace=.2)
            self.assertTrue(select.select([child_fd], [], [], 3)[0])
        finally:
            if child_fd is not None:
                if not select.select([child_fd], [], [], 0)[0]:
                    process_guard.signal_pidfd(child_fd, signal.SIGKILL)
                os.close(child_fd)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()


class TestRecovery(unittest.TestCase):
    def test_skip_complete_held_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = item()
            self.assertEqual(guard.inspect_source(root, row), 'new')
            folder = synthetic_complete(root, row)
            self.assertEqual(guard.inspect_source(root, row), 'complete')
            (folder/'RAW_COMPLETE.json').unlink()
            self.assertEqual(guard.inspect_source(root, row), 'partial')
            (folder/'SOURCE_HELD.json').write_text('{}')
            self.assertEqual(guard.inspect_source(root, row), 'held')

    def test_invalid_complete_fails_without_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = item()
            folder = synthetic_complete(root, row)
            path = folder/'SOURCE.json'
            data = path.read_bytes()
            path.write_text('{"sha256":"wrong"}')
            before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(ValueError): guard.inspect_source(root, row)
            after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            path.write_bytes(data)
            (folder/'samples'/'source'/'pre'/'00'/'RESPONSE.json').unlink()
            with self.assertRaises(ValueError): guard.inspect_source(root, row)

    def test_no_rerun_partial_or_held(self):
        class Client:
            async def close(self): pass
        calls = []
        async def run_one(*args): calls.append(args)
        fake = types.SimpleNamespace(openai=types.SimpleNamespace(AsyncClient=lambda **_: Client()),
                                     local_url=lambda x:x, run_one=run_one)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = item('A'), item('B')
            for row in (first, second):
                folder = root/'sources'/guard.digest(row['source_id'].encode())[:24]
                folder.mkdir(parents=True)
            (root/'sources'/guard.digest(b'B')[:24]/'SOURCE_HELD.json').write_text('{}')
            stats = asyncio.run(guard.run([first, second], root, fake, Path('/model'), 'http://127.0.0.1:1/v1', 'http://127.0.0.1:2/v1'))
            self.assertEqual(stats['partial'], 1)
            self.assertEqual(stats['held'], 1)
            self.assertEqual(calls, [])

    def test_source_only_may_resume_but_any_request_prevents_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = item()
            folder = root/'sources'/guard.digest(row['source_id'].encode())[:24]
            folder.mkdir(parents=True)
            (folder/'SOURCE.json').write_text(json.dumps({'source_id': row['source_id'],
                                                         'sha256': guard.digest(guard.stable(row))}))
            self.assertEqual(guard.inspect_source(root, row), 'source_only')
            (folder/'samples').mkdir()
            self.assertEqual(guard.inspect_source(root, row), 'partial')
            (folder/'samples').rmdir()
            (folder/'SOURCE.json').write_text('{"source_id":"wrong"}')
            with self.assertRaises(ValueError): guard.inspect_source(root, row)

    def test_pilot_one_new_then_bulk_reuses_completed(self):
        class Client:
            async def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = item('PILOT'), item('BULK')
            calls = []
            async def run_one(row, *_):
                calls.append(row['source_id'])
                synthetic_complete(root, row)
            fake = types.SimpleNamespace(openai=types.SimpleNamespace(AsyncClient=lambda **_: Client()),
                                         local_url=lambda x:x, run_one=run_one)
            selected = []
            pilot = asyncio.run(guard.run([first, second], root, fake, Path('/model'),
                                          't', 's', max_new=1, on_attempt=selected.append))
            self.assertEqual(calls, ['PILOT'])
            self.assertEqual(len(selected), 1)
            self.assertEqual(pilot['new_complete'], 1)
            bulk = asyncio.run(guard.run([first, second], root, fake, Path('/model'), 't', 's'))
            self.assertEqual(calls, ['PILOT', 'BULK'])
            self.assertEqual(bulk['complete'], 1)
            self.assertEqual(bulk['new_complete'], 1)

    def test_one_invalid_or_failed_source_does_not_prevent_next(self):
        class Client:
            async def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken, failing, succeeding = item('BROKEN'), item('FAIL'), item('OK')
            folder = synthetic_complete(root, broken)
            (folder/'SOURCE.json').write_text('{"sha256":"wrong"}')
            calls = []
            async def run_one(row, *_):
                calls.append(row['source_id'])
                if row['source_id'] == 'FAIL':
                    raise RuntimeError('synthetic failure')
                synthetic_complete(root, row)
            def source_dir(row):
                return root/'sources'/guard.digest(row['source_id'].encode())[:24]
            def put_once(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as f:
                    json.dump(value, f)
            fake = types.SimpleNamespace(openai=types.SimpleNamespace(AsyncClient=lambda **_: Client()),
                       local_url=lambda x:x, run_one=run_one, source_dir=source_dir, put_once=put_once)
            stats = asyncio.run(guard.run([broken, failing, succeeding], root, fake, Path('/model'),
                                          'http://127.0.0.1:1/v1', 'http://127.0.0.1:2/v1'))
            self.assertEqual(stats['invalid'], 1)
            self.assertEqual(stats['failed'], 1)
            self.assertEqual(stats['new_complete'], 1)
            self.assertEqual(calls, ['FAIL', 'OK'])
            self.assertTrue((source_dir(failing)/'SOURCE_HELD.json').exists())
            self.assertFalse((source_dir(broken)/'SOURCE_HELD.json').exists())

    def test_joint_watcher_evaluator_receipts_no_real_api_or_regrading(self):
        parent = Path(__file__).resolve().parents[1]
        choices = (parent / 'agnes_eval_v3', parent / 'eduardo27b-github-update-20261109' / 'agnes_eval_v3')
        v3 = next(p for p in choices if p.is_dir())
        sys.path.insert(0, str(v3))
        try:
            import agnes_judge as judge
            import watch_sources as watcher
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                stage = root/'EASY'
                row = item('JOINT')
                source = synthetic_complete(stage, row)
                self.assertEqual(guard.inspect_source(stage, row), 'complete')
                calls = []
                def caller(*_):
                    calls.append(1)
                    return 200, {'model': judge.MODEL, 'choices': [
                        {'finish_reason': 'stop', 'message': {'content': 'true'}}]}
                with mock.patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                    self.assertTrue(watcher.process_source(source, root/'AGNES'/'EASY',
                        root/'AGNES'/'receipts'/'EASY', root/'AGNES'/'locks', 'fake-key', caller=caller, workers=5))
                    self.assertEqual(len(calls), 96)
                    results = json.loads((root/'AGNES'/'EASY'/'sources'/source.name/'RESULTS.json').read_bytes())
                    self.assertEqual(sum(r['status'] == 'correct' for r in results['results']), 96)
                    before = (root/'AGNES'/'EASY'/'sources'/source.name/'RESULTS.json').read_bytes()
                    self.assertTrue(watcher.process_source(source, root/'AGNES'/'EASY',
                        root/'AGNES'/'receipts'/'EASY', root/'AGNES'/'locks', 'fake-key',
                        caller=lambda *_: self.fail('must not regrade'), workers=5))
                    self.assertEqual(before, (root/'AGNES'/'EASY'/'sources'/source.name/'RESULTS.json').read_bytes())
                    self.assertEqual(sup.verify_pilot_grading(root, 'EASY', source.name)['correct'], 96)

                    # Semantic null is allowed; auth and widespread technical
                    # failures block bulk and are not mathematical judgments.
                    result_path = root/'AGNES'/'EASY'/'sources'/source.name/'RESULTS.json'
                    def set_unknown(index, code, content='null'):
                        r = results['results'][index]
                        r['status'] = 'unknown'
                        r['reason'] = 'undecidable_or_http_error'
                        receipt = (root/'AGNES'/'receipts'/'EASY'/source.name/
                                   judge.digest({'identity': r['identity']})/'ATTEMPT-0001-RESPONSE.json')
                        receipt.write_text(json.dumps({'attempt': 1, 'http_status': code,
                            'raw_response': {'model': judge.MODEL, 'choices': [
                                {'finish_reason': 'stop', 'message': {'content': content}}]}}))
                        result_path.write_text(json.dumps(results))
                    set_unknown(0, 200)
                    counts = sup.verify_pilot_grading(root, 'EASY', source.name)
                    self.assertEqual(counts['semantic_unknown'], 1)
                    self.assertEqual(counts['technical_unknown'], 0)
                    set_unknown(0, 401)
                    with self.assertRaises(ValueError): sup.verify_pilot_grading(root, 'EASY', source.name)
                    for i in range(10):
                        set_unknown(i, 500)
                    with self.assertRaises(ValueError): sup.verify_pilot_grading(root, 'EASY', source.name)
        finally:
            sys.path.remove(str(v3))


if __name__ == '__main__':
    unittest.main()
