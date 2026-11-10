#!/usr/bin/env python3
"""Four/five-GPU supervisor; requires an explicitly hashed watcher.

Three process-isolated teachers and one or two shared student services. Original
result stores, source-level receipts, prompts and sampling logic remain in
place. Start only after all previous controllers and workers have stopped.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
import process_guard

STAGES = ('EASY', 'MEDIUM', 'HARD')
V4_HASHES = {'coordinator.py': '29be65ad54276e426b221b9d89e404f20522450f63ac220080a8475785f88456',
             'lane_launch.py': '8b446a74b293a382160b7a6c139995172b538fa7f19f388c4f19c6578ea577e8'}
# Only the pair that passed offline joint receipt tests and target synthetic
# API testing is accepted. Caller-provided hashes alone are not sufficient.
APPROVED_GRADER_PAIRS = frozenset({
    ('8141d1205d2d47172aa058fa28882dae925c4082cae391b12427499c650d0af6',
     '2ffe746444939555b85e3557d2ddc3b61e3cd59d7997263909668876b924c871')})


class PreflightFailure(Exception):
    """Fixed step and exception class only; never expose exception text."""
    def __init__(self, step, error_type):
        super().__init__(step)
        self.step, self.error_type = step, error_type


def preflight(step, operation):
    try:
        return operation()
    except Exception as exc:
        raise PreflightFailure(step, type(exc).__name__) from exc


def model_signature(teacher, student):
    """Cheap identity snapshot for exactly the files verified by v4."""
    names = (
        (teacher, ('config.json', 'model-00001-of-00002.safetensors',
                   'model-00002-of-00002.safetensors', 'tokenizer.json',
                   'tokenizer_config.json', 'chat_template.jinja')),
        (student, ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                   'model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors',
                   'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors')),
    )
    out = []
    for root, files in names:
        for name in files:
            s = (root / name).stat()
            out.append((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return tuple(out)


def verify_models_stable(old, teacher, student):
    before = model_signature(teacher, student)
    old.verify_models(teacher, student)
    after = model_signature(teacher, student)
    if before != after:
        raise ValueError('model changed while hashing')
    return after


def choose_free_four(snapshot):
    """Return 4 distinct fully free A100s, or None while resources are busy."""
    selected, seen = [], set()
    for index in sorted(snapshot):
        item = snapshot[index]
        if (not isinstance(item, dict) or 'A100' not in item.get('name', '') or
                item.get('total', 0) < 75000 or item.get('free', 0) < .93 * item.get('total', 0) or
                item.get('util', 100) >= 10 or not item.get('uuid', '').startswith('GPU-') or
                item['uuid'] in seen):
            continue
        selected.append(index)
        seen.add(item['uuid'])
        if len(selected) == 4:
            return selected
    return None


def choose_free_elastic(snapshot):
    """Prefer five free devices, then four, three or two."""
    selected, seen = [], set()
    for index in sorted(snapshot):
        item = snapshot[index]
        if (not isinstance(item, dict) or 'A100' not in item.get('name', '') or
                item.get('total', 0) < 75000 or item.get('free', 0) < .93 * item.get('total', 0) or
                item.get('util', 100) >= 10 or not item.get('uuid', '').startswith('GPU-') or
                item['uuid'] in seen):
            continue
        selected.append(index)
        seen.add(item['uuid'])
        if len(selected) == 5:
            return selected
    return selected if len(selected) >= 2 else None


def wait_for_four(old, args, signature, cancelled, sleep=time.sleep, *, elastic=False, port_pool=None):
    """Retry only shortage of genuinely free cards, never other failures."""
    while True:
        if cancelled():
            raise InterruptedError('stopped while waiting for GPU')
        snapshot = preflight('GPU_QUERY', old.gpu_snapshot)
        chosen = choose_free_elastic(snapshot) if elastic else choose_free_four(snapshot)
        if chosen is not None:
            def no_existing_workers():
                if (old.old_process_running(args.root, args.repo) or watcher_conflict(args.root) or
                        process_guard.prior_conflict(args.root)):
                    raise ValueError('previous worker still running')
            preflight('WORKER_CONFLICT', no_existing_workers)
            # These are cheap and must still match after potentially hours waiting.
            preflight('CODE_BINDING', lambda: old.verify_old_binding(args.repo, args.root))
            preflight('DATA_BINDING', lambda: old.verify_inputs(args.root, args.data))
            preflight('PILOT', lambda: old.verify_pilot(args.root))
            preflight('ENVIRONMENT', lambda: old.verify_versions(Path(sys.executable), args.root))
            current = preflight('MODEL_METADATA', lambda: model_signature(args.teacher, args.student))
            if current != signature:
                signature = preflight('MODEL_HASH', lambda: verify_models_stable(old, args.teacher, args.student))
            # Re-snapshot after rechecking; only GPU shortage loops.
            fresh = preflight('GPU_QUERY', old.gpu_snapshot)
            try:
                ids = verify_gpu_snapshot(chosen, fresh)
            except ValueError as exc:
                if str(exc) != 'gpu unavailable':
                    raise PreflightFailure('GPU_VALIDATION', type(exc).__name__) from exc
            else:
                args.gpus[:] = chosen
                if elastic:
                    if port_pool is None or len(port_pool) < len(chosen):
                        raise PreflightFailure('PORTS', 'ValueError')
                    args.ports = list(port_pool[:len(chosen)])
                print(json.dumps({'state': 'GPU_READY', 'available_gpus': len(chosen)}), flush=True)
                return ids, signature
        print(json.dumps({'state': 'WAITING_FOR_FREE_GPUS',
                          'available_gpus': len(chosen) if chosen is not None else ('fewer_than_2' if elastic else 'fewer_than_4')}), flush=True)
        for _ in range(30):
            if cancelled():
                raise InterruptedError('stopped while waiting for GPU')
            sleep(1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def import_checked(path, expected, name):
    if not path.is_file() or path.is_symlink() or sha(path) != expected:
        raise ValueError('frozen v4 helper mismatch')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def take_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except Exception:
        os.close(fd)
        raise


def parse_devices(values, ports=False):
    if len(values) not in (2, 3, 4, 5) or any(type(x) is not int or x < 0 for x in values) or len(set(values)) != len(values):
        raise ValueError('two to five distinct values required')
    if ports and any(x < 1024 or x > 65535 for x in values):
        raise ValueError('invalid port')
    return tuple(values)


def verify_gpu_snapshot(gpus, snapshot):
    ids = []
    for index in parse_devices(gpus):
        d = snapshot.get(index)
        if (not d or 'A100' not in d.get('name', '') or d.get('total', 0) < 75000 or
            d.get('free', 0) < .93 * d.get('total', 0) or d.get('util', 100) >= 10 or
            not d.get('uuid', '').startswith('GPU-')):
            raise ValueError('gpu unavailable')
        ids.append(d['uuid'])
    if len(set(ids)) != len(gpus):
        raise ValueError('duplicated physical GPU')
    return ids


def verify_same_devices(gpus, uuids, snapshot):
    current_uuids = verify_gpu_snapshot(gpus, snapshot)
    if current_uuids != uuids:
        raise PreflightFailure('GPU_VALIDATION', 'ValueError')
    return current_uuids


def assert_ports_free(ports):
    socks = []
    try:
        for port in parse_devices(ports, ports=True):
            sock = socket.socket()
            socks.append(sock)
            sock.bind(('127.0.0.1', port))
    finally:
        for sock in socks:
            sock.close()


def watcher_conflict(root, proc=Path('/proc')):
    expected = str(root / 'AGNES' / 'receipts').encode()
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            raise ValueError('cannot inspect processes')
        if b'--receipts-dir' in argv:
            i = argv.index(b'--receipts-dir')
            if i + 1 < len(argv) and argv[i + 1] == expected:
                return True
    return False


def service_plans(python, teacher, student, gpu, ports):
    teacher_count = 1 if len(gpu) == 2 else (2 if len(gpu) == 3 else 3)
    tgpu, sgpu = gpu[:teacher_count], gpu[teacher_count:]
    tp, sp = ports[:teacher_count], ports[teacher_count:]
    prefix = [str(python), '-m', 'vllm.entrypoints.openai.api_server', '--host', '127.0.0.1']
    teacher_commands = [prefix + ['--port', str(port), '--model', str(teacher),
        '--served-model-name', 'Eduardo-27b', '--dtype', 'bfloat16',
        '--tensor-parallel-size', '1', '--max-model-len', '20000', '--gpu-memory-utilization', '0.85',
        '--max-num-seqs', '1', '--enforce-eager', '--limit-mm-per-prompt', '{"image":0,"video":0}', '--seed', '42']
        for port in tp]
    student_commands = [prefix + ['--port', str(port), '--model', str(student),
        '--served-model-name', 'Llama-3.1-8B-Instruct', '--dtype', 'float16',
        '--max-model-len', '20000', '--gpu-memory-utilization', '0.90', '--seed', '42']
        for port in sp]
    return list(zip(teacher_commands + student_commands, tgpu + sgpu))


def evaluator_plans(python, guard, root, repo, data, teacher, ports):
    if len(ports) == 2:
        teacher_route, student_route = (ports[0],) * 3, (ports[1],) * 3
    elif len(ports) == 3:
        teacher_route, student_route = (ports[0], ports[1], ports[0]), (ports[2],) * 3
    else:
        teacher_route = ports[:3]
        student_route = (ports[3], ports[3], ports[4] if len(ports) == 5 else ports[3])
    result = []
    for i, stage in enumerate(STAGES):
        source = root / 'easy-rest.jsonl' if stage == 'EASY' else data
        result.append([str(python), str(guard), '--repo', str(repo), '--source', str(source),
            '--results', str(root / stage), '--teacher-model', str(teacher),
            '--teacher-url', f'http://127.0.0.1:{teacher_route[i]}/v1',
            '--student-url', f'http://127.0.0.1:{student_route[i]}/v1', '--difficulty', stage.lower()])
    return result


def call_local(port, model, tokens=16):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': 'What is 2+3?'}],
                       'temperature': 0, 'max_tokens': tokens}).encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer loopback-only'})
    with opener.open(req, timeout=180) as response:
        if response.status != 200:
            raise ValueError('local smoke status')
        data = json.loads(response.read(262145))
    choices = data.get('choices') if isinstance(data, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('local smoke response')
    msg = choices[0].get('message')
    if not isinstance(msg, dict) or not any(isinstance(msg.get(k), str) and msg[k].strip()
                                            for k in ('content', 'reasoning_content', 'reasoning')):
        raise ValueError('empty smoke response')


def ready_local(proc, port, expected_model, cancelled, timeout=1800):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancelled() or proc.poll() is not None:
            raise RuntimeError('model start interrupted')
        try:
            with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                healthy = response.status == 200
            if healthy:
                with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
                    data = json.loads(response.read(1048577))
                if (isinstance(data, dict) and isinstance(data.get('data'), list) and
                    any(isinstance(x, dict) and x.get('id') == expected_model for x in data['data']) and
                    proc.poll() is None and not cancelled()):
                    return
        except (OSError, ValueError, TypeError):
            pass
        time.sleep(2)
    raise TimeoutError('model start timed out')


def smoke_services(ports):
    # Use synthetic inputs only; verify both shared student replicas handle eight
    # concurrent requests; this is not a real-data or max-context verification.
    teacher_count = 1 if len(ports) == 2 else (2 if len(ports) == 3 else 3)
    for port in ports[:teacher_count]:
        call_local(port, 'Eduardo-27b', 32)
    requests_per_student = 12 if len(ports) in (2, 3, 4) else 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_student * len(ports[teacher_count:])) as pool:
        futures = [pool.submit(call_local, port, 'Llama-3.1-8B-Instruct', 32)
                   for port in ports[teacher_count:] for _ in range(requests_per_student)]
        for f in futures:
            f.result()


def terminate_own(proc, run_tag):
    process_guard.terminate_tagged(proc, run_tag)


class WatcherManager:
    """Keep at most one grader alive; generation is independent of restarts."""
    def __init__(self, starter):
        self.starter = starter
        self.proc = None
        self.failures = 0
        self.started_at = None
        self.retry_at = 0.0

    def tick(self, now):
        if self.proc is not None:
            if self.proc.poll() is None:
                if now - self.started_at >= 30:
                    self.failures = 0
                return True
            self.proc = None
            self.failures += 1
            self.retry_at = now + min(60, 2 ** min(self.failures - 1, 6))
        if now < self.retry_at:
            return False
        try:
            self.proc = self.starter()
            self.started_at = now
            return True
        except (OSError, RuntimeError):
            self.failures += 1
            self.retry_at = now + min(60, 2 ** min(self.failures - 1, 6))
            return False


def read_evaluator_result(path, pilot=False):
    required = {'complete', 'new_complete', 'held', 'partial', 'invalid', 'failed'}
    with path.open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 1048576))
        lines = f.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict) or data.get('state') != 'DONE':
            continue
        counts = data.get('counts')
        if (isinstance(counts, dict) and set(counts) == required and
                all(type(n) is int and n >= 0 for n in counts.values())):
            if not pilot:
                return {'counts': counts}
            trial = data.get('pilot')
            if not isinstance(trial, dict) or trial.get('state') not in ('NO_NEW', 'COMPLETE', 'FAILED'):
                break
            key = trial.get('source_key')
            if ((trial['state'] == 'NO_NEW' and key is not None) or
                    (trial['state'] != 'NO_NEW' and (not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{24}', key)))):
                break
            return {'counts': counts, 'pilot': trial}
    raise ValueError('missing evaluator summary')


def verify_pilot_grading(root, stage, key):
    source = root / stage / 'sources' / key
    out = root / 'AGNES' / stage / 'sources' / key
    if (out / 'HELD.json').exists():
        raise ValueError('pilot grading held')
    result_path = out / 'RESULTS.json'
    if not result_path.is_file():
        return None
    raw = (source / 'SCORE_INPUT.json').read_bytes()
    rows = json.loads(raw)
    doc = json.loads(result_path.read_bytes())
    if (not isinstance(rows, list) or len(rows) != 96 or not isinstance(doc, dict) or
            doc.get('input_sha256') != hashlib.sha256(raw).hexdigest() or
            doc.get('completion_parser_version') != 'agnes_full_response_v2_topbool_v1' or
            not rows or doc.get('source_id') != rows[0].get('source_id')):
        raise ValueError('pilot grading binding mismatch')
    def identity(r):
        return (r.get('source_id'), r.get('target'), r.get('phase'), r.get('index'))
    results = doc.get('results')
    if (not isinstance(results, list) or len(results) != 96 or
            any(not isinstance(r, dict) or not isinstance(r.get('identity'), dict) or
                r.get('status') not in ('correct', 'incorrect', 'unknown') for r in results) or
            len({identity(r) for r in rows}) != 96 or
            len({identity(r['identity']) for r in results}) != 96 or
            {identity(r['identity']) for r in results} != {identity(r) for r in rows}):
        raise ValueError('pilot grading identity mismatch')
    counts = {s: sum(r['status'] == s for r in results) for s in ('correct', 'incorrect', 'unknown')}
    counts.update(semantic_unknown=0, technical_unknown=0, auth_failure=0)
    for result in results:
        if result['status'] != 'unknown':
            continue
        request = json.dumps({'identity': result['identity']}, ensure_ascii=False,
                             sort_keys=True, separators=(',', ':')).encode('utf-8')
        request_key = hashlib.sha256(request).hexdigest()
        directory = root / 'AGNES' / 'receipts' / stage / key / request_key
        receipts = sorted(directory.glob('ATTEMPT-????-RESPONSE.json'))
        response = json.loads(receipts[-1].read_bytes()) if receipts else {}
        code = response.get('http_status') if isinstance(response, dict) else None
        if code in (401, 403):
            counts['auth_failure'] += 1
        semantic = False
        try:
            raw = response['raw_response']
            choice = raw['choices'][0]
            text = choice['message']['content'].strip()
            if text.startswith('```') and text.endswith('```'):
                lines = text.splitlines()
                if len(lines) >= 3 and lines[0].strip().lower() in ('```', '```json') and lines[-1].strip() == '```':
                    text = '\n'.join(lines[1:-1]).strip()
            # Accept only exact null forms; no JSON object with duplicate keys.
            semantic = (code == 200 and raw['model'] == 'agnes-3.0-flash' and
                        choice['finish_reason'] == 'stop' and
                        (text == 'null' or re.fullmatch(r'\{\s*"correct"\s*:\s*null\s*\}', text) is not None))
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        counts['semantic_unknown' if semantic else 'technical_unknown'] += 1
    if not counts['correct'] + counts['incorrect']:
        raise ValueError('pilot grading has no definitive response')
    # 96 responses: >=10 technical unknowns indicates a service-format outage.
    # This gate evaluates availability, not mathematical judging quality.
    if counts['auth_failure'] or counts['technical_unknown'] >= 10:
        raise ValueError('pilot grading service unavailable')
    return counts


def wait_evaluators(evaluators, services, watcher, cancelled, student_count, run_tag, sleep=time.sleep):
    """Only the evaluators depending on a failed service are terminated."""
    failed_services = set()
    while any(p.poll() is None for p in evaluators):
        watcher.tick(time.monotonic())
        if cancelled():
            raise InterruptedError('cancelled during generation')
        for i, evaluator in enumerate(evaluators):
            teacher_index = 0 if len(services) == 2 else ((0, 1, 0)[i] if len(services) == 3 else i)
            student_index = (1 if len(services) == 2 else
                             (2 if len(services) == 3 else (3 if student_count == 1 or i < 2 else 4)))
            if evaluator.poll() is None and (services[teacher_index].poll() is not None or
                                             services[student_index].poll() is not None):
                failed_services.add(i)
                terminate_own(evaluator, run_tag)
        if any(p.poll() is None for p in evaluators):
            sleep(2)
    if failed_services or any(p.returncode != 0 for p in evaluators):
        raise RuntimeError('one or more evaluators failed or lost services')


def run(args):
    os.umask(0o077)
    if not args.run:
        raise ValueError('run flag required')
    # Delimit this invocation so a status check cannot report an older topology.
    print(json.dumps({'state': 'STARTING'}), flush=True)
    wait_enabled = bool(getattr(args, 'wait_gpus', False))
    elastic = bool(getattr(args, 'elastic', False))
    if elastic and not wait_enabled:
        raise ValueError('elastic requires waiting mode')
    paths = (args.root, args.repo, args.v4_dir, args.data, args.teacher, args.student, args.key, args.watcher, args.judge)
    if any(p is None or not p.is_absolute() for p in paths):
        raise ValueError('absolute paths and watcher required')
    if args.root != Path('/workspace/results/own-test-499-v1'):
        raise ValueError('unexpected result root')
    if not all(re.fullmatch(r'[0-9a-f]{64}', h or '') for h in (args.watcher_sha256, args.judge_sha256)):
        raise ValueError('watcher and judge SHA required')
    if (args.watcher_sha256, args.judge_sha256) not in APPROVED_GRADER_PAIRS:
        raise ValueError('grader pair not yet approved by integration tests')
    if (args.watcher.is_symlink() or not args.watcher.is_file() or sha(args.watcher) != args.watcher_sha256 or
        args.judge.is_symlink() or not args.judge.is_file() or sha(args.judge) != args.judge_sha256 or
        args.judge.name != 'agnes_judge.py' or args.judge.parent != args.watcher.parent):
        raise ValueError('grader file binding mismatch')
    if not args.key.is_file() or not args.key.stat().st_size:
        raise PreflightFailure('KEY', 'ValueError')
    def probe_pidfd():
        probe_fd = process_guard.open_pidfd(os.getpid())
        try:
            process_guard.signal_pidfd(probe_fd, 0)
        finally:
            os.close(probe_fd)
    preflight('PIDFD', probe_pidfd)
    parse_devices(args.gpus)
    parse_devices(args.ports, ports=True)
    if len(args.gpus) != len(args.ports):
        raise ValueError('GPU and port counts must match')
    if elastic and len(args.gpus) != 5:
        raise ValueError('elastic waiting requires five GPU placeholders')
    port_pool = tuple(args.ports)
    old = preflight('V4_CONTROLLER', lambda: import_checked(args.v4_dir / 'coordinator.py', V4_HASHES['coordinator.py'], 'five_old'))
    lane = preflight('V4_LAUNCHER', lambda: import_checked(args.v4_dir / 'lane_launch.py', V4_HASHES['lane_launch.py'], 'five_lane'))
    locks, jobs = [], []
    interrupted = [False]
    def stop(*_):
        interrupted[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        locks.append(preflight('ROOT_LOCK', lambda: take_lock(Path('/workspace/.own-test-499.lock'))))
        for stage in STAGES:
            directory = args.root / stage
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            locks.append(preflight('STAGE_LOCK', lambda: take_lock(directory / '.run.lock')))
        if (old.old_process_running(args.root, args.repo) or watcher_conflict(args.root) or
                process_guard.prior_conflict(args.root)):
            raise PreflightFailure('WORKER_CONFLICT', 'ValueError')
        print(json.dumps({'state': 'VERIFYING_FROZEN_INPUTS'}), flush=True)
        preflight('CODE_BINDING', lambda: old.verify_old_binding(args.repo, args.root))
        preflight('DATA_BINDING', lambda: old.verify_inputs(args.root, args.data))
        preflight('PILOT', lambda: old.verify_pilot(args.root))
        preflight('ENVIRONMENT', lambda: old.verify_versions(Path(sys.executable), args.root))
        if wait_enabled:
            print(json.dumps({'state': 'VERIFYING_MODEL_HASH'}), flush=True)
            signature = preflight('MODEL_HASH', lambda: verify_models_stable(old, args.teacher, args.student))
            uuids, signature = wait_for_four(old, args, signature, lambda: interrupted[0],
                                             elastic=elastic, port_pool=port_pool)
        else:
            preflight('MODEL_HASH', lambda: old.verify_models(args.teacher, args.student))
            snapshot = preflight('GPU_QUERY', old.gpu_snapshot)
            uuids = preflight('GPU_VALIDATION', lambda: verify_gpu_snapshot(args.gpus, snapshot))
        preflight('PORTS', lambda: assert_ports_free(args.ports))
        env = lane.clean_env(args.repo, args.root)
        run_tag = uuid.uuid4().hex
        env.update(FIVEGPU_RUN_ID=run_tag, FIVEGPU_RUN_ROOT=str(args.root))
        while True:
            # Elastic re-selection may expand the port set (e.g. two to four GPUs).
            # Check the currently selected set after every resource re-selection.
            preflight('PORTS', lambda: assert_ports_free(args.ports))
            for index, device_uuid in zip(args.gpus, uuids):
                preflight('CUDA_DEVICE', lambda: lane.verify_numeric_cuda(index, device_uuid, env))
            if not wait_enabled:
                break
            fresh = preflight('GPU_QUERY', old.gpu_snapshot)
            try:
                verify_same_devices(args.gpus, uuids, fresh)
            except ValueError as exc:
                if str(exc) != 'gpu unavailable':
                    raise PreflightFailure('GPU_VALIDATION', type(exc).__name__) from exc
                uuids, signature = wait_for_four(old, args, signature, lambda: interrupted[0],
                                                 elastic=elastic, port_pool=port_pool)
                continue
            break
        for stage in STAGES:
            source = args.root / 'easy-rest.jsonl' if stage == 'EASY' else args.data
            # Compare with the exact v4 configuration; never overwrite it.
            class ConfigArgs:
                pass
            cfg_args = ConfigArgs()
            cfg_args.source, cfg_args.teacher_model, cfg_args.student_model = source, args.teacher, args.student
            cfg_args.difficulty = stage.lower()
            versions = old.load(args.root / 'PILOT' / 'RUN_CONFIGURATION.json')['installed_versions']
            cfg = lane.config_for(cfg_args, versions)
            lane.bind_config(args.root / stage / 'RUN_CONFIGURATION.json', cfg,
                             lambda path, value: old.write_json(path, value, once=True))
        bind = {'coordinator_sha256': sha(Path(__file__)),
                'evaluator_guard_sha256': sha(Path(__file__).with_name('evaluator_guard.py')),
                'process_guard_sha256': sha(Path(__file__).with_name('process_guard.py')),
                'watcher_sha256': args.watcher_sha256, 'judge_sha256': args.judge_sha256, 'model_student_semaphore': 4,
                'source_dataset_sha256': old.DATA_SHA, 'old_code_binding_sha256': old.sha(args.root / 'CODE_BINDING.json'),
                'gpus': list(args.gpus), 'gpu_uuids': uuids, 'ports': list(args.ports),
                'gpu_selection': ('wait_for_free_two_to_five' if elastic else
                                  ('wait_for_free_four' if wait_enabled else 'specified')),
                'routing': ({'teacher': {s: args.ports[0] for s in STAGES},
                             'student': {s: args.ports[1] for s in STAGES}}
                            if len(args.ports) == 2 else
                            ({'teacher': dict(zip(STAGES, (args.ports[0], args.ports[1], args.ports[0]))),
                              'student': {s: args.ports[2] for s in STAGES}}
                             if len(args.ports) == 3 else
                            {'teacher': dict(zip(STAGES, args.ports[:3])),
                             'student': {'EASY': args.ports[3], 'MEDIUM': args.ports[3],
                                         'HARD': args.ports[4] if len(args.ports) == 5 else args.ports[3]}}))}
        stamp = f'{time.time_ns()}-{os.getpid()}.json'
        old.write_json(args.root / 'FIVEGPU_EXECUTIONS' / stamp, bind, once=True)
        print(json.dumps({'state': 'STARTING_SERVICES'}), flush=True)
        def start(cmd, name, proc_env):
            with (args.root / f'fivegpu-{name}.log').open('ab') as log:
                proc = subprocess.Popen(cmd, env=proc_env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, close_fds=True)
            jobs.append(proc)
            return proc
        services = []
        plans = service_plans(sys.executable, args.teacher, args.student, args.gpus, args.ports)
        for i, (cmd, gpu) in enumerate(plans):
            services.append(start(cmd, f'service-{i}', lane.numeric_device_env(env, gpu)))
        teacher_count = 1 if len(args.ports) == 2 else (2 if len(args.ports) == 3 else 3)
        for i, proc in enumerate(services):
            ready_local(proc, args.ports[i], 'Eduardo-27b' if i < teacher_count else 'Llama-3.1-8B-Instruct',
                        lambda: interrupted[0])
        smoke_services(args.ports)
        if interrupted[0]:
            raise InterruptedError('interrupted before evaluation')
        agnes = args.root / 'AGNES'
        (agnes / 'locks').mkdir(mode=0o700, parents=True, exist_ok=True)
        watcher_cmd = [sys.executable, str(args.watcher), '--stages-root', str(args.root),
            '--output-dir', str(agnes), '--receipts-dir', str(agnes / 'receipts'),
            '--lock-dir', str(agnes / 'locks'), '--key-file', str(args.key), '--workers', '5']
        watcher = WatcherManager(lambda: start(watcher_cmd, 'watcher', env))
        watcher.tick(time.monotonic())
        commands = evaluator_plans(sys.executable, Path(__file__).with_name('evaluator_guard.py'),
                                   args.root, args.repo, args.data, args.teacher, args.ports)
        def evaluate_phase(phase, pilot=False):
            evaluators, eval_logs = [], []
            for i, cmd in enumerate(commands):
                log_name = f'{phase}-{STAGES[i]}-{stamp}'
                eval_logs.append(args.root / f'fivegpu-{log_name}.log')
                evaluators.append(start(cmd + (['--max-new', '1'] if pilot else []), log_name, env))
            wait_evaluators(evaluators, services, watcher, lambda: interrupted[0],
                            len(args.ports) - teacher_count, run_tag)
            return {stage: read_evaluator_result(path, pilot=pilot) for stage, path in zip(STAGES, eval_logs)}

        print(json.dumps({'state': 'PILOT_RUNNING'}), flush=True)
        pilots = evaluate_phase('pilot', pilot=True)
        pilot_keys = {}
        for stage, value in pilots.items():
            info = value['pilot']
            if info['state'] == 'FAILED':
                raise RuntimeError('pilot source failed')
            if info['state'] == 'COMPLETE':
                pilot_keys[stage] = info['source_key']
        pilot_counts = {}
        pilot_deadline = time.monotonic() + 1800
        while len(pilot_counts) != len(pilot_keys):
            watcher.tick(time.monotonic())
            if interrupted[0]:
                raise InterruptedError('interrupted during pilot grading')
            for stage, key in pilot_keys.items():
                if stage not in pilot_counts:
                    graded = verify_pilot_grading(args.root, stage, key)
                    if graded is not None:
                        pilot_counts[stage] = graded
            if len(pilot_counts) == len(pilot_keys):
                break
            if time.monotonic() >= pilot_deadline:
                print(json.dumps({'state': 'PILOT_GRADING_PENDING',
                                  'pending_sources': len(pilot_keys) - len(pilot_counts)}), flush=True)
                return
            time.sleep(5)
        print(json.dumps({'state': 'PILOT_GRADED', 'sources': len(pilot_keys),
                          'counts': list(pilot_counts.values())}), flush=True)
        print(json.dumps({'state': 'BULK_RUNNING'}), flush=True)
        eval_summary = {stage: value['counts'] for stage, value in evaluate_phase('bulk').items()}
        # Monitor the existing grader until completed sources have RESULTS or HELD.
        unavailable_since = None
        last_pending = old.grading_pending(args.root)
        last_progress = time.monotonic()
        while old.grading_pending(args.root):
            alive = watcher.tick(time.monotonic())
            if interrupted[0]:
                raise InterruptedError('interrupted while grading')
            pending = old.grading_pending(args.root)
            if pending < last_pending:
                last_progress = time.monotonic()
            last_pending = pending
            if not alive and watcher.failures >= 3:
                unavailable_since = unavailable_since or time.monotonic()
            elif alive and watcher.failures == 0:
                unavailable_since = None
            if ((unavailable_since is not None and time.monotonic() - unavailable_since >= 300) or
                    time.monotonic() - last_progress >= 2700):
                print(json.dumps({'state': 'GENERATION_DONE_GRADING_PENDING',
                                  'pending_sources': old.grading_pending(args.root),
                                  'evaluators': eval_summary}), flush=True)
                return
            time.sleep(15)
        summary = {'easy': old.aggregate(args.root, ('PILOT', 'EASY')),
                   'medium': old.aggregate(args.root, ('MEDIUM',)), 'hard': old.aggregate(args.root, ('HARD',))}
        unresolved = any(any(counts[k] for k in ('held', 'partial', 'invalid', 'failed'))
                         for counts in eval_summary.values())
        unresolved |= any(counts['unknown'] or counts['grading_held'] for counts in summary.values())
        print(json.dumps({'state': 'SCANNED_WITH_UNRESOLVED' if unresolved else 'SCANNED_COMPLETE',
                          'summary': summary, 'evaluators': eval_summary}), flush=True)
    finally:
        for p in reversed(jobs):
            terminate_own(p, run_tag)
        for fd in reversed(locks):
            os.close(fd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true')
    p.add_argument('--root', type=Path, default=Path('/workspace/results/own-test-499-v1'))
    p.add_argument('--repo', type=Path, default=Path('/workspace/eduardo27b-isolated-eval'))
    p.add_argument('--v4-dir', type=Path, default=Path('/workspace/eduardo27b-parallel-code/parallel_eval_v4'))
    p.add_argument('--data', type=Path, default=Path('/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl'))
    p.add_argument('--teacher', type=Path, default=Path('/workspace/models/Eduardo-27b-3bbf17e'))
    p.add_argument('--student', type=Path, default=Path('/workspace/models/Meta-Llama-3.1-8B-Instruct'))
    p.add_argument('--key', type=Path, default=Path('/workspace/.secrets/agnes_api_key'))
    p.add_argument('--watcher', type=Path)
    p.add_argument('--watcher-sha256')
    p.add_argument('--judge', type=Path)
    p.add_argument('--judge-sha256')
    p.add_argument('--gpus', type=int, nargs='+', required=True, metavar='GPU')
    p.add_argument('--wait-gpus', action='store_true', help='Wait for any four idle A100 GPUs after one model validation.')
    p.add_argument('--elastic', action='store_true', help='Start with two to five idle A100s.')
    p.add_argument('--ports', type=int, nargs='+', metavar='PORT')
    args = p.parse_args()
    if args.ports is None:
        args.ports = tuple(range(18001, 18001 + len(args.gpus)))
    if args.wait_gpus and len(args.gpus) != (5 if args.elastic else 4):
        raise ValueError('elastic waiting requires five GPU placeholders; other waiting requires four')
    run(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if isinstance(exc, PreflightFailure):
            output = {'state': 'ERROR', 'step': exc.step, 'error_type': exc.error_type}
        else:
            output = {'state': 'ERROR', 'error_type': type(exc).__name__}
        print(json.dumps(output), file=sys.stderr, flush=True)
        sys.exit(1)
