#!/usr/bin/env python3
"""Three-lane handoff; preserves the serial run and its generation receipts."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = Path('/workspace')
DATA_SHA = '5e5aa211549d221d2287529e5a4bac03006ba37d0dc7041c19b386f0eb4a5c01'
STAGES = ('EASY', 'MEDIUM', 'HARD')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data, once=False):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.parallel-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if once:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def take_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except Exception:
        os.close(fd)
        raise


def parse_pairs(values, label):
    try:
        pairs = [tuple(int(n) for n in x.split(',')) for x in values]
    except ValueError as exc:
        raise ValueError('invalid_' + label) from exc
    if len(pairs) != 3 or any(len(p) != 2 for p in pairs):
        raise ValueError('invalid_' + label)
    flat = [n for pair in pairs for n in pair]
    if len(set(flat)) != 6 or any(n < 0 for n in flat):
        raise ValueError('duplicate_or_invalid_' + label)
    if label == 'ports' and any(n < 1024 or n > 65535 for n in flat):
        raise ValueError('invalid_ports')
    return dict(zip(STAGES, pairs))


def verify_old_binding(repo, root):
    binding = load(root / 'CODE_BINDING.json')
    judge = repo / 'agnes_eval_v2'
    files = [(p.relative_to(repo).as_posix(), p) for d in ('app', 'official')
             for p in (repo / d).rglob('*.py') if p.is_file()]
    files += [('judge/' + n, judge / n) for n in
              ('run.sh', 'agnes_judge.py', 'watch_sources.py', 'status.py')]
    actual = {'files': {name: sha(p) for name, p in sorted(files)}}
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=False)
    actual['git_commit'] = commit.stdout.strip() if commit.returncode == 0 else None
    if actual != binding:
        raise ValueError('old_code_binding_mismatch')


def verify_inputs(root, data):
    if sha(data) != DATA_SHA:
        raise ValueError('dataset_hash_mismatch')
    lines = data.read_bytes().splitlines(keepends=True)
    if len(lines) != 499:
        raise ValueError('dataset_count_mismatch')
    expected = {'pilot.jsonl': lines[0], 'easy-rest.jsonl': b''.join(lines[1:])}
    for name, content in expected.items():
        if (root / name).read_bytes() != content:
            raise ValueError('frozen_input_mismatch')


def verify_models(teacher, student):
    expected = {
        teacher / 'config.json': 'ddeb906bee82e415882d04790e045a961503ac86ec08e107010dffc18cd8c83b',
        teacher / 'model-00001-of-00002.safetensors': '626abc50901718cd92703c081963ff16f42dff5ec45133d76f201999ec7ac4b4',
        teacher / 'model-00002-of-00002.safetensors': '7be58da8a22afb664ebdfdc9226c2ae30790ccd2a74ff076f8cfff92174d6cbf',
        teacher / 'tokenizer.json': '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
        teacher / 'tokenizer_config.json': '792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b',
        teacher / 'chat_template.jinja': '2fbf099c2eb9284b19cc6188139e6f8aa840a93f4f9fcc937b3b89ae9b4fa837',
        student / 'config.json': '29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e',
        student / 'tokenizer.json': '79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4',
        student / 'tokenizer_config.json': '177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424',
        student / 'model-00001-of-00004.safetensors': '2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668',
        student / 'model-00002-of-00004.safetensors': '09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15',
        student / 'model-00003-of-00004.safetensors': 'fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa',
        student / 'model-00004-of-00004.safetensors': '92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b'}
    for path, wanted in expected.items():
        if sha(path) != wanted:
            raise ValueError('model_hash_mismatch')


def verify_pilot(root):
    base = root / 'PILOT' / 'sources'
    groups = [p for p in base.iterdir() if p.is_dir()] if base.is_dir() else []
    if len(groups) != 1:
        raise ValueError('pilot_incomplete')
    p = groups[0]
    marker = p / 'RAW_COMPLETE.json'
    if (p / 'SOURCE_HELD.json').exists() or not marker.is_file() or load(marker).get('samples') != 96:
        raise ValueError('pilot_incomplete')
    if len(list((p / 'samples').glob('*/*/*/RESPONSE.json'))) != 96:
        raise ValueError('pilot_incomplete')
    grade = root / 'AGNES' / 'PILOT' / 'sources' / p.name
    if (grade / 'HELD.json').exists() or not (grade / 'RESULTS.json').is_file():
        raise ValueError('pilot_grading_incomplete')
    results = load(grade / 'RESULTS.json').get('results', [])
    if len(results) != 96 or sum(x.get('status') in ('correct', 'incorrect') for x in results) < 48:
        raise ValueError('pilot_grading_incomplete')


def verify_versions(python, root):
    """Require the worker interpreter to match the completed pilot's versions."""
    names = ('vllm', 'torch', 'transformers', 'openai', 'math-verify')
    program = ('import importlib.metadata as m,json; '
               'print(json.dumps({n:m.version(n) for n in '
               "('vllm','torch','transformers','openai','math-verify')}))")
    try:
        current = json.loads(subprocess.check_output([str(python), '-c', program],
                                                      text=True, timeout=30))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError('environment_check_failed') from exc
    try:
        pilot = load(root / 'PILOT' / 'RUN_CONFIGURATION.json')['installed_versions']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError('pilot_config_missing') from exc
    if (not isinstance(current, dict) or set(current) != set(names) or
            any(not isinstance(current[n], str) or not current[n] for n in names) or
            current.get('vllm') != '0.18.0' or current != pilot):
        raise ValueError('environment_version_mismatch')
    for stage in STAGES:
        config = root / stage / 'RUN_CONFIGURATION.json'
        if config.exists():
            try:
                saved = load(config)['installed_versions']
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError('environment_version_mismatch') from exc
            if saved != pilot:
                raise ValueError('environment_version_mismatch')


def possible_old_service(argv, extra_ports=()):
    # Prior v3/v4 may have launched either role on any one of its lane ports.
    ports = {b'8001', b'8002', b'8011', b'8012', b'8021', b'8022'}
    ports.update(str(p).encode() for p in extra_ports if type(p) is int and 1 <= p <= 65535)
    old_models = {
        (b'Eduardo-27b', b'/workspace/models/Eduardo-27b-3bbf17e'),
        (b'Llama-3.1-8B-Instruct', b'/workspace/models/Meta-Llama-3.1-8B-Instruct'),
    }

    def option(argv, name):
        try:
            return argv[argv.index(name) + 1]
        except (ValueError, IndexError):
            return None

    return (b'vllm.entrypoints.openai.api_server' in argv and
            option(argv, b'--port') in ports and
            (option(argv, b'--served-model-name'), option(argv, b'--model')) in old_models)


def old_process_running(root, repo):
    expected = str(root).encode()
    serial_script = str(repo / 'agnes_eval_v2' / 'run.sh').encode()
    app_scripts = (str(repo / 'app' / 'launch.py').encode(),
                   str(repo / 'app' / 'offline_eval.py').encode())
    prior_ports = []
    for version in ('V3', 'V4'):
        for receipt in (root / f'PARALLEL_{version}_EXECUTIONS').glob('*.json'):
            try:
                ports = load(receipt).get('port_pair', [])
                if isinstance(ports, list):
                    prior_ports.extend(ports)
            except (OSError, ValueError, AttributeError):
                raise ValueError('prior_execution_receipt_unreadable')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes()
        except (OSError, ProcessLookupError):
            continue
        if (b'watch_sources.py' in cmd and expected in cmd) or serial_script in cmd:
            return True
        argv = cmd.split(b'\0')
        # An orphaned prior lane may not yet have launched a model service.
        if any(x.endswith((b'/parallel_eval_v3/lane_launch.py',
                           b'/parallel_eval_v4/lane_launch.py')) for x in argv):
            try:
                output = argv[argv.index(b'--results') + 1]
            except (ValueError, IndexError):
                output = b''
            if output in {str(root / stage).encode() for stage in STAGES}:
                return True
        possible_app = any(script in argv for script in app_scripts)
        if possible_app or possible_old_service(argv, prior_ports):
            try:
                env = (proc / 'environ').read_bytes().split(b'\0')
            except FileNotFoundError:
                continue
            except OSError as exc:
                # An inaccessible possible worker cannot safely be assumed stopped.
                raise ValueError('process_state_unreadable') from exc
            if any(x.startswith(b'RESULTS_DIR=' + expected + b'/') for x in env):
                return True
    return False


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=30)
    result = {}
    for line in raw.splitlines():
        values = [s.strip() for s in line.split(',')]
        if len(values) != 6:
            raise ValueError('gpu_query_invalid')
        idx, uuid, name, total, free, util = values
        result[int(idx)] = {'uuid': uuid, 'name': name, 'total': int(total),
                            'free': int(free), 'util': int(util)}
    return result


def check_resources(gpus, ports, snapshot=None):
    snapshot = gpu_snapshot() if snapshot is None else snapshot
    for idx in [n for pair in gpus.values() for n in pair]:
        device = snapshot.get(idx)
        if not device or 'A100' not in device['name'] or device['total'] < 75000:
            raise ValueError('gpu_not_a100_80gb')
        if device['free'] < .93 * device['total'] or device['util'] >= 10:
            raise ValueError('gpu_busy')
    # Bind, not just connect: catch listeners and other bound local services.
    sockets = []
    try:
        for port in [n for pair in ports.values() for n in pair]:
            s = socket.socket()
            sockets.append(s)
            s.bind(('127.0.0.1', port))
    finally:
        for s in sockets:
            s.close()
    return snapshot


def available_pair(snapshot, reserved, preferred=None):
    """Return two free physical A100s, excluding GPUs reserved by active lanes."""
    candidates = list(preferred) if preferred is not None else sorted(snapshot)
    result = []
    uuids = {snapshot[i].get('uuid') for i in reserved if i in snapshot}
    for idx in candidates:
        d = snapshot.get(idx)
        if (idx in reserved or not d or 'A100' not in d.get('name', '') or
                d.get('total', 0) < 75000 or d.get('free', 0) < .93 * d.get('total', 0) or
                d.get('util', 100) >= 10 or not d.get('uuid', '').startswith('GPU-') or
                d.get('uuid') in uuids):
            continue
        result.append(idx)
        uuids.add(d['uuid'])
        if len(result) == 2:
            return tuple(result)
    return None


def ports_available(pair):
    sockets = []
    try:
        for port in pair:
            s = socket.socket()
            sockets.append(s)
            s.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        for s in sockets:
            s.close()


def check_stage_locks(root):
    for stage in STAGES:
        directory = root / stage
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = take_lock(directory / '.run.lock')
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def started_watcher_command(args):
    judge = args.repo / 'agnes_eval_v2' / 'watch_sources.py'
    base = args.root / 'AGNES'
    return [str(args.python), str(judge), '--stages-root', str(args.root), '--output-dir', str(base),
            '--receipts-dir', str(base / 'receipts'), '--lock-dir', str(base / 'locks'),
            '--key-file', str(args.key), '--workers', '5']


def lane_command(args, stage, gpus, ports):
    input_path = args.root / 'easy-rest.jsonl' if stage == 'EASY' else args.data
    return [str(args.python), str(args.launcher), '--difficulty', stage.lower(),
            '--teacher-gpu', str(gpus[0]), '--student-gpu', str(gpus[1]),
            '--teacher-port', str(ports[0]), '--student-port', str(ports[1]),
            '--source', str(input_path), '--results', str(args.root / stage),
            '--repo', str(args.repo), '--teacher-model', str(args.teacher),
            '--student-model', str(args.student)]


def source_keys(data_path):
    rows = [json.loads(x) for x in data_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    ids = [r['source_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_source_ids')
    return {hashlib.sha256(s.encode()).hexdigest()[:24] for s in ids}


def verify_terminal(root, stage, data_path):
    cfg = load(root / stage / 'RUN_CONFIGURATION.json')
    if (cfg.get('protocol') != 'own-transfer-fresh-offline-v1' or
            cfg.get('transfer_difficulty') != stage.lower() or cfg.get('source_sha256') != sha(data_path)):
        raise ValueError('stage_configuration_mismatch')
    expected = source_keys(data_path)
    base = root / stage / 'sources'
    found = {p.name: p for p in base.iterdir() if p.is_dir()} if base.is_dir() else {}
    if set(found) != expected:
        raise ValueError('incomplete_stage_sources')
    counts = {'complete': 0, 'held': 0}
    for folder in found.values():
        if (folder / 'RAW_COMPLETE.json').is_file():
            if load(folder / 'RAW_COMPLETE.json').get('samples') != 96:
                raise ValueError('invalid_source_receipt')
            if len(list((folder / 'samples').glob('*/*/*/RESPONSE.json'))) != 96:
                raise ValueError('invalid_source_receipt')
            counts['complete'] += 1
        elif (folder / 'SOURCE_HELD.json').is_file():
            counts['held'] += 1
        else:
            raise ValueError('incomplete_stage_sources')
    return counts


def grading_pending(root):
    pending = 0
    for stage in STAGES:
        for marker in (root / stage / 'sources').glob('*/RAW_COMPLETE.json'):
            dst = root / 'AGNES' / stage / 'sources' / marker.parent.name
            if not (dst / 'RESULTS.json').is_file() and not (dst / 'HELD.json').is_file():
                pending += 1
    return pending


def aggregate(root, stages):
    counts = {'sources_started': 0, 'generation_complete': 0, 'generation_held': 0,
              'grading_complete': 0, 'grading_held': 0, 'correct': 0, 'incorrect': 0, 'unknown': 0}
    for stage in stages:
        base = root / stage / 'sources'
        for source in base.iterdir() if base.is_dir() else ():
            if not source.is_dir():
                continue
            counts['sources_started'] += 1
            if (source / 'RAW_COMPLETE.json').is_file():
                counts['generation_complete'] += 1
            elif (source / 'SOURCE_HELD.json').is_file():
                counts['generation_held'] += 1
            grade = root / 'AGNES' / stage / 'sources' / source.name
            if (grade / 'RESULTS.json').is_file():
                counts['grading_complete'] += 1
                for row in load(grade / 'RESULTS.json').get('results', []):
                    if row.get('status') in ('correct', 'incorrect', 'unknown'):
                        counts[row['status']] += 1
            elif (grade / 'HELD.json').is_file():
                counts['grading_held'] += 1
    return counts


def status_summary(root):
    marker = root / 'PARALLEL_V4_STATUS.json'
    return {'controller': load(marker) if marker.is_file() else {'state': 'NOT_STARTED'},
            'easy': aggregate(root, ('PILOT', 'EASY')),
            'medium': aggregate(root, ('MEDIUM',)), 'hard': aggregate(root, ('HARD',))}


ERROR_CODES = frozenset(('old_results_missing', 'old_process_still_running', 'old_code_binding_mismatch',
    'process_state_unreadable', 'dataset_hash_mismatch', 'dataset_count_mismatch', 'frozen_input_mismatch',
    'prior_execution_receipt_unreadable',
    'model_hash_mismatch', 'pilot_incomplete', 'pilot_grading_incomplete', 'api_key_missing',
    'gpu_not_a100_80gb', 'gpu_busy', 'gpu_query_invalid', 'watcher_start_failed',
    'incomplete_stage_sources', 'invalid_source_receipt', 'stage_configuration_mismatch',
    'duplicate_source_ids', 'parallel_code_changed', 'environment_check_failed',
    'pilot_config_missing', 'environment_version_mismatch'))


def run(args):
    os.umask(0o077)
    gpus = parse_pairs(args.gpu_pairs, 'gpus') if args.gpu_pairs else None
    ports = parse_pairs(args.port_pairs or ['8001,8002', '8011,8012', '8021,8022'], 'ports')
    if not 1 <= args.max_lanes <= 2 or not 2 <= args.poll_seconds <= 300:
        raise ValueError('invalid_scheduler_settings')
    if not args.root.is_dir():
        raise ValueError('old_results_missing')
    lock = take_lock(BASE / '.own-test-499.lock')
    watcher = None
    children = {}
    logs = []
    status = args.root / 'PARALLEL_V4_STATUS.json'
    interrupted = [False]
    def stop(*_):
        interrupted[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        write_json(status, {'state': 'PRECHECK'})
        if old_process_running(args.root, args.repo):
            raise ValueError('old_process_still_running')
        verify_old_binding(args.repo, args.root)
        verify_inputs(args.root, args.data)
        verify_pilot(args.root)
        verify_versions(args.python, args.root)
        verify_models(args.teacher, args.student)
        if not args.key.is_file() or not args.key.stat().st_size:
            raise ValueError('api_key_missing')
        check_stage_locks(args.root)
        pending_stages = [s for s in STAGES if not (args.root / s / 'STAGE_FINISHED').is_file()]
        for stage in set(STAGES) - set(pending_stages):
            input_path = args.root / 'easy-rest.jsonl' if stage == 'EASY' else args.data
            verify_terminal(args.root, stage, input_path)
        binding = {'dataset_sha256': DATA_SHA, 'old_code_binding_sha256': sha(args.root / 'CODE_BINDING.json'),
                   'coordinator_sha256': sha(Path(__file__)), 'launcher_sha256': sha(args.launcher)}
        binding_path = args.root / 'PARALLEL_V4_BINDING.json'
        if binding_path.exists():
            if load(binding_path) != binding:
                raise ValueError('parallel_code_changed')
        else:
            write_json(binding_path, binding, once=True)
        exe_dir = args.root / 'PARALLEL_V4_EXECUTIONS'
        exe_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        state = {'state': 'PARALLEL_STARTING', 'stages': {s: 'pending' for s in STAGES}}
        write_json(status, state)
        agnes = args.root / 'AGNES'
        (agnes / 'locks').mkdir(mode=0o700, parents=True, exist_ok=True)
        watchlog = (agnes / 'parallel-v4-watch.log').open('ab')
        logs.append(watchlog)
        watcher = subprocess.Popen(started_watcher_command(args), stdout=watchlog, stderr=subprocess.STDOUT,
                                   close_fds=True, start_new_session=True)
        time.sleep(0.5)
        if watcher.poll() is not None:
            raise RuntimeError('watcher_start_failed')
        queue = []
        assignments = {}
        attempts = {s: 0 for s in STAGES}
        ready_at = {s: 0.0 for s in STAGES}
        ended_once = set()
        for stage in STAGES:
            if (args.root / stage / 'STAGE_FINISHED').is_file():
                input_path = args.root / 'easy-rest.jsonl' if stage == 'EASY' else args.data
                counts = verify_terminal(args.root, stage, input_path)
                state['stages'][stage] = 'already_terminal_with_held' if counts['held'] else 'already_complete'
                ended_once.add(stage)
                continue
            queue.append(stage)
            state['stages'][stage] = 'queued'
        state['state'] = 'WAITING_FOR_GPUS' if queue else 'PARALLEL_STARTING'
        write_json(status, state)
        watcher_restarts = 0

        def retry_or_fail(stage, final_state):
            # Preserve every existing request/receipt; the launcher reuses frozen output.
            if attempts[stage] < 4:
                ready_at[stage] = time.monotonic() + 2 ** (attempts[stage] - 1)
                state['stages'][stage] = 'retry_wait'
                queue.append(stage)
            else:
                state['stages'][stage] = final_state
                # No active process exists; permit other pending stages to proceed.
                ended_once.add(stage)

        while (children or queue) and not interrupted[0]:
            for stage, child in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                del children[stage]
                assignments.pop(stage, None)
                ended_once.add(stage)
                if code:
                    retry_or_fail(stage, 'failed')
                else:
                    try:
                        data_path = args.root / 'easy-rest.jsonl' if stage == 'EASY' else args.data
                        counts = verify_terminal(args.root, stage, data_path)
                        # This is an actual completed pass, not a fabricated skip marker.
                        marker = args.root / stage / 'STAGE_FINISHED'
                        try:
                            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                            os.close(fd)
                        except FileExistsError:
                            pass
                        state['stages'][stage] = 'terminal_with_held' if counts['held'] else 'complete'
                    except Exception:
                        retry_or_fail(stage, 'receipt_check_failed')
                write_json(status, state)
            if watcher.poll() is not None:
                if watcher_restarts < 3:
                    time.sleep(2 ** watcher_restarts)
                    watcher_restarts += 1
                    watcher = subprocess.Popen(started_watcher_command(args), stdout=watchlog,
                        stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
                    state['watcher'] = 'restarting'
                else:
                    state['watcher'] = 'failed'
                write_json(status, state)
            if queue and len(children) < args.max_lanes:
                try:
                    snapshot = gpu_snapshot()
                except Exception:
                    snapshot = {}
                    state['resources'] = 'gpu_query_failed'
                else:
                    state['resources'] = 'ok'
                reserved = {n for pair in assignments.values() for n in pair}
                for stage in list(queue):
                    if len(children) >= args.max_lanes:
                        break
                    if stage == 'HARD' and not ({'EASY', 'MEDIUM'} & ended_once):
                        continue
                    if ready_at[stage] > time.monotonic():
                        continue
                    pair = available_pair(snapshot, reserved, gpus[stage] if gpus else None)
                    if pair is None:
                        state['stages'][stage] = 'waiting_for_gpus'
                        continue
                    if not ports_available(ports[stage]):
                        state['stages'][stage] = 'waiting_for_ports'
                        continue
                    # Reserve the chosen devices before any model service starts.
                    reserved.update(pair)
                    assignments[stage] = pair
                    queue.remove(stage)
                    attempts[stage] += 1
                    execution = {'stage': stage, 'gpu_pair': pair,
                        'gpu_uuids': [snapshot[i]['uuid'] for i in pair], 'port_pair': ports[stage],
                        'launcher_sha256': sha(args.launcher), 'coordinator_sha256': sha(Path(__file__)),
                        'dataset_sha256': DATA_SHA}
                    write_json(exe_dir / f'{time.time_ns()}-{os.getpid()}-{stage}.json', execution, once=True)
                    log = (args.root / stage / 'parallel-v4-controller.log').open('ab')
                    logs.append(log)
                    try:
                        child = subprocess.Popen(lane_command(args, stage, pair, ports[stage]),
                            stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
                    except Exception:
                        assignments.pop(stage, None)
                        reserved.difference_update(pair)
                        retry_or_fail(stage, 'failed_to_start')
                    else:
                        children[stage] = child
                        state['stages'][stage] = 'running'
            state['state'] = 'PARALLEL_RUNNING' if children else ('WAITING_FOR_GPUS' if queue else 'PARALLEL_STARTING')
            state['gpu_pairs'] = {s: list(pair) for s, pair in assignments.items()}
            state['attempts'] = dict(attempts)
            write_json(status, state)
            if children or queue:
                time.sleep(args.poll_seconds)
        if interrupted[0]:
            state['state'] = 'STOPPING'
        elif watcher.poll() is not None:
            state['state'] = 'GRADER_FAILED_GENERATION_TERMINAL'
        else:
            state['state'] = 'WAITING_FOR_GRADES'
            write_json(status, state)
            while grading_pending(args.root) and not interrupted[0]:
                if watcher.poll() is not None:
                    if watcher_restarts >= 3:
                        break
                    time.sleep(2 ** watcher_restarts)
                    watcher_restarts += 1
                    watcher = subprocess.Popen(started_watcher_command(args), stdout=watchlog,
                        stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
                time.sleep(15)
            if interrupted[0]:
                state['state'] = 'STOPPING'
            elif watcher.poll() is not None:
                state['state'] = 'GRADER_FAILED_GENERATION_TERMINAL'
            else:
                summaries = {'easy': aggregate(args.root, ('PILOT', 'EASY')),
                             'medium': aggregate(args.root, ('MEDIUM',)),
                             'hard': aggregate(args.root, ('HARD',))}
                state['summary'] = summaries
                issues = any(v not in ('complete', 'already_complete') for v in state['stages'].values())
                for counts in summaries.values():
                    issues |= (counts['generation_complete'] != 499 or counts['grading_complete'] != 499 or
                               counts['generation_held'] > 0 or counts['grading_held'] > 0 or counts['unknown'] > 0)
                state['state'] = 'FINISHED_WITH_FAILURES_OR_UNRESOLVED' if issues else 'FINISHED'
        write_json(status, state)
    except Exception as exc:
        code = str(exc) if str(exc) in ERROR_CODES else type(exc).__name__
        write_json(status, {'state': 'FAILED', 'code': code})
        raise
    finally:
        # Do not SIGKILL or edit unknown requests; terminate only processes this coordinator started.
        for child in children.values():
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in children.values():
            child.wait()
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
            watcher.wait()
        if interrupted[0]:
            write_json(status, {'state': 'INTERRUPTED'})
        for log in logs:
            log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def main():
    p = argparse.ArgumentParser(description='Run three isolated two-GPU lanes after safe handoff.')
    p.add_argument('--status', action='store_true', help='Print aggregate local progress only.')
    p.add_argument('--gpu-pairs', nargs=3, metavar=('EASY', 'MEDIUM', 'HARD'))
    p.add_argument('--port-pairs', nargs=3, metavar=('EASY', 'MEDIUM', 'HARD'))
    p.add_argument('--max-lanes', type=int, default=2, help='At most two simultaneous two-GPU lanes.')
    p.add_argument('--poll-seconds', type=int, default=10, help='Seconds between resource checks.')
    p.add_argument('--root', type=Path, default=BASE / 'results/own-test-499-v1')
    p.add_argument('--repo', type=Path, default=BASE / 'eduardo27b-isolated-eval')
    p.add_argument('--python', type=Path, default=BASE / 'eduardo-vllm018/bin/python')
    p.add_argument('--data', type=Path, default=BASE / 'data/math-transfer-test-499-49d3420/data/test.jsonl')
    p.add_argument('--teacher', type=Path, default=BASE / 'models/Eduardo-27b-3bbf17e')
    p.add_argument('--student', type=Path, default=BASE / 'models/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--key', type=Path, default=BASE / '.secrets/agnes_api_key')
    p.add_argument('--launcher', type=Path, default=Path(__file__).resolve().with_name('lane_launch.py'))
    args = p.parse_args()
    if any(not x.is_absolute() for x in (args.root, args.repo, args.python, args.data, args.teacher,
                                          args.student, args.key, args.launcher)):
        raise ValueError('absolute_paths_required')
    if args.status:
        print(json.dumps(status_summary(args.root), ensure_ascii=False, sort_keys=True))
        return
    run(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never print paths, credentials, questions, answers, responses, or logs.
        code = str(exc) if str(exc) in ERROR_CODES else type(exc).__name__
        print(json.dumps({'state': 'ERROR', 'code': code}), file=sys.stderr)
        raise SystemExit(1)
