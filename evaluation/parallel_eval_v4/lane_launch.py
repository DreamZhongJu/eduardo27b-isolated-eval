#!/usr/bin/env python3
"""Run one frozen-protocol lane with independent GPUs and loopback ports.

The source and result directory must already be assigned exclusively by the
coordinator. This file never modifies the original app or protocol.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid


CUDA_UUID_PROBE = r'''
import ctypes
import uuid

class CudaUUID(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_ubyte * 16)]

lib = ctypes.CDLL('libcuda.so.1')
lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.cuDeviceGetUuid.argtypes = [ctypes.POINTER(CudaUUID), ctypes.c_int]

def check(code):
    if code != 0:
        raise RuntimeError('cuda_driver_probe_failed')

check(lib.cuInit(0))
count = ctypes.c_int()
check(lib.cuDeviceGetCount(ctypes.byref(count)))
if count.value != 1:
    raise RuntimeError('cuda_visible_count_mismatch')
device = ctypes.c_int()
check(lib.cuDeviceGet(ctypes.byref(device), 0))
identifier = CudaUUID()
check(lib.cuDeviceGetUuid(ctypes.byref(identifier), device))
print('GPU-' + str(uuid.UUID(bytes=bytes(identifier.bytes))))
'''


def numeric_device_env(env, index):
    """Use numeric CUDA ordinals because vLLM 0.18 parses the mask as int."""
    if type(index) is not int or index < 0:
        raise ValueError('invalid_cuda_ordinal')
    return dict(env, CUDA_VISIBLE_DEVICES=str(index), CUDA_DEVICE_ORDER='PCI_BUS_ID')


def verify_numeric_cuda(index, expected_uuid, env):
    """Refuse to start if the CUDA ordinal selects a different physical GPU."""
    if not isinstance(expected_uuid, str) or not expected_uuid.startswith('GPU-'):
        raise ValueError('invalid_expected_gpu_uuid')
    try:
        expected = uuid.UUID(expected_uuid[4:])
    except ValueError as exc:
        raise ValueError('invalid_expected_gpu_uuid') from exc
    result = subprocess.run([sys.executable, '-c', CUDA_UUID_PROBE],
        env=numeric_device_env(env, index), capture_output=True, text=True,
        timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError('cuda_uuid_probe_failed')
    output = result.stdout.strip()
    try:
        actual = uuid.UUID(output[4:]) if output.startswith('GPU-') else None
    except ValueError as exc:
        raise RuntimeError('cuda_uuid_probe_invalid') from exc
    if actual is None or actual != expected:
        raise RuntimeError('cuda_nvml_uuid_mismatch')


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_for(args, versions):
    """Exact fields from the frozen app/launch.py; do not add lane fields."""
    cfg = {'protocol': 'author-transfer-fresh-offline-v1',
           'source_sha256': file_sha(args.source),
           'teacher_config_sha256': file_sha(args.teacher_model / 'config.json'),
           'student_config_sha256': file_sha(args.student_model / 'config.json'),
           'teacher_dtype': 'bfloat16', 'teacher_tp': 1, 'student_dtype': 'float16',
           'engine': 'vllm-0.18.0', 'teacher_max_model_len': 20000,
           'teacher_max_tokens': 4192, 'teacher_timeout': 1800,
           'student_max_tokens': 1024, 'student_samples_per_condition': 16,
           'student_temperature': 0.6, 'grading': 'ungraded-local-only'}
    if args.difficulty != 'paper':
        cfg['protocol'] = 'own-transfer-fresh-offline-v1'
        cfg['transfer_difficulty'] = args.difficulty
    cfg['installed_versions'] = versions
    return cfg


def bind_config(path, cfg, put_once):
    try:
        put_once(path, cfg)
    except FileExistsError:
        if json.loads(path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('saved run configuration does not match')


def lock_results(directory):
    fd = os.open(str(directory / '.run.lock'), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def gpu_uuids(teacher_gpu, student_gpu):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15, check=True)
    found = {}
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].startswith('GPU-'):
            found[int(fields[0])] = fields[1]
    if teacher_gpu not in found or student_gpu not in found:
        raise ValueError('assigned GPU not visible')
    if found[teacher_gpu] == found[student_gpu]:
        raise ValueError('teacher and student require distinct GPUs')
    return found[teacher_gpu], found[student_gpu]


def check_ports(ports):
    sockets = []
    try:
        for port in ports:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(s)
            s.bind(('127.0.0.1', port))
    finally:
        for s in sockets:
            s.close()


def service_commands(args):
    common = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--host', '127.0.0.1']
    teacher = common + ['--port', str(args.teacher_port), '--model', str(args.teacher_model),
        '--served-model-name', 'Eduardo-27b', '--dtype', 'bfloat16',
        '--tensor-parallel-size', '1', '--max-model-len', '20000',
        '--gpu-memory-utilization', '0.85', '--max-num-seqs', '1', '--enforce-eager',
        '--limit-mm-per-prompt', '{"image":0,"video":0}', '--seed', '42']
    student = common + ['--port', str(args.student_port), '--model', str(args.student_model),
        '--served-model-name', 'Llama-3.1-8B-Instruct', '--dtype', 'float16',
        '--max-model-len', '20000', '--gpu-memory-utilization', '0.90', '--seed', '42']
    evaluate = [sys.executable, str(args.repo / 'app' / 'offline_eval.py'),
        '--sources', str(args.source), '--teacher-model', str(args.teacher_model),
        '--difficulty', args.difficulty,
        '--teacher-url', f'http://127.0.0.1:{args.teacher_port}/v1',
        '--student-url', f'http://127.0.0.1:{args.student_port}/v1']
    return teacher, student, evaluate


def clean_env(repo, out):
    env = os.environ.copy()
    for k in ('OPENAI_API_KEY', 'AGNES_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
              'ANTHROPIC_API_KEY', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
              'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(k, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               VLLM_NO_USAGE_STATS='1', PYTHONPATH=str(repo / 'official'), RESULTS_DIR=str(out),
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    return env


def ready(proc, port, expected_model, timeout=1800):
    """Require this live server to report the exact expected served model."""
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('model server exited during initialization')
        try:
            with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                healthy = response.status == 200
            if healthy:
                with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
                    raw = response.read(1048577)
                    if response.status != 200 or len(raw) > 1048576:
                        raise ValueError('invalid model list')
                    data = json.loads(raw.decode('utf-8'))
                models = data.get('data') if isinstance(data, dict) else None
                if (isinstance(models, list) and
                        any(isinstance(model, dict) and model.get('id') == expected_model
                            for model in models)):
                    if proc.poll() is not None:
                        raise RuntimeError('model server exited during initialization')
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError('model server did not become ready')


def terminate_group(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run one lane with two GPUs and independent loopback ports.')
    p.add_argument('--difficulty', choices=('paper', 'easy', 'medium', 'hard'), required=True)
    p.add_argument('--teacher-gpu', type=int, required=True)
    p.add_argument('--student-gpu', type=int, required=True)
    p.add_argument('--teacher-port', type=int, required=True)
    p.add_argument('--student-port', type=int, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('/workspace/eduardo27b-isolated-eval'))
    p.add_argument('--teacher-model', type=Path, default=Path('/workspace/models/Eduardo-27b-3bbf17e'))
    p.add_argument('--student-model', type=Path, default=Path('/workspace/models/Meta-Llama-3.1-8B-Instruct'))
    args = p.parse_args(argv)
    if (args.teacher_gpu < 0 or args.student_gpu < 0 or args.teacher_gpu == args.student_gpu or
            args.teacher_port == args.student_port or
            not all(1 <= v <= 65535 for v in (args.teacher_port, args.student_port))):
        p.error('GPU IDs and ports must be distinct and valid')
    if not all(x.is_absolute() for x in (args.source, args.results, args.repo,
                                         args.teacher_model, args.student_model)):
        p.error('all paths must be absolute')
    return args


def main(argv=None):
    os.umask(0o077)
    args = parse_args(argv)
    if not all((p / 'config.json').is_file() for p in (args.teacher_model, args.student_model)):
        raise ValueError('model configuration missing')
    if not args.source.is_file() or not (args.repo / 'app' / 'offline_eval.py').is_file():
        raise ValueError('source or application missing')
    sys.path.insert(0, str(args.repo / 'app'))
    from offline_eval import parse_sources
    from scoring import put_once
    parse_sources(args.source, args.difficulty)
    args.results.mkdir(parents=True, exist_ok=True)
    lock_fd = lock_results(args.results)
    jobs = []
    try:
        versions = {name: importlib.metadata.version(name)
                    for name in ('vllm', 'torch', 'transformers', 'openai', 'math-verify')}
        cfg = config_for(args, versions)
        bind_config(args.results / 'RUN_CONFIGURATION.json', cfg, put_once)
        teacher_uuid, student_uuid = gpu_uuids(args.teacher_gpu, args.student_gpu)
        check_ports((args.teacher_port, args.student_port))
        env = clean_env(args.repo, args.results)
        verify_numeric_cuda(args.teacher_gpu, teacher_uuid, env)
        verify_numeric_cuda(args.student_gpu, student_uuid, env)
        execution = {'difficulty': args.difficulty, 'teacher_gpu': args.teacher_gpu,
            'teacher_gpu_uuid': teacher_uuid, 'student_gpu': args.student_gpu,
            'student_gpu_uuid': student_uuid, 'teacher_port': args.teacher_port,
            'student_port': args.student_port, 'start_time_unix': time.time(),
            'source_sha256': cfg['source_sha256'], 'cuda_device_order': 'PCI_BUS_ID',
            'cuda_uuid_verified': True}
        put_once(args.results / 'LANE_EXECUTIONS' / (uuid.uuid4().hex + '.json'), execution)
        teacher_cmd, student_cmd, eval_cmd = service_commands(args)
        # The driver probe above proves that these numeric masks select the chosen UUIDs.
        for name, cmd, gpu in (('teacher', teacher_cmd, args.teacher_gpu),
                               ('student', student_cmd, args.student_gpu)):
            log = (args.results / f'{name}-service.log').open('ab')
            try:
                proc = subprocess.Popen(cmd, env=numeric_device_env(env, gpu),
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            jobs.append((proc, log))
        ready(jobs[0][0], args.teacher_port, 'Eduardo-27b')
        ready(jobs[1][0], args.student_port, 'Llama-3.1-8B-Instruct')
        with (args.results / 'eval.log').open('ab') as eval_log:
            proc = subprocess.Popen(eval_cmd, env=env, stdout=eval_log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            jobs.append((proc, None))
            if proc.wait() != 0:
                raise RuntimeError('evaluation process failed')
    finally:
        for proc, log in reversed(jobs):
            terminate_group(proc)
            if log is not None:
                log.close()
        os.close(lock_fd)


def _terminate(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminate)
    try:
        main()
        print(json.dumps({'state': 'RUN_FINISHED'}))
    except KeyboardInterrupt:
        print(json.dumps({'state': 'INTERRUPTED'}), file=sys.stderr)
        sys.exit(130)
    except Exception:
        # Complete logs stay on the server; stdout is a fixed aggregate code.
        try:
            error_dir = parse_args().results
            error_dir.mkdir(parents=True, exist_ok=True)
            with (error_dir / 'lane-error.log').open('a', encoding='utf-8') as log:
                traceback.print_exc(file=log)
        except Exception:
            pass
        print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_LANE_FAILED'}), file=sys.stderr)
        sys.exit(1)
