"""Start local vLLM teacher/student services and a fresh offline evaluation."""
import argparse
import atexit
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get('RESULTS_DIR', '/results'))


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind_config(marker, config):
    from scoring import put_once
    try:
        put_once(marker, config)
    except FileExistsError:
        if json.loads(marker.read_text(encoding='utf-8')) != config:
            raise ValueError('saved run configuration does not match; use a new output directory')


def lock_results(directory):
    fd = os.open(str(directory / '.run.lock'), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        os.close(fd)
        raise
    return fd


def gpu_layout(teacher_gpus):
    if teacher_gpus not in (1, 2):
        raise ValueError('teacher needs one or two GPUs')
    return ','.join(str(i) for i in range(teacher_gpus)), str(teacher_gpus)


def ready(proc, port, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('model server exited during initialization')
        try:
            with urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError('model server did not become ready')


def main():
    from offline_eval import parse_sources
    os.umask(0o077)
    parser = argparse.ArgumentParser(description='Run one transfer difficulty in an independent result directory.')
    parser.add_argument('--difficulty', choices=('paper', 'easy', 'medium', 'hard'), default='paper',
                        help='paper: author transfer; easy/medium/hard: own dataset position')
    parser.add_argument('--teacher-gpus', type=int, choices=(1, 2), default=2,
                        help='teacher tensor-parallel GPU count; student uses one additional GPU')
    args = parser.parse_args()
    teacher = Path(os.environ.get('TEACHER_MODEL', '/models/teacher'))
    student = Path(os.environ.get('STUDENT_MODEL', '/models/student'))
    sources = Path(os.environ.get('SOURCE_FILE', '/input/sources.jsonl'))
    if not all(p.is_dir() and (p / 'config.json').is_file() for p in (teacher, student)):
        raise ValueError('local teacher and student model directories must contain config.json')
    if not sources.is_file():
        raise ValueError('source JSONL is missing')
    parse_sources(sources, args.difficulty)
    OUT.mkdir(parents=True, exist_ok=True)
    lock_fd = lock_results(OUT)
    atexit.register(os.close, lock_fd)
    config = {'protocol': 'author-transfer-fresh-offline-v1', 'source_sha256': file_sha(sources),
              'teacher_config_sha256': file_sha(teacher / 'config.json'),
              'student_config_sha256': file_sha(student / 'config.json'),
              'teacher_dtype': 'bfloat16', 'teacher_tp': args.teacher_gpus, 'student_dtype': 'float16',
              'engine': 'vllm-0.18.0', 'teacher_max_model_len': 20000,
              'teacher_max_tokens': 4192, 'teacher_timeout': 1800,
              'student_max_tokens': 1024, 'student_samples_per_condition': 16,
              'student_temperature': 0.6, 'grading': 'ungraded-local-only'}
    # Preserve the existing paper configuration so old checkpoints still resume.
    if args.difficulty != 'paper':
        config['protocol'] = 'own-transfer-fresh-offline-v1'
        config['transfer_difficulty'] = args.difficulty
    config['installed_versions'] = {name: importlib.metadata.version(name)
        for name in ('vllm', 'torch', 'transformers', 'openai', 'math-verify')}
    marker = OUT / 'RUN_CONFIGURATION.json'
    bind_config(marker, config)
    env = os.environ.copy()
    for k in ('OPENAI_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'ANTHROPIC_API_KEY',
              'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(k, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               VLLM_NO_USAGE_STATS='1', PYTHONPATH=str(ROOT / 'official'), RESULTS_DIR=str(OUT),
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    common = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--host', '127.0.0.1']
    jobs = []
    try:
        teacher_devices, student_device = gpu_layout(args.teacher_gpus)
        teacher_env = dict(env, CUDA_VISIBLE_DEVICES=teacher_devices)
        teacher_cmd = common + ['--port', '8001', '--model', str(teacher), '--served-model-name',
            'Eduardo-27b', '--dtype', 'bfloat16', '--tensor-parallel-size', str(args.teacher_gpus), '--max-model-len',
            '20000', '--gpu-memory-utilization', '0.85', '--max-num-seqs', '1', '--enforce-eager',
            '--limit-mm-per-prompt', '{"image":0,"video":0}', '--seed', '42']
        student_env = dict(env, CUDA_VISIBLE_DEVICES=student_device)
        student_cmd = common + ['--port', '8002', '--model', str(student), '--served-model-name',
            'Llama-3.1-8B-Instruct', '--dtype', 'float16', '--max-model-len', '20000',
            '--gpu-memory-utilization', '0.90', '--seed', '42']
        for name, cmd, process_env in [('teacher', teacher_cmd, teacher_env), ('student', student_cmd, student_env)]:
            log = (OUT / f'{name}-service.log').open('ab')
            proc = subprocess.Popen(cmd, env=process_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((proc, log))
        ready(jobs[0][0], 8001); ready(jobs[1][0], 8002)
        with (OUT / 'eval.log').open('ab') as eval_log:
            subprocess.run([sys.executable, str(ROOT / 'app' / 'offline_eval.py'), '--sources', str(sources),
                            '--teacher-model', str(teacher), '--difficulty', args.difficulty], env=env, check=True,
                           stdout=eval_log, stderr=subprocess.STDOUT)
    finally:
        for proc, log in jobs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()


if __name__ == '__main__':
    try:
        main()
        print(json.dumps({'state': 'RUN_FINISHED'}))
    except Exception:
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            with (OUT / 'launch-error.log').open('a', encoding='utf-8') as log:
                traceback.print_exc(file=log)
        except Exception:
            pass
        print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_RUN_FAILED'}), file=sys.stderr)
        sys.exit(1)
