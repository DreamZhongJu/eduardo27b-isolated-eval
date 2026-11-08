"""Start local vLLM teacher/student services and a fresh offline evaluation."""
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
    os.umask(0o077)
    teacher = Path(os.environ.get('TEACHER_MODEL', '/models/teacher'))
    student = Path(os.environ.get('STUDENT_MODEL', '/models/student'))
    sources = Path(os.environ.get('SOURCE_FILE', '/input/sources.jsonl'))
    if not all(p.is_dir() and (p / 'config.json').is_file() for p in (teacher, student)):
        raise ValueError('local teacher and student model directories must contain config.json')
    if not sources.is_file():
        raise ValueError('source JSONL is missing')
    rows = [json.loads(line) for line in sources.read_text(encoding='utf-8').splitlines() if line.strip()]
    keys = ('source_id', 'problem', 'answer', 'transfer_problem', 'transfer_answer')
    if not rows or any(any(not isinstance(row.get(k), str) or not row[k].strip() for k in keys) for row in rows):
        raise ValueError('invalid source JSONL')
    if len({row['source_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate source IDs')
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'protocol': 'author-transfer-fresh-offline-v1', 'source_sha256': file_sha(sources),
              'teacher_config_sha256': file_sha(teacher / 'config.json'),
              'student_config_sha256': file_sha(student / 'config.json'),
              'teacher_dtype': 'bfloat16', 'teacher_tp': 2, 'student_dtype': 'float16',
              'engine': 'vllm-0.18.0', 'teacher_max_model_len': 20000,
              'teacher_max_tokens': 4192, 'teacher_timeout': 1800,
              'student_max_tokens': 1024, 'student_samples_per_condition': 16,
              'student_temperature': 0.6, 'grading': 'ungraded-local-only'}
    config['installed_versions'] = {name: importlib.metadata.version(name)
        for name in ('vllm', 'torch', 'transformers', 'openai', 'math-verify')}
    marker = OUT / 'RUN_CONFIGURATION.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != config:
            raise ValueError('saved run configuration does not match; use a new output directory')
    else:
        marker.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
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
        teacher_env = dict(env, CUDA_VISIBLE_DEVICES='0,1')
        teacher_cmd = common + ['--port', '8001', '--model', str(teacher), '--served-model-name',
            'Eduardo-27b', '--dtype', 'bfloat16', '--tensor-parallel-size', '2', '--max-model-len',
            '20000', '--gpu-memory-utilization', '0.85', '--max-num-seqs', '1', '--enforce-eager',
            '--limit-mm-per-prompt', '{"image":0,"video":0}', '--seed', '42']
        student_env = dict(env, CUDA_VISIBLE_DEVICES='2')
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
                            '--teacher-model', str(teacher)], env=env, check=True,
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
