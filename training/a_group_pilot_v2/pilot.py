#!/usr/bin/env python3
"""One isolated EduardoRL single-node pilot. Requires prior A100 and environment verification."""
import argparse, fcntl, hashlib, json, os, secrets, signal, socket, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path('/workspace/A_group_repro_20261117')
PYTHON = ROOT / 'envs/env-candidate-vllm0191/bin/python'
REPO = ROOT / 'source/EduardoRL'
DATA = ROOT / 'data/author_prepared'
TUTOR = ROOT / 'models/Qwen3.5-4B'
STUDENT = Path('/workspace/models/Meta-Llama-3.1-8B-Instruct')
JUDGE = ROOT / 'models/Qwen3.6-27B'
TRAINER_ENTRY = Path(__file__).with_name('trainer_entry.py')
COMMIT = 'e8c89b7293e65e632149571b16a1c2d4e33c77fe'
CHILDREN = []

def require(ok, msg):
    if not ok: raise RuntimeError(msg)

def state(work, phase, message):
    p = work / 'status.json'
    fd, n = tempfile.mkstemp(prefix='.status-', dir=str(work))
    with os.fdopen(fd, 'w') as f:
        json.dump({'phase': phase, 'message': message, 'timestamp': int(time.time()), 'pid': os.getpid()}, f, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(n, p)

def gpu_snapshot():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=15)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=15)
    occupied = set()
    for line in apps.stdout.strip().splitlines():
        if not line.strip() or line.lower().startswith('no running processes'): continue
        parts = [x.strip() for x in line.split(',')]
        require(len(parts) == 2 and parts[0].startswith('GPU-'), 'GPU 进程状态无法解析')
        occupied.add(parts[0])
    result = {}
    for row in p.stdout.strip().splitlines():
        cols = [x.strip() for x in row.split(',')]
        require(len(cols) == 5 and cols[1].startswith('GPU-'), 'GPU 状态无法解析')
        i, total, free, util = int(cols[0]), int(cols[2]), int(cols[3]), int(cols[4])
        result[i] = (total, free, util, cols[1] in occupied)
    return result

def eligible(info):
    total, free, util, process = info
    return total >= 78000 and total - free <= 1024 and util == 0 and not process

def check_cards(ids):
    for _ in range(2):
        info = gpu_snapshot()
        for i in ids:
            require(i in info, 'GPU 编号不存在')
            require(eligible(info[i]), f'GPU {i} 非空闲或非 80GB，未启动')
        if _ == 0: time.sleep(5)

def wait_cards(work):
    while True:
        first = gpu_snapshot()
        time.sleep(5)
        second = gpu_snapshot()
        free = [i for i in sorted(first) if i in second and eligible(first[i]) and eligible(second[i])]
        if len(free) >= 4:
            chosen = free[:6]
            state(work, 'starting', f'已找到 {len(chosen)} 张候选空闲卡，正进行启动前复核')
            try:
                check_cards(chosen)
                return chosen
            except RuntimeError:
                pass
        state(work, 'waiting_gpu', f'当前仅有 {len(free)} 张候选空闲卡；至少需要 4 张')
        time.sleep(60)

def check_ports():
    socks = []
    try:
        for p in (18861, 18862):
            s = socket.socket(); socks.append(s); s.bind(('127.0.0.1', p))
    except OSError:
        raise RuntimeError('本机服务端口被占用')
    finally:
        for s in socks: s.close()

def validate():
    for p in (PYTHON, REPO/'eduardo/configs/eduardo_base.yaml', DATA/'train.parquet', DATA/'val.parquet', DATA/'materialization_manifest.json', TUTOR/'config.json', STUDENT/'config.json', JUDGE/'config.json'):
        require(p.is_file(), f'必要文件缺失：{p}')
    p = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=30)
    require(p.stdout.strip() == COMMIT, '作者代码版本不符')
    p = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True, timeout=30)
    require(not p.stdout.strip(), '作者代码存在修改')
    manifest = json.loads((DATA/'materialization_manifest.json').read_text())
    require(manifest['output']['train.parquet']['rows'] == 4712 and manifest['output']['val.parquet']['rows'] == 100, '题池数量不符')
    for name in ('train.parquet', 'val.parquet'):
        h = hashlib.sha256()
        with (DATA/name).open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
        require(h.hexdigest() == manifest['output'][name]['sha256'], f'题池文件校验不符：{name}')
    student_config = json.loads((STUDENT/'config.json').read_text())
    require(student_config.get('model_type') == 'llama' and
            student_config.get('hidden_size') == 4096 and
            student_config.get('num_hidden_layers') == 32 and
            student_config.get('vocab_size') == 128256, '学生模型配置与 Llama 3.1 8B 不一致')
    index = json.loads((STUDENT/'model.safetensors.index.json').read_text())
    shards = set(index.get('weight_map', {}).values())
    require(bool(shards) and all((STUDENT/name).is_file() for name in shards), '学生模型分片缺失')
    # The environment preparer must mark readiness only after import checks.
    envstatus = json.loads((ROOT/'env-candidate-status.json').read_text())
    require(envstatus.get('status') in ('完成', 'complete'), '候选训练环境尚未完成')

def service(work, name, cards, model, served, port, tp, dp, base):
    env = dict(base, CUDA_VISIBLE_DEVICES=','.join(map(str, cards)))
    cmd = [str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model),
           '--served-model-name', served, '--port', str(port), '--host', '127.0.0.1',
           '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.9', '--max-model-len', '32768',
           '--max-num-seqs', '256', '--tensor-parallel-size', str(tp), '--data-parallel-size', str(dp),
           '--enable-chunked-prefill', '--enable-prefix-caching']
    log = (work / (name + '.log')).open('xb')
    try: p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=work)
    finally: log.close()
    CHILDREN.append(p); return p

def ready(p, port, timeout=1200):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        require(p.poll() is None, '模型服务启动失败，请检查本地服务日志')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as r:
                if r.status == 200: return
        except Exception: pass
        time.sleep(5)
    raise RuntimeError('模型服务加载超时')

def smoke(port, name, token):
    payload = json.dumps({'model': name, 'messages': [{'role': 'user', 'content': 'What is 2+3? Answer briefly.'}],
                          'max_tokens': 64, 'temperature': 0.0,
                          'chat_template_kwargs': {'enable_thinking': False}}).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=payload,
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token}, method='POST')
    with urllib.request.urlopen(request, timeout=180) as r:
        response = json.load(r)
    content = response['choices'][0]['message'].get('content')
    require(isinstance(content, str) and bool(content.strip()), f'{name} 虚构题回答为空')

def stop_children():
    for p in reversed(CHILDREN):
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    deadline = time.monotonic() + 30
    for p in reversed(CHILDREN):
        try: p.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass

def run(ids, steps, work):
    require(ids is None or (4 <= len(ids) <= 6 and len(set(ids)) == len(ids) and all(i >= 0 for i in ids)), '需指定四至六个不同 GPU 编号')
    require(work.parent == ROOT / 'pilots' and not work.exists(), '试点目录必须为本任务下不存在的新目录')
    validate(); check_ports()
    work.mkdir(mode=0o700)
    for d in ('tmp', 'cache'): (work / d).mkdir()
    os.umask(0o077)
    sanitized = {k: v for k, v in os.environ.items()
                 if not (k.upper().endswith(('_TOKEN', '_API_KEY', '_SECRET')) or
                         k.upper() in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'))}
    # The author's separate interaction YAML uses this exact USER-based name.
    # Keep its model identifier identical to the vLLM service and reward calls.
    user = sanitized.get('USER')
    require(bool(user), 'USER 未设置，无法匹配作者学生交互配置')
    student_name = 'Llama-3.1-8B-Instruct-' + user
    token = secrets.token_urlsafe(32)
    base = dict(sanitized, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONUNBUFFERED='1',
      PYTHONDONTWRITEBYTECODE='1', VLLM_API_KEY=token,
      HF_HOME=str(work/'cache/hf'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
      HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1', TMPDIR=str(work/'tmp'),
      XDG_CACHE_HOME=str(work/'cache'), VLLM_CACHE_ROOT=str(work/'cache/vllm'),
      TORCH_HOME=str(work/'cache/torch'), TRITON_CACHE_DIR=str(work/'cache/triton'),
      TORCHINDUCTOR_CACHE_DIR=str(work/'cache/inductor'), CUDA_CACHE_PATH=str(work/'cache/cuda'), WANDB_MODE='disabled')
    try:
        # Ray Unix sockets must fit 107 bytes, including the session suffix.
        # Keep this short, random, task-owned directory inside the A-group root.
        ray_tmp = ROOT / ('r' + secrets.token_hex(3))
        worst = str(ray_tmp) + '/session_2099-12-31_23-59-59_999999_4194304/sockets/plasma_store'
        require(len(worst.encode()) <= 107, 'Ray 临时目录的 socket 路径过长')
        ray_tmp.mkdir(mode=0o700, exist_ok=False)
        (work/'ray_tmp_path.txt').write_text(str(ray_tmp) + '\n')
        if ids is None:
            state(work, 'waiting_gpu', '等待至少四张候选空闲卡')
            ids = wait_cards(work)
        else:
            check_cards(ids)
        # 4 cards: tutor 2, student 1, judge 1. 5: 2+1+2. 6: 2+2+2.
        tutor_cards = ids[:2]
        student_cards = ids[2:4] if len(ids) == 6 else ids[2:3]
        judge_cards = ids[len(tutor_cards) + len(student_cards):]
        role_text = f'教师卡 {tutor_cards}、学生卡 {student_cards}、裁判卡 {judge_cards}'
        check_ports()
        state(work, 'starting', f'正在加载模型；{role_text}')
        s = service(work,'student',student_cards,STUDENT,student_name,18861,1,len(student_cards),base)
        j = service(work,'judge',judge_cards,JUDGE,'judge-local',18862,len(judge_cards),1,base)
        ready(s,18861); ready(j,18862)
        smoke(18861, student_name, token); smoke(18862, 'judge-local', token)
        state(work, 'training', f'正在进行 {steps} 步试点；{role_text}')
        env = dict(base, CUDA_VISIBLE_DEVICES=','.join(map(str,tutor_cards)), PYTHONPATH=str(REPO),
          WORK_DIR=str(work), EDUARDO_RECIPE_DIR=str(REPO/'eduardo'), MODEL_PATH=str(TUTOR),
          EXPERIMENT=f'paper4b-pilot{steps}', SERVING_API_KEY=token,
          STUDENT_API_BASE='http://127.0.0.1:18861/v1', JUDGE_API_BASE='http://127.0.0.1:18862/v1',
          RAY_ADDRESS='local', RAY_TMPDIR=str(ray_tmp))
        overrides = ['trainer.nnodes=1','trainer.n_gpus_per_node=2','actor_rollout_ref.actor.fsdp_config.fsdp_size=2',
          'actor_rollout_ref.rollout.n=16','custom_reward_function.reward_kwargs.num_student_attempts=16',
          f'vars.student_model_name={student_name}','vars.judge_model_name=judge-local',
          f'data.train_files={DATA}/train.parquet', f'data.val_files={DATA}/val.parquet',
          f'trainer.total_training_steps={steps}','trainer.resume_mode=disable','trainer.logger=[console]',
          'custom_reward_function.reward_kwargs.eval_judge_model=', '+ray_kwargs.ray_init.address=local',
          f'+ray_kwargs.ray_init._temp_dir={ray_tmp}']
        if steps == 1: overrides.append('trainer.save_freq=1')
        cmd = [str(PYTHON),str(TRAINER_ENTRY),'--config-path',str(REPO/'eduardo/configs'),
               '--config-name','eduardo_base', *overrides]
        with (work/'trainer.log').open('xb') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=work)
            CHILDREN.append(p)
            while p.poll() is None:
                require(s.poll() is None and j.poll() is None, '学生或门槛服务中断，训练已停止')
                time.sleep(5)
            require(p.returncode == 0, '试点训练失败，请检查本地训练日志')
        state(work, 'finished', '试点进程正常退出，仍需检查步骤数、模型更新和检查点')
    except Exception as e:
        state(work, 'failed', str(e)); raise
    finally: stop_children()

def main():
    p = argparse.ArgumentParser(); choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--gpus', type=int, nargs='+')
    choice.add_argument('--wait-gpus', action='store_true')
    p.add_argument('--steps', type=int, choices=(1,10), default=1); p.add_argument('--name', required=True)
    a = p.parse_args()
    require(a.name.isidentifier() and len(a.name) <= 50, '试点名称必须仅由字母数字下划线组成')
    require(ROOT.is_dir() and not ROOT.is_symlink(), '任务目录不存在')
    (ROOT/'pilots').mkdir(mode=0o700,exist_ok=True)
    lock = (ROOT/'.pilot.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('已有试点启动器运行')
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig, lambda sig,frame: (_ for _ in ()).throw(RuntimeError('收到停止信号')))
    run(None if a.wait_gpus else a.gpus,a.steps,ROOT/'pilots'/a.name)
if __name__ == '__main__':
    try: main()
    except Exception as e:
        print('未完成：'+str(e),file=sys.stderr); sys.exit(1)
