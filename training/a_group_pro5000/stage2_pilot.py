"""Stage 2 (3-step pilot with step-2 model+extra checkpoint resume) orchestrator for Pro 5000."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pro5000_config as pcfg
from pro5000_trainer_entry import verify_model_extra_checkpoint, write_json
from stage1_collect import find_local_model
from stage2_precheck import core_gpus_clean, snapshot_gpus
from telemetry import Sampler

EDUARDO_REMOTE = 'https://gh-proxy.com/https://github.com/eth-lre/EduardoRL.git'
AUTHOR_COMMIT = pcfg.AUTHOR_COMMIT
DISK_FLOOR_BYTES = 45 * 10**9
PARQUET_SHA256 = {
    'train.parquet': 'f31edcf7bd845515b753acdf3fe15de820c9d7cefd579c626688ccb3068aeede',
    'val.parquet': '56c89d329818482bd5e314ea1908c44fb277d3a7207f72cbd352d9b465841d4b',
}


def default_base() -> Path:
    return Path.home() / 'bzhang' / 'a_group_pro5000'


def base_subprocess_env(base: Path) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not k.upper().endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))
        and k.upper() not in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')
    }
    env_lib = Path(sys.prefix) / 'lib'
    ld_parts = [str(env_lib)] if env_lib.is_dir() else []
    if env.get('LD_LIBRARY_PATH'):
        ld_parts.append(env['LD_LIBRARY_PATH'])
    env.update(
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',
        USER=os.environ.get('USER') or 'root',
        HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',
        VLLM_NO_USAGE_STATS='1',
        VLLM_LOGGING_LEVEL='INFO',
        WANDB_MODE='disabled',
        RAY_USAGE_STATS_ENABLED='0',
        RAY_DEDUP_LOGS='0',
        NO_PROXY='127.0.0.1,localhost,::1',
        no_proxy='127.0.0.1,localhost,::1',
        TMPDIR=str(base / 'tmp'),
        HF_HOME=str(base / 'cache' / 'hf'),
        TRITON_CACHE_DIR=str(base / 'cache' / 'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(base / 'cache' / 'inductor'),
        VLLM_CACHE_ROOT=str(base / 'cache' / 'vllm'),
        XDG_CACHE_HOME=str(base / 'cache' / 'xdg'),
    )
    env_bin = str(Path(sys.executable).resolve().parent)
    env['PATH'] = env_bin + os.pathsep + env.get('PATH', '')
    if ld_parts:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(ld_parts)
    return env


def ensure_eduardo_repo(base: Path) -> Path:
    import hashlib

    repo = base / 'source' / 'EduardoRL'
    repo.parent.mkdir(parents=True, exist_ok=True)
    git_bin = str(Path(sys.executable).resolve().parent / 'git')
    if not Path(git_bin).is_file():
        git_bin = 'git'
    git_env = base_subprocess_env(base)
    if not (repo / '.git').exists():
        print(f'[setup] cloning EduardoRL@{AUTHOR_COMMIT[:8]} via gh-proxy...', flush=True)
        subprocess.run([git_bin, 'clone', '--quiet', EDUARDO_REMOTE, str(repo)], env=git_env, check=True, timeout=300)
    rev = subprocess.check_output([git_bin, '-C', str(repo), 'rev-parse', 'HEAD'], env=git_env, text=True, timeout=30).strip()
    if rev != AUTHOR_COMMIT:
        subprocess.run([git_bin, '-C', str(repo), 'fetch', '--quiet', 'origin', AUTHOR_COMMIT], env=git_env, check=True, timeout=180)
        subprocess.run([git_bin, '-C', str(repo), 'checkout', '--quiet', '--detach', AUTHOR_COMMIT], env=git_env, check=True, timeout=60)
    status = subprocess.check_output([git_bin, '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], env=git_env, text=True, timeout=30)
    if status.strip():
        raise RuntimeError('EduardoRL 源码存在未跟踪或本地修改')
    return repo


def verify_model_dir(label: str, model_dir: Path) -> Path:
    """Verify model dir (including symlink targets and sharded safetensors) exists and is non-empty."""
    if model_dir.is_symlink() and not model_dir.exists():
        raise RuntimeError(f'{label} 模型软链接失效（目标不存在，可能节点重启过 /dev/shm）: {model_dir}')
    if not model_dir.is_dir():
        raise RuntimeError(f'缺少 {label} 模型目录: {model_dir}')
    resolved = model_dir.resolve(strict=True)
    cfg = resolved / 'config.json'
    if not cfg.is_file() or cfg.stat().st_size <= 0:
        raise RuntimeError(f'{label} 模型缺少或空 config.json: {model_dir}')
    for root, dirs, files in os.walk(resolved):
        for name in (*dirs, *files):
            p = Path(root) / name
            if p.is_symlink() and not p.exists():
                raise RuntimeError(f'{label} 模型包含失效软链接: {p}')
    shards = [p for p in resolved.glob('*.safetensors') if p.is_file() and p.stat().st_size > 0]
    if not shards:
        raise RuntimeError(f'{label} 模型缺少有效 *.safetensors 权重文件: {model_dir}')
    idx_file = resolved / 'model.safetensors.index.json'
    if idx_file.is_file():
        weight_map = json.loads(idx_file.read_text(encoding='utf-8')).get('weight_map', {})
        for shard_name in sorted(set(weight_map.values())):
            sp = resolved / shard_name
            if not sp.is_file() or sp.stat().st_size <= 0:
                raise RuntimeError(f'{label} 模型分片缺失或无效: {sp}')
    return model_dir


def verify_stage1_prereqs(base: Path, tutor_model: Path):
    import hashlib

    receipt = base / 'envs' / 'pro5000-fast' / 'kernel_equivalence.json'
    if not receipt.is_file():
        raise RuntimeError('缺少阶段 1 快内核等价回执 kernel_equivalence.json')
    data = json.loads(receipt.read_text(encoding='utf-8'))
    if not data.get('accepted') or data.get('fast_kernel_verified') is not True:
        raise RuntimeError('阶段 1 等价回执未通过')
    verify_model_dir('Tutor', tutor_model)
    for fname, expected_sha in PARQUET_SHA256.items():
        fpath = base / 'data' / 'author_prepared' / fname
        if not fpath.is_file():
            raise RuntimeError(f'缺少数据集文件 {fname}')
        h = hashlib.sha256()
        with fpath.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected_sha:
            raise RuntimeError(f'数据集校验不一致 {fname}')


def pick_free_ports(preferred: list[int]) -> list[int]:
    ports = []
    socks = []
    try:
        for port in preferred:
            s = socket.socket()
            try:
                s.bind(('127.0.0.1', port))
                ports.append(port)
                socks.append(s)
            except OSError:
                s.close()
                s2 = socket.socket()
                s2.bind(('127.0.0.1', 0))
                ports.append(int(s2.getsockname()[1]))
                socks.append(s2)
        return ports
    finally:
        for s in socks:
            s.close()


def write_access_log_config(work: Path) -> Path:
    from uvicorn.config import LOGGING_CONFIG

    cfg = deepcopy(LOGGING_CONFIG)
    cfg['disable_existing_loggers'] = False
    cfg['formatters']['access']['fmt'] = '[C_HTTP %(created).6f] "%(request_line)s" %(status_code)s'
    cfg['formatters']['access']['use_colors'] = False
    path = work / 'access-log-config.json'
    write_json(path, cfg)
    return path


def write_interaction_yaml(repo: Path, work: Path, student_name: str, student_api_base: str, student_conc: int) -> Path:
    import yaml

    src = repo / 'eduardo' / 'eduardo_interaction_config.yaml'
    obj = yaml.safe_load(src.read_text(encoding='utf-8'))
    rows = obj.get('interaction')
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get('name') != 'eduardo':
        raise RuntimeError('interaction 配置结构异常')
    rows[0]['config']['student_model'] = student_name
    rows[0]['config']['student_api_base'] = student_api_base
    rows[0]['config']['student_concurrency'] = student_conc
    out = work / 'interaction_pro5000.yaml'
    out.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return out


def wait_for_http_health(url: str, proc: subprocess.Popen, label: str, timeout_s: float = 600.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f'{label} 服务启动失败，提前退出 rc={rc}')
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f'{label} 服务健康检查超时 ({timeout_s}s)')


_RAY_SOCK_SUFFIX = '/session_2099-12-31_23-59-59_999999_4194304/sockets/plasma_store'


def make_ray_tmp(base: Path, max_socket_bytes: int = 107) -> Path:
    # Prefer /data (under ~/bzhang/) while strictly respecting AF_UNIX's 107-byte limit.
    tag = 'r' + secrets.token_hex(2)
    seen: set[Path] = set()
    candidates = [
        p
        for p in [base, *base.parents, Path('/tmp'), Path(tempfile.gettempdir())]
        if str(p) != '/' and not (p in seen or seen.add(p))
    ]
    for parent in candidates:
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            continue
        raytmp = parent / tag
        if len((str(raytmp) + _RAY_SOCK_SUFFIX).encode('utf-8')) <= max_socket_bytes:
            raytmp.mkdir(mode=0o700, exist_ok=False)
            return raytmp
    raise RuntimeError(f'无可用的 Ray 临时目录 (均超出 {max_socket_bytes} bytes UNIX socket 限制): base={base}')


def build_trainer_env(
    *,
    base: Path,
    work: Path,
    ckpt_dir: Path,
    raytmp: Path,
    tutor_gpus: list[int],
    all_gpus: list[int],
    student_gpus: list[int],
    layout: dict,
    repo: Path,
    tutor_model: Path,
    student_api_base: str,
    judge_api_base: str,
    start_step: int,
    total_steps: int,
    save_freq: int,
    test_freq: int = 25,
    max_ckpt_keep: int | None = None,
    check_step1_judge: bool = False,
    check_step2_ratio: bool = False,
    experiment: str = 'pro5000-stage2',
    ckpt_hint: int | None = None,
) -> dict[str, str]:
    """Build environment variables required by pro5000_trainer_entry.py (both preflight and run)."""
    out = {
        'CUDA_VISIBLE_DEVICES': ','.join(map(str, tutor_gpus)),
        'PYTHONPATH': f'{HERE}:{repo}',
        'PRO5000_BASE': str(base),
        'WORK_DIR': str(work),
        'PRO5000_CKPT_DIR': str(ckpt_dir),
        'RAY_TMPDIR': str(raytmp),
        'PRO5000_START_STEP': str(start_step),
        'PRO5000_TOTAL_STEPS': str(total_steps),
        'PRO5000_SAVE_FREQ': str(save_freq),
        'PRO5000_TEST_FREQ': str(test_freq),
        'PRO5000_TUTOR_GPUS': str(len(tutor_gpus)),
        'PRO5000_ALL_GPUS': ','.join(map(str, all_gpus)),
        'PRO5000_STUDENT_REPLICAS': str(len(student_gpus)),
        'PRO5000_OFFLOAD': 'true',
        'PRO5000_STUDENT_CONC': str(layout['student_concurrency']),
        'PRO5000_JUDGE_CONC': str(layout['judge_concurrency']),
        'PRO5000_MAX_CKPT_KEEP': 'none' if max_ckpt_keep is None else str(max_ckpt_keep),
        'PRO5000_CHECK_STEP1_JUDGE': '1' if check_step1_judge else '0',
        'PRO5000_CHECK_STEP2_RATIO': '1' if check_step2_ratio else '0',
        'EDUARDO_RECIPE_DIR': str(repo / 'eduardo'),
        'MODEL_PATH': str(tutor_model),
        'EXPERIMENT': experiment,
        'STUDENT_API_BASE': student_api_base,
        'JUDGE_API_BASE': judge_api_base,
    }
    if ckpt_hint is not None:
        out['PRO5000_CKPT_HINT_BYTES'] = str(ckpt_hint)
    return out


def terminate_pg(pgid: int):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        if sig == signal.SIGTERM:
            time.sleep(2)


def terminate_procs(procs: list[tuple[str, subprocess.Popen]]):
    for _, proc in reversed(procs):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for _, proc in reversed(procs):
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(0.5, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)


def summarize_stage2(work: Path, layout: dict, step2_ckpt_bytes: int) -> dict:
    s1 = json.loads((work / 'step-statistics' / '1.json').read_text(encoding='utf-8'))
    s2 = json.loads((work / 'step-statistics' / '2.json').read_text(encoding='utf-8'))
    s3 = json.loads((work / 'step-statistics' / '3.json').read_text(encoding='utf-8'))
    restore = json.loads((work / 'restore_step_2.json').read_text(encoding='utf-8'))
    cmp2 = json.loads((work / 'step2-comparison.json').read_text(encoding='utf-8'))

    steps_summary = []
    gpu_peak_overall = {}
    for s in (s1, s2, s3):
        m = s['metrics']
        pt = s['phase_timings']
        telem = s['telemetry']
        per_gpu_peak = {}
        for phase in ('gen', 'old_log_prob', 'update_actor'):
            for gid, gvals in telem.get(phase, {}).get('gpus', {}).items():
                mx = gvals.get('used_mib', {}).get('max')
                if mx is not None:
                    per_gpu_peak[gid] = max(per_gpu_peak.get(gid, 0.0), mx)
                    gpu_peak_overall[gid] = max(gpu_peak_overall.get(gid, 0.0), mx)

        jp = s['judge_proxy']
        sp = s['student_proxy']
        qm = s['queues'].get('groups', {})
        judge_parse = s.get('vote_log_counts', {}).get('parse', 0)
        judge_recov = s.get('vote_log_counts', {}).get('recovery', 0)
        judge_req = jp.get('total_chat_requests') or s.get('vote_log_counts', {}).get('requests', 0)
        parse_fail_rate = (judge_parse / judge_req) if judge_req else 0.0

        gen_sec = m.get('timing_s/gen', 0.0)
        tutor_busy_s = None
        gen_gpus = telem.get('gen', {}).get('gpus', {})
        tutor_utils = [
            gen_gpus[str(g)]['util_pct']['mean']
            for g in layout['tutor_gpus']
            if str(g) in gen_gpus and gen_gpus[str(g)]['util_pct']['mean'] is not None
        ]
        if tutor_utils and gen_sec > 0:
            tutor_busy_s = round(gen_sec * (sum(tutor_utils) / len(tutor_utils)) / 100.0, 2)

        student_qm_int = qm.get('student_interaction', {})
        student_qm_fin = qm.get('student_final', {})
        student_timeouts = student_qm_int.get('attempt_errors', 0) + student_qm_fin.get('attempt_errors', 0)

        steps_summary.append({
            'step': s['step'],
            'total_num_tokens': m.get('perf/total_num_tokens'),
            'step_seconds': m.get('timing_s/step'),
            'phases': {
                p: {
                    'minutes': round(pt[p]['minutes'], 3) if p in pt else None,
                    'ms_per_token': round(pt[p]['ms_per_token'], 4) if p in pt else None,
                }
                for p in ('gen', 'old_log_prob', 'update_actor')
            },
            'gpu_peak_mib': per_gpu_peak,
            'judge_calls': judge_req,
            'judge_non_2xx': jp.get('non_2xx', 0),
            'judge_parse_failures': judge_parse,
            'judge_parse_failure_rate': round(parse_fail_rate, 6),
            'judge_regex_recoveries': judge_recov,
            'judge_busy_wall_s': jp.get('combined_busy_wall_s'),
            'student_total_calls': sp.get('total_chat_requests'),
            'student_non_2xx': sp.get('non_2xx', 0),
            'student_attempt_errors': student_timeouts,
            'student_avg_response_tokens': sp.get('avg_completion_tokens'),
            'student_avg_interaction_tokens': sp.get('avg_interaction_tokens'),
            'student_avg_final_tokens': sp.get('avg_final_tokens'),
            'student_combined_busy_wall_s': sp.get('combined_busy_wall_s'),
            'student_backends': sp.get('backends', {}),
            'tutor_gen_busy_est_s': tutor_busy_s,
            'tutor_gen_mean_util_pct': round(sum(tutor_utils) / len(tutor_utils), 2) if tutor_utils else None,
            'actor_grad_norm': m.get('actor/grad_norm'),
            'actor_pg_loss': m.get('actor/pg_loss'),
        })

    step_secs = [row['step_seconds'] for row in steps_summary if row['step_seconds']]
    # Ignore step 1 cold compile overhead if steps 2 and 3 are steady, or provide both averages.
    steady_secs = [row['step_seconds'] for row in steps_summary[1:] if row['step_seconds']] or step_secs
    avg_step_s = sum(steady_secs) / len(steady_secs)
    est_100_hours = round(100 * avg_step_s / 3600.0, 2)
    return {
        'layout': layout,
        'restore': restore,
        'step2_comparison': cmp2,
        'step2_ckpt_gb': round(step2_ckpt_bytes / 1e9, 2),
        'gpu_peak_overall_mib': gpu_peak_overall,
        'steps': steps_summary,
        'steady_avg_step_s': round(avg_step_s, 2),
        'estimated_100_steps_hours': est_100_hours,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=default_base())
    parser.add_argument('--tutor-model', type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    parser.add_argument('--gpus-mode', choices=('auto', '8', '7'), default='auto')
    args = parser.parse_args()
    os.umask(0o077)

    base = args.base.resolve()
    py_exe = Path(sys.executable).resolve()
    verify_stage1_prereqs(base, args.tutor_model)
    repo = ensure_eduardo_repo(base)

    free_bytes = shutil.disk_usage('/data').free
    print(f'[preflight] /data free = {free_bytes / 1e9:.2f} GB ({free_bytes / 1024**3:.2f} GiB)', flush=True)
    if free_bytes < DISK_FLOOR_BYTES:
        print('STOP: /data 剩余低于 45G')
        return 3

    first = snapshot_gpus()
    time.sleep(2)
    second = snapshot_gpus()
    core_ok, issues = core_gpus_clean(first, second)
    if not core_ok:
        print('STOP: GPU 0-5 或 7 存在外部进程/占用:', issues)
        return 2

    if args.gpus_mode == '8':
        use_8 = True
    elif args.gpus_mode == '7':
        use_8 = False
    else:
        precheck_file = base / 'runs' / 'stage2_precheck.json'
        if precheck_file.is_file():
            pre = json.loads(precheck_file.read_text(encoding='utf-8'))
            use_8 = (pre.get('verdict') == 'PRECHECK_PASS_8GPU') and not second[6]['apps']
        else:
            use_8 = not second[6]['apps']

    tutor_gpus = [0, 1, 2, 3]
    student_gpus = [4, 6] if use_8 else [4]
    judge_gpus = [5, 7]
    all_gpus = sorted(tutor_gpus + student_gpus + judge_gpus)
    layout = {
        'mode': '8gpu' if use_8 else '7gpu',
        'tutor_gpus': tutor_gpus,
        'student_gpus': student_gpus,
        'judge_gpus': judge_gpus,
        'judge_tp': 2,
        'optimizer_offload': True,
        'student_concurrency': 32,
        'judge_concurrency': 16,
    }
    print(f'[preflight] 选用布局: {layout}', flush=True)

    run_name = f'stage2_pilot_{time.strftime("%Y%m%d_%H%M%S")}'
    work = base / 'runs' / run_name
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    (work / 'queue-metrics').mkdir(mode=0o700, exist_ok=True)
    ckpt_dir = work / 'checkpoints'
    ckpt_dir.mkdir(mode=0o700, exist_ok=True)

    models_root = base / 'models'
    if models_root.is_dir():
        for entry in sorted(models_root.iterdir()):
            if entry.is_symlink() and not entry.exists():
                raise RuntimeError(f'models 目录下存在失效软链接（可能节点重启过 /dev/shm）: {entry}')
    student_model = verify_model_dir(
        'Student',
        find_local_model(models_root, models_root / 'Meta-Llama-3.1-8B-Instruct', ('llama-3.1-8b', '8b-instruct')),
    )
    judge_model = verify_model_dir(
        'Judge',
        find_local_model(models_root, models_root / 'Qwen3.6-27B', ('qwen3.6-27b', '27b')),
    )
    print(f'[preflight] 模型就绪: student={student_model} -> {student_model.resolve()} ; judge={judge_model} -> {judge_model.resolve()}', flush=True)

    ports = pick_free_ports([18861, 18862, 18871, 18872, 18873])
    s_lb_port, j_proxy_port, s0_port, s1_port, j_port = ports
    student_api_base = f'http://127.0.0.1:{s_lb_port}/v1'
    judge_api_base = f'http://127.0.0.1:{j_proxy_port}/v1'

    token = secrets.token_urlsafe(32)
    user_tag = os.environ.get('USER') or 'root'
    student_name = f'Llama-3.1-8B-Instruct-{user_tag}'
    interaction = write_interaction_yaml(repo, work, student_name, student_api_base, layout['student_concurrency'])
    access_cfg = write_access_log_config(work)
    env_base = base_subprocess_env(base)
    env_base['VLLM_API_KEY'] = token
    env_base['SERVING_API_KEY'] = token
    env_base['EDUARDO_CONV_DIR'] = '/dev/null'  # Do not write rollout transcripts to disk.

    def trainer_env(raytmp: Path, start_step: int, total_steps: int, save_freq: int, ckpt_hint: int | None = None) -> dict[str, str]:
        return build_trainer_env(
            base=base, work=work, ckpt_dir=ckpt_dir, raytmp=raytmp,
            tutor_gpus=tutor_gpus, all_gpus=all_gpus, student_gpus=student_gpus,
            layout=layout, repo=repo, tutor_model=args.tutor_model,
            student_api_base=student_api_base, judge_api_base=judge_api_base,
            start_step=start_step, total_steps=total_steps, save_freq=save_freq,
            ckpt_hint=ckpt_hint,
        )

    # Hydra CPU preflight for Phase A (steps 1-2)
    raytmp_a = make_ray_tmp(base)
    try:
        overrides_a = pcfg.build_overrides(
            base=base, work=work, ckpt_dir=ckpt_dir, raytmp=raytmp_a,
            student_name=student_name, interaction=interaction,
            total_steps=2, save_freq=2, test_freq=25,
            tutor_gpus=len(tutor_gpus), optimizer_offload=True,
            student_concurrency=layout['student_concurrency'],
            judge_concurrency=layout['judge_concurrency'],
        )
        trainer_entry = HERE / 'pro5000_trainer_entry.py'
        pcfg.run_preflight(
            python_exe=py_exe, trainer_entry=trainer_entry, repo=repo, work=work,
            tutor_model=args.tutor_model, student_api_base=student_api_base,
            judge_api_base=judge_api_base, overrides=overrides_a,
            check_env=dict(env_base, **trainer_env(raytmp_a, 0, 2, 2)),
        )
    except Exception:
        shutil.rmtree(raytmp_a, ignore_errors=True)
        raise
    print('[preflight] Hydra 冻结配方预检通过', flush=True)

    managed_procs: list[tuple[str, subprocess.Popen]] = []
    sampler: Sampler | None = None
    step2_ckpt_bytes = 0
    try:
        def spawn(name: str, cmd: list[str], extra_env: dict[str, str], log_name: str) -> subprocess.Popen:
            log = (work / log_name).open('ab')
            try:
                proc = subprocess.Popen(
                    cmd, env=dict(env_base, **extra_env), cwd=work,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log.close()
            managed_procs.append((name, proc))
            return proc

        # 1. Start Student vLLM replica(s)
        student_backends = []
        for idx, (gpu_id, port) in enumerate(zip(student_gpus, (s0_port, s1_port))):
            cmd = [
                str(py_exe), '-B', '-m', 'vllm.entrypoints.openai.api_server',
                '--model', str(student_model),
                '--served-model-name', student_name,
                '--host', '127.0.0.1', '--port', str(port),
                '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.9',
                '--max-model-len', '32768', '--max-num-seqs', '256',
                '--tensor-parallel-size', '1', '--data-parallel-size', '1',
                '--enable-chunked-prefill', '--enable-prefix-caching',
                '--log-config-file', str(access_cfg),
            ]
            spawn(f'student_{idx}', cmd, {'CUDA_VISIBLE_DEVICES': str(gpu_id)}, f'student_{idx}.log')
            student_backends.append(f'http://127.0.0.1:{port}')

        # 2. Start Judge vLLM (TP=2 on GPU 5,7)
        judge_cmd = [
            str(py_exe), '-B', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(judge_model),
            '--served-model-name', 'judge-local',
            '--host', '127.0.0.1', '--port', str(j_port),
            '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.9',
            '--max-model-len', '32768', '--max-num-seqs', '256',
            '--tensor-parallel-size', '2', '--data-parallel-size', '1',
            '--enable-chunked-prefill', '--enable-prefix-caching',
            '--log-config-file', str(access_cfg),
        ]
        spawn('judge', judge_cmd, {'CUDA_VISIBLE_DEVICES': ','.join(map(str, judge_gpus))}, 'judge.log')

        # 3. Start loopback serving_proxy
        proxy_cmd = [
            str(py_exe), '-B', str(HERE / 'serving_proxy.py'),
            '--work', str(work),
            '--student-port', str(s_lb_port), '--student-backends', *student_backends,
            '--judge-port', str(j_proxy_port), '--judge-backends', f'http://127.0.0.1:{j_port}',
        ]
        spawn('proxy', proxy_cmd, {'CUDA_VISIBLE_DEVICES': ''}, 'proxy.log')

        # 4. Health checks
        for name, proc in managed_procs:
            if name.startswith('student_'):
                port = s0_port if name == 'student_0' else s1_port
                print(f'[serving] waiting for {name} (GPU {student_gpus[int(name[-1])]})...', flush=True)
                wait_for_http_health(f'http://127.0.0.1:{port}/health', proc, name, timeout_s=420)
            elif name == 'judge':
                print(f'[serving] waiting for judge TP=2 (GPUs {judge_gpus})...', flush=True)
                wait_for_http_health(f'http://127.0.0.1:{j_port}/health', proc, name, timeout_s=600)
            elif name == 'proxy':
                wait_for_http_health(f'http://127.0.0.1:{s_lb_port}/health', proc, 'proxy-student', timeout_s=30)
                wait_for_http_health(f'http://127.0.0.1:{j_proxy_port}/health', proc, 'proxy-judge', timeout_s=30)
        print('[serving] all serving backends & proxies healthy!', flush=True)

        # 5. Start 10s telemetry across all active GPUs
        def task_roots():
            return ()

        sampler = Sampler(work / 'tutor-telemetry.jsonl', tuple(all_gpus), task_roots, interval=10)
        sampler.start()

        # 6. Phase A: steps 1-2 + save step 2 [model, extra]
        print('[phase A] starting steps 1-2 (will save step-2 model+extra checkpoint and stop)...', flush=True)
        cmd_a = [
            str(py_exe), '-B', str(trainer_entry),
            '--config-path', str(repo / 'eduardo' / 'configs'), '--config-name', 'eduardo_base',
            *overrides_a,
        ]
        log_a = (work / 'trainer_phaseA.log').open('xb')
        proc_a = None
        try:
            proc_a = subprocess.Popen(
                cmd_a, env=dict(env_base, **trainer_env(raytmp_a, 0, 2, 2)),
                cwd=work, stdin=subprocess.DEVNULL, stdout=log_a, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_a.close()
        try:
            rc_a = proc_a.wait()
        finally:
            terminate_pg(proc_a.pid)
            shutil.rmtree(raytmp_a, ignore_errors=True)
        if rc_a != 0:
            raise RuntimeError(f'阶段 2 前两步训练退出异常 rc={rc_a}')

        ckpt_step2 = ckpt_dir / 'global_step_2'
        ckpt_info = verify_model_extra_checkpoint(ckpt_step2, world_size=len(tutor_gpus))
        step2_ckpt_bytes = ckpt_info['size_bytes']
        print(f'[phase A] step-2 [model,extra] checkpoint verified: {ckpt_info}', flush=True)

        # 7. Phase B: resume from global_step_2 -> step 3
        print('[phase B] starting fresh trainer to resume from global_step_2 -> step 3...', flush=True)
        raytmp_b = make_ray_tmp(base)
        try:
            overrides_b = pcfg.build_overrides(
                base=base, work=work, ckpt_dir=ckpt_dir, raytmp=raytmp_b,
                student_name=student_name, interaction=interaction,
                total_steps=3, save_freq=25, test_freq=25,
                tutor_gpus=len(tutor_gpus), optimizer_offload=True,
                student_concurrency=layout['student_concurrency'],
                judge_concurrency=layout['judge_concurrency'],
                resume_step=2,
            )
            env_b = trainer_env(raytmp_b, 2, 3, 25, ckpt_hint=step2_ckpt_bytes)
            pcfg.run_preflight(
                python_exe=py_exe, trainer_entry=trainer_entry, repo=repo, work=work,
                tutor_model=args.tutor_model, student_api_base=student_api_base,
                judge_api_base=judge_api_base, overrides=overrides_b,
                check_env=dict(env_base, **env_b),
            )
            cmd_b = [
                str(py_exe), '-B', str(trainer_entry),
                '--config-path', str(repo / 'eduardo' / 'configs'), '--config-name', 'eduardo_base',
                *overrides_b,
            ]
            log_b = (work / 'trainer_phaseB.log').open('xb')
            proc_b = None
            try:
                proc_b = subprocess.Popen(
                    cmd_b, env=dict(env_base, **env_b),
                    cwd=work, stdin=subprocess.DEVNULL, stdout=log_b, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_b.close()
            rc_b = proc_b.wait()
        finally:
            if 'proc_b' in locals() and proc_b is not None:
                terminate_pg(proc_b.pid)
            shutil.rmtree(raytmp_b, ignore_errors=True)
        if rc_b != 0:
            raise RuntimeError(f'阶段 2 第 3 步续训退出异常 rc={rc_b}')

        # Stop telemetry & serving cleanly before summarizing and deleting pilot checkpoint
        if sampler is not None:
            sampler.stop()
            sampler = None
        terminate_procs(managed_procs)
        managed_procs.clear()

        summary = summarize_stage2(work, layout, step2_ckpt_bytes)
        for gdir in sorted(ckpt_dir.glob('global_step_*')):
            if gdir.is_dir():
                shutil.rmtree(gdir)
        summary['pilot_ckpts_deleted'] = not any(ckpt_dir.glob('global_step_*'))
        summary['post_cleanup_disk_free_gb'] = round(shutil.disk_usage('/data').free / 1e9, 2)
        summary['work_dir'] = str(work)
        write_json(work / 'stage2_summary.json', summary)

        print('=== STAGE 2 SUMMARY ===')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        if sampler is not None:
            sampler.stop()
        terminate_procs(managed_procs)


if __name__ == '__main__':
    sys.exit(main())
