"""Stage 3 (formal 100-step DPPO run with resume, step-1 judge gate, and HF export) for Pro 5000."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pro5000_config as pcfg
from pro5000_trainer_entry import verify_model_extra_checkpoint, write_json
from stage1_collect import find_local_model
from stage2_pilot import (
    base_subprocess_env,
    build_trainer_env,
    default_base,
    ensure_eduardo_repo,
    make_ray_tmp,
    pick_free_ports,
    terminate_pg,
    terminate_procs,
    verify_model_dir,
    verify_stage1_prereqs,
    wait_for_http_health,
    write_access_log_config,
    write_interaction_yaml,
)
from stage2_precheck import core_gpus_clean, snapshot_gpus
from telemetry import Sampler

STAGE3_DISK_FLOOR_BYTES = 70 * 10**9
STAGE3_EXPORT_MIN_FREE_BYTES = 32 * 10**9
STAGE3_CKPT_HINT_BYTES = 21_000_000_000


def parse_step_num(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    return int(tail) if tail.isascii() and tail.isdigit() else None


def find_latest_stage3_run(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        raise RuntimeError(f'找不到运行目录: {runs_dir}')
    candidates = sorted(
        (
            p for p in runs_dir.glob('stage3_formal_*')
            if p.is_dir() and not p.is_symlink() and ((p / 'run_meta.json').is_file() or any((p / 'checkpoints').glob('global_step_*')))
        ),
        key=lambda p: p.name,
    )
    if not candidates:
        raise RuntimeError(f'找不到可续训的阶段 3 历史运行目录: {runs_dir}')
    return candidates[-1]


def list_verified_checkpoints(work: Path, ckpt_dir: Path, world_size: int = 4) -> list[int]:
    if not ckpt_dir.is_dir():
        return []
    steps = []
    for p in ckpt_dir.iterdir():
        if not p.is_dir() or p.is_symlink():
            continue
        step = parse_step_num(p.name, 'global_step_')
        if step is None or step < 1:
            continue
        if not (work / f'ckpt-verified-{step}.json').is_file():
            continue
        try:
            verify_model_extra_checkpoint(p, world_size=world_size)
        except RuntimeError:
            continue
        steps.append(step)
    return sorted(steps)


def prepare_resume_checkpoint(work: Path, ckpt_dir: Path, world_size: int = 4) -> int:
    """Validate latest checkpoint, quarantine any unverified higher step, and sync pointer."""
    verified = list_verified_checkpoints(work, ckpt_dir, world_size=world_size)
    if not verified:
        raise RuntimeError(f'目录 {ckpt_dir} 中没有已核对完整的阶段 3 检查点，无法续训')
    latest = verified[-1]

    quarantine = work / 'uncommitted'
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_dir() or p.is_symlink():
            continue
        step = parse_step_num(p.name, 'global_step_')
        if step is not None and step > latest:
            quarantine.mkdir(mode=0o700, exist_ok=True)
            dst = quarantine / f'{p.name}_{int(time.time())}'
            p.rename(dst)

    pointer = ckpt_dir / 'latest_checkpointed_iteration.txt'
    if pointer.is_symlink():
        raise RuntimeError(f'检查点指针不可为软链接: {pointer}')
    tmp = work / '.latest_iter.tmp'
    tmp.write_text(f'{latest}\n', encoding='utf-8')
    os.replace(tmp, pointer)
    return latest


def verify_stage3_gpus_clean(first: dict[int, dict], second: dict[int, dict]):
    core_ok, issues = core_gpus_clean(first, second)
    if not core_ok:
        raise RuntimeError(f'GPU 0-5 或 7 存在占用/进程: {issues}')
    # GPU 6 had an idle 100% sensor quirk in Stage 1/2; guard strictly against active apps or memory.
    for snap_name, snap in (('first', first), ('second', second)):
        g6 = snap.get(6)
        if g6 is None:
            raise RuntimeError(f'GPU 6 缺失 ({snap_name})')
        if g6['apps'] or g6['used_mib'] >= 500:
            raise RuntimeError(f'GPU 6 存在外部占用 ({snap_name}): apps={g6["apps"]} used={g6["used_mib"]} MiB')


def export_hf_step(
    py_exe: Path,
    work: Path,
    ckpt_step_dir: Path,
    target_dir: Path,
    env_base: dict[str, str],
) -> dict:
    if (target_dir / 'config.json').is_file() and any(target_dir.glob('*.safetensors')):
        return {'already_exported': True, 'target_dir': str(target_dir)}
    free_bytes = shutil.disk_usage('/data').free
    if free_bytes < STAGE3_EXPORT_MIN_FREE_BYTES:
        raise RuntimeError(
            f'导出前 /data 剩余 {free_bytes / 1e9:.2f} GB < {STAGE3_EXPORT_MIN_FREE_BYTES / 1e9:.0f} GB'
        )
    target_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = work / f'.export-staging-{ckpt_step_dir.name}'
    if staging.exists():
        shutil.rmtree(staging)
    t0 = time.monotonic()
    cmd = [
        str(py_exe), '-B', '-m', 'verl.model_merger', 'merge',
        '--backend', 'fsdp',
        '--local_dir', str(ckpt_step_dir / 'actor'),
        '--target_dir', str(staging),
    ]
    merge_env = dict(env_base, CUDA_VISIBLE_DEVICES='')
    with (work / 'export_step100.log').open('ab') as log:
        proc = subprocess.run(cmd, env=merge_env, cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'第 100 步 HF 权重导出失败 rc={proc.returncode}')
    shards = [p for p in staging.glob('*.safetensors') if p.is_file() and p.stat().st_size > 0]
    if not (staging / 'config.json').is_file() or not shards:
        raise RuntimeError('导出目录缺少 config.json 或有效 *.safetensors')
    os.replace(staging, target_dir)
    total_bytes = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
    return {
        'target_dir': str(target_dir),
        'shards': len(shards),
        'size_gb': round(total_bytes / 1e9, 2),
        'merge_seconds': round(time.monotonic() - t0, 2),
    }


def format_stage3_status(base: Path, run_dir: Path | None = None, gpu_lines: list[str] | None = None, proc_lines: list[str] | None = None) -> str:
    work = run_dir or find_latest_stage3_run(base / 'runs')
    meta_file = work / 'run_meta.json'
    meta = json.loads(meta_file.read_text(encoding='utf-8')) if meta_file.is_file() else {}
    total_steps = int(meta.get('total_steps', 100))

    steps_dir = work / 'step-statistics'
    rows = []
    if steps_dir.is_dir():
        numbered = []
        for p in steps_dir.glob('*.json'):
            if p.stem.isascii() and p.stem.isdigit():
                numbered.append((int(p.stem), p))
        for _, p in sorted(numbered):
            try:
                rows.append(json.loads(p.read_text(encoding='utf-8')))
            except ValueError:
                pass

    latest_step = rows[-1]['step'] if rows else 0
    recent = rows[-5:]
    steady = [r['metrics']['timing_s/step'] for r in rows[1:] if r.get('metrics', {}).get('timing_s/step')]
    if not steady and rows and rows[0].get('metrics', {}).get('timing_s/step'):
        steady = [rows[0]['metrics']['timing_s/step']]
    eta_str = 'N/A'
    if steady:
        avg_s = sum(steady) / len(steady)
        rem_steps = max(0, total_steps - latest_step)
        eta_h = rem_steps * avg_s / 3600.0
        eta_str = f'{eta_h:.2f}h (均步 {avg_s:.1f}s, 余 {rem_steps} 步)'

    ckpt_steps = []
    for p in work.glob('ckpt-verified-*.json'):
        s = parse_step_num(p.stem, 'ckpt-verified-')
        if s is not None:
            ckpt_steps.append(s)
    latest_ckpt = f'global_step_{max(ckpt_steps)}' if ckpt_steps else '无'

    du_target = Path('/data') if Path('/data').exists() else base
    du_data = shutil.disk_usage(du_target)
    lines = [
        f'=== 阶段 3 进度 ({work.name}) ===',
        f'进程状态: {" | ".join(proc_lines) if proc_lines else "无运行中 stage3 进程"}',
        f'最新步号: {latest_step} / {total_steps} | 最近存盘: {latest_ckpt} | 预计剩余: {eta_str}',
        f'/data 剩余: {du_data.free / 1e9:.2f} GB ({du_data.free / 1024**3:.2f} GiB)',
    ]
    if (work / 'step1_judge_test.json').is_file():
        s1 = json.loads((work / 'step1_judge_test.json').read_text(encoding='utf-8'))
        lines.append(
            f"第1步裁判测试: conc={s1.get('judge_concurrency')} passed={s1.get('passed')} "
            f"step_s={s1.get('step_seconds')} p95_s={s1.get('service_p95_s')} non_2xx={s1.get('judge_non_2xx')}"
        )
    if (work / 'step2-comparison.json').is_file():
        c2 = json.loads((work / 'step2-comparison.json').read_text(encoding='utf-8'))
        lines.append(f"第2/1步比值: passed={c2.get('passed')} ratios={c2.get('ratios')}")
    if recent:
        lines.append('最近几步耗时:')
        for r in recent:
            pt = r.get('phase_timings', {})
            sec = r.get('metrics', {}).get('timing_s/step')
            lines.append(
                f"  step {r['step']:3d}: {sec:.1f}s ({sec / 60:.2f}m) | "
                f"gen={pt.get('gen', {}).get('minutes', 0):.2f}m "
                f"old={pt.get('old_log_prob', {}).get('minutes', 0):.2f}m "
                f"upd={pt.get('update_actor', {}).get('minutes', 0):.2f}m"
                if sec is not None else f"  step {r['step']:3d}: timing N/A"
            )
    if gpu_lines:
        lines.append('8 卡显存与利用率:')
        lines.extend(f'  GPU {ln}' for ln in gpu_lines)
    return '\n'.join(lines)


def run_status(base: Path, run_dir: Path | None) -> int:
    proc_out = []
    try:
        out = subprocess.check_output(
            ['pgrep', '-af', 'stage3_formal.py|pro5000_trainer_entry.py'],
            text=True, timeout=5,
        )
        for ln in out.splitlines():
            if '--status' not in ln:
                proc_out.append(ln[:100])
    except subprocess.CalledProcessError:
        pass
    gpu_out = []
    try:
        gpu_csv = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
            text=True, timeout=5,
        )
        gpu_out = [ln.strip() for ln in gpu_csv.splitlines() if ln.strip()]
    except (OSError, subprocess.CalledProcessError):
        pass
    print(format_stage3_status(base, run_dir=run_dir, gpu_lines=gpu_out, proc_lines=proc_out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=default_base())
    parser.add_argument('--tutor-model', type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    parser.add_argument('--total-steps', type=int, default=100)
    parser.add_argument('--save-freq', type=int, default=5)
    parser.add_argument('--test-freq', type=int, default=25)
    parser.add_argument('--keep-ckpts', type=int, default=1)
    parser.add_argument('--student-concurrency', type=int, default=32)
    parser.add_argument('--judge-concurrency', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)

    base = args.base.resolve()
    if args.status:
        return run_status(base, args.run_dir)

    py_exe = Path(sys.executable).resolve()
    verify_stage1_prereqs(base, args.tutor_model)

    models_root = base / 'models'
    if models_root.is_dir():
        for entry in sorted(models_root.iterdir()):
            if entry.is_symlink() and not entry.exists():
                raise RuntimeError(f'models 目录下存在失效软链接（节点若重启过请先恢复 /dev/shm）: {entry}')
    student_model = verify_model_dir(
        'Student',
        find_local_model(models_root, models_root / 'Meta-Llama-3.1-8B-Instruct', ('llama-3.1-8b', '8b-instruct')),
    )
    judge_model = verify_model_dir(
        'Judge',
        find_local_model(models_root, models_root / 'Qwen3.6-27B', ('qwen3.6-27b', '27b')),
    )
    print(f'[preflight] 模型与软链接就绪: student={student_model} -> {student_model.resolve()} ; judge={judge_model} -> {judge_model.resolve()}', flush=True)

    free_bytes = shutil.disk_usage('/data').free
    print(f'[preflight] /data free = {free_bytes / 1e9:.2f} GB ({free_bytes / 1024**3:.2f} GiB)', flush=True)
    if free_bytes < STAGE3_DISK_FLOOR_BYTES:
        print('STOP: /data 剩余低于 70G')
        return 3

    repo = ensure_eduardo_repo(base)

    is_resume = bool(args.resume)
    if is_resume:
        work = args.run_dir.resolve() if args.run_dir else find_latest_stage3_run(base / 'runs')
        ckpt_dir = work / 'checkpoints'
        resume_step = prepare_resume_checkpoint(work, ckpt_dir, world_size=4)
        meta_file = work / 'run_meta.json'
        prior_meta = json.loads(meta_file.read_text(encoding='utf-8')) if meta_file.is_file() else {}
        judge_conc = args.judge_concurrency if args.judge_concurrency is not None else int(prior_meta.get('layout', {}).get('judge_concurrency', 64))
        print(f'[resume] 从目录 {work} 的 global_step_{resume_step} 续训 (judge_concurrency={judge_conc})', flush=True)
        if resume_step >= args.total_steps:
            print(f'[resume] 检查点已达目标步数 {resume_step}/{args.total_steps}，跳过训练直接核对导出', flush=True)
    else:
        run_name = f'stage3_formal_{time.strftime("%Y%m%d_%H%M%S")}'
        if args.check_only and not args.run_dir:
            work = base / 'tmp' / f'.stage3_preflight_{os.getpid()}'
        else:
            work = args.run_dir.resolve() if args.run_dir else (base / 'runs' / run_name)
        if work.exists() and any(work.iterdir()):
            raise RuntimeError(f'全新阶段 3 目录已存在非空文件: {work}（如要续训请加 --resume）')
        ckpt_dir = work / 'checkpoints'
        resume_step = None
        judge_conc = args.judge_concurrency if args.judge_concurrency is not None else 64

    tutor_gpus = [0, 1, 2, 3]
    student_gpus = [4, 6]
    judge_gpus = [5, 7]
    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    layout = {
        'mode': '8gpu',
        'tutor_gpus': tutor_gpus,
        'student_gpus': student_gpus,
        'judge_gpus': judge_gpus,
        'judge_tp': 2,
        'optimizer_offload': True,
        'student_concurrency': args.student_concurrency,
        'judge_concurrency': judge_conc,
    }
    print(f'[preflight] 固定 8 卡布局: {layout}', flush=True)

    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    (work / 'queue-metrics').mkdir(mode=0o700, exist_ok=True)
    ckpt_dir.mkdir(mode=0o700, exist_ok=True)

    ports = pick_free_ports([18861, 18862, 18871, 18872, 18873])
    s_lb_port, j_proxy_port, s0_port, s1_port, j_port = ports
    student_api_base = f'http://127.0.0.1:{s_lb_port}/v1'
    judge_api_base = f'http://127.0.0.1:{j_proxy_port}/v1'

    token = os.environ.get('SERVING_API_KEY') or ''
    if not token:
        import secrets
        token = secrets.token_urlsafe(32)
    user_tag = os.environ.get('USER') or 'root'
    student_name = f'Llama-3.1-8B-Instruct-{user_tag}'
    interaction = write_interaction_yaml(repo, work, student_name, student_api_base, layout['student_concurrency'])
    access_cfg = write_access_log_config(work)
    env_base = base_subprocess_env(base)
    env_base['VLLM_API_KEY'] = token
    env_base['SERVING_API_KEY'] = token
    env_base['EDUARDO_CONV_DIR'] = '/dev/null'

    trainer_entry = HERE / 'pro5000_trainer_entry.py'
    need_step1_gate = (resume_step is None or resume_step < 1) and judge_conc >= 64
    need_step2_gate = resume_step is None or resume_step < 2

    def make_env(raytmp: Path) -> dict[str, str]:
        return build_trainer_env(
            base=base, work=work, ckpt_dir=ckpt_dir, raytmp=raytmp,
            tutor_gpus=tutor_gpus, all_gpus=all_gpus, student_gpus=student_gpus,
            layout=layout, repo=repo, tutor_model=args.tutor_model,
            student_api_base=student_api_base, judge_api_base=judge_api_base,
            start_step=resume_step or 0, total_steps=args.total_steps,
            save_freq=args.save_freq, test_freq=args.test_freq,
            max_ckpt_keep=args.keep_ckpts,
            check_step1_judge=need_step1_gate,
            check_step2_ratio=need_step2_gate,
            experiment='pro5000-stage3',
            ckpt_hint=STAGE3_CKPT_HINT_BYTES,
        )

    raytmp = make_ray_tmp(base)
    try:
        overrides = pcfg.build_overrides(
            base=base, work=work, ckpt_dir=ckpt_dir, raytmp=raytmp,
            student_name=student_name, interaction=interaction,
            total_steps=args.total_steps, save_freq=args.save_freq, test_freq=args.test_freq,
            tutor_gpus=len(tutor_gpus), optimizer_offload=True,
            student_concurrency=layout['student_concurrency'],
            judge_concurrency=layout['judge_concurrency'],
            max_actor_ckpt_to_keep=args.keep_ckpts,
            resume_step=resume_step,
        )
        trainer_env_vars = make_env(raytmp)
        pcfg.run_preflight(
            python_exe=py_exe, trainer_entry=trainer_entry, repo=repo, work=work,
            tutor_model=args.tutor_model, student_api_base=student_api_base,
            judge_api_base=judge_api_base, overrides=overrides,
            check_env=dict(env_base, **trainer_env_vars),
        )
        print('[preflight] Hydra 阶段 3 冻结配方预检通过', flush=True)
    except Exception:
        shutil.rmtree(raytmp, ignore_errors=True)
        raise

    if args.check_only:
        shutil.rmtree(raytmp, ignore_errors=True)
        if not is_resume and not args.run_dir:
            shutil.rmtree(work, ignore_errors=True)
        print(f'STAGE3_PREFLIGHT_OK resume_step={resume_step} judge_concurrency={judge_conc}')
        return 0

    write_json(work / 'run_meta.json', {
        'layout': layout,
        'total_steps': args.total_steps,
        'save_freq': args.save_freq,
        'test_freq': args.test_freq,
        'keep_ckpts': args.keep_ckpts,
        'started_at': time.time(),
        'resume_step': resume_step,
    })

    managed_procs: list[tuple[str, subprocess.Popen]] = []
    sampler: Sampler | None = None
    try:
        if resume_step is None or resume_step < args.total_steps:
            first = snapshot_gpus()
            time.sleep(2)
            second = snapshot_gpus()
            verify_stage3_gpus_clean(first, second)

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

            student_backends = []
            for idx, (gpu_id, port) in enumerate(zip(student_gpus, (s0_port, s1_port))):
                cmd = [
                    str(py_exe), '-B', '-m', 'vllm.entrypoints.openai.api_server',
                    '--model', str(student_model), '--served-model-name', student_name,
                    '--host', '127.0.0.1', '--port', str(port),
                    '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.9',
                    '--max-model-len', '32768', '--max-num-seqs', '256',
                    '--tensor-parallel-size', '1', '--data-parallel-size', '1',
                    '--enable-chunked-prefill', '--enable-prefix-caching',
                    '--log-config-file', str(access_cfg),
                ]
                spawn(f'student_{idx}', cmd, {'CUDA_VISIBLE_DEVICES': str(gpu_id)}, f'student_{idx}.log')
                student_backends.append(f'http://127.0.0.1:{port}')

            judge_cmd = [
                str(py_exe), '-B', '-m', 'vllm.entrypoints.openai.api_server',
                '--model', str(judge_model), '--served-model-name', 'judge-local',
                '--host', '127.0.0.1', '--port', str(j_port),
                '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.9',
                '--max-model-len', '32768', '--max-num-seqs', '256',
                '--tensor-parallel-size', '2', '--data-parallel-size', '1',
                '--enable-chunked-prefill', '--enable-prefix-caching',
                '--log-config-file', str(access_cfg),
            ]
            spawn('judge', judge_cmd, {'CUDA_VISIBLE_DEVICES': ','.join(map(str, judge_gpus))}, 'judge.log')

            proxy_cmd = [
                str(py_exe), '-B', str(HERE / 'serving_proxy.py'),
                '--work', str(work),
                '--student-port', str(s_lb_port), '--student-backends', *student_backends,
                '--judge-port', str(j_proxy_port), '--judge-backends', f'http://127.0.0.1:{j_port}',
            ]
            spawn('proxy', proxy_cmd, {'CUDA_VISIBLE_DEVICES': ''}, 'proxy.log')

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

            sampler = Sampler(work / 'tutor-telemetry.jsonl', tuple(all_gpus), lambda: (), interval=10, append=is_resume)
            sampler.start()

            cmd_train = [
                str(py_exe), '-B', str(trainer_entry),
                '--config-path', str(repo / 'eduardo' / 'configs'), '--config-name', 'eduardo_base',
                *overrides,
            ]
            log_t = (work / 'trainer.log').open('ab')
            proc_t = None
            try:
                proc_t = subprocess.Popen(
                    cmd_train, env=dict(env_base, **trainer_env_vars),
                    cwd=work, stdin=subprocess.DEVNULL, stdout=log_t, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_t.close()
            try:
                rc = proc_t.wait()
            finally:
                terminate_pg(proc_t.pid)
            if rc != 0:
                raise RuntimeError(f'阶段 3 训练退出异常 rc={rc}')

            if sampler is not None:
                sampler.stop()
                sampler = None
            terminate_procs(managed_procs)
            managed_procs.clear()

        final_ckpt = ckpt_dir / f'global_step_{args.total_steps}'
        final_info = verify_model_extra_checkpoint(final_ckpt, world_size=len(tutor_gpus))
        export_info = export_hf_step(
            py_exe, work, final_ckpt, work / 'exports' / f'step_{args.total_steps}', env_base
        )
        # Explicitly keep /dev/shm/zhujie_a_group/ intact for downstream evaluation per Stage 2 review.
        summary = {
            'work_dir': str(work),
            'layout': layout,
            'total_steps': args.total_steps,
            'final_checkpoint': final_info,
            'hf_export': export_info,
            'shm_models_preserved': True,
            'post_run_disk_free_gb': round(shutil.disk_usage('/data').free / 1e9, 2),
        }
        write_json(work / 'stage3_summary.json', summary)
        print('=== STAGE 3 SUMMARY ===')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        shutil.rmtree(raytmp, ignore_errors=True)
        if sampler is not None:
            sampler.stop()
        terminate_procs(managed_procs)


if __name__ == '__main__':
    sys.exit(main())
