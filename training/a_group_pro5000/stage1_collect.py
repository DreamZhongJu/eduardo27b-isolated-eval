"""Collect Stage 1 environment, disk, smoke-test, and memory numbers safely on RTX PRO 5000."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from memory_pro5000 import summarize_pro5000

A100_TARGET_PACKAGES = {
    'torch': '2.10.0',
    'vllm': '0.19.1',
    'transformers': '5.7.0',
    'numpy': '2.1.0',
    'ray': '2.55.1',
    'verl': '0.7.1',
    'flash-attn': '2.8.3',
    'flash-linear-attention': '0.5.2',
    'fla-core': '0.5.2',
    'causal-conv1d': '1.6.2.post1',
    'math-verify': '0.8.0',
    'datasets': '4.8.5',
    'accelerate': '1.13.0',
    'codetiming': '1.4.0',
    'dill': '0.4.0',
    'hydra-core': '1.3.2',
    'pandas': '2.3.3',
    'peft': '0.19.1',
    'pyarrow': '22.0.0',
    'pybind11': '3.0.1',
    'pylatexenc': '2.10',
    'tensordict': '0.10.0',
    'torchdata': '0.11.0',
    'wandb': '0.26.1',
    'packaging': '25.0',
    'tensorboard': '2.20.0',
    'setuptools': '80.9.0',
    'wheel': 'any',
    'openai': '2.33.0',
    'ninja': '1.13.0',
    'triton': 'any',
}

SYNTHETIC_PROMPTS = (
    'Explain in two concise sentences why the sum of two odd integers is always even.',
    'Compute the derivative of f(x) = 3x^2 - 5x + 2 and evaluate it at x = 4 in two sentences.',
)


def default_base() -> Path:
    return Path.home() / 'bzhang' / 'a_group_pro5000'


def sanitized_env(base: Path, visible_gpus: str) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not k.upper().endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))
    }
    env_lib = Path(sys.prefix) / 'lib'
    ld_parts = [str(env_lib)] if env_lib.is_dir() else []
    if env.get('LD_LIBRARY_PATH'):
        ld_parts.append(env['LD_LIBRARY_PATH'])
    env.update(
        CUDA_VISIBLE_DEVICES=visible_gpus,
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',
        VLLM_NO_USAGE_STATS='1',
        DO_NOT_TRACK='1',
        TMPDIR=str(base / 'tmp'),
        TRITON_CACHE_DIR=str(base / 'cache' / 'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(base / 'cache' / 'inductor'),
        VLLM_CACHE_ROOT=str(base / 'cache' / 'vllm'),
        XDG_CACHE_HOME=str(base / 'cache' / 'xdg'),
    )
    if ld_parts:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(ld_parts)
    return env


def parse_gpu_snapshot(gpu_csv: str, app_csv: str) -> dict[int, dict]:
    apps_by_uuid: dict[str, list[dict]] = {}
    for line in app_csv.splitlines():
        if not line.strip():
            continue
        uuid, pid_s, used_s = [p.strip() for p in line.split(',')]
        apps_by_uuid.setdefault(uuid, []).append({'pid': int(pid_s), 'used_mib': int(used_s)})
    gpus = {}
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        idx_s, uuid, name, used_s, free_s, total_s, util_s = [p.strip() for p in line.split(',')]
        idx = int(idx_s)
        gpus[idx] = {
            'index': idx,
            'uuid': uuid,
            'name': name,
            'used_mib': int(used_s),
            'free_mib': int(free_s),
            'total_mib': int(total_s),
            'util': int(util_s),
            'apps': apps_by_uuid.get(uuid, []),
        }
    return gpus


def snapshot_gpus() -> dict[int, dict]:
    gpu_csv = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'],
        text=True, timeout=15,
    )
    app_csv = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
         '--format=csv,noheader,nounits'],
        text=True, timeout=15,
    )
    return parse_gpu_snapshot(gpu_csv, app_csv)


def detect_idle_gpus(first: dict[int, dict], second: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    idle, busy = [], []
    for idx in sorted(second):
        a, b = first.get(idx), second[idx]
        is_idle = (
            a is not None
            and not a['apps'] and not b['apps']
            and a['used_mib'] < 500 and b['used_mib'] < 500
            and a['util'] < 10 and b['util'] < 10
            and b['total_mib'] >= 70000
        )
        if is_idle:
            idle.append(b)
        else:
            busy.append(b)
    return idle, busy


def select_idle_group(idle_gpus: list[dict], count: int) -> tuple[list[dict], bool]:
    if len(idle_gpus) < count:
        raise RuntimeError(f'需要 {count} 张空闲整卡，当前只有 {len(idle_gpus)} 张')
    by_idx = {g['index']: g for g in idle_gpus}
    if count <= 4:
        for group in ((0, 1, 2, 3), (4, 5, 6, 7)):
            avail = [by_idx[i] for i in group if i in by_idx]
            if len(avail) >= count:
                return avail[:count], True
    chosen = [by_idx[i] for i in sorted(by_idx)][:count]
    idxs = {g['index'] for g in chosen}
    same_numa = idxs <= {0, 1, 2, 3} or idxs <= {4, 5, 6, 7}
    return chosen, same_numa


def collect_packages() -> dict:
    import torch

    versions = {}
    for pkg, expected in A100_TARGET_PACKAGES.items():
        try:
            ver = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            ver = 'MISSING'
        versions[pkg] = {'actual': ver, 'a100_target': expected}

    pip_check = subprocess.run(
        [sys.executable, '-m', 'pip', 'check'],
        capture_output=True, text=True, timeout=120,
    )

    def tool_line(cmd):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=15)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            return lines[-1] if cmd[0] == 'nvcc' and len(lines) > 1 else (lines[0] if lines else '')
        except Exception as exc:
            return f'ERR:{type(exc).__name__}'

    return {
        'python': platform.python_version(),
        'torch_version': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cxx11_abi': bool(torch.compiled_with_cxx11_abi()),
        'nvcc': tool_line(['nvcc', '--version']),
        'gxx': tool_line(['g++', '--version']),
        'git': tool_line(['git', '--version']),
        'pip_check_rc': pip_check.returncode,
        'pip_check_msg': ' '.join(pip_check.stdout.strip().splitlines()),
        'packages': versions,
    }


def collect_disk(base: Path) -> dict:
    df_out = subprocess.check_output(['df', '-h', '/', '/data'], text=True, timeout=15)
    subdirs = {}
    for sub in ('envs', 'models', 'data', 'cache', 'tmp', 'runs'):
        p = base / sub
        if p.exists():
            out = subprocess.check_output(['du', '-sh', str(p)], text=True, timeout=120)
            subdirs[sub] = out.split()[0]
        else:
            subdirs[sub] = '0'
    total_out = subprocess.check_output(['du', '-sh', str(base)], text=True, timeout=120)
    return {
        'base_total': total_out.split()[0],
        'subdirs': subdirs,
        'df_h': df_out.strip().splitlines(),
    }


def parse_vllm_kv_from_logs(text: str) -> dict:
    kv_tokens = None
    kv_gib = None
    max_conc = None
    kv_lines = []
    for line in text.splitlines():
        if any(k in line for k in ('KV cache', 'concurrency', 'kv_cache')):
            clean = line.strip()
            if clean and clean not in kv_lines:
                kv_lines.append(clean[:200])
    m_tok = re.search(r'GPU KV cache size:\s*([\d,]+)\s*tokens', text)
    if m_tok:
        kv_tokens = int(m_tok.group(1).replace(',', ''))
    m_gib = re.search(r'Available KV cache memory:\s*([0-9.]+)\s*GiB', text)
    if m_gib:
        kv_gib = float(m_gib.group(1))
    m_conc = re.search(r'Maximum concurrency for [\d,]+ tokens per request:\s*([0-9.]+)x', text)
    if m_conc:
        max_conc = float(m_conc.group(1))
    return {
        'log_kv_tokens': kv_tokens,
        'log_kv_gib': kv_gib,
        'log_max_concurrency': max_conc,
        'kv_log_lines': kv_lines[-6:],
    }


def worker_flash_attn() -> dict:
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_varlen_func

    torch.manual_seed(42)
    lens = [256, 768, 1536]
    nheads, nheads_k, headdim = 16, 4, 128
    cu = torch.tensor([0, 256, 1024, 2560], device='cuda', dtype=torch.int32)
    total = int(cu[-1].item())
    q = torch.randn(total, nheads, headdim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(total, nheads_k, headdim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(total, nheads_k, headdim, device='cuda', dtype=torch.bfloat16)

    fa_out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), causal=True)

    refs = []
    for i in range(len(lens)):
        s, e = int(cu[i].item()), int(cu[i + 1].item())
        qi = q[s:e].transpose(0, 1).unsqueeze(0)
        ki = k[s:e].transpose(0, 1).unsqueeze(0)
        vi = v[s:e].transpose(0, 1).unsqueeze(0)
        ref = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, enable_gqa=True)
        refs.append(ref.squeeze(0).transpose(0, 1))
    sdpa_out = torch.cat(refs, dim=0)
    diff = (fa_out.float() - sdpa_out.float()).abs()
    return {
        'device_name': torch.cuda.get_device_name(0),
        'capability': list(torch.cuda.get_device_capability(0)),
        'lengths': lens,
        'max_abs_diff': float(diff.max().item()),
        'mean_abs_diff': float(diff.mean().item()),
    }


def _nccl_rank(rank: int, world_size: int, port: int, queue):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
    )
    numel = 128 * 1024 * 1024  # 256 MiB in bf16
    buf = torch.full((numel,), float(rank + 1), device='cuda', dtype=torch.bfloat16)
    for _ in range(5):
        buf.fill_(float(rank + 1))
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        buf.fill_(float(rank + 1))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    expected = sum(range(1, world_size + 1))
    sample_val = float(buf[0].item())
    dist.destroy_process_group()
    if rank == 0:
        avg_s = sum(times) / len(times)
        nbytes = numel * 2
        algbw_gb_s = nbytes / avg_s / 1e9
        busbw_gb_s = algbw_gb_s * 2 * (world_size - 1) / world_size
        queue.put({
            'world_size': world_size,
            'tensor_mib': nbytes / (1024 ** 2),
            'iters': len(times),
            'avg_ms': avg_s * 1000.0,
            'algbw_gb_s': algbw_gb_s,
            'busbw_gb_s': busbw_gb_s,
            'sum_check_ok': abs(sample_val - expected) < 0.1,
        })


def worker_nccl(world_size: int) -> dict:
    import torch.multiprocessing as mp

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = int(sock.getsockname()[1])
    sock.close()
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    mp.spawn(_nccl_rank, args=(world_size, port, queue), nprocs=world_size, join=True)
    return queue.get(timeout=30)


def worker_vllm(model_path: str, tp: int, gpu_util: float, max_model_len: int) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        dtype='bfloat16',
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=256,
        trust_remote_code=False,
    )
    load_s = time.monotonic() - t0
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    num_blocks = getattr(cache_cfg, 'num_gpu_blocks', None)
    block_size = getattr(cache_cfg, 'block_size', None)
    kv_tokens = int(num_blocks) * int(block_size) if num_blocks and block_size else None

    params = SamplingParams(temperature=0.0, max_tokens=48)
    t1 = time.monotonic()
    outputs = llm.generate(list(SYNTHETIC_PROMPTS), params, use_tqdm=False)
    gen_s = time.monotonic() - t1
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    previews = [
        ' '.join(o.outputs[0].text.strip().splitlines())[:120]
        for o in outputs
    ]
    return {
        'model': Path(model_path).name,
        'tp': tp,
        'gpu_util': gpu_util,
        'max_model_len': max_model_len,
        'load_seconds': round(load_s, 2),
        'gen_seconds': round(gen_s, 2),
        'gen_tokens': gen_tokens,
        'tok_per_s': round(gen_tokens / max(gen_s, 1e-6), 2),
        'num_gpu_blocks': num_blocks,
        'block_size': block_size,
        'kv_cache_tokens': kv_tokens,
        'max_concurrency_at_len': round(kv_tokens / max_model_len, 2) if kv_tokens else None,
        'previews': previews,
    }


def run_subworker(base: Path, visible_gpus: str, args_list: list[str], capture_logs: bool = False) -> tuple[dict, str]:
    with tempfile.NamedTemporaryFile(dir=base / 'tmp', suffix='.json', delete=False) as tf:
        out_path = Path(tf.name)
    try:
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--out-json', str(out_path), *args_list]
        proc = subprocess.run(
            cmd, env=sanitized_env(base, visible_gpus),
            capture_output=True, text=True, timeout=900,
        )
        combined = (proc.stdout or '') + '\n' + (proc.stderr or '')
        if proc.returncode != 0:
            tail = '\n'.join(combined.splitlines()[-80:])
            raise RuntimeError(f'子进程失败 rc={proc.returncode}: {tail}')
        data = json.loads(out_path.read_text(encoding='utf-8'))
        return data, (combined if capture_logs else '')
    finally:
        if out_path.exists():
            out_path.unlink()


def find_local_model(models_root: Path, preferred: Path, keywords: tuple[str, ...]) -> Path:
    if (preferred / 'config.json').is_file():
        return preferred
    if not models_root.is_dir():
        return preferred
    candidates = []
    for cfg in sorted(models_root.rglob('config.json')):
        rel = str(cfg.parent.relative_to(models_root)).lower()
        if any(kw.lower() in rel for kw in keywords):
            candidates.append(cfg.parent)
    return candidates[0] if candidates else preferred


def run_all(base: Path, tutor_model: Path, skip_vllm: bool = False) -> dict:
    first = snapshot_gpus()
    time.sleep(2)
    second = snapshot_gpus()
    idle, busy = detect_idle_gpus(first, second)

    pkg_info = collect_packages()
    disk_info = collect_disk(base)

    # 1. Flash-attn vs SDPA (1 idle GPU)
    g1, _ = select_idle_group(idle, 1)
    print(f'[stage1] 1. flash-attn vs SDPA on GPU {g1[0]["index"]}...', flush=True)
    fa_res, _ = run_subworker(base, str(g1[0]['index']), ['--worker-fa'])

    # 2. 4-GPU NCCL all-reduce (prefer same NUMA group)
    nccl_res = None
    if len(idle) >= 4:
        g4, same_numa = select_idle_group(idle, 4)
        idxs4 = [g['index'] for g in g4]
        devs4 = ','.join(str(i) for i in idxs4)
        print(f'[stage1] 2. NCCL 4-GPU all-reduce on GPUs {idxs4} (same_numa={same_numa})...', flush=True)
        nccl_res, _ = run_subworker(base, devs4, ['--worker-nccl', '4'])
        nccl_res['gpu_indices'] = idxs4
        nccl_res['same_numa'] = same_numa
    else:
        nccl_res = {'skipped': f'only {len(idle)} idle GPUs'}

    # 3. vLLM smoke & 27B KV concurrency
    vllm_cases = []
    if not skip_vllm:
        models_root = base / 'models'
        student_dir = find_local_model(
            models_root,
            models_root / 'LLM-Research' / 'Meta-Llama-3.1-8B-Instruct',
            ('llama-3.1-8b', 'llama3.1-8b', '8b-instruct'),
        )
        judge_dir = find_local_model(
            models_root,
            models_root / 'Qwen' / 'Qwen3.6-27B',
            ('qwen3.6-27b', 'qwen3_6-27b', '27b'),
        )
        plan = [
            ('tutor_4b_tp1_058', tutor_model, 1, 0.58, 10240),
            ('student_8b_tp1_090', student_dir, 1, 0.90, 32768),
            ('judge_27b_tp1_086', judge_dir, 1, 0.86, 32768),
            ('judge_27b_tp1_090', judge_dir, 1, 0.90, 32768),
        ]
        if len(idle) >= 2:
            plan.append(('judge_27b_tp2_090', judge_dir, 2, 0.90, 32768))

        for label, mpath, tp, util, maxlen in plan:
            if not (mpath / 'config.json').is_file():
                vllm_cases.append({'case': label, 'skipped': f'missing {mpath}'})
                continue
            chosen, same_numa = select_idle_group(idle, tp)
            idxs = [g['index'] for g in chosen]
            devs = ','.join(str(i) for i in idxs)
            print(f'[stage1] 3. vLLM {label} on GPUs {idxs} (util={util}, maxlen={maxlen})...', flush=True)
            res, logs = run_subworker(
                base, devs,
                ['--worker-vllm', str(mpath), str(tp), str(util), str(maxlen)],
                capture_logs=True,
            )
            res.update({'case': label, 'gpu_indices': idxs, 'same_numa': same_numa})
            res.update(parse_vllm_kv_from_logs(logs))
            vllm_cases.append(res)

    device_mib = float(g1[0]['total_mib'])
    memory_est = summarize_pro5000(device_mib)

    return {
        'gpu_snapshot': {
            'idle_indices': [g['index'] for g in idle],
            'busy': [
                {'index': g['index'], 'used_mib': g['used_mib'], 'total_mib': g['total_mib'],
                 'util': g['util'], 'app_count': len(g['apps'])}
                for g in busy
            ],
            'device_name': g1[0]['name'],
            'device_total_mib': g1[0]['total_mib'],
        },
        'packages': pkg_info,
        'disk': disk_info,
        'smoke_flash_attn': fa_res,
        'smoke_nccl_4gpu': nccl_res,
        'smoke_vllm': vllm_cases,
        'memory_estimate_tutor_4gpu': memory_est,
    }


def format_report(summary: dict) -> str:
    lines = ['=== Stage 1 Collected Summary (Pro 5000) ===']
    gs = summary['gpu_snapshot']
    lines.append(f"GPUs idle={gs['idle_indices']} busy={gs['busy']} device={gs['device_name']} ({gs['device_total_mib']} MiB)")

    pk = summary['packages']
    lines.append(
        f"Python={pk['python']} torch={pk['torch_version']} cuda={pk['torch_cuda']} "
        f"cxx11_abi={pk['cxx11_abi']} pip_check_rc={pk['pip_check_rc']} msg={pk['pip_check_msg']}"
    )
    lines.append(f"nvcc={pk['nvcc']} | g++={pk['gxx']} | git={pk['git']}")
    lines.append('--- Package Versions vs A100 ---')
    for name, info in pk['packages'].items():
        diff = '' if info['a100_target'] in ('any', info['actual'].split('+', 1)[0]) else '  <-- DEVIATION'
        lines.append(f"  {name:24s}: actual={info['actual']:16s} a100={info['a100_target']}{diff}")

    dk = summary['disk']
    lines.append(f"--- Disk: base_total={dk['base_total']} subdirs={dk['subdirs']} ---")
    for row in dk['df_h']:
        lines.append('  ' + row)

    fa = summary['smoke_flash_attn']
    lines.append(
        f"--- FlashAttn vs SDPA ({fa['device_name']} cc={fa['capability']}) ---\n"
        f"  max_abs_diff={fa['max_abs_diff']:.6g} mean_abs_diff={fa['mean_abs_diff']:.6g}"
    )

    nc = summary['smoke_nccl_4gpu']
    if 'algbw_gb_s' in nc:
        lines.append(
            f"--- NCCL 4-GPU AllReduce GPUs={nc['gpu_indices']} same_numa={nc['same_numa']} ---\n"
            f"  tensor={nc['tensor_mib']:.0f} MiB avg={nc['avg_ms']:.2f} ms "
            f"algbw={nc['algbw_gb_s']:.2f} GB/s busbw={nc['busbw_gb_s']:.2f} GB/s sum_ok={nc['sum_check_ok']}"
        )
    else:
        lines.append(f"--- NCCL 4-GPU AllReduce: {nc} ---")

    lines.append('--- vLLM Smoke & 27B KV Concurrency ---')
    for vc in summary['smoke_vllm']:
        if 'skipped' in vc:
            lines.append(f"  {vc['case']}: SKIPPED ({vc['skipped']})")
            continue
        lines.append(
            f"  {vc['case']} (gpus={vc['gpu_indices']}, tp={vc['tp']}, util={vc['gpu_util']}, "
            f"maxlen={vc['max_model_len']}): load={vc['load_seconds']}s tok/s={vc['tok_per_s']} "
            f"kv_tokens={vc['kv_cache_tokens']} (log={vc['log_kv_tokens']}, {vc['log_kv_gib']} GiB) "
            f"max_conc={vc['max_concurrency_at_len']}x (log={vc['log_max_concurrency']}x)"
        )
        if vc.get('kv_log_lines'):
            lines.append(f"    kv_logs={vc['kv_log_lines']}")
        lines.append(f"    previews={vc['previews']}")

    lines.append('--- Tutor 4-GPU Memory Estimate (73,415 MiB) ---')
    for k, m in summary['memory_estimate_tutor_4gpu'].items():
        lines.append(
            f"  {k}: update={m['update_gib']} rollout={m['rollout_gib']} weight_sync={m['weight_sync_gib']} "
            f"peak={m['peak_gib']} margin={m['margin_gib']} (>=8GiB={m['meets_8gib_reserve']}, <=95%={m['within_95_pct']})"
        )
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=default_base())
    parser.add_argument('--tutor-model', type=Path, default=Path('/data/models/Qwen3.5-4B'))
    parser.add_argument('--skip-vllm', action='store_true')
    parser.add_argument('--out-json', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-fa', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-nccl', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--worker-vllm', nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)

    if args.worker_fa:
        args.out_json.write_text(json.dumps(worker_flash_attn()), encoding='utf-8')
        return 0
    if args.worker_nccl:
        args.out_json.write_text(json.dumps(worker_nccl(args.worker_nccl)), encoding='utf-8')
        return 0
    if args.worker_vllm:
        mpath, tp_s, util_s, maxlen_s = args.worker_vllm
        res = worker_vllm(mpath, int(tp_s), float(util_s), int(maxlen_s))
        args.out_json.write_text(json.dumps(res), encoding='utf-8')
        return 0

    base = args.base.resolve()
    for sub in ('runs', 'tmp', 'cache/triton', 'cache/inductor', 'cache/vllm', 'cache/xdg'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    summary = run_all(base, args.tutor_model, skip_vllm=args.skip_vllm)
    report = format_report(summary)

    out_json = base / 'runs' / 'stage1_smoke_and_env.json'
    out_txt = base / 'runs' / 'stage1_smoke_and_env.txt'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    out_txt.write_text(report + '\n', encoding='utf-8')
    print(report, flush=True)
    print(f'\n已保存: {out_txt} 和 {out_json}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
