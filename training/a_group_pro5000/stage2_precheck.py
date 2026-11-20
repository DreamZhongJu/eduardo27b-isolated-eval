"""Stage 2 pre-launch check: /data >= 45G, GPU 0-5,7 clean, and GPU 6 app + 30s BF16 matmul check."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

CORE_GPUS = (0, 1, 2, 3, 4, 5, 7)
DISK_FLOOR_BYTES = 45 * 10**9
MATMUL_N = 8192
MATMUL_SECONDS = 30.0
GPU6_MIN_RATIO = 0.90


def default_base() -> Path:
    return Path.home() / 'bzhang' / 'a_group_pro5000'


def sanitized_env(base: Path, visible_gpu: str) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not k.upper().endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))
    }
    env_lib = Path(sys.prefix) / 'lib'
    ld_parts = [str(env_lib)] if env_lib.is_dir() else []
    if env.get('LD_LIBRARY_PATH'):
        ld_parts.append(env['LD_LIBRARY_PATH'])
    env.update(
        CUDA_VISIBLE_DEVICES=visible_gpu,
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(base / 'tmp'),
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
        parts = [p.strip() for p in line.split(',')]
        idx = int(parts[0])
        gpus[idx] = {
            'index': idx,
            'uuid': parts[1],
            'name': parts[2],
            'used_mib': int(parts[3]),
            'free_mib': int(parts[4]),
            'total_mib': int(parts[5]),
            'util': int(parts[6]),
            'sm_clock_mhz': int(parts[7]),
            'temp_c': int(parts[8]),
            'power_w': float(parts[9]),
            'apps': apps_by_uuid.get(parts[1], []),
        }
    return gpus


def snapshot_gpus() -> dict[int, dict]:
    gpu_csv = subprocess.check_output(
        ['nvidia-smi',
         '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,clocks.current.sm,temperature.gpu,power.draw',
         '--format=csv,noheader,nounits'],
        text=True, timeout=15,
    )
    app_csv = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
         '--format=csv,noheader,nounits'],
        text=True, timeout=15,
    )
    return parse_gpu_snapshot(gpu_csv, app_csv)


def core_gpus_clean(first: dict[int, dict], second: dict[int, dict]) -> tuple[bool, list[str]]:
    issues = []
    for idx in CORE_GPUS:
        a, b = first.get(idx), second.get(idx)
        if a is None or b is None:
            issues.append(f'GPU {idx} 缺失')
            continue
        if a['apps'] or b['apps']:
            issues.append(f'GPU {idx} 有外部进程 apps={b["apps"]}')
        elif a['used_mib'] >= 500 or b['used_mib'] >= 500:
            issues.append(f'GPU {idx} 已用显存过高 used={b["used_mib"]} MiB')
        elif a['util'] >= 10 or b['util'] >= 10:
            issues.append(f'GPU {idx} 利用率过高 util={b["util"]}%')
    return not issues, issues


def parse_nvidia_smi_q(text: str) -> dict:
    sections = ('Utilization', 'Clocks', 'Clocks Event Reasons', 'Clocks Throttle Reasons')
    extracted = []
    keep = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            keep = False
            continue
        if any(stripped == sec or stripped.startswith(sec + ' ') for sec in sections):
            keep = True
            extracted.append(stripped)
            continue
        if keep:
            if raw.startswith('    ') and not raw.startswith('        '):
                if any(k in stripped for k in ('Gpu', 'Memory', 'Graphics', 'SM', 'Video', 'Active')):
                    extracted.append('  ' + stripped)
                elif ': Active' in stripped:
                    extracted.append('  ' + stripped)
            elif ': Active' in stripped:
                extracted.append('  ' + stripped)
    return {'lines': extracted}


def query_gpu6_detail() -> dict:
    raw = subprocess.check_output(['nvidia-smi', '-i', '6', '-q'], text=True, timeout=15)
    return parse_nvidia_smi_q(raw)


def worker_matmul(seconds: float, n: int) -> dict:
    import torch

    a = torch.randn((n, n), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, n), device='cuda', dtype=torch.bfloat16)
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        c = torch.matmul(a, b)
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12
    return {
        'device_name': torch.cuda.get_device_name(0),
        'n': n,
        'iters': iters,
        'elapsed_s': round(elapsed, 3),
        'tflops': round(tflops, 2),
        'checksum_finite': bool(torch.isfinite(c[0, 0]).item()),
    }


def run_matmul_on_gpu(base: Path, gpu_idx: int, seconds: float) -> dict:
    with tempfile.NamedTemporaryFile(dir=base / 'tmp', suffix='.json', delete=False) as tf:
        out_path = Path(tf.name)
    try:
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()),
               '--out-json', str(out_path), '--worker-matmul', str(seconds)]
        proc = subprocess.run(
            cmd, env=sanitized_env(base, str(gpu_idx)),
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(f'GPU {gpu_idx} 矩阵乘测试失败 rc={proc.returncode}: {proc.stderr[-500:]}')
        return json.loads(out_path.read_text(encoding='utf-8'))
    finally:
        out_path.unlink(missing_ok=True)


def decide_layout(core_ok: bool, disk_ok: bool, gpu6_apps: list, gpu0_tflops: float | None, gpu6_tflops: float | None) -> tuple[int, str, dict]:
    if not core_ok:
        return 2, 'PRECHECK_STOP_CORE_GPU_BUSY', {}
    if not disk_ok:
        return 3, 'PRECHECK_STOP_DISK_LOW', {}
    if gpu6_apps:
        return 0, 'PRECHECK_PASS_7GPU', {'reason': f'GPU 6 有计算进程: {gpu6_apps}'}
    if not gpu0_tflops or not gpu6_tflops:
        return 0, 'PRECHECK_PASS_7GPU', {'reason': '未能取得 BF16 矩阵乘对比'}
    ratio = gpu6_tflops / gpu0_tflops
    if ratio >= GPU6_MIN_RATIO:
        return 0, 'PRECHECK_PASS_8GPU', {'gpu6_to_gpu0_ratio': round(ratio, 4)}
    return 0, 'PRECHECK_PASS_7GPU', {'gpu6_to_gpu0_ratio': round(ratio, 4), 'reason': 'GPU 6 速度低于 GPU 0 的 90%'}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=default_base())
    parser.add_argument('--seconds', type=float, default=MATMUL_SECONDS)
    parser.add_argument('--out-json', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-matmul', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_matmul is not None:
        res = worker_matmul(args.worker_matmul, MATMUL_N)
        args.out_json.write_text(json.dumps(res), encoding='utf-8')
        return 0

    base = args.base.resolve()
    (base / 'tmp').mkdir(parents=True, exist_ok=True)
    (base / 'runs').mkdir(parents=True, exist_ok=True)

    df_out = subprocess.check_output(['df', '-h', '/data'], text=True, timeout=15).strip()
    free_bytes = shutil.disk_usage('/data').free
    disk_ok = free_bytes >= DISK_FLOOR_BYTES
    print('=== 1. /data 磁盘检查 ===')
    print(df_out)
    print(f'/data free_GB={free_bytes / 1e9:.2f} free_GiB={free_bytes / (1024**3):.2f} >=45G={disk_ok}')

    first = snapshot_gpus()
    time.sleep(2)
    second = snapshot_gpus()
    core_ok, core_issues = core_gpus_clean(first, second)
    print('=== 2. GPU 0-7 进程与状态快照 ===')
    for idx in sorted(second):
        g = second[idx]
        print(f"GPU {idx}: used={g['used_mib']} MiB, util={g['util']}%, sm_clk={g['sm_clock_mhz']} MHz, "
              f"temp={g['temp_c']}C, power={g['power_w']}W, apps={g['apps']}")

    if not core_ok:
        print('STOP_ISSUES:', core_issues)
        print('PRECHECK_STOP_CORE_GPU_BUSY')
        return 2
    if not disk_ok:
        print('PRECHECK_STOP_DISK_LOW')
        return 3

    print('=== 3. GPU 6 详细查询 (nvidia-smi -i 6 -q) ===')
    gpu6_q = query_gpu6_detail()
    for line in gpu6_q['lines']:
        print(' ', line)

    gpu6_apps = second[6]['apps']
    gpu0_bench = gpu6_bench = None
    if gpu6_apps:
        print(f'GPU 6 发现计算进程 {gpu6_apps}，不触发 GPU 6 矩阵乘对比，直接选用 7 卡。')
    else:
        print(f'=== 4. GPU 0 与 GPU 6 各 {args.seconds:.0f}s BF16 矩阵乘对比 ({MATMUL_N}x{MATMUL_N}) ===', flush=True)
        gpu0_bench = run_matmul_on_gpu(base, 0, args.seconds)
        print(f"  GPU 0: {gpu0_bench['tflops']} TFLOPS ({gpu0_bench['iters']} iters in {gpu0_bench['elapsed_s']}s)")
        gpu6_bench = run_matmul_on_gpu(base, 6, args.seconds)
        print(f"  GPU 6: {gpu6_bench['tflops']} TFLOPS ({gpu6_bench['iters']} iters in {gpu6_bench['elapsed_s']}s)")

    rc, verdict, extra = decide_layout(
        core_ok, disk_ok, gpu6_apps,
        gpu0_bench['tflops'] if gpu0_bench else None,
        gpu6_bench['tflops'] if gpu6_bench else None,
    )
    summary = {
        'df_data': df_out.splitlines(),
        'data_free_bytes': free_bytes,
        'disk_ok': disk_ok,
        'core_gpus_ok': core_ok,
        'core_issues': core_issues,
        'gpus': second,
        'gpu6_detail': gpu6_q,
        'matmul_gpu0': gpu0_bench,
        'matmul_gpu6': gpu6_bench,
        'verdict': verdict,
        'extra': extra,
    }
    out_file = base / 'runs' / 'stage2_precheck.json'
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'结论: {verdict} {extra}')
    print(f'已保存: {out_file}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
