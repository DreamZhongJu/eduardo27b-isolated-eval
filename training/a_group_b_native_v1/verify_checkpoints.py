"""Verify the existing two-step B pilot with veRL's FSDP merger on CPU."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def config(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def inspect_input(step_dir):
    actor = step_dir / 'actor'
    if config(actor / 'fsdp_config.json').get('world_size') != 2:
        raise ValueError('world size')
    for rank in range(2):
        if not nonempty(actor / f'model_world_size_2_rank_{rank}.pt'):
            raise ValueError('model rank')
    state_files = [actor / f'{kind}_world_size_2_rank_{rank}.pt'
                   for kind in ('optim', 'extra_state') for rank in range(2)]
    state_files.append(step_dir / 'data.pt')
    return config(actor / 'huggingface/config.json'), sum(nonempty(path) for path in state_files)


def inspect_export(folder, source_config):
    from safetensors import safe_open

    saved = config(folder / 'config.json')
    for key in ('model_type', 'architectures'):
        if not saved.get(key) or saved[key] != source_config.get(key):
            raise ValueError('HF config')
    shards = sorted(folder.glob('*.safetensors'))
    index = folder / 'model.safetensors.index.json'
    expected = config(index)['weight_map'] if index.is_file() else None
    if not shards or (expected is None and len(shards) != 1):
        raise ValueError('shard list')
    actual = {}
    for shard in shards:
        sampled = False
        with safe_open(str(shard), framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in actual:
                    raise ValueError('duplicate key')
                actual[key] = shard.name
                if sampled:
                    continue
                view = handle.get_slice(key)
                shape = view.get_shape()
                if 0 in shape:
                    continue
                cell = view[tuple(slice(0, 1) for _ in shape)] if shape else handle.get_tensor(key)
                if not math.isfinite(float(cell.item())):
                    raise ValueError('nonfinite sample')
                sampled = True
        if not sampled:
            raise ValueError('unreadable sample')
    if expected is not None and actual != expected:
        raise ValueError('weight map')
    return len(shards), len(actual)


def verify(work, output):
    work = work.resolve(strict=True)
    if output.is_symlink() or output.resolve().is_relative_to(work):
        raise ValueError('output location')
    output = output.resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               HF_HOME=str(output / 'cache'), XDG_CACHE_HOME=str(output / 'cache'),
               PYTHONPATH=str(work.parent.parent / 'source/EduardoRL'))
    with (output / '验收.md').open('x', encoding='utf-8') as report:
        def row(text):
            print(text, flush=True)
            print(text, file=report, flush=True)

        row('| 步 | 项目 | 结果 |')
        row('|---:|---|---|')
        for step in (1, 2):
            step_dir = work / 'verl_ckpts/b-functional' / f'global_step_{step}'
            target = output / f'step_{step}'
            stage = '输入分片和配置'
            try:
                source, count = inspect_input(step_dir)
                row(f'| {step} | FSDP 模型分片 | 2/2 |')
                row(f'| {step} | 优化器、额外状态与数据加载器文件非空 | {count}/5 |')
                stage = 'CPU 合并'
                with (output / f'step_{step}.log').open('xb') as log:
                    proc = subprocess.run([sys.executable, '-B', '-m', 'verl.model_merger',
                                           'merge', '--backend', 'fsdp', '--local_dir',
                                           str(step_dir / 'actor'), '--target_dir', str(target)],
                                          cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
                if proc.returncode:
                    row(f'| {step} | {stage} | 退出码 {proc.returncode} |')
                    return 1
                stage = 'HF 配置、索引与 CPU 张量读取'
                shards, keys = inspect_export(target, source)
                row(f'| {step} | {stage} | 通过，{shards} 个分片，{keys} 个张量键，每片抽读一个数值 |')
                if count != 5:
                    row(f'| {step} | 训练状态文件 | 存在缺项，已停止 |')
                    return 1
            except Exception as exc:
                row(f'| {step} | {stage} | 失败（{type(exc).__name__}） |')
                return 1
        row('| 合计 | 两个检查点的模型权重 CPU 合并读取 | 2/2 通过 |')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        sys.exit(verify(args.work, args.output))
    except (OSError, ValueError) as exc:
        print('检查未启动或无法写入验收目录（' + type(exc).__name__ + '）', file=sys.stderr)
        sys.exit(1)
