"""Single-GPU Qwen3.5-4B fast/slow training-kernel comparison for RTX PRO 5000."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

MIB = 1024 ** 2
GIB = 1024 ** 3
PROTOCOL = 'qwen35_pro5000_one_card_v1'
HARD_SECONDS = 7200
SOAK_SECONDS = {'slow': 40 * 60, 'fast': 20 * 60}
LIMIT = 10240
SEED = 42
KEYS = ('logprob', 'loss', 'grad_norm')
MODE_PACKAGES = {
    'transformers': '5.7.0',
    'flash-linear-attention': '0.5.2',
    'fla-core': '0.5.2',
    'causal-conv1d': '1.6.2.post1',
}


class CompareFailure(RuntimeError):
    """A diagnostic label safe for the terminal."""


@dataclass(frozen=True)
class Sequence:
    ids: tuple[int, ...]
    mask: tuple[bool, ...]
    label: str


def default_base() -> Path:
    return Path.home() / 'bzhang' / 'a_group_pro5000'


def snapshot_gpus():
    gpu_out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'],
        text=True, timeout=15,
    )
    app_out = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
         '--format=csv,noheader,nounits'],
        text=True, timeout=15,
    )
    apps_by_uuid = {}
    for line in app_out.splitlines():
        if not line.strip():
            continue
        uuid, pid, used = [p.strip() for p in line.split(',')]
        apps_by_uuid.setdefault(uuid, []).append({'pid': int(pid), 'used_mib': int(used)})
    gpus = {}
    for line in gpu_out.splitlines():
        if not line.strip():
            continue
        idx_s, uuid, name, used_s, free_s, total_s, util_s = [p.strip() for p in line.split(',')]
        idx = int(idx_s)
        gpus[idx] = {
            'index': idx, 'uuid': uuid, 'name': name,
            'used_mib': int(used_s), 'free_mib': int(free_s), 'total_mib': int(total_s),
            'util': int(util_s), 'apps': apps_by_uuid.get(uuid, []),
        }
    return gpus


def pick_idle_gpu(requested=None):
    first = snapshot_gpus()
    time.sleep(2)
    second = snapshot_gpus()
    idle = []
    for idx in sorted(second):
        a, b = first.get(idx), second[idx]
        if (a and not a['apps'] and not b['apps']
                and a['used_mib'] < 500 and b['used_mib'] < 500
                and a['util'] < 10 and b['util'] < 10
                and b['total_mib'] >= 70000):
            idle.append(b)
    if requested is not None:
        match = next((g for g in idle if g['index'] == requested), None)
        if match is None:
            raise CompareFailure(f'指定 GPU {requested} 当前非空闲整卡')
        return match
    if not idle:
        raise CompareFailure('未找到空闲的 RTX PRO 5000 整卡')
    return idle[0]


def parse_gpu_uuid(raw: str) -> str:
    text = raw.strip().lower()
    if text.startswith('gpu-'):
        text = text[4:]
    return text.replace('-', '')


def _build_synthetic_messages(base_prompt, turns: int, variant: int, scale: int):
    messages = list(base_prompt)
    think_unit = (
        'Let us analyze the structure of the problem carefully. '
        'First, identify the given quantities, constraints, and what needs to be shown. '
        'Second, test small cases or algebraic transformations without revealing the final answer directly. '
        'Third, check whether an intermediate hint helps the student discover the next valid step on their own. '
    )
    reply_body = (
        'Good start. Look closely at the relationship between the main terms and try rewriting them '
        'in an equivalent form before substituting values. What pattern do you notice?'
    )
    student_unit = (
        'I tried expanding the expression and grouping similar terms together. '
        'After simplifying one side, I can see a common factor, though I am not yet sure '
        'how to handle the remaining boundary condition. '
    )
    for t in range(turns):
        thought = think_unit * (scale + ((t + variant) % 2))
        messages.append({
            'role': 'assistant',
            'content': f'<think>{thought.strip()}</think>\n{reply_body}',
        })
        if t < turns - 1:
            messages.append({
                'role': 'user',
                'content': (student_unit * max(1, scale - 1)).strip(),
            })
    return messages


def checked_ids(ids):
    if isinstance(ids, Mapping):
        ids = ids.get('input_ids')
    if isinstance(ids, (tuple, list)) and ids and isinstance(ids[0], (tuple, list)):
        ids = ids[0]
    if not isinstance(ids, (tuple, list)) or any(type(x) is not int or x < 0 for x in ids):
        raise CompareFailure('对话重分词格式异常')
    return list(ids)


def reconstruct(messages, tokenizer, label: str):
    if (not isinstance(messages, list) or len(messages) < 3
            or any(not isinstance(row, dict) or not isinstance(row.get('content'), str) for row in messages)
            or [row.get('role') for row in messages[:2]] != ['system', 'user']):
        return None
    ids = checked_ids(tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=True,
    ))
    if not ids or len(ids) > 2048:
        return None
    mask = [False] * len(ids)
    separator = checked_ids(tokenizer.encode('<|im_end|>\n', add_special_tokens=False))
    turns = 0
    for index, row in enumerate(messages[2:], start=2):
        role = 'assistant' if index % 2 == 0 else 'user'
        if row.get('role') != role:
            return None
        if role == 'assistant':
            current = checked_ids(tokenizer.encode(row['content'], add_special_tokens=False))
            ids.extend(current)
            mask.extend([True] * len(current))
            turns += bool(current)
            ids.extend(separator)
            mask.extend([False] * len(separator))
        else:
            current = checked_ids(tokenizer.apply_chat_template(
                [row], tokenize=True, add_generation_prompt=True, enable_thinking=True,
            ))
            ids.extend(current)
            mask.extend([False] * len(current))
    if turns == 0 or len(ids) > LIMIT:
        return None
    return Sequence(tuple(ids), tuple(mask), label)


def build_mixed_batch(train_parquet: Path, tokenizer):
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(train_parquet, columns=['prompt'])
    prompts = table['prompt'].to_pylist()
    if len(prompts) < 3:
        raise CompareFailure('公开训练集行数不足 3 条')
    chosen = np.random.default_rng(SEED).choice(len(prompts), size=3, replace=False)

    specs = [('short', 2, 5), ('medium', 5, 7), ('long', 9, 9)]
    sequences = []
    for variant, ((label, turns, scale), idx) in enumerate(zip(specs, chosen)):
        seq = None
        for sc in range(scale, 0, -1):
            for t in range(turns, 0, -1):
                msgs = _build_synthetic_messages(prompts[int(idx)], t, variant, sc)
                cand = reconstruct(msgs, tokenizer, label)
                if cand is not None:
                    seq = cand
                    break
            if seq is not None:
                break
        if seq is None:
            raise CompareFailure(f'无法构造 {label} 对话测试样本')
        sequences.append(seq)
    if len(sequences[-1].ids) - len(sequences[0].ids) < 512:
        raise CompareFailure('长短混合批次长度差不足 512 token')
    return sequences


class Kernels:
    """Switch linear-attention fast/slow path per layer sharing identical weights."""
    def __init__(self, model, native):
        self.layers = [layer for layer in model.modules() if isinstance(layer, native.Qwen3_5GatedDeltaNet)]
        if not self.layers or not native.is_fast_path_available or native.FusedRMSNormGated is None:
            raise CompareFailure('原生 Qwen3.5 快路径不可用')
        self.counts = Counter()
        self.hooks = []
        self.orig = []
        for layer in self.layers:
            fast_conv, fast_chunk, fast_norm = layer.causal_conv1d_fn, layer.chunk_gated_delta_rule, layer.norm
            if (not callable(fast_conv) or not callable(fast_chunk)
                    or not isinstance(fast_norm, native.FusedRMSNormGated)
                    or getattr(fast_norm, 'bias', None) is not None
                    or not fast_conv.__module__.startswith('causal_conv1d')
                    or not fast_chunk.__module__.startswith('fla.')):
                raise CompareFailure('线性注意力未绑定预期第三方内核')
            slow_norm = native.Qwen3_5RMSNormGated(layer.head_v_dim, eps=layer.layer_norm_epsilon)
            slow_norm.weight = fast_norm.weight
            self.orig.append((layer, fast_conv, fast_chunk, fast_norm, slow_norm))
            self.hooks.append(layer.conv1d.register_forward_hook(self.hook('slow_conv')))
            self.hooks.append(fast_norm.register_forward_hook(self.hook('fast_norm')))
            self.hooks.append(slow_norm.register_forward_hook(self.hook('slow_norm')))
        self.slow_chunk = native.torch_chunk_gated_delta_rule

    def hook(self, name):
        def count(_module, _inputs, _result):
            self.counts[name] += 1
        return count

    def wrap(self, fn, name):
        def measured(*args, **kwargs):
            if name == 'fast_conv':
                kwargs['weight'] = kwargs['weight'].to(kwargs['x'].dtype)
            output = fn(*args, **kwargs)
            self.counts[name] += 1
            return output
        return measured

    def use(self, mode):
        for layer, fast_conv, fast_chunk, fast_norm, slow_norm in self.orig:
            layer.causal_conv1d_fn = self.wrap(fast_conv, 'fast_conv') if mode == 'fast' else None
            layer.chunk_gated_delta_rule = (self.wrap(fast_chunk, 'fast_chunk') if mode == 'fast'
                                            else self.wrap(self.slow_chunk, 'slow_chunk'))
            layer.norm = fast_norm if mode == 'fast' else slow_norm

    def verified(self, mode, trajectories):
        chosen = {f'{mode}_{suffix}' for suffix in ('conv', 'chunk', 'norm')}
        return set(self.counts) == chosen and all(
            self.counts[name] >= len(self.layers) * trajectories for name in chosen
        )


def one_batch(torch, model, kernels, batch, mode, check, *, keep_values):
    from torch.utils.checkpoint import checkpoint

    def log_probs(hidden, targets):
        with torch.autocast(device_type='cuda', enabled=False):
            logits = model.lm_head(hidden.float())
            return -torch.nn.functional.cross_entropy(logits, targets, reduction='none')

    kernels.use(mode)
    kernels.counts.clear()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    combined, samples = [], []
    tokens = sum(len(positions) for _, positions in batch)
    seconds = 0.0
    for ids, positions in batch:
        check()
        started = time.monotonic()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model.model.language_model(input_ids=ids, use_cache=False, return_dict=True)
        targets = ids[0, positions]
        effective = output.last_hidden_state[0, positions - 1]
        chunks = [checkpoint(log_probs, x, y, use_reentrant=False)
                  for x, y in zip(effective.split(256), targets.split(256))]
        values = torch.cat(chunks)
        loss = -values.sum() / tokens
        loss.backward()
        combined.append(loss.detach())
        if keep_values:
            samples.extend(values.detach().cpu().tolist())
        elif not bool(torch.isfinite(values.detach()).all().item()):
            raise CompareFailure('耐久阶段有效 token logprob 非有限')
        torch.cuda.synchronize()
        seconds += time.monotonic() - started
        del output, targets, effective, chunks, values, loss
    started = time.monotonic()
    grads = [torch.linalg.vector_norm(p.grad.detach().float()) for p in model.parameters() if p.grad is not None]
    if not grads:
        raise CompareFailure('模型未产生梯度')
    grad_norm = float(torch.linalg.vector_norm(torch.stack(grads)).item())
    loss = float(torch.stack(combined).sum().item())
    torch.cuda.synchronize()
    seconds += time.monotonic() - started
    if not (math.isfinite(grad_norm) and grad_norm > 0 and math.isfinite(loss) and loss > 0):
        raise CompareFailure('实际损失或梯度不是有限正数')
    if not kernels.verified(mode, len(batch)):
        raise CompareFailure('实际慢或快路径函数调用与目标不符')
    row = {
        'mode': mode, 'values': samples, 'loss': loss, 'grad_norm': grad_norm,
        'seconds_per_token': seconds / tokens,
        'kernel_call_counts': dict(kernels.counts),
        'peak_alloc_mib': torch.cuda.max_memory_allocated() / MIB,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / MIB,
    }
    model.zero_grad(set_to_none=True)
    return row


def comparisons(rows):
    base, repeat, fast = rows
    if not base['values'] or len({len(item['values']) for item in rows}) != 1:
        raise CompareFailure('有效 token 的对齐长度不一致')
    result = {}
    for key in KEYS:
        if key == 'logprob':
            values = [r['values'] for r in rows]
            distances = [max(abs(a - b) for a, b in zip(values[0], target)) for target in values[1:]]
        else:
            distances = [abs(base[key] - target[key]) for target in (repeat, fast)]
        if any(not math.isfinite(value) or value < 0 for value in distances):
            raise CompareFailure('内核对照出现非有限值')
        result[key] = {'slow_repeat_abs_diff': distances[0], 'fast_slow_abs_diff': distances[1]}
    return result


def evaluate_equivalence(rows, distances, fast_kernel_verified):
    """Apply original within_repeat first; if failed, apply Section 14 tolerance."""
    within = all(v['fast_slow_abs_diff'] <= v['slow_repeat_abs_diff'] for v in distances.values())
    base, repeat, fast = rows
    slow_mean = math.fsum(base['values']) / len(base['values'])
    fast_mean = math.fsum(fast['values']) / len(fast['values'])
    rel = {
        'loss_rel_diff': abs(fast['loss'] - base['loss']) / max(abs(base['loss']), 1e-12),
        'logprob_mean_rel_diff': abs(fast_mean - slow_mean) / max(abs(slow_mean), 1e-12),
        'grad_norm_rel_diff': abs(fast['grad_norm'] - base['grad_norm']) / max(abs(base['grad_norm']), 1e-12),
        'logprob_token_max_abs_diff': distances['logprob']['fast_slow_abs_diff'],
    }
    fast_counts = fast.get('kernel_call_counts', {})
    slow_counts_ok = all(
        r.get('kernel_call_counts', {}).get('fast_conv', 0) == 0
        and r.get('kernel_call_counts', {}).get('fast_chunk', 0) == 0
        for r in (base, repeat)
    )
    fast_counts_ok = (fast_counts.get('fast_conv', 0) > 0
                      and fast_counts.get('fast_chunk', 0) > 0
                      and fast_counts.get('fast_norm', 0) > 0)
    all_finite = all(
        all(math.isfinite(x) for x in r['values'])
        and math.isfinite(r['loss']) and math.isfinite(r['grad_norm'])
        for r in rows
    )
    sec14 = (
        bool(fast_kernel_verified) and slow_counts_ok and fast_counts_ok and all_finite
        and rel['loss_rel_diff'] <= 1e-2
        and rel['logprob_mean_rel_diff'] <= 1e-2
        and rel['grad_norm_rel_diff'] <= 5e-2
        and rel['logprob_token_max_abs_diff'] <= 1.0
    )
    criterion = 'within_repeat' if within else ('section14_tolerance' if sec14 else 'failed')
    return (within or sec14), criterion, rel


def round_trend(cycles):
    count = len(cycles)
    if not count:
        raise CompareFailure('没有耐久轮次')
    third = max(1, count // 3)
    first, last = cycles[:third], cycles[-third:]
    keys = ('round_seconds', 'ms_per_token', 'loadavg_1')
    averages = lambda group: {k: math.fsum(r[k] for r in group) / third for k in keys}
    a, b = averages(first), averages(last)
    return {
        'rounds_per_third': third, 'first': a, 'last': b,
        'last_to_first_wall': b['round_seconds'] / a['round_seconds'],
        'last_to_first_ms_per_token': b['ms_per_token'] / a['ms_per_token'],
    }


def endurance(torch, model, kernels, batch, mode, check, report_dir, soak_seconds):
    torch.cuda.empty_cache()
    started = time.monotonic()
    cycles = []
    peak_alloc = peak_reserved = 0.0
    with (report_dir / f'soak-{mode}-cycles.jsonl').open('x', encoding='utf-8') as stream:
        while time.monotonic() - started < soak_seconds[mode]:
            round_start = time.monotonic()
            current = one_batch(torch, model, kernels, batch, mode, check, keep_values=False)
            sample = {
                'round': len(cycles) + 1,
                'round_seconds': time.monotonic() - round_start,
                'ms_per_token': current['seconds_per_token'] * 1000,
                'loadavg_1': os.getloadavg()[0],
            }
            stream.write(json.dumps(sample, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            cycles.append(sample)
            peak_alloc = max(peak_alloc, current['peak_alloc_mib'])
            peak_reserved = max(peak_reserved, current['peak_reserved_mib'])
            print(f'[soak {mode}] round={len(cycles)} elapsed={time.monotonic()-started:.0f}s '
                  f'ms_per_tok={sample["ms_per_token"]:.3f} load1={sample["loadavg_1"]:.2f}', flush=True)
    return {
        'cycles': len(cycles), 'elapsed_seconds': time.monotonic() - started,
        'peak_alloc_mib': peak_alloc, 'peak_reserved_mib': peak_reserved,
        'seconds_per_token': math.fsum(s['ms_per_token'] for s in cycles) / len(cycles) / 1000,
        'round_trend': round_trend(cycles),
    }


def score_result(result, *, soak_seconds=SOAK_SECONDS, max_mem_mib=80000):
    if not isinstance(result, dict) or result.get('fast_kernel_verified') is not True:
        raise CompareFailure('实际内核调用未核实')
    rows = result['runs']
    if len(rows) != 3 or [r.get('mode') for r in rows] != ['slow', 'slow', 'fast']:
        raise CompareFailure('内核试验顺序或测量次数异常')
    expected_tokens = sum(result['teacher_tokens'])
    if len(result['sequence_lengths']) != 3 or expected_tokens < 1:
        raise CompareFailure('长短混合批次测量不完整')
    for r in rows:
        if (len(r.get('values', ())) != expected_tokens
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in r['values'])
                or any(type(r.get(k)) not in (int, float) or not math.isfinite(r[k]) or r[k] <= 0
                       for k in ('loss', 'grad_norm', 'seconds_per_token'))):
            raise CompareFailure('有效 token、损失、梯度或计时数值异常')
        for alias in ('peak_alloc_mib', 'peak_reserved_mib'):
            if type(r.get(alias)) not in (int, float) or not 0 < r[alias] < max_mem_mib:
                raise CompareFailure('显存峰值数值异常')
    distances = comparisons(rows)
    equivalent, criterion, rel = evaluate_equivalence(rows, distances, result['fast_kernel_verified'])
    soak = result['endurance']
    stopped_early = result.get('endurance_skipped') is True
    if stopped_early:
        if soak or equivalent:
            raise CompareFailure('耐久提前停止与数值对照不一致')
    elif set(soak) != set(soak_seconds) or any(
        type(soak[m].get('elapsed_seconds')) not in (int, float)
        or not math.isfinite(soak[m]['elapsed_seconds'])
        or soak[m]['elapsed_seconds'] < dur
        or type(soak[m].get('cycles')) is not int or soak[m]['cycles'] < 1
        or type(soak[m].get('seconds_per_token')) not in (int, float)
        or not math.isfinite(soak[m]['seconds_per_token'])
        or soak[m]['seconds_per_token'] <= 0
        or type(soak[m].get('peak_alloc_mib')) not in (int, float)
        or not 0 < soak[m]['peak_alloc_mib'] < max_mem_mib
        or type(soak[m].get('peak_reserved_mib')) not in (int, float)
        or not 0 < soak[m]['peak_reserved_mib'] < max_mem_mib
        for m, dur in soak_seconds.items()
    ):
        raise CompareFailure('耐久对照未完成')
    for r in rows:
        vals = r.pop('values')
        r['logprob_mean'] = math.fsum(vals) / len(vals)
    result['comparisons'] = distances
    result['relative_diffs'] = rel
    result['criterion_used'] = criterion
    return equivalent


def display(result, passed):
    print('对照结果', '通过' if passed else '不等价', '判据:', result.get('criterion_used'))
    print('实测有效长度', *result['sequence_lengths'], '目标教师 token', *result['teacher_tokens'])
    print('| 指标 | 慢路径重复绝对差 | 快慢绝对差 |')
    print('|---|---:|---:|')
    for key, row in result['comparisons'].items():
        print(f"| {key} | {row['slow_repeat_abs_diff']:.9g} | {row['fast_slow_abs_diff']:.9g} |")
    rel = result.get('relative_diffs', {})
    print(f"补充判据指标: loss_rel={rel.get('loss_rel_diff', 0):.6g}  "
          f"logprob_mean_rel={rel.get('logprob_mean_rel_diff', 0):.6g}  "
          f"grad_norm_rel={rel.get('grad_norm_rel_diff', 0):.6g}  "
          f"logprob_token_max_abs={rel.get('logprob_token_max_abs_diff', 0):.6g}")
    print('| 阶段 | 平均 logprob | NLL | 梯度范数 | 秒/有效 token | 显存峰值 MiB（alloc / reserved） |')
    print('|---|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(result['runs'], 1):
        print(f"| {row['mode']} {idx} | {row['logprob_mean']:.8g} | {row['loss']:.8g} | "
              f"{row['grad_norm']:.8g} | {row['seconds_per_token']:.6g} | "
              f"{row['peak_alloc_mib']:.0f} / {row['peak_reserved_mib']:.0f} |")
        print(f"  kernel_calls({row['mode']} {idx}): {row.get('kernel_call_counts', {})}")
    for mode, row in result['endurance'].items():
        print(f"| {mode} 耐久 {row['elapsed_seconds']:.0f}s / {row['cycles']} 轮 | — | — | — | "
              f"{row['seconds_per_token']:.6g} | {row['peak_alloc_mib']:.0f} / "
              f"{row['peak_reserved_mib']:.0f} |")
    if result.get('endurance_skipped'):
        print('两条数值判据均未通过，耐久未开始')
    if result['endurance']:
        print('| 耐久 | 每端轮数 | 首段/末段整轮秒数 | 末/首用时 | 首段/末段 ms/token | 末/首 ms/token | 首段/末段负载(1m) |')
        print('|---|---:|---:|---:|---:|---:|---:|')
        for mode, row in result['endurance'].items():
            tr = row['round_trend']
            a, b = tr['first'], tr['last']
            print(f"| {mode} | {tr['rounds_per_third']} | {a['round_seconds']:.2f} / {b['round_seconds']:.2f} | "
                  f"{tr['last_to_first_wall']:.3f} | {a['ms_per_token']:.3f} / {b['ms_per_token']:.3f} | "
                  f"{tr['last_to_first_ms_per_token']:.3f} | {a['loadavg_1']:.2f} / {b['loadavg_1']:.2f} |")


def run_worker(model_dir: Path, train_parquet: Path, index: int, uuid: str,
               deadline: float, report_dir: Path, soak_seconds: dict):
    from importlib import metadata
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
        import torch
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    for name, expected in MODE_PACKAGES.items():
        if metadata.version(name).split('+', 1)[0] != expected:
            raise CompareFailure(f'快路径依赖版本不符: {name}')
    if torch.__version__.split('+', 1)[0] != '2.10.0' or torch.version.cuda != '12.8' or not torch.compiled_with_cxx11_abi():
        raise CompareFailure('Torch/CUDA/ABI 版本不符')
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise CompareFailure('要求可见一张支持 BF16 的卡')
    props = torch.cuda.get_device_properties(0)
    if 'RTX PRO 5000' not in props.name.upper() or props.total_memory < 70000 * MIB:
        raise CompareFailure(f'运行单卡不符: {props.name}')
    if parse_gpu_uuid(str(props.uuid)) != parse_gpu_uuid(uuid):
        raise CompareFailure('单卡 UUID 不符')

    external_apps_seen = 0
    last_snap = [0.0]

    def check(*, inspect_gpu=True):
        nonlocal external_apps_seen
        if time.monotonic() >= deadline:
            raise TimeoutError()
        if inspect_gpu and time.monotonic() - last_snap[0] >= 15:
            last_snap[0] = time.monotonic()
            row = snapshot_gpus().get(index)
            if row and len(row['apps']) > 1:
                external_apps_seen += 1

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    selected = build_mixed_batch(train_parquet, tokenizer)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = native.Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='sdpa',
    ).to('cuda')
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.zero_grad(set_to_none=True)

    batch = []
    for row in selected:
        ids = torch.tensor([row.ids], device='cuda', dtype=torch.long)
        positions = torch.tensor([i for i, m in enumerate(row.mask) if m], device='cuda', dtype=torch.long)
        if int(positions[0]) < 1:
            raise CompareFailure('首个有效 token 没有可用上下文')
        batch.append((ids, positions))

    kernels = Kernels(model, native)
    runs = []
    for mode, trials in (('slow', 2), ('fast', 1)):
        print(f'[compare] warmup mode={mode}', flush=True)
        one_batch(torch, model, kernels, batch, mode, check, keep_values=False)
        for t in range(trials):
            torch.cuda.empty_cache()
            row = one_batch(torch, model, kernels, batch, mode, check, keep_values=True)
            print(f'[compare] {mode} trial={t+1}: loss={row["loss"]:.6f} grad={row["grad_norm"]:.6f} '
                  f's/tok={row["seconds_per_token"]:.6f} counts={row["kernel_call_counts"]}', flush=True)
            runs.append(row)

    distances = comparisons(runs)
    equivalent, _, _ = evaluate_equivalence(runs, distances, True)
    stability = {}
    if equivalent:
        for mode in ('slow', 'fast'):
            print(f'[soak] 开始 {mode} 耐久 ({soak_seconds[mode]}s)...', flush=True)
            stability[mode] = endurance(torch, model, kernels, batch, mode, check, report_dir, soak_seconds)
    return {
        'fast_kernel_verified': True,
        'gpu_name': props.name,
        'gpu_total_mib': props.total_memory / MIB,
        'external_apps_seen': external_apps_seen,
        'sequence_lengths': [len(r.ids) for r in selected],
        'teacher_tokens': [sum(r.mask) for r in selected],
        'runs': runs,
        'endurance': stability,
        'endurance_skipped': not equivalent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=default_base())
    parser.add_argument('--model-dir', type=Path, default=Path('/data/models/Qwen3.5-4B'))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--slow-seconds', type=int, default=SOAK_SECONDS['slow'])
    parser.add_argument('--fast-seconds', type=int, default=SOAK_SECONDS['fast'])
    parser.add_argument('--worker', nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    soak_seconds = {'slow': args.slow_seconds, 'fast': args.fast_seconds}

    if args.worker:
        idx, uuid, deadline_s, report_dir_s, model_dir_s, train_s = args.worker
        out_json = Path(report_dir_s) / 'raw_worker_result.json'
        res = run_worker(Path(model_dir_s), Path(train_s), int(idx), uuid,
                         float(deadline_s), Path(report_dir_s), soak_seconds)
        out_json.write_text(json.dumps(res, allow_nan=False))
        return 0

    base = args.base.resolve()
    env_prefix = base / 'envs' / 'pro5000-fast'
    receipt = env_prefix / 'kernel_equivalence.json'
    if receipt.exists():
        print('等价回执已存在，跳过重复运行:', receipt)
        return 0
    train_parquet = base / 'data' / 'author_prepared' / 'train.parquet'
    if not train_parquet.is_file() or not (args.model_dir / 'config.json').is_file():
        print('缺少 train.parquet 或 Qwen3.5-4B 目录')
        return 1

    gpu = pick_idle_gpu(args.gpu)
    print(f'单卡对照使用 GPU {gpu["index"]} ({gpu["name"]}, {gpu["total_mib"]} MiB)', flush=True)
    reports = base / 'runs' / 'kernel-compare-pro5000'
    reports.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_dir = Path(tempfile.mkdtemp(dir=reports, prefix='run-'))
    print('本轮数字记录目录', report_dir, flush=True)

    deadline = time.monotonic() + HARD_SECONDS
    env = {k: v for k, v in os.environ.items()
           if not k.upper().endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))}
    env_lib = Path(sys.prefix) / 'lib'
    ld_parts = [str(env_lib)] if env_lib.is_dir() else []
    if env.get('LD_LIBRARY_PATH'):
        ld_parts.append(env['LD_LIBRARY_PATH'])
    env.update(
        CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(base / 'tmp'), TRITON_CACHE_DIR=str(base / 'cache' / 'triton'),
        XDG_CACHE_HOME=str(base / 'cache' / 'xdg'),
    )
    if ld_parts:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(ld_parts)
    cmd = [
        sys.executable, '-B', str(Path(__file__).resolve()),
        '--slow-seconds', str(args.slow_seconds), '--fast-seconds', str(args.fast_seconds),
        '--worker', str(gpu['index']), gpu['uuid'], str(deadline),
        str(report_dir), str(args.model_dir), str(train_parquet),
    ]
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=HARD_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        print('单卡对照超时（2小时硬上限）')
        return 124
    if rc != 0:
        print('单卡对照子进程异常退出 rc=', rc)
        return rc

    result = json.loads((report_dir / 'raw_worker_result.json').read_text())
    passed = score_result(result, soak_seconds=soak_seconds, max_mem_mib=gpu['total_mib'])
    result.update(protocol=PROTOCOL, environment=str(env_prefix), gpu_index=gpu['index'], accepted=passed)
    summary_path = report_dir / 'comparison-summary.json'
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    display(result, passed)
    if passed:
        receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print('等价回执已落盘:', receipt)
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
