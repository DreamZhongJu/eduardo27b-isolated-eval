"""Inspect real Transformers kernel bindings without allocating an A100."""

import argparse
from contextlib import nullcontext
from importlib import import_module, metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch


MODEL = 'transformers.models.qwen3_5.modeling_qwen3_5'
KERNEL_WHEELS = {
    'flash-linear-attention': '0.5.2',
    'fla-core': '0.5.2',
    'causal-conv1d': '1.6.2.post1',
}
NATURAL = '无卡原样'
SIMULATED = '无卡模拟'
DEFAULT_PREFIX = Path('/workspace/A_group_repro_20261117/envs/env-a100-fast-v2')


class PrecheckError(ValueError):
    """A fixed diagnostic safe to show."""


def check_prefix(prefix):
    try:
        resolved = prefix.resolve(strict=True)
    except OSError:
        raise PrecheckError('独立候选环境目录不可读取') from None
    if Path(sys.prefix).resolve() != resolved:
        raise PrecheckError('解释器不属于独立候选环境')
    if sys.version_info[:2] != (3, 12) or platform.machine() != 'x86_64':
        raise PrecheckError('需要 Python 3.12 x86_64')
    # The same checks as the offline installer: inherited libraries are
    # identified by their actual runtime and pinned version. Only the three
    # new wheels must be installed inside the independent prefix.
    try:
        torch = import_module('torch')
    except (ImportError, OSError, RuntimeError):
        raise PrecheckError('独立环境不能导入 Torch') from None
    if (torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8'
            or not torch.compiled_with_cxx11_abi() or torch.cuda.is_initialized()):
        raise PrecheckError('Torch 运行版本、CUDA 12.8、ABI 或无卡状态不符')
    try:
        if metadata.version('transformers') != '5.7.0':
            raise PrecheckError('Transformers 版本需要 5.7.0')
    except metadata.PackageNotFoundError:
        raise PrecheckError('缺少 Transformers') from None
    for name, expected in KERNEL_WHEELS.items():
        try:
            installed = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            raise PrecheckError(f'缺少 {name}') from None
        if installed.version.split('+', 1)[0] != expected:
            raise PrecheckError(f'{name} 版本与离线包不符')
        try:
            Path(installed.locate_file('')).resolve().relative_to(resolved)
        except (OSError, ValueError):
            raise PrecheckError(f'{name} 未安装在独立候选环境中') from None


def check_bindings(model, layer, mode, *, fla_ops=None, causal=None):
    if mode == NATURAL:
        return (model.is_fast_path_available is False
                and model.chunk_gated_delta_rule is None and model.causal_conv1d_fn is None
                and layer.chunk_gated_delta_rule is model.torch_chunk_gated_delta_rule
                and layer.causal_conv1d_fn is None)
    if mode == SIMULATED:
        chunk = fla_ops.chunk_gated_delta_rule
        conv = causal.causal_conv1d_fn
        return (model.is_fast_path_available is True and callable(chunk) and callable(conv)
                and layer.chunk_gated_delta_rule is model.chunk_gated_delta_rule is chunk
                and layer.causal_conv1d_fn is model.causal_conv1d_fn is conv)
    raise ValueError('预检模式无效')


def probe(mode):
    import torch
    from transformers.utils import import_utils

    if torch.cuda.is_available() or torch.cuda.is_initialized() or MODEL in sys.modules:
        raise RuntimeError('隔离的无卡进程状态不符')
    if torch.version.cuda != '12.8' or not torch.compiled_with_cxx11_abi():
        raise ValueError('Torch 编译版本或 ABI 不符')

    simulated = mode == SIMULATED
    availability = patch.object(import_utils, 'is_torch_cuda_available', return_value=True)
    current_device = patch.object(torch.cuda, 'current_device', return_value=torch.device('meta'))
    # Only the Transformers availability gate is simulated. FLA sees the real
    # CPU-only runtime. Its own Triton/device discovery stays untouched.
    with availability if simulated else nullcontext():
        import_utils.is_flash_linear_attention_available.cache_clear()
        import_utils.is_causal_conv1d_available.cache_clear()
        model = import_module(MODEL)
    # The fused norm constructor asks PyTorch for a device ordinal. Its
    # parameter stays on meta during the simulation, like the rest of the layer.
    with torch.device('meta'), current_device if simulated else nullcontext():
        layer = model.Qwen3_5GatedDeltaNet(model.Qwen3_5TextConfig(), 0)

    if torch.cuda.is_initialized() or not all(p.is_meta for p in layer.parameters()):
        raise RuntimeError('无卡预检出现非 meta 参数或 CUDA 初始化')
    refs = (dict(fla_ops=import_module('fla.ops.gated_delta_rule'), causal=import_module('causal_conv1d'))
            if simulated else {})
    return check_bindings(model, layer, mode, **refs)


def run_worker(mode, prefix):
    code = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--prefix', str(prefix), '--mode', mode]
    try:
        run = subprocess.run(code, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return '预检超时'
    if run.returncode == 0:
        return '通过'
    known = {'导入失败', '绑定不符', '设备或二进制运行库错误', '其他预检异常', '此无卡进程无法完成模拟'}
    tail = run.stdout.splitlines()[-1:] if run.stdout else []
    return tail[0] if tail and tail[0] in known else '子进程异常'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=Path, default=DEFAULT_PREFIX)
    parser.add_argument('--mode', choices=(NATURAL, SIMULATED), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.mode:
        try:
            if probe(args.mode):
                return 0
            print('绑定不符')
        except ImportError:
            print('导入失败')
        except (OSError, RuntimeError) as error:
            message = str(error).lower()
            no_device = ('0 active drivers', 'cuda_error_no_device', 'no cuda gpus are available',
                         'found no nvidia driver', 'cannot access accelerator device when none is available')
            blocked_by_hidden_cuda = args.mode == SIMULATED and any(text in message for text in no_device)
            print('此无卡进程无法完成模拟' if blocked_by_hidden_cuda else '设备或二进制运行库错误')
        except Exception:
            print('其他预检异常')
        return 1

    try:
        check_prefix(args.prefix)
    except PrecheckError as error:
        print('候选环境 CPU 核验失败', error)
        return 1
    natural = run_worker(NATURAL, args.prefix)
    simulated = run_worker(SIMULATED, args.prefix)
    print('核对 Qwen3.5 真实层的 chunk_gated_delta_rule 和 causal_conv1d_fn')
    print('无卡原样（预期走 Torch，检查实际类的函数绑定）', natural)
    print('无卡模拟（仅替换 Transformers 可用性门槛和构造设备为 meta，检查真实函数绑定）', simulated)
    print('A100 实际内核调用、前反向数值、速度和显存均待实机验证')
    if natural != '通过' or simulated not in ('通过', '此无卡进程无法完成模拟'):
        return 1
    if simulated != '通过':
        print('独立环境版本及链接检查可继续，快内核绑定留给单卡实测')
    return 0


if __name__ == '__main__':
    sys.exit(main())
