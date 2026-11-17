#!/usr/bin/env python3
"""Read-only, redacted AssertionError locations from one local A-group pilot."""
import re
from pathlib import Path

ROOT = Path('/workspace/A_group_repro_20261117')
WORK = ROOT / 'pilots/smoke_shared2'
MAX_WORKER_FILES = 60
MAX_BYTES = 2_000_000
MAX_EVENTS = 20

MARKERS = (
    ('/site-packages/verl/', 'verl/'),
    ('/site-packages/vllm/', 'vllm/'),
    ('/site-packages/transformers/', 'transformers/'),
    ('/site-packages/torch/', 'torch/'),
    ('/site-packages/ray/', 'ray/'),
    ('/site-packages/triton/', 'triton/'),
    ('/source/verl-v0.7.1/verl/', 'verl/'),
    ('/source/EduardoRL/eduardo/', 'eduardo/'),
)

FRAME = re.compile(r'File ["\']([^"\']+)["\'], line (\d+), in ([A-Za-z_<>.0-9]+)')
ASSERTION = re.compile(r'(?m)^.{0,350}?\bAssertionError(?:\s*:\s*([^\r\n]*))?\s*$')
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
CATEGORIES = (
    ('共享显卡内存分析受干扰', r'Error in memory profiling|Memory usage increased after sleeping|memory profiling.*(?:failed|error)'),
    ('线性注意力缓存与批次不匹配', r'num_cache_lines\s*>=\s*batch|cache.lines.*batch'),
    ('注意力缓存规格不匹配', r'AttentionSpec|kv.cache.spec|kv_cache_spec|page.size'),
    ('显存或缓存不足', r'out of memory|insufficient.*memory|not enough memory|free memory.*(?:less|lower)|no available memory'),
    ('并行度或注意力头数不匹配', r'(?:attention|kv|key.value|num).*head.*(?:divis|multiple|parallel)|tensor.parallel.*(?:divis|multiple)|world.size.*(?:divis|multiple)'),
    ('权重或张量形状不匹配', r'size mismatch|shape mismatch|shape.*(?:incompatible|mismatch)|weight.*(?:not found|missing)'),
    ('模型架构不支持', r'architectur.*(?:not supported|unsupported)|model.*(?:not supported|unsupported)'),
    ('CUDA 图捕获', r'cuda.graph|graph.capture'),
    ('注意力或 Triton 算子', r'flash.attn|attention kernel|triton.*(?:error|fail)|causal_conv1d'),
    ('精度类型不兼容', r'(?:dtype|bfloat16|float16|fp16|bf16).*(?:unsupported|not supported|incompatible|must)'),
    ('CUDA 驱动或内核', r'cuda error|driver.*(?:insufficient|mismatch)|no kernel image|invalid device'),
)


def safe_frame(frame):
    raw, line, function = frame.groups()
    for marker, prefix in MARKERS:
        if marker in raw:
            suffix = raw.rsplit(marker, 1)[-1]
            if re.fullmatch(r'[A-Za-z0-9_./-]+\.py', suffix) and '..' not in suffix:
                return prefix + suffix, line, function
    return None


def log_paths():
    paths = [('主训练日志', WORK / 'trainer.log')]
    link = WORK / 'ray_tmp_path.txt'
    try:
        ray_dir = Path(link.read_text(encoding='utf-8').strip()).resolve(strict=True)
        ray_dir.relative_to(ROOT.resolve(strict=True))
        sessions = [p for p in ray_dir.glob('session_*') if p.is_dir() and not p.is_symlink()]
        if sessions:
            logs = max(sessions, key=lambda p: p.stat().st_mtime_ns) / 'logs'
            logs_real = logs.resolve(strict=True)
            candidates = []
            for pat in ('worker-*.err', 'worker-*.out', '*worker*.err', '*worker*.out', 'raylet.err'):
                for p in logs.glob(pat):
                    try:
                        if p.is_file() and p.resolve(strict=True).parent == logs_real:
                            candidates.append(p)
                    except OSError:
                        continue
            unique = {str(p): p for p in candidates}
            newest = sorted(unique.values(), key=lambda p: p.stat().st_mtime_ns, reverse=True)
            paths.extend(('Ray工作日志' + str(i), p) for i, p in enumerate(newest[:MAX_WORKER_FILES], 1))
    except (OSError, ValueError):
        pass
    return paths


def read_tail(path):
    with path.open('rb') as f:
        limit = 16_000_000 if path == WORK / 'trainer.log' else MAX_BYTES
        f.seek(max(0, path.stat().st_size - limit))
        return ANSI.sub('', f.read().decode('utf-8', 'replace'))


def gather():
    paths = log_paths()
    unique = {}
    readable = 0
    event_total = 0
    for label, path in paths:
        try:
            text = read_tail(path)
        except OSError:
            continue
        readable += 1
        for assertion in ASSERTION.finditer(text):
            event_total += 1
            context = text[max(0, assertion.start() - 12_000):assertion.end()]
            frames = []
            for match in FRAME.finditer(context):
                frame = safe_frame(match)
                if frame and (not frames or frames[-1] != frame):
                    frames.append(frame)
            # Fixed names only. Never display arbitrary exception messages.
            tags = tuple(name for name, pattern in CATEGORIES if re.search(pattern, context[-3000:], re.I))
            last = tuple(frames[-4:])
            key = (last, tags)
            if key not in unique:
                unique[key] = {'label': label, 'count': 0, 'frames': last, 'tags': tags}
            unique[key]['count'] += 1
    return readable, event_total, list(unique.values())[-MAX_EVENTS:]


def main():
    readable, event_total, events = gather()
    print('已检查日志数：', readable)
    print('发现 AssertionError 条目：', event_total)
    print('说明：Ray 可能重复记录同一断言；框架行号为定位信息。')
    if not events:
        print('未在检查范围内找到 AssertionError；请报告该结果。')
        return
    for i, event in enumerate(events, 1):
        print(f'\n断言 {i}｜来源：{event["label"]}｜相同条目数：{event["count"]}')
        print('固定类别提示：', '、'.join(event['tags']) if event['tags'] else '未匹配')
        print('| 框架文件 | 行号 | 函数 |')
        print('|----------|------|------|')
        for file, line, function in event['frames']:
            print(f'| {file} | {line} | {function} |')
        if not event['frames']:
            print('| 未识别内部框架位置 | — | — |')


if __name__ == '__main__':
    main()
