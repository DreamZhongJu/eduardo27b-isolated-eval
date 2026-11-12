#!/usr/bin/env python3
"""Read-only, source-weighted normalized gains and source bootstrap.

No source identifiers, student answers, prompts, or API responses are emitted.
"""
import argparse
import hashlib
import importlib.util
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
SUMMARY_SHA = '53879d59aa1ce8ae6cf85f4baea62e699c6ea0b21be43ed0a9bf870a5da069f8'


def import_checked(path, expected, name):
    if not path.is_absolute() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('script hash mismatch')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalized_gain(pre, post):
    if not (0 <= pre <= 1 and 0 <= post <= 1):
        raise ValueError('invalid probability')
    return (post - pre) / (1 - pre) if post > pre else post - pre


def p_value(c, unknown_as_wrong):
    correct, wrong, unknown = (c.get(k, 0) for k in ('correct', 'incorrect', 'unknown'))
    if any(not isinstance(x, int) or x < 0 for x in (correct, wrong, unknown)) or correct + wrong + unknown != 16:
        raise ValueError('sample count mismatch')
    den = 16 if unknown_as_wrong else correct + wrong
    return correct / den if den else None


def percentile(xs, q):
    rank = q * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def bootstrap(values, seed, draws=10000):
    if not values:
        return None
    values = list(values)
    n = len(values)
    rand = random.Random(seed)
    samples = sorted(sum(rand.choices(values, k=n)) / n for _ in range(draws))
    return sum(values) / n, percentile(samples, .025), percentile(samples, .975), n


def estimate(records, tier, target, baseline, unknown_as_wrong, easy_filtered=False):
    """One value per ID. Combined original averages available tier deltas per ID."""
    per_id = defaultdict(list)
    excluded_zero = set()
    filtered = set()
    for g, sid, counts in records:
        if tier != '汇总' and g != tier:
            continue
        if tier == '汇总' and target != '原题':
            raise ValueError('invalid combined category')
        # Filter once against fixed independent pretest, regardless of baseline
        # or missing-score sensitivity. Missing pre answers count as wrong.
        pre = counts[(target, 'pre')]
        if easy_filtered and pre.get('correct', 0) >= 15:
            filtered.add(sid)
            continue
        a = p_value(counts[(target, baseline)], unknown_as_wrong)
        b = p_value(counts[(target, 'post')], unknown_as_wrong)
        if a is None or b is None:
            excluded_zero.add(sid)
            continue
        per_id[sid].append(normalized_gain(a, b))
    pairs = [(sid, sum(v) / len(v)) for sid, v in sorted(per_id.items())]
    return pairs, len(filtered), len(excluded_zero - set(per_id))


def collect_sources(s, inv, root, overlay, decisions):
    if any(not p.is_absolute() for p in (root, overlay, decisions)):
        raise ValueError('absolute paths required')
    if (not root.is_dir() or root.is_symlink() or not overlay.is_dir() or overlay.is_symlink() or
            decisions.is_symlink() or (decisions.exists() and not decisions.is_dir())):
        raise ValueError('invalid roots')
    rows, errors = [], Counter()
    seen = defaultdict(int)
    for stage in s.STAGES:
        tier = s.GROUP[stage]
        folder = root / stage / 'sources'
        if not folder.is_dir() or folder.is_symlink():
            errors['来源目录缺失或异常'] += 1
            continue
        for source in sorted(folder.iterdir()):
            key = source.name
            try:
                inv.no_symlink(source, root)
                if not source.is_dir() or not re.fullmatch(r'[0-9a-f]{24}', key):
                    errors['来源结构异常'] += 1
                    continue
                marker, held = source / 'RAW_COMPLETE.json', source / 'SOURCE_HELD.json'
                inv.no_symlink(marker, root)
                inv.no_symlink(held, root)
                if marker.is_file() and held.is_file():
                    errors['来源结构异常'] += 1
                elif held.is_file():
                    errors['暂停来源'] += 1
                elif not marker.is_file():
                    errors['不完整来源'] += 1
                elif not (root / 'AGNES' / stage / 'sources' / key / 'RESULTS.json').is_file():
                    errors['无判分记录'] += 1
                else:
                    sid, counts, errs = s.load_source(inv, root, overlay, decisions, stage, key)
                    seen[(tier, sid)] += 1
                    rows.append((tier, sid, counts, errs))
            except (OSError, ValueError, KeyError, TypeError, UnicodeError):
                errors['来源校验异常'] += 1
    result = []
    for tier, sid, counts, errs in rows:
        if seen[(tier, sid)] != 1:
            errors['重复来源记录'] += 1
            continue
        errors.update(errs)
        result.append((tier, sid, counts))
    return result, errors


def render(s, records, errors, draws, seed):
    labels = [('原题合并', '汇总', '原题', False),
              ('原题-简单对话', '简单', '原题', False),
              ('原题-中等对话', '中等', '原题', False),
              ('原题-困难对话', '困难', '原题', False),
              ('简单迁移题', '简单', '迁移题', False),
              ('中等迁移题', '中等', '迁移题', False),
              ('困难迁移题', '困难', '迁移题', False),
              ('简单迁移题-剔除高分', '简单', '迁移题', True)]
    lines = ['逐来源非对称归一化增益；百分数，区间为按来源 bootstrap 的 95% 百分位区间',
             f'重抽样次数={draws}，随机种子={seed}；每个计算独立派生可复现种子。']
    for unknown_as_wrong, policy in ((True, '未决按错'), (False, '排除未决')):
        for baseline, base_label in (('pre', '独立前测'), ('control', '无教学对照')):
            rows = []
            for label, tier, target, filtered in labels:
                pairs, n_filtered, n_no_scored = estimate(records, tier, target, baseline, unknown_as_wrong, filtered)
                # Stable independent seed, independent of Python's randomized hash.
                seedbytes = hashlib.sha256(f'{seed}|{policy}|{base_label}|{label}'.encode()).digest()
                result = bootstrap([v for _, v in pairs], int.from_bytes(seedbytes[:8], 'big'), draws)
                if result is None:
                    rows.append((label, '无可用来源', '-', n_filtered, n_no_scored))
                else:
                    mean, low, high, n = result
                    rows.append((label, f'{mean*100:.2f} [{low*100:.2f}, {high*100:.2f}]', n, n_filtered, n_no_scored))
            lines += ['', f'{policy}｜基线：{base_label}', s.table(('项目', '增益 % [95% CI]', '来源数', '高分剔除', '无已判定'), rows)]
    lines += ['', '来源质量核对', s.table(('项目', '数量'), [(k, errors[k]) for k in sorted(errors)] or [('异常及暂停', 0)]),
              '原题合并：先对同一来源在可用难度对话中的各自增益求平均，再以来源为单位等权重抽样；各档原题单列。',
              '高分剔除固定使用简单迁移题独立前测正确数≥15/16；未决按错确定筛选，敏感性分析沿用相同名单。',
              '排除未决时某来源的基线或后测无已判定作答则剔除该来源；未完成来源不计入。',
              '只核对来源记录身份与绑定，未与冻结 499 来源清单逐条核对；不证明全部题目或判分均正确。']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True, type=Path)
    p.add_argument('--inventory', required=True, type=Path)
    p.add_argument('--root', type=Path, default=Path('/workspace/results/own-test-499-v1'))
    p.add_argument('--overlay', type=Path, default=Path('/workspace/results/agnes-topbool-recovery-v1'))
    p.add_argument('--decisions', type=Path, default=Path('/workspace/results/agnes-rejudge-v1'))
    p.add_argument('--draws', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    try:
        if args.draws != 10000:
            raise ValueError('10000 draws required')
        s = import_checked(args.summary, SUMMARY_SHA, 'source_gain_summary')
        inv = s.load_inventory(args.inventory)
        records, errors = collect_sources(s, inv, args.root, args.overlay, args.decisions)
        if not records:
            raise ValueError('no records')
        print(render(s, records, errors, args.draws, args.seed))
    except Exception:
        print('只读计算失败：请检查目录、文件校验与来源结构；没有修改原始数据。')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
