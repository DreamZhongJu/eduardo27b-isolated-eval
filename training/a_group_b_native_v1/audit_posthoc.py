#!/usr/bin/env python3
"""Aggregate author rollouts and task-owned logs, without outputting contents.

This is intentionally offline, read-only, and does not instrument training.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

EXPECTED_COMMIT = 'e8c89b7293e65e632149571b16a1c2d4e33c77fe'
TOKENIZER_CONFIG_BLOB = 'eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1'
LABELS = ('judge', 'student_interaction', 'student_final')
WORKER_FILE = re.compile(r'^worker-.*\.(?:out|err)(?:\.\d+)?$')
PID_FILE = re.compile(r'^worker-.*-([0-9]+)\.(?:out|err)(?:\.\d+)?$')
RETRY = re.compile(r'\[(judge|student_interaction|student_final)\].*(?:transient failure|5xx).*\bretry\b', re.I)
EXHAUSTED = re.compile(r'\[(judge|student_interaction|student_final)\].*gave up after.*using fallback', re.I)
NONRETRY = re.compile(r'\[(judge|student_interaction|student_final)\].*non-retriable.*using fallback', re.I)
DEDUP = re.compile(r'\[repeated\s+\d+x\s+across\s+cluster\]', re.I)
SEMAPHORE = re.compile(r'created concurrency semaphore label=(judge|student) limit=(\d+) pid=(\d+)', re.I)


def load_stripper(repo: Path):
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=20).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=20).stdout.strip()
    if commit != EXPECTED_COMMIT or dirty:
        raise RuntimeError('作者源码版本或状态不符')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    from eduardo.utils import strip_teacher_controls
    return strip_teacher_controls


def load_tokenizer(model: Path):
    raw = (model / 'tokenizer_config.json').read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != TOKENIZER_CONFIG_BLOB:
        raise RuntimeError('tokenizer配置与冻结版本不符')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)


def inspect_rows(work: Path, strip, tokenizer=None):
    files = sorted((work / 'eduardo_conversations').glob('conversations_*.jsonl'))
    counts, seen, worker_pids = Counter(), set(), set()
    counts['文件数'] = len(files)
    for path in files:
        try:
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    counts['行数'] += 1
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError):
                        counts['无效行'] += 1
                        continue
                    if not isinstance(row, dict):
                        counts['无效行'] += 1
                        continue
                    uid = row.get('uuid')
                    if not isinstance(uid, str) or not uid:
                        counts['无UUID'] += 1
                        continue
                    if uid in seen:
                        counts['重复UUID'] += 1
                        continue
                    seen.add(uid)
                    counts['唯一轨迹'] += 1
                    pid = row.get('pid')
                    if type(pid) is int and pid > 0:
                        worker_pids.add(pid)
                    note = row.get('note')
                    counts['note=ok' if note == 'ok' else 'note=empty_transcript' if note == 'empty_transcript' else 'note其他'] += 1
                    msgs = row.get('messages')
                    if not isinstance(msgs, list) or any(not isinstance(m, dict) for m in msgs):
                        counts['消息字段异常'] += 1
                        continue
                    teachers = [m for m in msgs if m.get('role') == 'assistant']
                    if not teachers:
                        counts['无教师消息'] += 1
                        continue
                    if any(not isinstance(m.get('content'), str) for m in teachers):
                        counts['教师内容异常'] += 1
                        continue
                    raw = [m['content'] for m in teachers]
                    counts['教师消息完整轨迹'] += 1
                    if not any(strip(t).strip() for t in raw):
                        counts['教师全部不可见轨迹'] += 1
                    else:
                        counts['教师部分可见轨迹'] += 1
                    for t in raw:
                        counts['已保存教师回合'] += 1
                        if tokenizer is not None:
                            try:
                                n = len(tokenizer.encode(t, add_special_tokens=False))
                                counts['已重新分词回合'] += 1
                                if n >= 650: counts['近似650及以上'] += 1
                                if n >= 662: counts['近似662及以上'] += 1
                            except Exception:
                                counts['分词失败回合'] += 1
        except OSError:
            counts['不可读取文件'] += 1
    return counts, len(worker_pids)


def ray_logs(root: Path, work: Path):
    """Only accept a recorded Ray tempdir created directly inside root."""
    ray = Path((work / 'ray_tmp_path.txt').read_text(encoding='utf-8').strip()).resolve(strict=True)
    if ray.parent != root.resolve(strict=True) or not re.fullmatch(r'r[0-9a-f]{6}', ray.name):
        raise RuntimeError('Ray会话目录与任务不符')
    sessions = [p for p in ray.glob('session_*') if p.is_dir() and not p.is_symlink()]
    if len(sessions) != 1:
        raise RuntimeError('Ray会话不唯一或缺失')
    logroot = (sessions[0] / 'logs').resolve(strict=True)
    paths, seen, stats, pids = [], set(), Counter(), set()
    for p in sorted(logroot.iterdir()):
        if p.name.startswith('worker-') and re.search(r'\.(?:out|err)\.\d+\.gz$', p.name):
            stats['压缩轮转未扫描'] += 1
        if not WORKER_FILE.fullmatch(p.name): continue
        try:
            if not p.is_file() or p.resolve(strict=True).parent != logroot:
                stats['异常路径'] += 1
                continue
            st = p.stat(); inode = (st.st_dev, st.st_ino)
            if inode in seen:
                stats['同inode重复'] += 1
                continue
            seen.add(inode)
            match = PID_FILE.fullmatch(p.name)
            if match: pids.add(match.group(1))
            paths.append(p)
        except OSError:
            stats['异常路径'] += 1
    return paths, stats, len(pids)


def tally_logs(paths):
    counts = Counter()
    for path in paths:
        try:
            with path.open('r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    for key, pattern in (('重试日志行', RETRY), ('重试耗尽回退日志行', EXHAUSTED), ('不可重试回退日志行', NONRETRY)):
                        hit = pattern.search(line)
                        if hit: counts[(hit.group(1).lower(), key)] += 1
                    if DEDUP.search(line): counts['Ray合并提示行'] += 1
            counts['成功读取文件数'] += 1
        except OSError:
            counts['读取失败文件数'] += 1
    return counts


def semaphore_bindings(paths):
    """Observed logger records, not a claim of exhaustive process discovery."""
    bindings = {'judge': set(), 'student': set()}
    for path in paths:
        try:
            with path.open('r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    hit = SEMAPHORE.search(line)
                    if hit:
                        bindings[hit.group(1).lower()].add((int(hit.group(3)), int(hit.group(2))))
        except OSError:
            continue
    return bindings


def timing_rows(path):
    """Read structured numeric metrics only; retain no source log lines."""
    value = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    timing_pat = re.compile(r'timing_s/(step|testing|save_checkpoint)\s*[\"\']?\s*[:=]\s*' + value)
    step_pat = re.compile(r'training/global_step\s*[\"\']?\s*[:=]\s*(\d+)')
    rows = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            metrics = {match.group(1): float(match.group(2)) for match in timing_pat.finditer(line)}
            if metrics:
                m = step_pat.search(line)
                metrics['step_no'] = int(m.group(1)) if m else None
                rows.append(metrics)
    return rows


def probability_rows(path):
    """Extract only author's numeric metrics and step ids, without log contents."""
    keys = ('rollout_probs_diff_mean', 'rollout_probs_diff_max', 'rollout_probs_diff_std',
            'rollout_actor_probs_pearson_corr', 'rollout_probs_diff_valid')
    number = r'([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?))'
    metric_pat = re.compile(r'training/(' + '|'.join(keys) + r')\s*[\"\']?\s*[:=]\s*' + number, re.I)
    step_pat = re.compile(r'training/global_step\s*[\"\']?\s*[:=]\s*(\d+)')
    steps = {}
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            step = step_pat.search(line)
            if not step:
                continue  # do not infer an association from a neighboring line
            step_no = int(step.group(1))
            row = steps.setdefault(step_no, {})
            for match in metric_pat.finditer(line):
                row[match.group(1).lower()] = float(match.group(2))
    return steps


def pct(num, denom):
    return f'{num / denom * 100:.2f}%' if denom else '未知（无有效分母）'


def table(title, rows):
    print('\n' + title + '\n| 项目 | 数量 |\n|------|-----:|')
    for key, value in rows: print(f'| {key} | {value} |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/workspace/A_group_repro_20261117'))
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--model', type=Path)
    args = ap.parse_args()
    root = args.root.resolve(strict=True)
    work = args.work.resolve(strict=True)
    if work.parent != root / 'pilots':
        raise SystemExit('仅允许本任务 pilots 下的运行目录')
    try:
        strip = load_stripper(args.source or root / 'source/EduardoRL')
    except Exception as e:
        raise SystemExit('作者函数不可用；类型：' + type(e).__name__)
    try:
        tokenizer = load_tokenizer(args.model or root / 'models/Qwen3.5-4B')
        tok_status = '已加载冻结 tokenizer'
    except Exception as e:
        tokenizer = None
        tok_status = '未覆盖（' + type(e).__name__ + '）'
    counts, row_pids = inspect_rows(work, strip, tokenizer)
    table('对话（去重后）', [(k, counts[k]) for k in ('文件数','行数','唯一轨迹','重复UUID','无UUID','无效行',
          '不可读取文件','note=ok','note=empty_transcript','note其他','消息字段异常','无教师消息',
          '教师内容异常','教师消息完整轨迹','教师全部不可见轨迹','教师部分可见轨迹')]
          + [('全部不可见占完整轨迹比例', pct(counts['教师全部不可见轨迹'], counts['教师消息完整轨迹'])),
             ('记录中的不同 reward 进程数', row_pids)])
    table('已保存的教师回合（重新分词是近似）', [('tokenizer', tok_status),
          ('已保存教师回合', counts['已保存教师回合']),
          ('已重新分词回合', counts['已重新分词回合'] if tokenizer else '未覆盖'),
          ('分词失败回合', counts['分词失败回合'] if tokenizer else '未覆盖'),
          ('近似650及以上', counts['近似650及以上'] if tokenizer else '未覆盖'),
          ('近似662及以上', counts['近似662及以上'] if tokenizer else '未覆盖'),
          ('近似662及以上比例', pct(counts['近似662及以上'], counts['已重新分词回合']) if tokenizer else '未覆盖')])
    print('\n尾轮未追加轨迹数：未覆盖（原始对话无生成/终止回执，不能仅凭最后消息推断）。')
    print('逐步骤统计：未覆盖（对话记录没有可靠训练步号）。')
    groups = {'主训练日志': [work / 'trainer.log'], 'Ray worker日志': []}
    try:
        groups['Ray worker日志'], stats, pid_count = ray_logs(root, work)
        print('Ray worker 日志文件数：', len(groups['Ray worker日志']), '；文件名识别的不同 PID 数：', pid_count)
        print('Ray 日志异常路径：', stats['异常路径'], '；同 inode 重复：', stats['同inode重复'], '；压缩轮转未扫描：', stats['压缩轮转未扫描'])
    except (OSError, RuntimeError, ValueError):
        print('Ray 会话未能核验；worker 日志结果未覆盖。')
        groups['Ray worker日志'] = None
    results = {name: tally_logs(paths) if paths is not None else None for name, paths in groups.items()}
    all_logs = [p for paths in groups.values() if paths is not None for p in paths]
    bindings = semaphore_bindings(all_logs)
    print('\n作者 semaphore 初始化日志（只统计日志中实际观察到的 PID，可能不完整）')
    print('| 类型 | 观察到的不同进程数 | 观察到的每进程限制 |')
    print('|------|--------------------:|----------------------|')
    for label, observed in bindings.items():
        pids = {pid for pid, _ in observed}
        limits = sorted({limit for _, limit in observed})
        print(f'| {label} | {len(pids) if pids else "未知"} | {", ".join(map(str, limits)) if limits else "未知"} |')
    print('\nAPI调用异常日志行（各日志组可能互相复制，不能相加；行数并非独立请求数）')
    print('| 调用类型 | 日志组 | 重试行 | 耗尽回退行 | 不可重试回退行 |')
    print('|----------|--------|--------|------------|----------------|')
    for label in LABELS:
        for group, c in results.items():
            vals = [c[(label, k)] for k in ('重试日志行','重试耗尽回退日志行','不可重试回退日志行')] if c is not None else ['未覆盖'] * 3
            print('| ' + ' | '.join(map(str, [label, group] + vals)) + ' |')
    for group, c in results.items():
        if c is not None:
            print(f'{group}：读取 {c["成功读取文件数"]} 个；失败 {c["读取失败文件数"]} 个；Ray合并提示 {c["Ray合并提示行"]} 行。')
    try:
        rows = timing_rows(work / 'trainer.log')
        if rows:
            last = rows[-1]
            def seconds(key):
                return f'{last[key]:.2f}' if key in last else '未记录'
            table('最新训练指标计时（秒）', [('训练步号', last['step_no'] if last['step_no'] is not None else '未知'),
                  ('训练 step', seconds('step')), ('末尾验证 testing', seconds('testing')),
                  ('保存检查点', seconds('save_checkpoint'))])
            print('说明：作者 step 计时与 testing 计时平级；保存检查点计时属于 step 的一部分。')
        else:
            print('\n训练计时：未找到可解析的 timing_s 指标。')
    except OSError:
        print('\n训练计时：训练日志不可读取。')
    print('\nrollout 与 actor 训练指标（仅日志中同一行的有效步号；不是权重逐项一致性验证）')
    print('| 步数 | 概率差均值 | 概率差最大值 | 概率差标准差 | 概率相关系数 | 有效标记 |')
    print('|------|-----------:|-------------:|-------------:|-------------:|---------:|')
    try:
        rows = probability_rows(work / 'trainer.log')
        if not rows:
            print('| 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |')
        def fmt(row, key):
            x = row.get(key)
            return '未记录' if x is None else '非有限值' if not math.isfinite(x) else f'{x:.6g}'
        for step, row in sorted(rows.items()):
            print('| ' + ' | '.join([str(step)] + [fmt(row, k) for k in (
                  'rollout_probs_diff_mean', 'rollout_probs_diff_max', 'rollout_probs_diff_std',
                  'rollout_actor_probs_pearson_corr', 'rollout_probs_diff_valid')]) + ' |')
    except OSError:
        print('| 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |')
    print('说明：原始 finish_reason、未落盘教师回合、请求端无日志故障和缺失日志均无法由本统计复原。')
    print('每题重试与失败为日志行数，不能当成去重后的票数；静默异常不能判为零。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
