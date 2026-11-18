"""Numeric-only C audit. Runs after the trainer and both HTTP servers have exited."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

# The published bundle keeps the pinned B helpers alongside this file.
if not Path(__file__).with_name('audit_posthoc.py').is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a_group_b_native_v1'))
import audit_posthoc as audit

EXPECTED = {'train': 256, 'val': 100}
TIMERS = ('step', 'testing', 'gen', 'reward', 'update_actor', 'save_checkpoint', 'update_weights')
VALUE = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)'
STEP = re.compile(r"\btraining/global_step[\"']?\s*[:=]\s*1(?![.\d])")
TIME = re.compile(r"\btiming_s/(" + '|'.join(TIMERS) + r")[\"']?\s*[:=]\s*" + VALUE, re.I)
CHAT = re.compile(r'"POST /v1/chat/completions HTTP/[\d.]+"\s+(\d{3})\b')
ENGINE_QUEUE = re.compile(r'Running:\s*(\d+) reqs?,\s*Waiting:\s*(\d+) reqs?', re.I)
STAMP = re.compile(r'\[C_HTTP (\d+\.\d{6})\]')
UNAVAILABLE = '未知'


def write_access_config(work):
    """Keep the standard Uvicorn handlers and timestamp successful/failed accesses alike."""
    from uvicorn.config import LOGGING_CONFIG
    config = deepcopy(LOGGING_CONFIG)
    config['disable_existing_loggers'] = False
    config['formatters']['access']['fmt'] = '[C_HTTP %(created).6f] "%(request_line)s" %(status_code)s'
    config['formatters']['access']['use_colors'] = False
    path = Path(work) / 'c-access-log.json'
    with path.open('x', encoding='utf-8') as out:
        json.dump(config, out)
    return path


def read_timings(path):
    values = defaultdict(set)
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if STEP.search(line):
                for key, raw in TIME.findall(line):
                    value = float(raw)
                    if math.isfinite(value) and value >= 0:
                        values[key].add(value)
    # Ray copies of a line have no effect. Disagreeing copies cannot supply a trusted value.
    return {key: next(iter(vals)) for key, vals in values.items() if len(vals) == 1}


def source_hash(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


def load_splits(data_dir):
    """Author rollout rows omit split; bind the exact original problem to the verified parquets."""
    import pyarrow.parquet as pq
    pools = {}
    for split in EXPECTED:
        keys = set()
        for record in pq.read_table(data_dir / (split + '.parquet'), columns=['extra_info']).to_pylist():
            info = record['extra_info']
            if info['split'] != split or not isinstance(info['problem'], str) or not info['problem']:
                raise ValueError('冻结题单字段异常')
            keys.add(source_hash(info['problem']))
        pools[split] = keys
    if pools['train'] & pools['val']:
        raise ValueError('同一原题跨越两个冻结划分')
    return {key: split for split, keys in pools.items() for key in keys}


def tally_trajectories(paths, splits, strip, tokenizer):
    counts = {split: Counter() for split in EXPECTED}
    timestamps = {split: [] for split in EXPECTED}
    errors, sources, seen = Counter(), {split: Counter() for split in EXPECTED}, set()
    for path in paths:
        try:
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        uid, source = row['uuid'], row['problem']
                        if not isinstance(uid, str) or not uid or not isinstance(source, str) or not source:
                            raise ValueError
                        key = source_hash(source)
                    except (TypeError, KeyError, ValueError):
                        errors['记录格式'] += 1
                        continue
                    if uid in seen:
                        errors['重复UUID'] += 1
                        continue
                    seen.add(uid)
                    split = splits.get(key)
                    if split is None:
                        errors['题单未绑定'] += 1
                        continue
                    c = counts[split]
                    sources[split][key] += 1
                    c['rows'] += 1
                    for tag in ('ok', 'empty_transcript'):
                        if row.get('note') == tag:
                            c[tag] += 1
                    if row.get('note') not in ('ok', 'empty_transcript'):
                        c['other_note'] += 1
                    gate = row.get('gated')
                    if type(gate) in (int, bool) and gate in (0, 1):
                        c['gated' if gate else 'pass'] += 1
                    else:
                        errors['门槛字段'] += 1
                    ts = row.get('ts')
                    if type(ts) in (int, float) and math.isfinite(ts):
                        timestamps[split].append(ts)
                    messages = row.get('messages')
                    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
                        errors['教师消息'] += 1
                        continue
                    texts = [m.get('content') for m in messages if m.get('role') == 'assistant']
                    if not all(isinstance(t, str) for t in texts):
                        errors['教师消息'] += 1
                        continue
                    c['inspectable'] += 1
                    last = messages[-1] if messages else {}
                    if (texts and last.get('role') == 'user' and
                            isinstance(last.get('content'), str) and last['content'].strip()):
                        c['possible_omitted_teacher'] += 1
                    if not texts:
                        c['no_teacher'] += 1
                    if not any(strip(t).strip() for t in texts):
                        c['invisible'] += 1
                    for value in texts:
                        c['saved_turns'] += 1
                        if tokenizer is None:
                            continue
                        try:
                            length = len(tokenizer.encode(value, add_special_tokens=False))
                        except Exception:
                            errors['重分词'] += 1
                            continue
                        c['retokenized'] += 1
                        c['at_cap'] += int(length >= 662)
        except (OSError, UnicodeError):
            errors['读取失败'] += 1
    for split, group_counts in sources.items():
        counts[split]['sources'] = len(group_counts)
        # The training pool can contain multiple transfer variants of the same original.
        size = 16 if split == 'train' else 1
        errors['来源重复次数不符'] += sum(v % size != 0 or (split == 'val' and v != 1)
                                         for v in group_counts.values())
    return counts, timestamps, errors


def checkpoint_boundary(path, timestamps):
    """One step, synchronous checkpoint; validation generation follows its persisted iteration marker."""
    try:
        if path.is_symlink() or path.read_text().strip() != '1':
            return None
        cut = path.stat().st_mtime_ns / 1e9
    except (OSError, ValueError):
        return None
    if any(len(timestamps[split]) != n for split, n in EXPECTED.items()):
        return None
    if not max(timestamps['train']) < cut < min(timestamps['val']):
        return None
    return cut


def tally_access(path, cut):
    counts = Counter()
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = CHAT.search(line)
            if not match:
                continue
            counts['all'] += 1
            counts['non_2xx'] += int(not 200 <= int(match[1]) < 300)
            stamp = STAMP.search(line)
            if stamp is None or cut is None or abs(float(stamp[1]) - cut) <= 1:
                counts['unassigned'] += 1
            else:
                phase = 'train' if float(stamp[1]) < cut else 'val'
                counts[phase] += 1
    return counts


def estimate_seconds(timings):
    if any(k not in timings for k in ('step', 'testing', 'save_checkpoint')):
        return {}
    step, validation, checkpoint = (timings[k] for k in ('step', 'testing', 'save_checkpoint'))
    if step < checkpoint or step <= 0 or validation <= 0:
        return {}
    return {n: n * (step - checkpoint) + ((n + 24) // 25) * (checkpoint + validation)
            for n in (10, 100)}


def judge_queue(path):
    counts = Counter()
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            m = ENGINE_QUEUE.search(line)
            if m:
                running, waiting = map(int, m.groups())
                counts['samples'] += 1
                counts['active'] += int(running + waiting > 0)
                counts['waiting'] += int(waiting > 0)
                counts['peak_running'] = max(counts['peak_running'], running)
                counts['peak_waiting'] = max(counts['peak_waiting'], waiting)
    return counts


def report(root, work):
    timings = read_timings(work / 'trainer.log')
    audit.table('第 1 步 veRL 计时', [(key, timings.get(key, UNAVAILABLE)) for key in TIMERS],
                headers=('timing_s 指标', '秒'))
    print('step 与 testing 单独计时；gen 包括生成期间等待学生或奖励的时间，子项可能重叠。')

    estimates = estimate_seconds(timings)
    audit.table('以本次 C 单步推算训练和验证', [(n, round(estimates[n] / 3600, 2) if n in estimates else UNAVAILABLE)
                for n in (10, 100)], headers=('步数', '小时'))
    print('常规步取 step 减保存时间；每 25 步及末步各计一次保存和验证。'
          '启动和 CPU 合并单列，不计入推算。门槛放行比例与后测工作量随训练改变，实际时间会变动。')

    strip = audit.load_stripper(root / 'source/EduardoRL')
    tokenizer = audit.load_tokenizer(root / 'models/Qwen3.5-4B')
    splits = load_splits(root / 'data/author_prepared')
    paths = sorted((work / 'eduardo_conversations').glob('conversations_*.jsonl'))
    rows, timestamps, errors = tally_trajectories(paths, splits, strip, tokenizer)
    fields = (('rows', '按UUID去重记录'), ('sources', '不同原题'), ('ok', 'note=ok'),
              ('empty_transcript', 'note=empty_transcript'), ('other_note', '其他 note'),
              ('gated', '门槛拦截'), ('pass', '门槛放行'), ('no_teacher', '无教师回合'),
              ('invisible', '教师无可见文本（含无教师回合）'), ('inspectable', '可核对可见性的对话'),
              ('possible_omitted_teacher', '尾轮路径线索（末条非空学生发言，之前有教师）'),
              ('at_cap', '重分词至少 662 token 的回合'), ('retokenized', '成功重分词的已保存回合'))
    view = [(title, *(rows[s][key] for s in EXPECTED)) for key, title in fields]
    for num, den, title in [('invisible', 'inspectable', '教师无可见文本比例'),
                            ('at_cap', 'retokenized', '近似达到上限的教师回合比例')]:
        view.append((title, *(audit.pct(rows[s][num], rows[s][den]) for s in EXPECTED)))
    audit.table('训练和末尾验证轨迹', view, headers=('项目', '训练', '验证'))
    print('尾轮线索利用作者交互中正常终止会追加空学生消息的行为；若学生返回空串，计数可能漏掉。'
          '确切未保存回合及 finish_reason 均未知；长度仅针对已保存原文。')
    print('去重、身份绑定、读取及字段异常合计', sum(errors.values()))

    cut = checkpoint_boundary(work / 'verl_ckpts/c-paper-one-step/latest_checkpointed_iteration.txt', timestamps)
    valid = (all(rows[s]['rows'] == EXPECTED[s] for s in EXPECTED) and
             not any(errors.values()) and cut is not None and 'step' in timings and 'testing' in timings)
    audit_rows = []
    observations = {}
    for role, label in [('judge', '裁判'), ('student', '学生')]:
        a = tally_access(work / (role + '.log'), cut)
        split_ok = cut is not None and not a['unassigned'] and a['train'] > 0 and a['val'] > 0
        valid &= split_ok
        observations[role] = a
        audit_rows.append((label, a['all'], a['train'] if split_ok else UNAVAILABLE,
                           a['val'] if split_ok else UNAVAILABLE, a['unassigned'], a['non_2xx']))
    audit.table('本机服务已记录的 chat/completions POST 响应', audit_rows,
                headers=('服务', '总数', '训练', '末尾验证', '无法归属', '非 2xx'))
    print('用同机同步写出的第 1 步检查点标记分界；已按冻结题单与轨迹时间复核。接收前的失败不产生 HTTP 响应。'
          if cut is not None else '缺少可信阶段分界，分段请求数未知。')
    for split, title in [('train', '训练'), ('val', '末尾验证')]:
        expected_votes = 6 * rows[split]['ok']
        observed = observations['judge']
        if not observed['unassigned'] and observed[split] > 0:
            print(f'{title}六票计划 {expected_votes}，服务端观察 {observed[split]}。')

    queue = judge_queue(work / 'judge.log')
    audit.table('裁判服务 vLLM 队列（日志采样）',
                [(title, queue[key] if queue['samples'] else UNAVAILABLE) for key, title in [
                    ('samples', '统计样本'), ('active', '活跃样本'), ('waiting', '观察到等待的样本'),
                    ('peak_running', '运行中请求峰值'), ('peak_waiting', '等待请求峰值')]])
    if queue['waiting'] >= 3 and queue['waiting'] * 5 >= queue['active']:
        print('至少 3 次且至少五分之一活跃采样观察到等待，建议评估 2 张裁判卡 DP，维持同一总并发比较耗时。')
    else:
        print('当前先维持 1 张裁判卡；现有采样对持续服务端积压的证据有限。')
    print('服务端不统计客户端等待许可的时间，单次 1 卡运行不能测出双卡提速。')

    paths, coverage, _ = audit.ray_logs(root, work)
    logs = {'worker': audit.tally_logs(paths), 'trainer': audit.tally_logs([work / 'trainer.log'])}
    detail = []
    for group, item in logs.items():
        for label in audit.LABELS:
            detail.append((group, label, item[label, '重试日志行'], item[label, '重试耗尽回退日志行'],
                           item[label, '不可重试回退日志行']))
    audit.table('客户端技术异常（日志行，各组可能重复）', detail,
                headers=('日志', '调用', '重试', '耗尽 fallback', '不可重试 fallback'))
    terminal = audit.terminal_status(logs, coverage)
    print('日志技术终止检查', {0: '已扫描日志中为零', 2: '发现技术失败', 3: '覆盖不足'}[terminal])
    return terminal or (0 if valid else 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/A_group_repro_20261117'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    try:
        return report(args.root, args.work)
    except Exception as exc:
        print('C 汇总未完成；异常类型', type(exc).__name__, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main())
