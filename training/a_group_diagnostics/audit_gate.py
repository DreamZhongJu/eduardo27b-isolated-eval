#!/usr/bin/env python3
"""Read-only aggregate diagnostic for an A-group rollout; never prints raw data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

DEFAULT_ROOT = Path('/workspace/A_group_repro_20261117')
EXPECTED_COMMIT = 'e8c89b7293e65e632149571b16a1c2d4e33c77fe'
MAX_LOGS = 200


def output_table(title, rows):
    print('\n' + title)
    print('| 项目 | 数量 |')
    print('|------|-----:|')
    for name, value in rows:
        print(f'| {name} | {value} |')


def load_stripper(repo):
    import subprocess
    p = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                       capture_output=True, text=True, timeout=20, check=True)
    if p.stdout.strip() != EXPECTED_COMMIT:
        raise RuntimeError('作者代码版本不一致')
    status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
                            capture_output=True, text=True, timeout=20, check=True)
    if status.stdout.strip():
        raise RuntimeError('作者代码存在改动')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    from eduardo.utils import strip_teacher_controls
    return strip_teacher_controls


def audit_rows(work, strip):
    folder = work / 'eduardo_conversations'
    files = sorted(folder.glob('conversations_*.jsonl')) if folder.is_dir() else []
    seen = set()
    n = Counter()
    counts = Counter()
    joint = Counter()
    for path in files:
        try:
            f = path.open('r', encoding='utf-8')
        except OSError:
            n['unreadable_files'] += 1
            continue
        with f:
            for line in f:
                n['all_lines'] += 1
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    n['invalid_rows'] += 1
                    continue
                if not isinstance(row, dict):
                    n['invalid_rows'] += 1
                    continue
                uid = row.get('uuid')
                if not isinstance(uid, str) or not uid:
                    n['no_uuid'] += 1
                    continue
                if uid in seen:
                    n['duplicate'] += 1
                    continue
                seen.add(uid)
                n['unique'] += 1
                note = row.get('note')
                if note == 'ok':
                    counts['note_ok'] += 1
                elif note == 'empty_transcript':
                    counts['note_empty'] += 1
                else:
                    counts['note_other'] += 1
                gated = row.get('gated')
                if gated in (1, True):
                    counts['gated'] += 1
                elif gated in (0, False):
                    counts['not_gated'] += 1
                else:
                    counts['gated_unknown'] += 1
                messages = row.get('messages')
                if not isinstance(messages, list):
                    counts['messages_unknown'] += 1
                    messages = []
                assistants = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
                if not assistants:
                    counts['no_assistant'] += 1
                else:
                    raw = [m.get('content') if isinstance(m.get('content'), str) else '' for m in assistants]
                    if not any(s.strip() for s in raw):
                        counts['assistant_raw_empty'] += 1
                    elif not any(strip(s).strip() for s in raw):
                        counts['assistant_visible_empty'] += 1
                    else:
                        counts['assistant_visible_nonempty'] += 1
                turns = row.get('num_turns')
                if type(turns) is not int or turns < 0:
                    counts['turn_unknown'] += 1
                elif turns == 0:
                    counts['turn_0'] += 1
                elif turns == 1:
                    counts['turn_1'] += 1
                elif turns <= 4:
                    counts['turn_2_4'] += 1
                elif turns <= 9:
                    counts['turn_5_9'] += 1
                else:
                    counts['turn_10_plus'] += 1
                def decision(value):
                    return value if value in ('OK', 'REJECT') else 'OTHER'
                joint[(decision(row.get('judge_no_leak')), decision(row.get('judge_teaching')))] += 1
    n['file_count'] = len(files)
    return n, counts, joint


ACCESS = re.compile(r'"POST\s+/v1/chat/completions(?:\?[^\s"]*)?\s+HTTP/[^"\s]+"\s+(\d{3})\b')
PATTERNS = (
    ('judge 瞬态错误重试日志行', re.compile(r'\[judge\].*(?:transient failure|5xx).*retry', re.I)),
    ('judge 重试耗尽降级日志行', re.compile(r'\[judge\].*gave up after.*using fallback', re.I)),
    ('judge 不可重试降级日志行', re.compile(r'\[judge\].*non-retriable.*using fallback', re.I)),
    ('判决格式无法解析日志行', re.compile(r'Failed to parse judge response, defaulting to REJECT', re.I)),
    ('判决正则恢复日志行', re.compile(r'Judge JSON malformed, recovered decision via regex', re.I)),
)


def count_server(path):
    result = Counter()
    try:
        with path.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                for match in ACCESS.finditer(line):
                    result[int(match.group(1))] += 1
    except OSError:
        return None
    return result


def count_logfiles(files):
    counts = Counter()
    unread = 0
    for path in files:
        try:
            with path.open('r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    for name, pat in PATTERNS:
                        if pat.search(line):
                            counts[name] += 1
        except OSError:
            unread += 1
    return counts, unread


def ray_logs(root, work):
    try:
        raydir = Path((work / 'ray_tmp_path.txt').read_text(encoding='utf-8').strip()).resolve(strict=True)
        root_real = root.resolve(strict=True)
        if raydir.parent != root_real or not re.fullmatch(r'r[0-9a-f]{6}', raydir.name):
            raise ValueError('Ray 路径不是本任务专用目录')
        sessions = [p for p in raydir.glob('session_*') if p.is_dir() and not p.is_symlink()]
        if not sessions:
            return [], 0, '未发现本次 Ray 会话'
        logroot = (max(sessions, key=lambda p: p.stat().st_mtime_ns) / 'logs').resolve(strict=True)
        paths = []
        for p in logroot.iterdir():
            if p.name.startswith('worker-') and (p.name.endswith('.out') or p.name.endswith('.err')):
                try:
                    if p.is_file() and p.resolve(strict=True).parent == logroot:
                        paths.append(p)
                except OSError:
                    continue
        paths.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        return paths[:MAX_LOGS], len(paths), '已发现'
    except (OSError, ValueError):
        return [], 0, '未发现本次 Ray 会话'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    work = root / 'pilots/smoke_shared3'
    try:
        strip = load_stripper(root / 'source/EduardoRL')
    except Exception:
        raise SystemExit('无法加载或核验作者代码；未检查，请报告。')
    n, c, j = audit_rows(work, strip)
    try:
        import yaml
        cfg = yaml.safe_load((root / 'source/EduardoRL/eduardo/configs/user.yaml').read_text(encoding='utf-8'))
        configured = cfg['custom_reward_function']['reward_kwargs']['judge_votes']
        vote_text = str(configured) if type(configured) is int else '未知'
    except Exception:
        vote_text = '未知'
    runtime_votes = set()
    trainer_log = work / 'trainer.log'
    if trainer_log.is_file():
        try:
            with trainer_log.open('r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    for value in re.findall(r'''["']?judge_votes["']?\s*[:=]\s*["']?(\d{1,3})\b''', line):
                        runtime_votes.add(int(value))
        except OSError:
            pass
    print('作者配置 judge_votes：', vote_text)
    print('训练日志中出现的 judge_votes 值：', ', '.join(map(str, sorted(runtime_votes))) if runtime_votes else '未发现；仅凭日志不能排除覆盖')
    output_table('对话记录（按 UUID 去重）', [
        ('对话文件数', n['file_count']), ('记录行数', n['all_lines']),
        ('可解析且有 UUID 的唯一记录', n['unique']), ('重复 UUID 记录', n['duplicate']),
        ('缺 UUID 记录', n['no_uuid']), ('无效行', n['invalid_rows']),
        ('note=ok', c['note_ok']), ('note=empty_transcript', c['note_empty']), ('其他 note', c['note_other']),
        ('门槛拦下', c['gated']), ('未被门槛拦下', c['not_gated']), ('门槛字段未知', c['gated_unknown']),
    ])
    output_table('每条对话的教师内容（按作者函数去除思考）', [
        ('无教师消息', c['no_assistant']), ('有教师消息但原始内容全空', c['assistant_raw_empty']),
        ('原始内容非空、去除思考后全空', c['assistant_visible_empty']),
        ('去除思考后有可见教师内容', c['assistant_visible_nonempty']),
        ('消息字段无法读取', c['messages_unknown']),
    ])
    output_table('教师轮数分布（记录字段 num_turns）', [(label, c[key]) for label, key in (
        ('0 轮','turn_0'), ('1 轮','turn_1'), ('2–4 轮','turn_2_4'),
        ('5–9 轮','turn_5_9'), ('10 轮及以上','turn_10_plus'), ('未知','turn_unknown'))])
    output_table('两项门槛判决组合', [
        (f'不泄露={left}，知识正确={right}', j[(left,right)])
        for left in ('OK','REJECT','OTHER') for right in ('OK','REJECT','OTHER')
    ])
    statuses = count_server(work / 'judge.log')
    if statuses is None:
        print('\n裁判 HTTP 日志：未找到或无法读取')
    else:
        output_table('裁判 HTTP 响应行（含一次虚构测试，可能含重试）',
                     [(f'HTTP {status}', statuses[status]) for status in sorted(statuses)] + [('合计', sum(statuses.values()))])
    trainer = work / 'trainer.log'
    trainer_paths = [trainer] if trainer.is_file() else []
    trainer_counts, trainer_unread = count_logfiles(trainer_paths)
    rpaths, rtotal, rstate = ray_logs(root, work)
    rc, runread = count_logfiles(rpaths)
    output_table('日志扫描范围（各组可能有重复，不可相加）', [
        ('主训练日志文件数', len(trainer_paths)), ('主训练日志不可读文件数', trainer_unread),
        ('Ray 日志扫描文件数', len(rpaths)), ('Ray 日志发现文件数', rtotal),
        ('Ray 日志不可读文件数', runread), ('对话文件不可读数', n['unreadable_files']),
    ])
    print('Ray 会话：', rstate)
    output_table('主训练日志的裁判异常标记（行数）', [(name, trainer_counts[name]) for name, _ in PATTERNS])
    output_table('Ray worker 日志的裁判异常标记（行数）', [(name, rc[name]) for name, _ in PATTERNS])
    print('\n提示：日志行数不是唯一请求数；重试、跨文件复制与未捕获的错误会影响比较。')
    print('此检查只读取既有文件；不输出题目、答案、回复或原始日志。')


if __name__ == '__main__':
    main()
