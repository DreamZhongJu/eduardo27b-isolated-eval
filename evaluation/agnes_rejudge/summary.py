#!/usr/bin/env python3
"""Read-only aggregate of original, recovered and rejudged sample decisions.

Does not call APIs or write results. Output is fixed Chinese aggregate labels.
Missing, conflicting or unverified records remain unknown, never guessed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path('/workspace/results/own-test-499-v1')
OVERLAY = Path('/workspace/results/agnes-topbool-recovery-v1')
DECISIONS = Path('/workspace/results/agnes-rejudge-v1')
INVENTORY_SHA = '2e35edbc6105c99bd044347673ec4735061da1091149033a49077d803d090f78'
SCHEMA = 'standalone-rejudge-v1'
STAGES = ('PILOT', 'EASY', 'MEDIUM', 'HARD')
GROUP = {'PILOT':'简单', 'EASY':'简单', 'MEDIUM':'中等', 'HARD':'困难'}
PHASES = ('pre', 'control', 'post')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def safe_bytes(path, base):
    base, path = Path(base), Path(path)
    parts = path.relative_to(base).parts
    if base.is_symlink():
        raise ValueError('invalid root')
    for part in parts:
        base = base / part
        if base.is_symlink():
            raise ValueError('symlink refused')
    return path.read_bytes()


def read_json(path, base):
    raw = safe_bytes(path, base)
    return raw, json.loads(raw)


def load_inventory(path):
    if not path.is_absolute() or path.is_symlink() or sha(path.read_bytes()) != INVENTORY_SHA:
        raise ValueError('inventory binding')
    spec = importlib.util.spec_from_file_location('local_inventory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MODEL != 'agnes-3.0-flash':
        raise ValueError('model binding')
    return module


def strict_json(text):
    def parse_pairs(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('duplicate key')
            out[k] = v
        return out
    return json.loads(text, object_pairs_hook=parse_pairs)


def verdict(raw, model):
    try:
        if raw.get('model') != model or raw['choices'][0]['finish_reason'] != 'stop':
            return None
        text = raw['choices'][0]['message']['content'].strip()
        if text.startswith('```') and text.endswith('```'):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].strip().lower() in ('```', '```json') and lines[-1].strip() == '```':
                text = '\n'.join(lines[1:-1]).strip()
        result = strict_json(text)
        if type(result) is bool:
            return result
        if isinstance(result, dict) and set(result) == {'correct'} and type(result['correct']) is bool:
            return result['correct']
    except (AttributeError, TypeError, KeyError, IndexError, ValueError):
        pass
    return None


def verified_rejudge(inv, dest, stage, key, digest, original, decision):
    """Require a matching final result AND a matching successful API receipt."""
    rounds = decision.get('rounds')
    if type(rounds) is not int or not 1 <= rounds <= 3:
        return False
    wanted = decision['status']
    root = dest / 'receipts' / stage / key / f'round-{rounds}'
    if not root.is_dir():
        return False
    # Epochs are shared by every sample within the source. Select the latest
    # epoch that has an entry for *this* identity; a newer epoch belonging
    # exclusively to another sample must not invalidate this one's receipt.
    # If this identity has a newer incomplete epoch, retain it here so the
    # missing RESULT fails closed rather than silently using an older result.
    epochs = sorted(p for p in root.iterdir() if re.fullmatch(r'auth-\d{4}', p.name)
                    and ((p / digest).exists() or (p / digest).is_symlink()))
    if not epochs:
        return False
    folder = epochs[-1] / digest
    _, result = read_json(folder / 'RESULT.json', dest)
    for field in ('identity', 'request_sha256', 'model', 'grader'):
        if result.get(field) != original.get(field):
            return False
    if result.get('status') != wanted:
        return False
    responses = sorted(folder.glob('ATTEMPT-????-RESPONSE.json'))
    intents = sorted(folder.glob('ATTEMPT-????-INTENT.json'))
    numbers = {int(p.name[8:12]) for p in responses}
    if (not numbers or numbers != {int(p.name[8:12]) for p in intents} or
            numbers != set(range(1, max(numbers) + 1)) or max(numbers) > 4):
        return False
    for n in sorted(numbers):
        _, intent = read_json(folder / f'ATTEMPT-{n:04d}-INTENT.json', dest)
        _, response = read_json(folder / f'ATTEMPT-{n:04d}-RESPONSE.json', dest)
        if intent != {'attempt': n, 'request_sha256': original['request_sha256']} or response.get('attempt') != n:
            return False
        if n != max(numbers) and response.get('retryable') is not True:
            return False
    return (response.get('http_status') == 200 and
            verdict(response.get('raw_response'), inv.MODEL) is (wanted == 'correct'))


def load_source(inv, root, overlay, decisions, stage, key):
    src = root / stage / 'sources' / key
    score_raw, rows = read_json(src / 'SCORE_INPUT.json', root)
    report_raw, report = read_json(root / 'AGNES' / stage / 'sources' / key / 'RESULTS.json', root)
    _, marker = read_json(src / 'RAW_COMPLETE.json', root)
    _, source_record = read_json(src / 'SOURCE.json', root)
    if not all(isinstance(doc, dict) for doc in (report, marker, source_record)):
        raise ValueError('source documents')
    source_id = report.get('source_id')
    if (not isinstance(source_id, str) or not source_id or sha(source_id.encode())[:24] != key or
            marker.get('source_id') != source_id or marker.get('samples') != 96 or
            marker.get('grading_status') != 'ungraded' or
            source_record.get('source_id') != source_id or source_record.get('sha256') != marker.get('source_sha256') or
            (marker.get('score_input_sha256') is not None and marker['score_input_sha256'] != sha(score_raw)) or
            report.get('input_sha256') != sha(score_raw) or report.get('grader') != 'agnes_full_response_v2' or
            report.get('model') != inv.MODEL or not isinstance(rows, list) or len(rows) != 96 or
            not isinstance(report.get('results'), list) or len(report['results']) != 96):
        raise ValueError('source binding')
    expected_target = 'ours_' + ('easy' if stage in ('PILOT', 'EASY') else stage.lower())
    expected = {(t, p, n) for t in ('source', expected_target) for p in PHASES for n in range(16)}
    originals = {}
    entries = {}
    for row in rows:
        ident = inv.identity(row)
        token = inv.digest({'identity': ident})
        coord = (ident['target'], ident['phase'], ident['index'])
        if ident['source_id'] != source_id or coord not in expected or token in entries:
            raise ValueError('input identity')
        entries[token] = row
    if {(r['target'], r['phase'], r['index']) for r in entries.values()} != expected:
        raise ValueError('missing sample')
    source_rows = [r for r in rows if r['target'] == 'source']
    transfer_rows = [r for r in rows if r['target'] == expected_target]
    first, transfer = source_rows[0], transfer_rows[0]
    for target_rows in (source_rows, transfer_rows):
        first_row = target_rows[0]
        if any(r.get('problem') != first_row.get('problem') or
               r.get('reference_answer') != first_row.get('reference_answer') for r in target_rows):
            raise ValueError('inconsistent target')
    fields = ('candidate_id', 'difficulty', 'position_by_student_correct', 'historical_student_correct',
              'historical_student_n_scored', 'historical_group_score_tie')
    if (any(any(r.get(k) != transfer.get(k) for k in fields) for r in transfer_rows) or
            transfer.get('difficulty') != expected_target[5:]):
        raise ValueError('transfer metadata')
    selected = {'source_id': source_id, 'problem': first['problem'], 'answer': first['reference_answer'],
                'transfer_problem': transfer['problem'], 'transfer_answer': transfer['reference_answer'],
                'transfer_target': expected_target, 'transfer_candidate_id': transfer['candidate_id'],
                'transfer_position': transfer['position_by_student_correct'],
                'student_correct': transfer['historical_student_correct'],
                'student_n_scored': transfer['historical_student_n_scored'],
                'score_tie': transfer['historical_group_score_tie']}
    if inv.digest(selected) != marker.get('source_sha256'):
        raise ValueError('selected source binding')
    for result in report['results']:
        ident = inv.identity(result.get('identity'))
        token = inv.digest({'identity': ident})
        if (token in originals or token not in entries or result.get('status') not in ('correct','incorrect','unknown') or
                result.get('request_sha256') != inv.digest(inv.payload(entries[token])) or
                result.get('model') != inv.MODEL or result.get('grader') != 'agnes_full_response_v2'):
            raise ValueError('result identity')
        originals[token] = result
    if set(originals) != set(entries):
        raise ValueError('result coverage')

    recoveries = {}
    unresolved = any(r['status'] == 'unknown' for r in originals.values())
    errors = Counter()
    recovery_valid = True
    if unresolved:
        try:
            candidate = inv.source_overlay(root, stage, key)
            path = overlay / stage / key / 'OVERLAY.json'
            if candidate['recoveries']:
                _, old = read_json(path, overlay)
                if old != candidate:
                    raise ValueError('overlay mismatch')
            elif path.exists():
                _, old = read_json(path, overlay)
                if old != candidate:
                    raise ValueError('overlay mismatch')
            recoveries = {inv.digest({'identity': x['identity']}):x['decision']
                          for x in candidate['recoveries']}
        except (OSError, ValueError, KeyError, TypeError):
            recovery_valid = False
            errors['叠层未能核验来源'] += 1

    counts = {('原题' if t == 'source' else '迁移题', p):Counter()
              for t in ('source', expected_target) for p in PHASES}
    for token, original in originals.items():
        ident = original['identity']
        category = ('原题' if ident['target'] == 'source' else '迁移题', ident['phase'])
        status = original['status']
        if status == 'unknown' and recovery_valid:
            chosen = recoveries.get(token)
            candidate_status = ('correct' if chosen else 'incorrect') if type(chosen) is bool else None
            path = decisions / 'decisions' / stage / key / token / 'DECISION.json'
            if path.exists():
                try:
                    _, decision = read_json(path, decisions)
                    binding = {'schema':SCHEMA, 'stage':stage, 'source_key':key,
                               'identity':ident, 'request_sha256':original['request_sha256'],
                               'source_result_sha256':sha(report_raw), 'input_sha256':sha(score_raw)}
                    if any(decision.get(k) != v for k, v in binding.items()) or decision.get('status') not in ('correct','incorrect','unknown'):
                        raise ValueError('decision binding')
                    ds, method = decision['status'], decision.get('method')
                    if candidate_status is not None:
                        if ds != candidate_status or method != 'verified_topbool_overlay' or decision.get('rounds') != 0:
                            raise ValueError('conflicting decision')
                    elif ds != 'unknown':
                        if method != 'agnes_rejudge' or not verified_rejudge(inv, decisions, stage, key, token, original, decision):
                            raise ValueError('unverified decision')
                        candidate_status = ds
                    elif method != 'agnes_rejudge':
                        raise ValueError('invalid unknown')
                except (OSError, ValueError, KeyError, TypeError):
                    errors['补判记录异常作答'] += 1
                    candidate_status = None
            if candidate_status is not None:
                status = candidate_status
        counts[category][status] += 1
    # Refuse a snapshot that changed during scanning.
    if (safe_bytes(src / 'SCORE_INPUT.json', root) != score_raw or
        safe_bytes(root / 'AGNES' / stage / 'sources' / key / 'RESULTS.json', root) != report_raw):
        raise ValueError('source changed')
    return source_id, counts, errors


def collect(inv, root, overlay, decisions):
    if (not root.is_absolute() or not root.is_dir() or root.is_symlink() or
            not overlay.is_absolute() or overlay.is_symlink() or not overlay.is_dir() or
            not decisions.is_absolute() or decisions.is_symlink() or
            (decisions.exists() and not decisions.is_dir())):
        raise ValueError('invalid roots')
    groups = {g:Counter() for g in ('简单','中等','困难')}
    observed = {g:set() for g in groups}
    candidates = []
    for stage in STAGES:
        stats = groups[GROUP[stage]]
        folder = root / stage / 'sources'
        if not folder.is_dir():
            stats['来源目录缺失'] += 1
            continue
        if folder.is_symlink():
            stats['来源目录异常'] += 1
            continue
        for source in sorted(folder.iterdir()):
            key = source.name
            try:
                inv.no_symlink(source, root)
                if not source.is_dir() or not re.fullmatch(r'[0-9a-f]{24}', key):
                    stats['来源结构异常'] += 1
                    continue
                observed[GROUP[stage]].add(key)
                marker, held = source/'RAW_COMPLETE.json', source/'SOURCE_HELD.json'
                inv.no_symlink(marker, root)
                inv.no_symlink(held, root)
                if marker.is_file() and held.is_file():
                    stats['来源结构异常'] += 1
                elif held.is_file():
                    stats['暂停'] += 1
                elif not marker.is_file():
                    stats['不完整'] += 1
                elif not (root/'AGNES'/stage/'sources'/key/'RESULTS.json').is_file():
                    stats['无判分记录'] += 1
                else:
                    stats['完成标记'] += 1
                    try:
                        source_id, counts, errors = load_source(inv, root, overlay, decisions, stage, key)
                        candidates.append((GROUP[stage], source_id, counts, errors))
                    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
                        stats['判分或输入异常'] += 1
            except (OSError, ValueError, TypeError):
                stats['来源读取异常'] += 1
    for group in groups:
        groups[group]['已发现来源'] = len(observed[group])
        groups[group]['未找到来源'] = max(0, 499 - len(observed[group]))
    duplicates = Counter((group, source_id) for group, source_id, _, _ in candidates)
    totals = {g:{(target,phase):Counter() for target in ('原题','迁移题') for phase in PHASES}
              for g in groups}
    for group, source_id, counts, errors in candidates:
        stats = groups[group]
        if duplicates[(group, source_id)] != 1:
            stats['重复来源记录'] += 1
            continue
        stats['纳入来源'] += 1
        stats.update(errors)
        for category, outcome in counts.items():
            totals[group][category].update(outcome)
    return groups, totals


def table(headers, rows):
    import unicodedata
    def width(x):
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in str(x))
    cells = [tuple(map(str,headers))] + [tuple(map(str,row)) for row in rows]
    widths = [max(width(row[i]) for row in cells) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    def line(row):
        return '| ' + ' | '.join(c + ' ' * (widths[i] - width(c)) if i < 2 else ' ' * (widths[i] - width(c)) + c for i,c in enumerate(row)) + ' |'
    return '\n'.join([border,line(cells[0]),border] + [line(row) for row in cells[1:]] + [border])


def render(groups, totals):
    tiers = ('简单','中等','困难')
    fields = ('已发现来源','未找到来源','纳入来源','暂停','不完整','无判分记录',
              '判分或输入异常','叠层未能核验来源','补判记录异常作答','重复来源记录',
              '来源结构异常','来源读取异常','来源目录缺失','来源目录异常')
    source_rows = []
    for field in fields:
        values = [groups[g][field] for g in tiers]
        source_rows.append((field, *values, sum(values)))
    lines = ['来源与数据校验（每档预期499个来源）', table(('项目','简单','中等','困难','合计'), source_rows), '',
             '学生作答汇总（每个纳入来源每阶段每题16次；判分未决单列）']
    phase_zh = {'pre':'前测','control':'对照','post':'后测'}
    rows = []
    for g in tiers:
        for phase in PHASES:
            for target in ('原题','迁移题'):
                c = totals[g][(target,phase)]
                n = sum(c.values())
                rows.append((g, phase_zh[phase], target, n, c['correct'], c['incorrect'], c['unknown']))
    lines += [table(('难度','阶段','题目','作答数','判对','判错','未决'), rows), '',
              '说明：简单档包括 PILOT；只纳入完成且通过身份、样本和判分绑定核对的来源。',
              '来源数量仅依据目录与本地来源身份核验，未逐条对照冻结499清单，不证明官方名单完整覆盖。',
              '未决不算判错；未完成来源及无法校验记录未计入作答数。旧叠层和补判的数学判断未做人工确认。']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--overlay', type=Path, default=OVERLAY)
    p.add_argument('--decisions', type=Path, default=DECISIONS)
    p.add_argument('--inventory', type=Path, required=True)
    a = p.parse_args()
    try:
        if not all(path.is_absolute() for path in (a.root, a.overlay, a.decisions, a.inventory)):
            raise ValueError('absolute paths required')
        inv = load_inventory(a.inventory)
        groups, totals = collect(inv, a.root, a.overlay, a.decisions)
        print(render(groups, totals))
    except Exception:
        print('只读汇总失败，请检查本机目录和版本。')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
