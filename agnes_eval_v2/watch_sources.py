#!/usr/bin/env python3
"""Poll station-local completed sources; send only approved grading fields."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path

import agnes_judge as judge

running = True


def stop(*_):
    global running
    running = False


def check_own_source(rows, marker, folder):
    """Reconstruct the frozen selected row when old public markers lack input hash."""
    source_path = folder / 'SOURCE.json'
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError('source receipt missing')
    source = judge.read_json(source_path)
    if (marker.get('samples') != 96 or marker.get('grading_status') != 'ungraded' or
            source.get('source_id') != marker.get('source_id') or
            source.get('sha256') != marker.get('source_sha256')):
        raise ValueError('source receipt mismatch')
    source_rows = [r for r in rows if r.get('target') == 'source']
    transfer_rows = [r for r in rows if r.get('target') != 'source']
    if len(source_rows) != 48 or len(transfer_rows) != 48:
        raise ValueError('wrong target sample count')
    first, transfer = source_rows[0], transfer_rows[0]
    target = transfer.get('target')
    if target not in ('ours_easy', 'ours_medium', 'ours_hard'):
        raise ValueError('unsupported public marker target')
    for target_rows in (source_rows, transfer_rows):
        if {(r.get('phase'), r.get('index')) for r in target_rows} != {
                (phase, i) for phase in ('pre', 'control', 'post') for i in range(16)}:
            raise ValueError('sample identities mismatch')
        sample = target_rows[0]
        if any(r.get('target') != sample.get('target') or r.get('problem') != sample.get('problem')
               or r.get('reference_answer') != sample.get('reference_answer') for r in target_rows):
            raise ValueError('target inputs mismatch')
    fields = ('candidate_id', 'difficulty', 'position_by_student_correct', 'historical_student_correct',
              'historical_student_n_scored', 'historical_group_score_tie')
    if any(any(r.get(k) != transfer.get(k) for k in fields) for r in transfer_rows):
        raise ValueError('transfer metadata mismatch')
    if transfer.get('difficulty') != target[5:]:
        raise ValueError('transfer difficulty mismatch')
    selected = {'source_id': marker['source_id'], 'problem': first['problem'],
                'answer': first['reference_answer'], 'transfer_problem': transfer['problem'],
                'transfer_answer': transfer['reference_answer'], 'transfer_target': target,
                'transfer_candidate_id': transfer['candidate_id'],
                'transfer_position': transfer['position_by_student_correct'],
                'student_correct': transfer['historical_student_correct'],
                'student_n_scored': transfer['historical_student_n_scored'],
                'score_tie': transfer['historical_group_score_tie']}
    if judge.digest(selected) != marker.get('source_sha256'):
        raise ValueError('selected source content mismatch')


def process_source(folder, output_root, receipts_root, locks, key, caller=judge.api_call, workers=5, window=None):
    if folder.is_symlink() or not re.fullmatch(r'[0-9a-f]{24}', folder.name):
        raise ValueError('invalid source directory')
    marker_path, input_path = folder/'RAW_COMPLETE.json', folder/'SCORE_INPUT.json'
    if not marker_path.is_file() or not input_path.is_file():
        return False
    if marker_path.is_symlink() or input_path.is_symlink():
        raise ValueError('symlink input refused')
    marker = judge.read_json(marker_path)
    raw = input_path.read_bytes()
    if marker.get('score_input_sha256') is not None and marker.get('score_input_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('score input hash mismatch')
    rows = json.loads(raw)
    if (not isinstance(rows, list) or not rows or marker.get('samples') != len(rows) or
            not isinstance(marker.get('source_id'), str) or
            any(r.get('source_id') != marker['source_id'] for r in rows)):
        raise ValueError('source marker mismatch')
    source_key = hashlib.sha256(marker['source_id'].encode()).hexdigest()[:24]
    if folder.name != source_key:
        raise ValueError('source directory identity mismatch')
    if len({judge.digest(judge.identity(row)) for row in rows}) != len(rows):
        raise ValueError('duplicate identity')
    if marker.get('score_input_sha256') is None:
        check_own_source(rows, marker, folder)
    for row in rows:
        judge.payload(row)
    output_dir = output_root / 'sources' / source_key
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path = output_dir / 'RESULTS.json'
    input_hash = hashlib.sha256(raw).hexdigest()
    if output_path.exists():
        previous = judge.read_json(output_path)
        if previous.get('input_sha256') != input_hash or previous.get('source_id') != marker['source_id']:
            raise ValueError('source output identity mismatch')
        return True
    if not 1 <= workers <= 5:
        raise ValueError('invalid worker count')
    if window is None:
        window = judge.AdaptiveWindow(workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda row: judge.grade_one(row, key, receipts_root / source_key,
                                                           locks, caller=caller, window=window), rows))
    doc = {'source_id': marker['source_id'], 'input_sha256': input_hash,
           'grader': 'agnes_full_response_v2', 'model': judge.MODEL, 'results': results}
    try:
        judge.put_once(output_path, doc)
    except FileExistsError:
        # Concurrent watchers may finish the same source; verify rather than overwrite.
        if judge.read_json(output_path) != doc:
            raise ValueError('conflicting source result')
    counts = {s: sum(r['status'] == s for r in results) for s in ('correct', 'incorrect', 'unknown')}
    print(json.dumps({'state': 'LOCAL_AGNES_SOURCE_DONE', **counts}), flush=True)
    return True


def main():
    os.umask(0o077)
    p = judge.SafeParser(add_help=False, prog='LOCAL_AGNES_WATCH')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--sources-dir', type=Path)
    group.add_argument('--stages-root', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--receipts-dir', required=True, type=Path)
    p.add_argument('--lock-dir', required=True, type=Path)
    p.add_argument('--key-file', required=True, type=Path)
    p.add_argument('--poll-seconds', type=int, default=15)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    root = args.sources_dir if args.sources_dir is not None else args.stages_root
    paths = (root, args.output_dir, args.receipts_dir, args.lock_dir, args.key_file)
    if not all(x.is_absolute() for x in paths) or not root.is_dir() or not args.lock_dir.is_dir():
        raise ValueError('invalid paths')
    if not 1 <= args.poll_seconds <= 3600 or not 1 <= args.workers <= 5:
        raise ValueError('invalid poll interval')
    key = args.key_file.read_text(encoding='utf-8').strip()
    if not key:
        raise ValueError('missing key')
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.receipts_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    seen_errors = set()
    window = judge.AdaptiveWindow(args.workers)
    print(json.dumps({'state': 'LOCAL_AGNES_WATCH_RUNNING'}), flush=True)
    while running:
        scan = [(None, args.sources_dir)] if args.sources_dir is not None else [
            (stage, args.stages_root / stage / 'sources') for stage in ('EASY', 'MEDIUM', 'HARD')]
        for stage, source_dir in scan:
            if not source_dir.is_dir() or source_dir.is_symlink():
                continue
            out = args.output_dir if stage is None else args.output_dir / stage
            receipts = args.receipts_dir if stage is None else args.receipts_dir / stage
            for folder in sorted(source_dir.iterdir()):
                if not running:
                    break
                seen_key = (stage, folder.name)
                if not folder.is_dir() or folder.is_symlink() or seen_key in seen_errors:
                    continue
                held = out / 'sources' / folder.name / 'HELD.json'
                if held.is_file():
                    continue
                try:
                    process_source(folder, out, receipts, args.lock_dir, key, workers=args.workers, window=window)
                except Exception as exc:
                    # Leave receipts in place for a deliberate safe recovery; don't re-send blindly.
                    seen_errors.add(seen_key)
                    if re.fullmatch(r'[0-9a-f]{24}', folder.name):
                        try:
                            judge.put_once(held, {'state': 'held', 'error_type': type(exc).__name__})
                        except FileExistsError:
                            pass
                    print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_AGNES_SOURCE_HELD'}), flush=True)
        if args.once:
            break
        for _ in range(args.poll_seconds):
            if not running:
                break
            time.sleep(1)
    print(json.dumps({'state': 'LOCAL_AGNES_WATCH_STOPPED'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_AGNES_WATCH_FAILED'}), flush=True)
        raise SystemExit(1)
