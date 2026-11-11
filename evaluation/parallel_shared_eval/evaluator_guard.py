#!/usr/bin/env python3
"""Run one difficulty in its own process with conservative source recovery.

Frozen app code supplies all generation/protocol logic. Existing partial/held
sources are not retried. Output consists solely of aggregate counters.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def inspect_source(stage_dir, row):
    key = digest(row['source_id'].encode())[:24]
    folder = stage_dir / 'sources' / key
    if folder.is_symlink():
        raise ValueError('source symlink')
    if not folder.exists():
        return 'new'
    if not folder.is_dir():
        raise ValueError('source path')
    complete, held = folder / 'RAW_COMPLETE.json', folder / 'SOURCE_HELD.json'
    if complete.is_symlink() or held.is_symlink():
        raise ValueError('marker symlink')
    if complete.exists() and held.exists():
        raise ValueError('conflicting source markers')
    if held.exists():
        return 'held'
    if not complete.exists():
        children = list(folder.iterdir())
        if len(children) == 1 and children[0].name == 'SOURCE.json' and not children[0].is_symlink():
            source = json.loads(children[0].read_bytes())
            if source.get('source_id') != row['source_id'] or source.get('sha256') != digest(stable(row)):
                raise ValueError('source-only binding mismatch')
            # No request or dialogue intent exists, so resuming cannot repeat
            # any previously sent teacher/student request.
            return 'source_only'
        return 'partial'
    marker = json.loads(complete.read_bytes())
    source = json.loads((folder / 'SOURCE.json').read_bytes())
    expected_hash = digest(stable(row))
    if (marker.get('source_id') != row['source_id'] or marker.get('samples') != 96 or
        marker.get('source_sha256') != expected_hash or
        source.get('source_id') != row['source_id'] or source.get('sha256') != expected_hash):
        raise ValueError('completed source binding')
    targets = ('source', row['transfer_target'])
    expected = {(t, p, i) for t in targets for p in ('pre', 'control', 'post') for i in range(16)}
    entries = json.loads((folder / 'SCORE_INPUT.json').read_bytes())
    if not isinstance(entries, list) or len(entries) != 96:
        raise ValueError('completed source inputs')
    actual = {(r.get('target'), r.get('phase'), r.get('index')) for r in entries
              if isinstance(r, dict) and r.get('source_id') == row['source_id']}
    if actual != expected:
        raise ValueError('completed source identities')
    responses = list((folder / 'samples').glob('*/*/*/RESPONSE.json'))
    if len(responses) != 96:
        raise ValueError('completed source responses')
    return 'complete'


async def run(rows, stage_dir, evaluator, teacher_model, teacher_url, student_url, max_new=0, on_attempt=None):
    counts = {'complete': 0, 'new_complete': 0, 'held': 0, 'partial': 0, 'invalid': 0, 'failed': 0}
    attempted = 0
    student = evaluator.openai.AsyncClient(api_key='loopback-only', base_url=evaluator.local_url(student_url),
                                            timeout=300, max_retries=0)
    try:
        for row in rows:
            try:
                state = inspect_source(stage_dir, row)
            except (OSError, ValueError, TypeError, KeyError):
                # Bad source data is neither silently accepted nor altered; a
                # single source must not prevent later independent sources.
                counts['invalid'] += 1
                continue
            if state not in ('new', 'source_only'):
                counts[state] += 1
                continue
            if max_new and attempted >= max_new:
                break
            attempted += 1
            if on_attempt:
                on_attempt(digest(row['source_id'].encode())[:24])
            try:
                await evaluator.run_one(row, str(teacher_model), evaluator.local_url(teacher_url),
                                        evaluator.local_url(student_url), student)
            except Exception as exc:
                folder = evaluator.source_dir(row)
                marker = folder / 'SOURCE_HELD.json'
                if not marker.exists() and not (folder / 'RAW_COMPLETE.json').exists():
                    evaluator.put_once(marker, {'source_id': row['source_id'], 'status': 'held',
                                                'error_type': type(exc).__name__})
                counts['failed'] += 1
                continue
            try:
                state = inspect_source(stage_dir, row)
            except (OSError, ValueError, TypeError, KeyError):
                state = 'invalid'
            if state != 'complete':
                counts['invalid'] += 1
            else:
                counts['new_complete'] += 1
    finally:
        await student.close()
    return counts


def main():
    p = argparse.ArgumentParser()
    for name in ('repo', 'source', 'results', 'teacher-model', 'teacher-url', 'student-url', 'difficulty'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--max-new', type=int, default=0)
    args = p.parse_args()
    paths = [Path(args.repo), Path(args.source), Path(args.results), Path(args.teacher_model)]
    if (any(not path.is_absolute() for path in paths) or args.difficulty not in ('easy', 'medium', 'hard')
            or not 0 <= args.max_new <= 1):
        raise ValueError('invalid args')
    os.environ['RESULTS_DIR'] = args.results
    os.environ['LOCAL_ONLY_KEY'] = 'loopback-only'
    sys.path.insert(0, str(Path(args.repo) / 'app'))
    sys.path.insert(0, str(Path(args.repo) / 'official'))
    import offline_eval as evaluator
    rows = evaluator.parse_sources(args.source, args.difficulty)
    selected = []
    result = asyncio.run(run(rows, Path(args.results), evaluator, Path(args.teacher_model),
                             args.teacher_url, args.student_url, args.max_new, selected.append))
    output = {'state': 'DONE', 'counts': result}
    if args.max_new:
        output['pilot'] = {'source_key': selected[0] if selected else None,
                           'state': ('NO_NEW' if not selected else
                                     'COMPLETE' if result['new_complete'] == 1 else 'FAILED')}
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'ERROR', 'code': 'EVALUATOR_GUARD_FAILED'}), file=sys.stderr, flush=True)
        sys.exit(1)
