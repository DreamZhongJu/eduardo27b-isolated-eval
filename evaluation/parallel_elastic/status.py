#!/usr/bin/env python3
"""Read-only aggregate progress. Never emits paths, source IDs or contents."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path('/workspace/results/own-test-499-v1')
KEY_RE = re.compile(r'^[0-9a-f]{24}$')
HEX_RE = re.compile(r'^[0-9a-f]{64}$')
COUNTERS = ('sources_started', 'generation_complete', 'generation_held', 'generation_partial',
            'generation_invalid', 'grading_complete', 'grading_held', 'grading_pending', 'grading_invalid',
            'grading_orphan', 'correct', 'incorrect', 'unknown', 'read_errors')


def wait_phase(root):
    """Read fixed, aggregate supervisor events; never return arbitrary log text."""
    allowed = {'STARTING', 'VERIFYING_FROZEN_INPUTS', 'VERIFYING_MODEL_HASH', 'WAITING_FOR_FREE_GPUS',
               'GPU_READY', 'STARTING_SERVICES', 'PILOT_RUNNING', 'PILOT_GRADED',
               'PILOT_GRADING_PENDING', 'BULK_RUNNING', 'GENERATION_DONE_GRADING_PENDING',
               'SCANNED_COMPLETE', 'SCANNED_WITH_UNRESOLVED', 'ERROR'}
    allowed_steps = {'ROOT_LOCK', 'STAGE_LOCK', 'WORKER_CONFLICT', 'KEY', 'PIDFD',
                     'V4_CONTROLLER', 'V4_LAUNCHER', 'CODE_BINDING', 'DATA_BINDING',
                     'PILOT', 'ENVIRONMENT', 'MODEL_HASH', 'MODEL_METADATA', 'GPU_QUERY',
                     'GPU_VALIDATION', 'CUDA_DEVICE', 'PORTS'}
    path = root / 'shared-supervisor-elastic.log'
    try:
        if path.is_symlink():
            raise OSError('invalid log')
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 1000000))
            lines = f.read().decode('utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return {'state': 'not_recorded'}
    except OSError:
        return {'state': 'unreadable'}
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict) or value.get('state') not in allowed:
            continue
        events.append(value)
    if events:
        value = events[-1]
        result = {'state': value['state']}
        if value.get('step') in allowed_steps:
            result['step'] = value['step']
        # Selection is reported only after a GPU_READY in this invocation and
        # after the last wait. Later re-selection replaces the previous count.
        selected = None
        for event in events:
            if event['state'] in ('STARTING', 'VERIFYING_FROZEN_INPUTS', 'WAITING_FOR_FREE_GPUS'):
                selected = None
            elif (event['state'] == 'GPU_READY' and type(event.get('available_gpus')) is int and
                  event['available_gpus'] in (2, 3, 4, 5)):
                selected = event['available_gpus']
        if selected is not None and value['state'] in ('GPU_READY', 'STARTING_SERVICES', 'PILOT_RUNNING',
                                                         'PILOT_GRADED', 'BULK_RUNNING'):
            result['selected_gpus'] = selected
        return result
    return {'state': 'not_recorded'}


def process_status(script, proc=Path('/proc')):
    wanted = os.fsencode(str(script))
    errors = 0
    try:
        candidates = list(proc.iterdir())
    except OSError:
        return {'state': 'unknown', 'unreadable_processes': 1}
    running = False
    for p in candidates:
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            errors += 1
            continue
        if wanted in args and b'--run' in args:
            running = True
    return {'state': 'running' if running else ('unknown' if errors else 'not_running'),
            'unreadable_processes': errors}


def folders(path):
    try:
        if path.is_symlink():
            raise OSError('symlink')
        return list(path.iterdir()), 0
    except FileNotFoundError:
        return [], 0
    except OSError:
        return [], 1


def safe_exists(path):
    try:
        if path.is_symlink():
            raise OSError('symlink')
        path.stat()
        return True
    except FileNotFoundError:
        return False


def read_json(path):
    if path.is_symlink():
        raise OSError('symlink')
    return json.loads(path.read_bytes())


def validate_grade(doc, source_id):
    if not isinstance(doc, dict) or doc.get('source_id') != source_id:
        return False
    rows = doc.get('results')
    if (not isinstance(doc.get('input_sha256'), str) or not HEX_RE.fullmatch(doc['input_sha256']) or
            not isinstance(rows, list) or len(rows) != 96):
        return False
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get('status') not in ('correct', 'incorrect', 'unknown'):
            return False
        ident = row.get('identity')
        if (not isinstance(ident, dict) or ident.get('source_id') != source_id or
                not isinstance(ident.get('target'), str) or not ident['target'] or
                ident.get('phase') not in ('pre', 'control', 'post') or
                type(ident.get('index')) is not int or not 0 <= ident['index'] < 16):
            return False
        key = (ident['target'], ident['phase'], ident['index'])
        if key in seen:
            return False
        seen.add(key)
    targets = {x[0] for x in seen}
    return len(targets) == 2 and 'source' in targets and len(seen) == 96


def stage_snapshot(root, stage):
    counts = {name: 0 for name in COUNTERS}
    files, errors = folders(root / stage / 'sources')
    counts['read_errors'] += errors
    source_keys = set()
    for folder in files:
        try:
            if not KEY_RE.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
                counts['generation_invalid'] += 1
                continue
            source_keys.add(folder.name)
            counts['sources_started'] += 1
            marker = folder / 'RAW_COMPLETE.json'
            held = folder / 'SOURCE_HELD.json'
            marker_exists = safe_exists(marker)
            held_exists = safe_exists(held)
            if marker_exists and held_exists:
                counts['generation_invalid'] += 1
                continue
            if held_exists:
                counts['generation_held'] += 1
                continue
            if not marker_exists:
                counts['generation_partial'] += 1
                continue
            try:
                raw = read_json(marker)
                if (not isinstance(raw, dict) or raw.get('samples') != 96 or
                        not isinstance(raw.get('source_id'), str) or not raw['source_id'] or
                        hashlib.sha256(raw['source_id'].encode()).hexdigest()[:24] != folder.name):
                    counts['generation_invalid'] += 1
                    continue
            except (OSError, ValueError, TypeError):
                counts['generation_invalid'] += 1
                continue
            counts['generation_complete'] += 1
            grade = root / 'AGNES' / stage / 'sources' / folder.name
            result, held_grade = grade / 'RESULTS.json', grade / 'HELD.json'
            has_result, has_held = safe_exists(result), safe_exists(held_grade)
            if has_result and has_held:
                counts['grading_invalid'] += 1
            elif has_held:
                counts['grading_held'] += 1
            elif not has_result:
                counts['grading_pending'] += 1
            else:
                try:
                    doc = read_json(result)
                    if not validate_grade(doc, raw['source_id']):
                        counts['grading_invalid'] += 1
                        continue
                    counts['grading_complete'] += 1
                    for item in doc['results']:
                        counts[item['status']] += 1
                except (OSError, ValueError, TypeError):
                    counts['grading_invalid'] += 1
        except OSError:
            counts['read_errors'] += 1
    grade_entries, errors = folders(root / 'AGNES' / stage / 'sources')
    counts['read_errors'] += errors
    for folder in grade_entries:
        if KEY_RE.fullmatch(folder.name) and folder.name not in source_keys:
            counts['grading_orphan'] += 1
    return counts


def snapshot(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError('results root unavailable')
    groups = {'easy': ('PILOT', 'EASY'), 'medium': ('MEDIUM',), 'hard': ('HARD',)}
    out = {}
    for name, stages in groups.items():
        parts = [stage_snapshot(root, stage) for stage in stages]
        out[name] = {key: sum(part[key] for part in parts) for key in COUNTERS}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--supervisor', type=Path, default=Path(__file__).resolve().with_name('supervisor.py'))
    args = parser.parse_args()
    if not args.root.is_absolute() or not args.supervisor.is_absolute():
        raise ValueError('absolute paths required')
    supervisor = process_status(args.supervisor)
    phase = wait_phase(args.root)
    if supervisor.get('state') != 'running':
        phase.pop('selected_gpus', None)
    output = {'supervisor': supervisor, 'phase': phase, 'stages': snapshot(args.root),
              'scope': 'read-only snapshot; complete means marker/structure, not full data validation'}
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'STATUS_ERROR'}))
        sys.exit(1)
