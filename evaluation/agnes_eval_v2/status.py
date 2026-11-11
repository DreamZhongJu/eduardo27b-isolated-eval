#!/usr/bin/env python3
"""Print only experiment status and aggregate counts."""
import json
from pathlib import Path

ROOT = Path('/workspace/results/own-test-499-v1')


def collect(stages):
    result = {'sources_started': 0, 'generation_complete': 0, 'generation_held': 0,
              'grading_complete': 0, 'grading_held': 0, 'correct': 0,
              'incorrect': 0, 'unknown': 0}
    for stage in stages:
        base = ROOT / stage / 'sources'
        for source in base.iterdir() if base.is_dir() else ():
            if not source.is_dir():
                continue
            result['sources_started'] += 1
            if (source / 'RAW_COMPLETE.json').is_file():
                result['generation_complete'] += 1
            elif (source / 'SOURCE_HELD.json').is_file():
                result['generation_held'] += 1
            grade = ROOT / 'AGNES' / stage / 'sources' / source.name
            if (grade / 'RESULTS.json').is_file():
                doc = json.loads((grade / 'RESULTS.json').read_text(encoding='utf-8'))
                result['grading_complete'] += 1
                for row in doc.get('results', []):
                    if row.get('status') in ('correct', 'incorrect', 'unknown'):
                        result[row['status']] += 1
            elif (grade / 'HELD.json').is_file():
                result['grading_held'] += 1
    return result


def main():
    state = (ROOT / 'STATUS').read_text(encoding='utf-8').strip() if (ROOT / 'STATUS').is_file() else 'NOT_STARTED'
    print(json.dumps({'state': state, 'easy': collect(('PILOT', 'EASY')),
                      'medium': collect(('MEDIUM',)), 'hard': collect(('HARD',))}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'STATUS_UNAVAILABLE'}))
