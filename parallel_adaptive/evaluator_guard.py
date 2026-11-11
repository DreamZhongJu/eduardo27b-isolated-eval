#!/usr/bin/env python3
"""Source-boundary evaluator using two local student servers.

The same frozen evaluator, data, prompts, receipt names and sampling parameters
are used. Never start this writer while an old non-locking guard writes the
same stage. A controller must first enforce exclusive ownership of the entire
experiment, then start at most one guard per stage. SIGUSR1 requests a graceful
stop before the next source; no in-flight request is cancelled by this guard.
"""
import argparse
import asyncio
import fcntl
import json
import os
import signal
import sys
from pathlib import Path

from adaptive_base_guard import digest, inspect_source
from student_router import StudentRouter

draining = False


def drain(*_):
    global draining
    draining = True


async def run(rows, stage_dir, evaluator, teacher_model, teacher_url, student_url,
              student_url_secondary, max_new=0, on_attempt=None, router_factory=StudentRouter,
              should_drain=lambda: draining):
    counts = {'complete': 0, 'new_complete': 0, 'held': 0, 'partial': 0, 'invalid': 0, 'failed': 0}
    attempted = 0
    student = router_factory((student_url, student_url_secondary), slots_per_student=2)
    try:
        for row in rows:
            if should_drain():
                break
            try:
                state = inspect_source(stage_dir, row)
            except (OSError, ValueError, TypeError, KeyError):
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
                # The frozen dialogue uses the primary URL, while its student
                # samples call the router through the existing client API.
                await evaluator.run_one(row, str(teacher_model), evaluator.local_url(teacher_url),
                                        evaluator.local_url(student_url), student)
            except Exception as exc:
                folder = evaluator.source_dir(row)
                marker = folder / 'SOURCE_HELD.json'
                if not marker.exists() and not (folder / 'RAW_COMPLETE.json').exists():
                    evaluator.put_once(marker, {'source_id': row['source_id'], 'status': 'held',
                                                'error_type': type(exc).__name__})
                counts['failed'] += 1
                # Any request failure can leave sibling calls in flight. Stop
                # this guard rather than cascading held sources. Router.close
                # waits for dispatched requests to finish before exit.
                break
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
    for name in ('repo', 'source', 'results', 'teacher-model', 'teacher-url', 'student-url',
                 'student-url-secondary', 'difficulty'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--max-new', type=int, default=0)
    args = p.parse_args()
    paths = [Path(args.repo), Path(args.source), Path(args.results), Path(args.teacher_model)]
    if (any(not path.is_absolute() for path in paths) or args.difficulty not in ('easy', 'medium', 'hard')
            or not 0 <= args.max_new <= 1):
        raise ValueError('invalid args')
    # Stage lock only protects against cooperating new guards. The controller
    # must also guarantee no old non-locking worker is active before launch.
    result_dir = Path(args.results)
    result_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if result_dir.is_symlink():
        raise ValueError('symlink result directory')
    fd = os.open(result_dir / '.adaptive-evaluator.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGUSR1, drain)
        os.environ['RESULTS_DIR'] = args.results
        os.environ['LOCAL_ONLY_KEY'] = 'loopback-only'
        sys.path.insert(0, str(Path(args.repo) / 'app'))
        sys.path.insert(0, str(Path(args.repo) / 'official'))
        import offline_eval as evaluator
        rows = evaluator.parse_sources(args.source, args.difficulty)
        selected = []
        result = asyncio.run(run(rows, result_dir, evaluator, Path(args.teacher_model), args.teacher_url,
                                 args.student_url, args.student_url_secondary, args.max_new, selected.append))
        output = {'state': 'DONE', 'counts': result, 'drained': draining}
        if args.max_new:
            output['pilot'] = {'source_key': selected[0] if selected else None,
                               'state': ('NO_NEW' if not selected else
                                         'COMPLETE' if result['new_complete'] == 1 else 'FAILED')}
        print(json.dumps(output, sort_keys=True), flush=True)
    finally:
        os.close(fd)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'ERROR', 'code': 'ADAPTIVE_EVALUATOR_GUARD_FAILED'}), file=sys.stderr, flush=True)
        sys.exit(1)
