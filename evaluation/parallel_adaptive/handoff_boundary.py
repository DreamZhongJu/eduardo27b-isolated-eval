"""Conservative, read-only predicates for a proposed source-boundary handoff.

Not a stop/signal tool and not connected to any supervisor.  Call only while
all writers for the inspected stage are independently known to be quiescent.
A positive answer alone is never permission to stop or restart processes.
"""
import hashlib
import json
import re
from pathlib import Path

STAGES = ('EASY', 'MEDIUM', 'HARD')


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('invalid file')
    return json.loads(path.read_bytes())


def stage_quiescent(root, stage):
    """Refuse ambiguous/partial and held records, even if held looks old.

    Completed records must have a compatible marker and source binding. A
    SOURCE-only directory is safe only because the frozen evaluator writes
    each INTENT before sending any teacher or student request.  When this
    predicate runs the evaluator must be stopped, or scanning is racy.
    """
    if stage not in STAGES:
        raise ValueError('invalid stage')
    base = Path(root) / stage / 'sources'
    if base.is_symlink() or not base.is_dir():
        raise ValueError('invalid sources directory')
    for folder in base.iterdir():
        if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r'[0-9a-f]{24}', folder.name):
            raise ValueError('invalid source directory')
        files = {p.name for p in folder.iterdir()}
        if not files:
            return False
        if 'SOURCE_HELD.json' in files:
            # asyncio.gather may leave other model calls in flight when one
            # sample raises and the evaluator has already written HELD.
            return False
        if 'SOURCE.json' not in files:
            return False
        src = _read(folder / 'SOURCE.json')
        source_id, source_hash = src.get('source_id'), src.get('sha256')
        if (not isinstance(source_id, str) or not isinstance(source_hash, str) or
                not re.fullmatch(r'[0-9a-f]{64}', source_hash) or
                hashlib.sha256(source_id.encode()).hexdigest()[:24] != folder.name):
            raise ValueError('source binding')
        if files == {'SOURCE.json'}:
            continue
        if 'RAW_COMPLETE.json' not in files:
            return False
        marker = _read(folder / 'RAW_COMPLETE.json')
        if (marker.get('source_id') != source_id or marker.get('source_sha256') != source_hash
                or marker.get('samples') != 96 or 'SCORE_INPUT.json' not in files):
            raise ValueError('completion binding')
    return True


def grader_quiescent(root):
    """Detect incomplete API attempts when every grader writer is stopped.

    Does not cover an unknown grader outside this receipt tree; the caller
    must independently prove process identity and exclusive writer ownership.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('invalid receipts')
    for intent in root.rglob('ATTEMPT-????-INTENT.json'):
        parents = list(intent.parents)
        if intent.is_symlink() or any(p.is_symlink() for p in parents if p != root and root in p.parents):
            raise ValueError('invalid intent')
        number = intent.name[8:12]
        doc = _read(intent)
        if doc.get('attempt') != int(number):
            raise ValueError('intent binding')
        response = intent.with_name(f'ATTEMPT-{number}-RESPONSE.json')
        if not response.exists():
            return False
        outcome = _read(response)
        if outcome.get('attempt') != int(number):
            raise ValueError('response binding')
    return True
