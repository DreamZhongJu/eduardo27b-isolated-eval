#!/usr/bin/env python3
"""Append-only, station-local rejudging of previously unknown student responses.

Run only after the evaluation controller has finished. Never writes to old results.
Only aggregate counts are printed; prompt inputs and individual receipts stay local.
"""
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

STAGES = ('PILOT', 'EASY', 'MEDIUM', 'HARD')
SCHEMA = 'standalone-rejudge-v1'
MAX_ROUNDS = 3
APPROVED_JUDGE = '2ffe746444939555b85e3557d2ddc3b61e3cd59d7997263909668876b924c871'
APPROVED_RECOVERY = '5dd59dade528c45b395563d35bee3b63d3112703d5a5561eb446cecf696eaffc'
DEFERRED_SCHEMA = 'agnes-technical-deferred-v1'
FORMAT_CYCLES = 2
FORMAT_FUSE = 3
FORMAT_ISSUES = frozenset({'response_not_json', 'response_truncated', 'response_structure_invalid',
                           'response_content_empty', 'response_duplicate_keys', 'response_schema_invalid'})


def retryable_format(response):
    """Only format failures already identified as retryable by the pinned judge."""
    return (isinstance(response, dict) and response.get('http_status') == 200 and
            response.get('retryable') is True and response.get('content_issue') in FORMAT_ISSUES)


CHECKPOINT_SCHEMA = 'agnes-format-circuit-checkpoint-v1'


def _bound_record(path, doc, schema):
    ident = doc.get('identity')
    if (not isinstance(ident, dict) or set(ident) != {'source_id', 'target', 'phase', 'index'} or
            any(not isinstance(ident[k], str) or not ident[k] for k in ('source_id', 'target', 'phase')) or
            type(ident['index']) is not int or not 0 <= ident['index'] < 16):
        return False
    raw = json.dumps({'identity': ident}, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    key = hashlib.sha256(ident['source_id'].encode()).hexdigest()[:24]
    return (doc.get('schema') == schema and path.parent.name == hashlib.sha256(raw).hexdigest() and
            path.parent.parent.name == key and doc.get('source_key') == key and
            path.parent.parent.parent.name in STAGES and doc.get('stage') == path.parent.parent.parent.name and
            all(isinstance(doc.get(k), str) and re.fullmatch(r'[0-9a-f]{64}', doc[k])
                for k in ('request_sha256', 'source_result_sha256', 'input_sha256')))


@contextlib.contextmanager
def format_lock(dest):
    """Serialize deferred markers and clear checkpoints across workers/processes."""
    directory = dest / 'control'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('invalid circuit directory')
    fd = os.open(directory / '.format.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _format_uncovered(dest):
    """Caller holds format_lock. Never use filesystem/system clock ordering."""
    records = {}
    base = dest / 'deferred'
    for path in base.rglob('DEFERRED.json'):
        raw, doc = read_json_bytes(path)
        if (not isinstance(doc, dict) or not _bound_record(path, doc, DEFERRED_SCHEMA) or
                doc.get('reason') != 'retryable_response_format' or
                type(doc.get('format_cycles')) is not int or not FORMAT_CYCLES <= doc['format_cycles'] <= 1000):
            raise ValueError('invalid technical record')
        key = '/'.join(path.relative_to(base).parts[:-1])
        records[key] = hashlib.sha256(raw).hexdigest()
    covered = set()
    for path in (dest / 'control' / 'format-checkpoints').glob('*.json'):
        _, doc = read_json_bytes(path)
        if (not isinstance(doc, dict) or doc.get('schema') != CHECKPOINT_SCHEMA or
                doc.get('reason') not in ('api_verdict', 'operator_reset') or not isinstance(doc.get('covered'), dict)):
            raise ValueError('invalid circuit checkpoint')
        if doc['reason'] == 'api_verdict':
            proof = doc.get('proof')
            if (not isinstance(proof, dict) or set(proof) != {'sample', 'decision_sha256'} or
                    not isinstance(proof['sample'], str) or not re.fullmatch(r'(PILOT|EASY|MEDIUM|HARD)/[0-9a-f]{24}/[0-9a-f]{64}', proof['sample']) or
                    not isinstance(proof['decision_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', proof['decision_sha256'])):
                raise ValueError('invalid circuit proof')
            decision_path = dest / 'decisions' / proof['sample'] / 'DECISION.json'
            raw, decision = read_json_bytes(decision_path)
            if (not isinstance(decision, dict) or not _bound_record(decision_path, decision, SCHEMA) or
                    decision.get('method') != 'agnes_rejudge' or decision.get('status') not in ('correct', 'incorrect') or
                    hashlib.sha256(raw).hexdigest() != proof['decision_sha256']):
                raise ValueError('invalid circuit proof')
        elif 'proof' in doc:
            raise ValueError('invalid reset proof')
        for key, digest in doc['covered'].items():
            if (not isinstance(key, str) or not re.fullmatch(r'(PILOT|EASY|MEDIUM|HARD)/[0-9a-f]{24}/[0-9a-f]{64}', key) or
                    not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or records.get(key) != digest):
                raise ValueError('checkpoint references invalid deferred record')
            covered.add(key)
    return {key: digest for key, digest in records.items() if key not in covered}


def format_circuit_open(dest):
    with format_lock(dest):
        return len(_format_uncovered(dest)) >= FORMAT_FUSE


def _write_format_checkpoint_locked(dest, j, reason, proof_path=None):
    uncovered = _format_uncovered(dest)
    if reason == 'api_verdict' and (not uncovered or len(uncovered) >= FORMAT_FUSE):
        return False
    doc = {'schema':CHECKPOINT_SCHEMA, 'reason':reason, 'covered':uncovered}
    if reason == 'api_verdict':
        raw, proof = read_json_bytes(proof_path)
        if (not isinstance(proof, dict) or not _bound_record(proof_path, proof, SCHEMA) or
                proof.get('method') != 'agnes_rejudge' or proof.get('status') not in ('correct', 'incorrect')):
            raise ValueError('invalid new verdict')
        doc['proof'] = {'sample':'/'.join(proof_path.relative_to(dest / 'decisions').parts[:-1]),
                        'decision_sha256':hashlib.sha256(raw).hexdigest()}
    elif reason != 'operator_reset':
        raise ValueError('invalid checkpoint reason')
    stamp = f'{time.time_ns()}-{os.getpid()}-{threading.get_ident()}.json'
    j.put_once(dest / 'control' / 'format-checkpoints' / stamp, doc)
    return True


def write_format_checkpoint(dest, j, reason, proof_path=None):
    with format_lock(dest):
        return _write_format_checkpoint_locked(dest, j, reason, proof_path)

class ItemBudget:
    """Bound newly attempted API-eligible samples, not old-overlay recovery."""
    def __init__(self, limit):
        if type(limit) is not int or limit < 0:
            raise ValueError('invalid item limit')
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            if self.limit and self.used >= self.limit:
                return False
            self.used += 1
            return True

    def reached(self):
        with self.lock:
            return bool(self.limit and self.used >= self.limit)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json_bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('file unavailable')
    b = path.read_bytes()
    return b, json.loads(b)


def load_judge(path, expected):
    if not path.is_absolute() or path.is_symlink() or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('judge path or hash')
    if sha(path.read_bytes()) != expected:
        raise ValueError('judge hash')
    spec = importlib.util.spec_from_file_location('local_approved_agnes_judge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_recovery(path, expected):
    if expected != APPROVED_RECOVERY:
        raise ValueError('unapproved recovery')
    if not path.is_absolute() or path.is_symlink() or sha(path.read_bytes()) != expected:
        raise ValueError('recovery path or hash')
    spec = importlib.util.spec_from_file_location('local_approved_topbool_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(j, root, stage, key):
    folder = root / stage / 'sources' / key
    if (folder / 'SOURCE_HELD.json').exists():
        raise ValueError('held source')
    score_path = folder / 'SCORE_INPUT.json'
    result_path = root / 'AGNES' / stage / 'sources' / key / 'RESULTS.json'
    score_bytes, rows = read_json_bytes(score_path)
    result_bytes, result = read_json_bytes(result_path)
    _, marker = read_json_bytes(folder / 'RAW_COMPLETE.json')
    _, source_record = read_json_bytes(folder / 'SOURCE.json')
    if not all(isinstance(doc, dict) for doc in (result, marker, source_record)):
        raise ValueError('source documents')
    source_id = result.get('source_id')
    if (not isinstance(rows, list) or not isinstance(source_id, str) or
        hashlib.sha256(source_id.encode()).hexdigest()[:24] != key or
        marker.get('source_id') != source_id or marker.get('samples') != 96 or
        marker.get('grading_status') != 'ungraded' or
        source_record.get('source_id') != source_id or
        source_record.get('sha256') != marker.get('source_sha256') or
        (marker.get('score_input_sha256') is not None and marker.get('score_input_sha256') != sha(score_bytes)) or
        result.get('input_sha256') != sha(score_bytes) or result.get('model') != j.MODEL or
        result.get('grader') != 'agnes_full_response_v2' or
        not isinstance(result.get('results'), list) or len(rows) != 96 or len(result['results']) != 96):
        raise ValueError('source identity')
    if stage not in STAGES:
        raise ValueError('invalid stage')
    expected_target = 'ours_' + ('easy' if stage in ('PILOT', 'EASY') else stage.lower())
    expected = {(t, p, i) for t in ('source', expected_target) for p in ('pre', 'control', 'post') for i in range(16)}
    inputs = {}
    for row in rows:
        ident = j.identity(row)
        coordinate = (ident['target'], ident['phase'], ident['index'])
        if ident['source_id'] != source_id or coordinate not in expected:
            raise ValueError('source mismatch')
        digest = j.digest({'identity': ident})
        if digest in inputs:
            raise ValueError('duplicate input')
        sample = folder / 'samples' / ident['target'] / ident['phase'] / f"{ident['index']:02d}"
        _, intent = read_json_bytes(sample / 'INTENT.json')
        _, receipt = read_json_bytes(sample / 'RESPONSE.json')
        if (not isinstance(intent, dict) or set(intent) != {'source_id', 'target', 'phase', 'index', 'request_sha256'} or
                not isinstance(intent.get('request_sha256'), str) or
                re.fullmatch(r'[0-9a-f]{64}', intent['request_sha256']) is None or
                not isinstance(receipt, dict) or receipt.get('identity') != intent or
                j.identity(intent) != ident or receipt.get('text') != row.get('student_response') or
                receipt.get('finish_reason') != row.get('finish_reason')):
            raise ValueError('sample receipt mismatch')
        inputs[digest] = row
    if {(r['target'], r['phase'], r['index']) for r in inputs.values()} != expected:
        raise ValueError('missing sample')
    source_rows = [r for r in rows if r['target'] == 'source']
    transfer_rows = [r for r in rows if r['target'] == expected_target]
    first, transfer = source_rows[0], transfer_rows[0]
    for group in (source_rows, transfer_rows):
        if any(row.get('problem') != group[0].get('problem') or
               row.get('reference_answer') != group[0].get('reference_answer') for row in group):
            raise ValueError('inconsistent target')
    fields = ('candidate_id', 'difficulty', 'position_by_student_correct', 'historical_student_correct',
              'historical_student_n_scored', 'historical_group_score_tie')
    if (any(any(row.get(field) != transfer.get(field) for field in fields) for row in transfer_rows) or
            transfer.get('difficulty') != expected_target[5:]):
        raise ValueError('transfer metadata')
    selected = {'source_id':source_id, 'problem':first['problem'], 'answer':first['reference_answer'],
                'transfer_problem':transfer['problem'], 'transfer_answer':transfer['reference_answer'],
                'transfer_target':expected_target, 'transfer_candidate_id':transfer['candidate_id'],
                'transfer_position':transfer['position_by_student_correct'],
                'student_correct':transfer['historical_student_correct'],
                'student_n_scored':transfer['historical_student_n_scored'],
                'score_tie':transfer['historical_group_score_tie']}
    if j.digest(selected) != marker.get('source_sha256'):
        raise ValueError('selected source binding')
    results = {}
    for item in result['results']:
        ident = j.identity(item.get('identity'))
        digest = j.digest({'identity': ident})
        if digest in results or digest not in inputs:
            raise ValueError('duplicate result')
        if (item.get('request_sha256') != j.digest(j.payload(inputs[digest])) or
            item.get('model') != j.MODEL or item.get('grader') != 'agnes_full_response_v2' or
            item.get('status') not in ('correct', 'incorrect', 'unknown')):
            raise ValueError('result binding')
        results[digest] = item
    if set(results) != set(inputs):
        raise ValueError('missing result')
    return score_path, result_path, score_bytes, result_bytes, inputs, results


def read_recoveries(j, recovery, root, overlay_root, stage, key, score_bytes, result_bytes, results):
    # Recompute from all original request/response receipts. A missing overlay
    # means 'no candidates' only if the independently recomputed list is empty.
    if overlay_root is None or recovery is None:
        raise ValueError('verified overlay and recovery required')
    reference = recovery.source_overlay(root, stage, key)
    if (not isinstance(reference, dict) or reference.get('stage') != stage or
            reference.get('schema') != 'agnes-topbool-recovery-v1' or
            reference.get('input_sha256') != sha(score_bytes) or
            reference.get('source_result_sha256') != sha(result_bytes) or
            not isinstance(reference.get('recoveries'), list)):
        raise ValueError('recomputed overlay invalid')
    path = overlay_root / stage / key / 'OVERLAY.json'
    if not path.exists():
        if reference['recoveries']:
            raise ValueError('missing recoverable overlay')
        return {}
    _, overlay = read_json_bytes(path)
    if overlay != reference:
        raise ValueError('overlay differs from original receipts')
    if (overlay.get('schema') != 'agnes-topbool-recovery-v1' or overlay.get('stage') != stage or
        overlay.get('input_sha256') != sha(score_bytes) or
        overlay.get('source_result_sha256') != sha(result_bytes) or
        not isinstance(overlay.get('recoveries'), list)):
        raise ValueError('overlay binding')
    recovered = {}
    for item in overlay['recoveries']:
        ident = j.identity(item.get('identity'))
        k = j.digest({'identity': ident})
        if (k in recovered or k not in results or results[k]['status'] != 'unknown' or
            item.get('request_sha256') != results[k]['request_sha256'] or
            item.get('original_status') != 'unknown' or type(item.get('decision')) is not bool):
            raise ValueError('overlay result mismatch')
        folder = root / 'AGNES' / 'receipts' / stage / key / k
        attempts = sorted(folder.glob('ATTEMPT-????-RESPONSE.json'))
        if not attempts:
            raise ValueError('overlay response missing')
        b, response = read_json_bytes(attempts[-1])
        if (sha(b) != item.get('response_sha256') or response.get('http_status') != 200 or
            j.verdict(response.get('raw_response')) is not item['decision']):
            raise ValueError('overlay response mismatch')
        recovered[k] = item['decision']
    return recovered


def classify_response(j, response):
    """Distinguish an actual null judgment from any technical failure."""
    if not isinstance(response, dict):
        return 'technical'
    code = response.get('http_status')
    if code in (401, 403):
        return 'auth'
    if code != 200 or response.get('retryable') is True:
        return 'technical'
    raw = response.get('raw_response')
    try:
        if raw.get('model') != j.MODEL:
            return 'technical'
        choice = raw['choices'][0]
        if choice['finish_reason'] != 'stop':
            return 'technical'
        content = choice['message']['content'].strip()
        if content.startswith('```') and content.endswith('```'):
            lines = content.splitlines()
            if len(lines) < 3 or lines[0].strip().lower() not in ('```', '```json') or lines[-1].strip() != '```':
                return 'technical'
            content = '\n'.join(lines[1:-1]).strip()
        value = j.strict_json(content)
    except (ValueError, KeyError, AttributeError, TypeError, IndexError):
        return 'technical'
    if value is None or (isinstance(value, dict) and set(value) == {'correct'} and value['correct'] is None):
        return 'semantic_null'
    return 'verdict' if type(j.verdict(raw)) is bool else 'technical'


def would_send_request(j, attempt_dir):
    """Check whether grade_one has an API request left in this round epoch."""
    if (attempt_dir / 'RESULT.json').exists():
        return False
    for number in range(1, j.MAX_ATTEMPTS + 1):
        intent = attempt_dir / f'ATTEMPT-{number:04d}-INTENT.json'
        response = attempt_dir / f'ATTEMPT-{number:04d}-RESPONSE.json'
        if intent.exists() and not response.exists():
            # grade_one records this as interrupted and advances to next attempt.
            continue
        if response.exists():
            if not intent.exists():
                return False  # grade_one will reject this broken history.
            prior = j.read_json(response)
            code = prior.get('http_status')
            if prior.get('retryable') is True and (prior.get('status') == 'unknown' or
                    (type(code) is int and (code in (408, 429) or 500 <= code <= 599))):
                continue
            return False
        return True
    return False


def rejudge_source(j, recovery, root, dest, overlay_root, stage, key, api_key, window, workers,
                   caller, auth_failed, service_failed, budget, max_rounds):
    score_path, result_path, score_bytes, result_bytes, inputs, results = verify_source(j, root, stage, key)
    recovered = read_recoveries(j, recovery, root, overlay_root, stage, key, score_bytes, result_bytes, results)
    output = dest / 'decisions' / stage / key
    receipt_root = dest / 'receipts' / stage / key
    locks = root / 'AGNES' / 'locks'
    if not locks.is_dir():
        raise ValueError('shared slots missing')
    old_digest = sha(result_bytes)
    counts = Counter()
    api_counts = Counter()
    api_lock = threading.Lock()

    def work(k):
        row = inputs[k]
        old = results[k]
        if old['status'] != 'unknown':
            return 'preserved'
        binding = {'schema': SCHEMA, 'stage': stage, 'source_key': key, 'identity': old['identity'],
                   'request_sha256': old['request_sha256'], 'source_result_sha256': old_digest,
                   'input_sha256': sha(score_bytes)}
        final_path = output / k / 'DECISION.json'
        if final_path.exists():
            _, previous = read_json_bytes(final_path)
            if any(previous.get(a) != b for a, b in binding.items()) or previous.get('status') not in ('correct', 'incorrect', 'unknown'):
                raise ValueError('existing decision mismatch')
            return 'already_' + previous['status']
        invoked = False
        if k in recovered:
            decision = {**binding, 'status': 'correct' if recovered[k] else 'incorrect',
                        'method': 'verified_topbool_overlay', 'rounds': 0}
        else:
            # This is an append-only technical record, never a semantic verdict.
            deferred_path = dest / 'deferred' / stage / key / k / 'DEFERRED.json'
            def save_format_deferred(cycles):
                deferred = {**binding, 'schema': DEFERRED_SCHEMA,
                            'reason': 'retryable_response_format', 'format_cycles': cycles}
                try:
                    with format_lock(dest):
                        try:
                            j.put_once(deferred_path, deferred)
                        except FileExistsError:
                            if j.read_json(deferred_path) != deferred:
                                raise ValueError('technical record conflict')
                        tripped = len(_format_uncovered(dest)) >= FORMAT_FUSE
                except Exception:
                    service_failed.set()
                    raise
                if tripped:
                    service_failed.set()
                    return 'format_fuse_trip'
                return 'format_deferred'
            if deferred_path.exists():
                _, deferred = read_json_bytes(deferred_path)
                expected = {**binding, 'schema': DEFERRED_SCHEMA, 'reason': 'retryable_response_format'}
                if (any(deferred.get(a) != b for a, b in expected.items()) or
                        type(deferred.get('format_cycles')) is not int or
                        not FORMAT_CYCLES <= deferred['format_cycles'] <= 1000):
                    raise ValueError('existing technical record mismatch')
                return 'format_deferred'
            if auth_failed.is_set():
                return 'auth_deferred'
            if service_failed.is_set():
                return 'service_deferred'
            decision = None
            reserved = False
            def metered_caller(body, secret):
                nonlocal invoked
                with api_lock:
                    api_counts['api_requests'] += 1
                    if not invoked:
                        api_counts['api_samples'] += 1
                        invoked = True
                return caller(body, secret)
            for number in range(1, max_rounds + 1):
                if auth_failed.is_set():
                    return 'auth_deferred'
                if service_failed.is_set():
                    return 'service_deferred'
                # 401/403 receipts are retained in numbered auth epochs. A later run can
                # retry after credentials are fixed without deleting past receipts.
                request_key = j.digest({'identity': j.identity(row)})
                format_failures = 0
                for epoch in range(1000):
                    epoch_root = receipt_root / f'round-{number}' / f'auth-{epoch:04d}'
                    attempt_dir = epoch_root / request_key
                    existing = attempt_dir / 'RESULT.json'
                    attempts = sorted(attempt_dir.glob('ATTEMPT-????-RESPONSE.json'))
                    if (existing.exists() and j.read_json(existing).get('status') == 'unknown' and
                            attempts and classify_response(j, j.read_json(attempts[-1])) in ('auth', 'technical')):
                        if retryable_format(j.read_json(attempts[-1])):
                            format_failures += 1
                        if format_failures >= FORMAT_CYCLES:
                            return save_format_deferred(format_failures)
                        continue
                    break
                else:
                    raise ValueError('auth epoch limit')
                if not reserved and would_send_request(j, attempt_dir):
                    if not budget.acquire():
                        return 'limit_deferred'
                    reserved = True
                result = j.grade_one(row, api_key, epoch_root, locks, caller=metered_caller, window=window)
                if result['status'] == 'unknown':
                    attempts = sorted(attempt_dir.glob('ATTEMPT-????-RESPONSE.json'))
                    kind = classify_response(j, j.read_json(attempts[-1]) if attempts else None)
                    if kind == 'auth':
                        auth_failed.set()
                        return 'auth_deferred'
                    if kind != 'semantic_null':
                        if attempts and retryable_format(j.read_json(attempts[-1])):
                            format_failures += 1
                            if format_failures >= FORMAT_CYCLES:
                                return save_format_deferred(format_failures)
                        service_failed.set()
                        return 'service_deferred'
                if result['status'] != 'unknown':
                    decision = {**binding, 'status': result['status'], 'method': 'agnes_rejudge', 'rounds': number}
                    break
            if decision is None:
                decision = {**binding, 'status': 'unknown', 'method': 'agnes_rejudge', 'rounds': max_rounds}
        if score_path.read_bytes() != score_bytes or result_path.read_bytes() != result_bytes:
            raise ValueError('source changed')
        def store_decision():
            try:
                j.put_once(final_path, decision)
            except FileExistsError:
                if j.read_json(final_path) != decision:
                    raise ValueError('conflicting decision')
        if decision.get('method') == 'agnes_rejudge' and decision.get('status') in ('correct', 'incorrect') and invoked:
            try:
                with format_lock(dest):
                    store_decision()
                    _write_format_checkpoint_locked(dest, j, 'api_verdict', final_path)
            except Exception:
                service_failed.set()
                raise
        else:
            store_decision()
        return ('overlay_' if k in recovered else 'rejudged_') + decision['status']

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, k) for k in inputs]
        for future in concurrent.futures.as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception:
                counts['item_error'] += 1
    counts.update(api_counts)
    return counts


def run(j, root, dest, overlay_root, key, workers, caller=None, max_items=0, per_stage=None,
        max_rounds=1, recovery=None, reset_format_circuit=False):
    root, dest = root.resolve(), dest.resolve()
    if root == dest or root in dest.parents or dest in root.parents or not root.is_dir():
        raise ValueError('invalid paths')
    if overlay_root is None or recovery is None or not 1 <= max_rounds <= MAX_ROUNDS:
        raise ValueError('verified recovery and rounds required')
    overlay_root = overlay_root.resolve()
    if (not overlay_root.is_dir() or overlay_root == dest or dest in overlay_root.parents or
            overlay_root in dest.parents):
        raise ValueError('invalid overlay path')
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = j.lock_file(dest / '.rejudge.lock')
    counts = Counter()
    budget = ItemBudget(max_items)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if reset_format_circuit:
            write_format_checkpoint(dest, j, 'operator_reset')
            print('已记录人工熔断重置；原始判分及技术待处理记录不变。', flush=True)
        with format_lock(dest):
            uncovered = _format_uncovered(dest)
        if len(uncovered) >= FORMAT_FUSE:
            # Assign the global circuit to one affected tier without timestamps.
            stage = sorted(uncovered)[-1].split('/')[0]
            if stage not in STAGES:
                raise ValueError('invalid persistent fuse stage')
            counts.update(service_failed=1, format_fuse_trip=1)
            if per_stage is not None:
                per_stage[stage] = {'format_fuse_trip': 1, 'service_deferred': 1}
            return dict(counts)
        window = j.AdaptiveWindow(workers)
        auth_failed = threading.Event()
        service_failed = threading.Event()
        for stage in STAGES:
            stage_counts = Counter(visited=1)
            counts['visited'] += 1
            source_dir = root / 'AGNES' / stage / 'sources'
            if not source_dir.is_dir() or source_dir.is_symlink():
                stage_counts['missing_stage'] += 1
                counts['missing_stage'] += 1
                if per_stage is not None:
                    per_stage[stage] = dict(stage_counts)
                continue
            reports = sorted(source_dir.glob('*/RESULTS.json'))
            if not reports:
                stage_counts['empty_stage'] += 1
                counts['empty_stage'] += 1
            for path in reports:
                source_key = path.parent.name
                if not re.fullmatch(r'[0-9a-f]{24}', source_key):
                    counts['source_error'] += 1
                    stage_counts['source_error'] += 1
                    continue
                try:
                    outcome = rejudge_source(j, recovery, root, dest, overlay_root, stage, source_key,
                                             key, window, workers, caller or j.api_call, auth_failed,
                                             service_failed, budget, max_rounds)
                    counts.update(outcome)
                    stage_counts.update(outcome)
                    counts['sources_scanned'] += 1
                    stage_counts['sources_scanned'] += 1
                except Exception:
                    counts['source_error'] += 1
                    stage_counts['source_error'] += 1
                if auth_failed.is_set():
                    counts['authentication_failed'] += 1
                    stage_counts['authentication_failed'] += 1
                    if per_stage is not None:
                        per_stage[stage] = dict(stage_counts)
                    return dict(counts)
                if service_failed.is_set():
                    counts['service_failed'] += 1
                    stage_counts['service_failed'] += 1
                    if per_stage is not None:
                        per_stage[stage] = dict(stage_counts)
                    return dict(counts)
                # All workers for this source have finished and flushed their
                # records. Failure checks must precede the pilot limit check.
                if budget.reached():
                    if not counts.get('source_error') and not counts.get('item_error'):
                        counts['limit_reached'] += 1
                        stage_counts['limit_reached'] += 1
                    if per_stage is not None:
                        per_stage[stage] = dict(stage_counts)
                    return dict(counts)
            if per_stage is not None:
                per_stage[stage] = dict(stage_counts)
        return dict(counts)
    finally:
        os.close(fd)


def format_report(per_stage, state):
    """Terminal table contains aggregate counters only, never raw records."""
    groups = (('PILOT', 'EASY'), ('MEDIUM',), ('HARD',))
    headers = ('项目', '简单', '中等', '困难', '合计')
    definitions = (
        ('已扫描来源', ('sources_scanned',)),
        ('原有对错保留', ('preserved',)),
        ('旧回执恢复', ('overlay_correct', 'overlay_incorrect')),
        ('已有独立判定', ('already_correct', 'already_incorrect', 'already_unknown')),
        ('API尝试作答', ('api_samples',)),
        ('API请求次数', ('api_requests',)),
        ('本轮新判对', ('rejudged_correct',)),
        ('本轮新判错', ('rejudged_incorrect',)),
        ('本轮仍未决', ('rejudged_unknown',)),
        ('技术待处理', ('format_deferred', 'format_fuse_trip')),
        ('格式熔断', ('format_fuse_trip',)),
        ('达到上限暂缓', ('limit_deferred',)),
        ('鉴权异常暂缓', ('auth_deferred',)),
        ('服务异常暂缓', ('service_deferred',)),
        ('来源核验异常', ('source_error',)),
        ('作答处理异常', ('item_error',)),
        ('缺少目录', ('missing_stage',)),
        ('目录无结果', ('empty_stage',)),
    )
    visited = [sum(1 for stage in group if per_stage.get(stage, {}).get('visited') == 1)
               for group in groups]
    flags = ['是' if n == len(group) else ('部分' if n else '否')
             for n, group in zip(visited, groups)]
    rows = [('档位已检查', *flags, f'{sum(visited)}/4')]
    for label, keys in definitions:
        vals = [sum(per_stage.get(stage, {}).get(key, 0) for stage in group for key in keys)
                for group in groups]
        rows.append((label, *vals, sum(vals)))
    cells = [headers] + [tuple(str(v) for v in row) for row in rows]
    def width(s):
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1
                   for c in str(s))
    widths = [max(width(row[i]) for row in cells) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (n + 2) for n in widths) + '+'
    def line(row):
        cells2 = []
        for i, cell in enumerate(row):
            spaces = ' ' * (widths[i] - width(cell))
            cells2.append(cell + spaces if i == 0 else spaces + cell)
        return '| ' + ' | '.join(cells2) + ' |'
    return '\n'.join([f'本轮状态：{state}', border, line(cells[0]), border,
                       *(line(row) for row in cells[1:]), border,
                       '简单档包括 PILOT；部分或未扫描的档位不能视为完成。',
                       '候选与新判结果保存在独立层；原记录不变。',
                       '明确无法判断的未决会保存，不会在同一输出目录自动重判。'])


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--overlay', type=Path, required=True)
    p.add_argument('--judge', type=Path, required=True)
    p.add_argument('--judge-sha256', required=True)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--recovery-sha256', required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--workers', type=int, default=5, choices=range(1, 6))
    p.add_argument('--max-items', type=int, default=5,
                   help='maximum newly attempted samples; default five; zero means unlimited')
    p.add_argument('--max-rounds', type=int, default=1, choices=range(1, MAX_ROUNDS + 1))
    p.add_argument('--reset-format-circuit', action='store_true',
                   help='record an explicit operator reset after investigating service format failures')
    args = p.parse_args()
    if not all(p.is_absolute() for p in (args.root, args.output, args.judge, args.key_file, args.overlay, args.recovery)):
        raise ValueError('absolute paths required')
    if args.judge_sha256 != APPROVED_JUDGE:
        raise ValueError('unapproved judge')
    j = load_judge(args.judge, args.judge_sha256)
    recovery = load_recovery(args.recovery, args.recovery_sha256)
    if args.key_file.is_symlink():
        raise ValueError('key symlink refused')
    key = args.key_file.read_text().strip()
    if not key:
        raise ValueError('empty key')
    stage_counts = {}
    counts = run(j, args.root, args.output, args.overlay, key, args.workers,
                 max_items=args.max_items, per_stage=stage_counts, max_rounds=args.max_rounds,
                 recovery=recovery, reset_format_circuit=args.reset_format_circuit)
    errors = (counts.get('source_error') or counts.get('item_error') or counts.get('missing_stage') or
              counts.get('empty_stage') or counts.get('authentication_failed') or counts.get('service_failed'))
    unresolved = (counts.get('rejudged_unknown') or counts.get('already_unknown') or
                  counts.get('format_deferred') or counts.get('format_fuse_trip'))
    state = ('出现异常' if errors else ('已到本轮上限' if counts.get('limit_deferred') or counts.get('limit_reached') else
             ('仍有未决' if unresolved else '本轮扫描完成')))
    print(format_report(stage_counts, state), flush=True)
    return 2 if errors else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('补判未启动或中断，请检查本机权限与配置。', flush=True)
        raise SystemExit(1)
