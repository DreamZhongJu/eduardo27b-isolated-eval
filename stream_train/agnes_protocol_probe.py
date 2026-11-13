"""Probe the frozen pipeline with synthetic questions and no production rows.

Each stage uses exactly the corresponding production pipeline function. The
in-memory sink deliberately discards all API output and child jobs. Only a
fixed stage, category and allowlisted reason may be printed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

from stream_train import __main__ as app
from stream_train import pipeline as p


KINDS = ('generate', 'review', 'equivalence', 'grade')
REASONS = frozenset(('invalid_json_response', 'response_schema',
    'unexpected_provider_model', 'incomplete_response', 'missing_content',
    'unparseable_content', 'nonobject_response', 'generation_schema',
    'empty_candidate', 'review_schema', 'review_enum', 'equivalence_schema',
    'grade_schema', 'URLError', 'TimeoutError', 'ConnectionError',
    'ConnectionResetError', 'ConnectionAbortedError', 'ConnectionRefusedError',
    'RemoteDisconnected', 'OSError', 'SSLError', 'BrokenPipeError',
    'InterruptedError', 'IncompleteRead', 'ProtocolError'))


class Sink:
    """Accept one result without persisting payloads, API receipts or children."""
    def __init__(self):
        self.finished = False

    def finish(self, job, result, children=(), state='done'):
        if self.finished or state not in ('done', 'rejected', 'unknown'):
            raise ValueError('unexpected probe state')
        self.finished = True


def payload(kind):
    source = {'source_id': 'synthetic-protocol-probe', 'source_row_index': 0,
              'source_split': 'train', 'problem': 'What is 2 + 3?',
              'answer': '5', 'direction_band': 'mixed'}
    candidate = {'candidate_id': 'synthetic-probe-candidate', 'origin': 'self_generated',
                 'question': 'What is 2 + 4?', 'answer': '6',
                 'solution': '2 + 4 = 6.', 'construction_status': 'proposed',
                 'notes': 'synthetic'}
    base = {'source': source, 'candidate': candidate, 'proposal': 1, 'feedback': []}
    if kind == 'generate':
        return {'source': source, 'proposal': 1, 'feedback': []}
    if kind == 'review':
        return base
    if kind == 'equivalence':
        return {**base, 'review': {'independent_answer': '6'}}
    if kind == 'grade':
        return {**base, 'student': {'finish_reason': 'stop',
                                    'text': '2 + 4 = 6. Final answer: \\boxed{6}.'}}
    raise ValueError('invalid probe kind')


def classify(exc):
    # Never print exception messages or arbitrary provider content.
    category = app.api_outcome(exc)
    if category not in ('success', 'neutral', 'auth', 'model', 'protocol', 'congestion'):
        category = 'neutral'
    reason = 'other'
    if isinstance(exc, p.UnknownOutcome):
        raw = str(exc)
        if raw in REASONS:
            reason = raw
        elif re.fullmatch(r'http_[1-5][0-9][0-9]', raw):
            reason = raw
    return category, reason


def probe(cfg, key, kinds, timeout=60):
    if not key or not 1 <= timeout <= 900:
        raise ValueError('invalid probe arguments')
    if not kinds or any(kind not in KINDS for kind in kinds):
        raise ValueError('invalid probe kind')
    # Probe process only: reuse the unmodified transport with a shorter
    # timeout, so one hung synthetic request cannot stall all retries.
    request = p.request_json
    def bounded(url, body, secret, **kwargs):
        kwargs['timeout'] = timeout
        return request(url, body, secret, **kwargs)
    p.request_json = bounded
    try:
        for kind in kinds:
            sink = Sink()
            try:
                getattr(p, kind)(sink, 'synthetic-probe-job', payload(kind), cfg, key)
                if not sink.finished:
                    raise ValueError('missing probe result')
            except Exception as exc:
                category, reason = classify(exc)
                print(json.dumps({'stage': kind, 'status': 'failed',
                                  'category': category, 'reason': reason}), flush=True)
                return False
            # Semantic rejection or uncertainty is still a valid protocol response.
            print(json.dumps({'stage': kind, 'status': 'ok'}), flush=True)
        return True
    finally:
        p.request_json = request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--kind', action='append', choices=KINDS)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    try:
        path = Path(args.config)
        if not path.is_absolute():
            raise ValueError('absolute config path required')
        cfg = app.config(path)
        key = os.environ.get('AGNES_API_KEY', '').strip()
        return 0 if probe(cfg, key, args.kind or KINDS, args.timeout) else 1
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({'stage': 'configuration', 'status': 'failed',
                          'category': 'configuration', 'reason': 'other'}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
