"""Conservative local boxed + math_verify grader for single numeric answers.

This is NOT the project's GPT v5 grader. Any non-numeric, multi-part, missing
box, parsing failure, or ambiguous trailing text stays unknown for review.
"""
import argparse
import json
import re
from pathlib import Path

NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d+)?|\\frac\{[+-]?\d+\}\{[+-]?\d+\})$')


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid arguments')


def score(row):
    try:
        from protocol import last_boxed_only_string, _remove_boxed
    except Exception:
        return 'unknown', 'verifier_failed'
    raw = row['student_response']
    gold = row['reference_answer'].strip().replace(' ', '')
    boxed = last_boxed_only_string(raw)
    if not boxed:
        return 'unknown', 'no_box'
    pos = raw.rfind(boxed)
    if pos < 0 or not re.fullmatch(r'[\s.$\\)\]]*', raw[pos + len(boxed):]):
        return 'unknown', 'ambiguous_after_box'
    pred = _remove_boxed(boxed).strip().replace(' ', '')
    if not NUMBER.fullmatch(gold) or not NUMBER.fullmatch(pred):
        return 'unknown', 'non_simple_numeric'
    try:
        from math_verify import parse, verify
        a, b = parse(gold), parse(pred)
        if not a or not b:
            return 'unknown', 'parse_failed'
        return ('correct' if bool(verify(a, b)) else 'incorrect'), 'boxed_math_verify_numeric'
    except Exception:
        return 'unknown', 'verifier_failed'


def main():
    from scoring import put_once
    p = SafeParser(add_help=False, prog='LOCAL_SCORE')
    p.add_argument('score_input', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    rows = json.loads(args.score_input.read_text(encoding='utf-8'))
    verdicts = []
    for row in rows:
        decision, reason = score(row)
        verdicts.append({'source_id': row['source_id'], 'target': row['target'], 'phase': row['phase'],
                         'index': row['index'], 'status': decision, 'reason': reason})
    put_once(args.output, {'grader': 'local_boxed_math_verify_simple_numeric_v1', 'results': verdicts})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_SCORING_FAILED'}))
        raise SystemExit(1)
