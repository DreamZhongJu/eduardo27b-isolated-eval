"""Print only approved aggregate integers and fixed state codes; no source text or paths."""
import json
import os
from pathlib import Path


def main():
    base = Path(os.environ.get('RESULTS_DIR', '/results')) / 'sources'
    groups = [p for p in base.iterdir() if p.is_dir()] if base.is_dir() else []
    done = sum((p / 'RAW_COMPLETE.json').is_file() for p in groups)
    held = sum((p / 'SOURCE_HELD.json').is_file() and not (p / 'RAW_COMPLETE.json').is_file() for p in groups)
    outputs = sum(sum(1 for _ in (p / 'samples').glob('*/*/*/RESPONSE.json')) for p in groups)
    print(json.dumps({'state': 'LOCAL_SUMMARY', 'sources_started': len(groups),
                      'sources_complete': done, 'sources_held': held, 'student_responses': outputs}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_STATUS_FAILED'}))
        raise SystemExit(1)
