"""Local immutable source-level receipts; no grading or remote output."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_dir(source):
    return Path(os.environ['RESULTS_DIR']) / 'sources' / sha(source['source_id'].encode())[:24]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.receipt-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(stable(value) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
