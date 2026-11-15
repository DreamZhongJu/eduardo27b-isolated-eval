"""Create a consistent, private, local-only training queue snapshot.

This tool does not stop processes, make network requests, read credential files,
or print records. It refuses a live controller or worker for this run. A source
database is copied with SQLite's backup API, including committed WAL pages.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from urllib.parse import quote, urlsplit
import zipfile


class SnapshotError(Exception):
    pass


KINDS = frozenset(('generate', 'review', 'equivalence', 'student', 'grade'))
STATES = frozenset(('pending', 'started', 'done', 'rejected', 'unknown'))
CONFIG_KEYS = ('namespace', 'api_base', 'api_model', 'student_base', 'student_model')
RECEIPT_KEYS = ('sources_sha256', 'config_sha256', 'source_count', 'run_mode',
                'student_port', 'student_model', 'workers', 'pilot_timeout_seconds')
POLICY_KEYS = ('policy', 'maximum', 'initial', 'protocol_sha256',
               'producer_sha256', 'utc_started')
SECRET_PATTERN = re.compile(rb'(?:sk|cpk)-[A-Za-z0-9_-]{24,}', re.I)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def option(args, name):
    values = []
    for i, item in enumerate(args):
        if item == name and i + 1 < len(args):
            values.append(args[i + 1])
        elif item.startswith(name + '='):
            values.append(item[len(name) + 1:])
    return values[0] if len(values) == 1 else None


def same_path(value, expected, proc_dir):
    if not value:
        raise SnapshotError('相关进程缺少队列路径，无法确认已停止')
    path = Path(value)
    if not path.is_absolute():
        # A process may have been launched from any directory. Resolve paths
        # relative to its cwd, never relative to this exporter.
        path = Path(os.readlink(str(proc_dir / 'cwd'))) / path
    return path.resolve() == expected


def live_run_processes(db, run, proc_root=Path('/proc')):
    """Never expose command lines; treat unreadable own processes as unsafe."""
    found = []
    for p in proc_root.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            args = [x.decode('utf-8', 'surrogateescape') for x in
                    (p / 'cmdline').read_bytes().split(b'\0') if x]
            modules = {args[i + 1] for i, a in enumerate(args[:-1]) if a == '-m'}
            watcher = any(os.path.basename(x) == 'auto_stop_student.py' for x in args)
            if (watcher or ('stream_train' in modules and any(x in args for x in ('api', 'student')))
                    or modules.intersection(('stream_train.api_adaptive_external',
                                             'stream_train.api_retry_supervisor',
                                             'stream_train.api_handoff',
                                             'stream_train.api_handoff_immediate',
                                             'stream_train.api_cancel_waiter'))):
                if same_path(option(args, '--db'), db, p):
                    found.append('worker_or_retry')
            if ('stream_train.launch_supervised' in modules and
                    same_path(option(args, '--run-dir'), run, p)):
                found.append('controller')
        except FileNotFoundError:
            continue
        except OSError:
            raise SnapshotError('无法确认运行进程，未创建快照') from None
    return found


def open_ro(path):
    return sqlite3.connect('file:' + quote(str(path), safe='/') + '?mode=ro',
                           uri=True, timeout=5)


def backup_db(source, destination, timeout):
    started = time.monotonic()
    src = open_ro(source)
    dst = sqlite3.connect(str(destination), timeout=5)
    def progress(status, remaining, total):
        if time.monotonic() - started > timeout:
            raise SnapshotError('数据库快照超时')
    try:
        src.backup(dst, pages=4096, progress=progress, sleep=.05)
    finally:
        dst.close()
        src.close()


def metadata_and_counts(db):
    con = open_ro(db)
    try:
        check = con.execute('PRAGMA integrity_check').fetchall()
        if check != [('ok',)]:
            raise SnapshotError('数据库完整性检查未通过')
        metadata = dict(con.execute("SELECT key,value FROM metadata WHERE key IN ('protocol','source_sha256','held_source_ids_sha256')"))
        if not {'protocol', 'source_sha256', 'held_source_ids_sha256'} <= metadata.keys():
            raise SnapshotError('数据库缺少运行绑定')
        protocol = json.loads(metadata['protocol'])
        source_hash = json.loads(metadata['source_sha256'])
        counts = []
        for kind, state, count in con.execute('SELECT kind,state,COUNT(*) FROM jobs GROUP BY kind,state'):
            if kind not in KINDS or state not in STATES or not isinstance(count, int):
                raise SnapshotError('队列状态无法识别')
            counts.append({'kind': kind, 'state': state, 'count': count})
        if not counts or not isinstance(protocol, dict) or not isinstance(source_hash, str):
            raise SnapshotError('队列数据或元数据无效')
        return protocol, source_hash, counts
    finally:
        con.close()


def reject_embedded_secrets(db):
    """Best-effort scan all text and blob cells; report no matching data."""
    con = open_ro(db)
    try:
        tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            quoted_table = '"' + table.replace('"', '""') + '"'
            cols = [r[1] for r in con.execute('PRAGMA table_info(' + quoted_table + ')')]
            if not cols:
                continue
            quoted = ','.join('"' + c.replace('"', '""') + '"' for c in cols)
            for row in con.execute('SELECT ' + quoted + ' FROM ' + quoted_table):
                for value in row:
                    if isinstance(value, str):
                        value = value.encode('utf-8', 'surrogatepass')
                    if isinstance(value, bytes) and SECRET_PATTERN.search(value):
                        raise SnapshotError('数据内出现疑似密钥，未发布压缩包')
    finally:
        con.close()


def safe_config(config_path, protocol):
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    config = {}
    for key in CONFIG_KEYS:
        value = raw.get(key)
        if not isinstance(value, str) or not value or value != protocol.get(key):
            raise SnapshotError('配置与数据库绑定不匹配')
        if SECRET_PATTERN.search(value.encode()):
            raise SnapshotError('配置内出现疑似密钥')
        config[key] = value
    for key in ('api_base', 'student_base'):
        parsed = urlsplit(config[key])
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise SnapshotError('配置地址包含不允许的认证或参数')
    return config


def whitelist_file(path, keys):
    if not path.is_file() or path.is_symlink():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    result = {}
    for key in keys:
        value = raw.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and SECRET_PATTERN.search(value.encode()):
                raise SnapshotError('运行回执内出现疑似密钥')
            result[key] = value
    return result


def check_disk(run_dir, out_path):
    run = Path(run_dir).resolve()
    output = Path(out_path)
    db = run / 'formal.sqlite'
    if (not output.is_absolute() or not output.parent.is_dir() or
            not db.is_file() or db.is_symlink()):
        raise SnapshotError('数据库或目标目录不可用')
    wal = Path(str(db) + '-wal')
    source_bytes = db.stat().st_size + (wal.stat().st_size if wal.is_file() else 0)
    # Snapshot copy + potentially uncompressed ZIP + fixed overhead, before
    # asking users to stop any work. Some legitimate archives need less.
    needed = 2 * source_bytes + 128 * 1024 * 1024
    free = shutil.disk_usage(str(output.parent)).free
    if free < needed:
        raise SnapshotError('空间不足：保守预计至少需要 %d MiB，当前可用 %d MiB' %
                            ((needed + 1048575) // 1048576, free // 1048576))
    return needed, free


def runtime_environment(model_dir):
    versions = {}
    for name in ('vllm', 'torch', 'transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
            versions[name] = 'unknown'
    model = {'config_sha256': None, 'tokenizer_config_sha256': None,
             'weights_checksum': 'not_checked'}
    if model_dir is not None:
        if not Path(model_dir).is_absolute():
            raise SnapshotError('模型目录必须是绝对路径')
        root = Path(model_dir).resolve()
        if root.is_dir():
            for name, key in (('config.json', 'config_sha256'),
                              ('tokenizer_config.json', 'tokenizer_config_sha256')):
                path = root / name
                if path.is_file():
                    model[key] = digest(path)
    return {'python_version': '.'.join(str(x) for x in sys.version_info[:3]),
            'package_versions': versions, 'model': model}


def create_snapshot(run_dir, source_path, code_dir, out_path, timeout, model_dir=None):
    run = Path(run_dir).resolve()
    db = (run / 'formal.sqlite').resolve()
    source = Path(source_path).resolve()
    code = Path(code_dir).resolve()
    output = Path(out_path)
    if (not output.is_absolute() or output.exists() or output.is_symlink() or
            not db.is_file() or not source.is_file() or not output.parent.is_dir() or
            not 1 <= timeout <= 3600):
        raise SnapshotError('输入或输出路径无效；不会覆盖已有文件')
    check_disk(run, output)
    if live_run_processes(db, run):
        raise SnapshotError('正式任务仍有控制器或工作进程，先安全停止再快照')
    lock_path = Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'
    lock_fd = None
    try:
        if lock_path.exists():
            # NFS may require a writable descriptor for an exclusive flock;
            # opening O_RDWR does not write to the existing lock file.
            flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
            lock_fd = os.open(str(lock_path), flags)
            st = os.fstat(lock_fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
                raise SnapshotError('API 锁文件身份异常')
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SnapshotError('API 生产者仍持锁，未创建快照') from None
        with tempfile.TemporaryDirectory(prefix='.math-snapshot-', dir=str(output.parent)) as tmp:
            work = Path(tmp)
            dest = work / 'formal.sqlite'
            print('正在生成数据库一致性快照（包括已提交的 WAL）。', flush=True)
            backup_db(db, dest, timeout)
            print('正在检查数据库完整性、来源身份与敏感格式。', flush=True)
            protocol, expected_source_hash, counts = metadata_and_counts(dest)
            if digest(source) != expected_source_hash:
                raise SnapshotError('来源文件与队列绑定不一致')
            config = safe_config(run / 'config.json', protocol)
            reject_embedded_secrets(dest)
            if live_run_processes(db, run):
                raise SnapshotError('快照期间检测到任务启动，未发布快照')
            config_out = work / 'config.json'
            config_out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            expected_code = protocol.get('code_sha256', {})
            code_hashes = {}
            for name in ('__main__.py', 'pipeline.py', 'store.py'):
                path = code / name
                actual = digest(path) if path.is_file() and not path.is_symlink() else None
                code_hashes[name] = {'expected': expected_code.get(name), 'actual': actual,
                                     'matches': bool(actual and actual == expected_code.get(name))}
            receipts = []
            r = whitelist_file(run / 'receipt.json', RECEIPT_KEYS)
            if r is not None:
                receipts.append({'type': 'run', 'value': r})
            for path in sorted(run.glob('api-policy-*.json')):
                r = whitelist_file(path, POLICY_KEYS)
                if r is not None:
                    receipts.append({'type': 'api_policy', 'value': r})
            receipt_out = work / 'runtime_receipts.json'
            receipt_out.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            # The source input is already present on the receiving workstation.
            # Verify the identity but do not put a duplicate copy in the archive.
            names = ('formal.sqlite', 'config.json', 'runtime_receipts.json')
            manifest = {'format': 'math-transfer-queue-snapshot-v1',
                        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'queue_counts': counts, 'source_sha256': expected_source_hash,
                        'code_sha256': code_hashes,
                        'environment': runtime_environment(model_dir),
                        'credential_check': 'key files excluded; best-effort token-pattern scan, not a guarantee',
                        'files': {name: {'sha256': digest(work / name), 'bytes': (work / name).stat().st_size}
                                  for name in names}}
            manifest_out = work / 'manifest.json'
            manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary_zip = work / 'archive.zip'
            print('正在压缩并校验，数据量大时需要一些时间。', flush=True)
            with zipfile.ZipFile(temporary_zip, 'x', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as z:
                for name in (*names, 'manifest.json'):
                    z.write(work / name, arcname=name)
            with zipfile.ZipFile(temporary_zip) as z:
                if z.testzip() is not None:
                    raise SnapshotError('压缩包校验失败')
            if live_run_processes(db, run):
                raise SnapshotError('打包期间检测到任务启动，未发布快照')
            os.link(str(temporary_zip), str(output))  # Atomic publication, never overwrite.
            os.chmod(output, 0o600)
            print('快照已完成；数据库完整性和压缩包校验通过。')
            print('队列状态：待处理=%d，进行中=%d，未决=%d' % tuple(
                sum(x['count'] for x in counts if x['state'] == s)
                for s in ('pending', 'started', 'unknown')))
            print('压缩包字节数：%d' % output.stat().st_size)
            print('压缩包 SHA-256：%s' % digest(output))
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--sources', required=True)
    p.add_argument('--code-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-dir', help='可选；仅记录模型和 tokenizer 配置哈希，不检查或打包权重')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--space-only', action='store_true',
                   help='只检查本地空间，不需要停止任务，不创建快照')
    args = p.parse_args()
    try:
        if args.space_only:
            needed, free = check_disk(args.run_dir, args.output)
            print('空间检查通过：保守预计至少需要 %d MiB，当前可用 %d MiB。' %
                  ((needed + 1048575) // 1048576, free // 1048576))
        else:
            create_snapshot(args.run_dir, args.sources, args.code_dir, args.output, args.timeout, args.model_dir)
    except (SnapshotError, OSError, ValueError, sqlite3.Error, zipfile.BadZipFile) as exc:
        if isinstance(exc, SnapshotError):
            print(str(exc), file=sys.stderr)
        else:
            print('快照未完成：文件、数据库或空间异常', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
