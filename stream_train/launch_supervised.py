"""One synthetic end-to-end gate, then one isolated train run.

Launch only after the site authorizes sending source and candidate texts to
Agnes for generation and review as well as grading. Run inside the container,
using an already-running dedicated student service. No historical data are
modified; an empty hold file records an independent rerun, not freshness.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from urllib.parse import quote, urlsplit

from stream_train import auto_stop_student as guard
from stream_train import __main__ as app


class LaunchError(Exception):
    pass


def preflight_processes():
    # No existing training student consumer may be reused or silently shared.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
            args = [x.decode('utf-8', 'surrogateescape') for x in
                    (entry / 'cmdline').read_bytes().split(b'\0') if x]
            if guard.has_module(args, 'stream_train') and args.count('student') == 1:
                raise LaunchError('已存在训练学生消费者，未启动新任务')
        except FileNotFoundError:
            continue
        except OSError:
            raise LaunchError('无法安全核查现有消费者') from None
    # The real producer independently takes the same lock at launch, so this
    # preflight improves diagnostics but is not relied on for race exclusion.
    lock = Path.home() / '.cache' / 'math-transfer-stream' / 'api.lock'
    if lock.exists():
        try:
            with lock.open('a') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
        except OSError:
            raise LaunchError('已有训练 API 生产者占用锁') from None


def preflight_pidfd(pid, model, alias, port, stamp):
    fd = guard.open_pidfd(pid)
    try:
        guard.validate_service(pid, model, alias, port, stamp)
        signal.pidfd_send_signal(fd, 0)
    except (OSError, AttributeError):
        raise LaunchError('容器不支持安全进程句柄或无权检测，未启动试点') from None
    finally:
        os.close(fd)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def connect_readonly(path):
    return sqlite3.connect('file:' + quote(str(path), safe='/') + '?mode=ro', uri=True, timeout=5)


def pilot_state(path):
    """Return only aggregate readiness; never emit task contents."""
    with connect_readonly(path) as db:
        rows = db.execute('SELECT kind,state,payload,result FROM jobs').fetchall()
    states = Counter((kind, state) for kind, state, _, _ in rows)
    if any(k not in guard.KINDS or s not in guard.STATES for k, s in states):
        raise LaunchError('试点队列出现无法识别的状态')
    pending = sum(n for (k, s), n in states.items() if s in ('pending', 'started'))
    unknown = sum(n for (k, s), n in states.items() if s == 'unknown')
    samples = defaultdict(lambda: {'student': {}, 'grade': {}})
    for kind, state, payload_text, result_text in rows:
        if kind not in ('student', 'grade') or state != 'done':
            continue
        payload = json.loads(payload_text)
        candidate = payload['candidate']
        if candidate.get('origin') != 'self_generated':
            continue
        key = candidate['candidate_id']
        index = payload['sample_index']
        sample_id = payload['sample_id']
        result = json.loads(result_text)
        samples[key][kind][index] = (sample_id, result)
    passed = False
    for group in samples.values():
        student, grade = group['student'], group['grade']
        if set(student) != set(range(16)) or set(grade) != set(range(16)):
            continue
        if any(student[i][0] != grade[i][0] for i in range(16)):
            continue
        if any(grade[i][1].get('decision') not in ('correct', 'incorrect') for i in range(16)):
            continue
        # At least one complete response must have been judged by the real API.
        if not any(isinstance(grade[i][1].get('meta'), dict) and
                   isinstance(grade[i][1].get('verdict'), dict) for i in range(16)):
            continue
        passed = True
    return pending, unknown, passed


def stop_children(children):
    for proc in children:
        if proc.poll() is None:
            proc.terminate()
    for proc in children:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise LaunchError('进程未正常退出，已保留记录；请人工检查') from None


def launch(args):
    if not args.authorized_external or not args.confirm_dedicated:
        raise LaunchError('需要确认站点授权 Agnes 出题、审核、判分和专用学生服务')
    if not os.environ.get('AGNES_API_KEY', '').strip():
        raise LaunchError('未设置 Agnes 密钥环境变量')
    for item in (args.sources, args.config, args.model_path, args.run_dir):
        if not Path(item).is_absolute():
            raise LaunchError('所有输入和运行路径必须是绝对路径')
    source, config = Path(args.sources).resolve(), Path(args.config).resolve()
    model, run = Path(args.model_path).resolve(), Path(args.run_dir)
    if not source.is_file() or not config.is_file() or not model.is_dir():
        raise LaunchError('输入、配置或模型路径不可用')
    with source.open(encoding='utf-8') as f:
        source_count = sum(1 for line in f if line.strip())
    if source_count != args.expected_sources:
        raise LaunchError('来源数与预期不一致，未启动')
    if digest(source) != args.sources_sha256.lower():
        raise LaunchError('来源文件哈希与冻结清单不一致，未启动')
    if not 1 <= args.workers <= 64 or args.pilot_timeout < 60:
        raise LaunchError('参数范围无效')
    cfg = app.config(config)
    endpoint = urlsplit(cfg['student_base'])
    if endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.port is None:
        raise LaunchError('学生服务必须使用明确的本机地址和端口')
    port, alias = endpoint.port, cfg['student_model']
    # The matching process and socket are checked again before formal launch.
    service_pid = guard.find_service(model, alias, port)
    service_stamp = guard.validate_service(service_pid, model, alias, port)
    preflight_pidfd(service_pid, model, alias, port, service_stamp)
    preflight_processes()
    if any(guard.metrics(port)):
        raise LaunchError('学生服务有在途或排队请求，请等待空闲')
    if run.exists():
        raise LaunchError('运行目录已存在：拒绝覆盖或重复提交')
    run.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(run, 0o700)
    frozen_config = run / 'config.json'
    config_bytes = config.read_bytes()
    frozen_config.write_bytes(config_bytes)
    receipt = {'sources_sha256': digest(source), 'config_sha256': digest(frozen_config),
               'source_count': source_count,
               'run_mode': 'independent rerun; empty hold; historical duplicates possible',
               'student_service_pid': service_pid, 'student_service_start': service_stamp,
               'student_port': port, 'student_model': alias, 'workers': args.workers,
               'pilot_timeout_seconds': args.pilot_timeout}
    (run / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    hold = run / 'empty-hold.txt'; hold.touch(mode=0o600, exist_ok=False)
    fake = run / 'synthetic-source.jsonl'
    fake.write_text(json.dumps({'source_id': 'synthetic-autostart-1', 'source_row_index': 0,
                               'source_split': 'train', 'problem': 'What is 2+3?', 'answer': '5'}) + '\n')
    if app.config(frozen_config) != cfg:
        raise LaunchError('配置在准备期间发生变化')
    pilot_config = run / 'pilot-config.json'
    pilot_cfg = json.loads(config_bytes)
    pilot_cfg['namespace'] = cfg['namespace'] + '-synthetic-pilot'
    pilot_config.write_text(json.dumps(pilot_cfg, ensure_ascii=False))
    def command(db, *tail):
        chosen = pilot_config if db == pilot else frozen_config
        return [sys.executable, '-m', 'stream_train', '--db', str(db), '--config', str(chosen), *tail]
    pilot = run / 'pilot.sqlite'
    formal = run / 'formal.sqlite'
    children = []
    formal_active = False
    def start(name, cmd):
        out = (run / (name + '.log')).open('ab', buffering=0)
        try:
            return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        finally:
            out.close()
    def initialize(db, data):
        name = 'pilot-init.log' if db == pilot else 'formal-init.log'
        with (run / name).open('ab') as log:
            result = subprocess.run(command(db, 'init', '--sources', str(data), '--hold-source-ids', str(hold)),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise LaunchError('初始化失败，保留目录供检查')
    try:
        initialize(pilot, fake)
        api = start('pilot-api', command(pilot, 'api'))
        children.append(api)
        student = start('pilot-student', command(pilot, 'student', '--workers', str(args.workers)))
        children.append(student)
        deadline = time.monotonic() + args.pilot_timeout
        print('试点运行中：独立合成来源、真实 Agnes 和学生服务。', flush=True)
        while time.monotonic() < deadline:
            if api.poll() is not None or student.poll() is not None:
                raise LaunchError('试点进程退出，请检查站内日志')
            pending, unknown, passed = pilot_state(pilot)
            if not pending:
                if unknown or not passed:
                    raise LaunchError('试点未通过：存在未决或没有完整的 16 次作答及明确判分')
                break
            time.sleep(5)
        else:
            raise LaunchError('试点超时，未启动正式任务')
        stop_children(children); children = []
        guard.validate_service(service_pid, model, alias, port, service_stamp)
        if any(guard.metrics(port)):
            raise LaunchError('试点结束后学生服务尚未空闲')
        if digest(source) != receipt['sources_sha256'] or digest(frozen_config) != receipt['config_sha256']:
            raise LaunchError('输入或配置已变化，未启动正式任务')
        initialize(formal, source)
        if digest(source) != receipt['sources_sha256'] or digest(frozen_config) != receipt['config_sha256']:
            raise LaunchError('正式初始化期间输入或配置发生变化')
        api = start('formal-api', command(formal, 'api'))
        children.append(api)
        student = start('formal-student', command(formal, 'student', '--workers', str(args.workers)))
        children.append(student)
        time.sleep(3)
        if any(p.poll() is not None for p in children):
            raise LaunchError('正式进程启动失败，已保留记录')
        watcher = start('auto-stop', [sys.executable, str(Path(guard.__file__).resolve()),
            '--db', str(formal), '--namespace', cfg['namespace'], '--model-path', str(model),
            '--student-model', alias, '--port', str(port), '--service-pid', str(service_pid),
            '--consumer-pid', str(student.pid), '--confirm-initialized', '--confirm-dedicated'])
        formal_active = True
        print('试点通过；正式任务已启动，自动停卡监控已附加。', flush=True)
        warned_student = False
        while watcher.poll() is None:
            if api.poll() is not None:
                print('API 进程已退出，自动停卡仅在队列满足条件时执行。', flush=True)
                watcher.wait()
                break
            if student.poll() is not None and not warned_student:
                print('学生消费者已退出；如尚未进入自动停卡，请检查队列与站内日志。', flush=True)
                warned_student = True
            time.sleep(15)
        if watcher.returncode == 0:
            stop_children([api])
            children = []
            print('流程耗尽，学生服务已退出；请核查未决和显存。', flush=True)
        else:
            raise LaunchError('自动停卡监控退出，任务保留，请查看站内状态')
    except Exception:
        # Do not touch a running formal workload on monitoring failures.
        if not formal_active and children:
            stop_children(children)
        raise


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--expected-sources', type=int, default=7246)
    p.add_argument('--sources-sha256', default='9fe5db7cb6be61d5e1d33986de1c21257ac3996ee1440fe3630bad0924d4ccae')
    p.add_argument('--pilot-timeout', type=int, default=3600)
    p.add_argument('--authorized-external', action='store_true')
    p.add_argument('--confirm-dedicated', action='store_true')
    args = p.parse_args()
    try:
        launch(args)
    except (LaunchError, guard.GuardError, OSError, sqlite3.Error, ValueError, KeyError) as exc:
        print('未继续：' + str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
