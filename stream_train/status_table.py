"""只输出流式训练任务数量和进程概况；不读取题目、作答或日志。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import unicodedata
from urllib.parse import quote


KINDS = [('generate', '出题'), ('review', '审核'), ('equivalence', '答案核对'),
         ('student', '学生作答'), ('grade', '判分')]
STATES = ('pending', 'started', 'done', 'rejected', 'unknown')


def read_queue(path):
    if not path.is_file():
        return '尚未创建', None
    try:
        db = sqlite3.connect('file:' + quote(str(path), safe='/') + '?mode=ro',
                             uri=True, timeout=1, isolation_level=None)
        try:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            rows = db.execute('SELECT kind,state,COUNT(*) FROM jobs GROUP BY kind,state').fetchall()
            bound = db.execute("SELECT 1 FROM metadata WHERE key='protocol'").fetchone()
            db.commit()
        finally:
            db.close()
        if not bound or any(k not in dict(KINDS) or s not in STATES for k, s, _ in rows):
            return '结构或状态异常', None
        counts = {(k, s): n for k, s, n in rows}
        return '已创建队列', counts
    except (OSError, sqlite3.Error):
        return '无法读取或数据库暂时锁定', None


def option(args, key):
    found = []
    for i, value in enumerate(args):
        if value == key and i + 1 < len(args):
            found.append(args[i + 1])
        elif value.startswith(key + '='):
            found.append(value[len(key) + 1:])
    return found[0] if len(found) == 1 else None


def matches_path(value, expected):
    return bool(value and os.path.isabs(value) and os.path.realpath(value) == str(expected))


def scan_processes(run):
    found = Counter()
    unreadable = 0
    try:
        entries = list(Path('/proc').iterdir())
    except OSError:
        return found, None
    for p in entries:
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.geteuid():
                continue
            args = [v.decode('utf-8', 'surrogateescape') for v in
                    (p / 'cmdline').read_bytes().split(b'\0') if v]
            if not args:
                continue
            if ('stream_train.launch_supervised' in args and
                    matches_path(option(args, '--run-dir'), run)):
                found['supervisor'] += 1
            db = option(args, '--db')
            for phase in ('pilot', 'formal'):
                if not matches_path(db, run / (phase + '.sqlite')):
                    continue
                if 'stream_train' in args and 'api' in args:
                    found[phase + '_api'] += 1
                if 'stream_train' in args and 'student' in args:
                    found[phase + '_student'] += 1
                if any(os.path.basename(a) == 'auto_stop_student.py' for a in args):
                    found['watcher'] += 1
        except FileNotFoundError:
            continue
        except OSError:
            unreadable += 1
    return found, unreadable


def service_state(run):
    try:
        receipt = json.loads((run / 'receipt.json').read_text())
        pid = receipt['student_service_pid']
        start = receipt['student_service_start']
        if type(pid) is not int or pid <= 1 or type(start) is not int:
            return '无法确认'
        p = Path('/proc') / str(pid)
        if p.stat().st_uid != os.geteuid():
            return '无法确认'
        tail = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        args = (p / 'cmdline').read_bytes().split(b'\0')
        if int(tail[19]) != start:
            return '未运行或进程已变化'
        if not any(b'vllm' in x for x in args):
            return '无法确认'
        return '进程存在'
    except FileNotFoundError:
        return '未发现'
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return '无法确认'


def display(run):
    pilot, _ = read_queue(run / 'pilot.sqlite')
    formal, counts = read_queue(run / 'formal.sqlite')
    processes, unreadable = scan_processes(run)
    print('试点：%s  |  正式：%s' % (pilot, formal))
    print('服务：%s  |  监督器：%d  |  自动停卡：%d' %
          (service_state(run), processes['supervisor'], processes['watcher']))
    print('进程数：试点 API %d、学生 %d；正式 API %d、学生 %d' %
          (processes['pilot_api'], processes['pilot_student'],
           processes['formal_api'], processes['formal_student']))
    if unreadable is None or unreadable:
        print('部分进程无法读取，进程数可能不完整。')
    if counts is None:
        print('正式队列暂无可显示的统计。')
        return
    print('+----------+--------+--------+--------+--------+--------+')
    print('| 阶段     | 待处理 | 进行中 | 已完成 | 已拒绝 | 未决   |')
    print('+----------+--------+--------+--------+--------+--------+')
    for kind, name in KINDS:
        values = [counts.get((kind, state), 0) for state in STATES]
        width = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in name)
        print('| %s | %6d | %6d | %6d | %6d | %6d |' %
              (name + ' ' * max(0, 8 - width), *values))
    print('+----------+--------+--------+--------+--------+--------+')
    print('单位是任务，不是来源或完整题数；已创建队列不代表初始化已完成。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    path = Path(args.run_dir)
    if not path.is_absolute():
        parser.error('运行目录必须使用绝对路径')
    display(path.resolve())


if __name__ == '__main__':
    main()
