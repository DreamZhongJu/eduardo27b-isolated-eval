#!/usr/bin/env python3
"""Stage A: verify a task-owned GPU reservation dies with its controller.
No model load, training, remote API, or termination of unrelated processes.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import random
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path('/workspace/A_group_repro_20261117')
LOCK = ROOT / '.pilot.lock'
SCRIPT = Path(__file__).resolve()
MIB = 1024 * 1024


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def no_old_smoke_controllers():
    unreadable = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            unreadable += 1
            continue
        # Match only this A-group's exact pilot paths and explicit smoke name.
        candidate = any(a.startswith((str(ROOT) + '/').encode()) and
                        a.endswith(b'/pilot.py') for a in args)
        for i, arg in enumerate(args):
            if candidate and arg == b'--name' and i + 1 < len(args) and args[i+1].startswith(b'smoke'):
                return False, unreadable
    return True, unreadable


def service_ports_free():
    holders = []
    try:
        for port in (18861, 18862):
            sock = socket.socket(); holders.append(sock); sock.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        for sock in holders: sock.close()


def check_existing():
    old_free, unreadable = no_old_smoke_controllers()
    require(unreadable == 0, '无法检查全部进程，未启动测试')
    require(old_free, '仍有旧 A 组 smoke 控制器运行，未启动测试')
    require(service_ports_free(), '本机模型服务端口仍在使用，未启动测试')
    require(probe_via_child() == 0, '已有任务持有控制器锁，未启动测试')


def parent_death(parent):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    require(os.getppid() == parent, '父进程已退出')


def snapshot():
    def query(args):
        p = subprocess.run(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                           text=True, capture_output=True, timeout=20, check=True)
        return p.stdout.strip().splitlines()
    devices = {}
    for row in query(['--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu']):
        col = [x.strip() for x in row.split(',')]
        require(len(col) == 6 and col[1].startswith('GPU-'), '无法解析GPU设备状态')
        idx, uuid, name, total, free, util = int(col[0]), col[1], col[2], int(col[3]), int(col[4]), int(col[5])
        devices[idx] = {'uuid': uuid, 'name': name, 'total': total, 'free': free, 'util': util, 'apps': []}
    by_uuid = {v['uuid']: v for v in devices.values()}
    for row in query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory']):
        if not row or row.lower().startswith('no running processes'):
            continue
        col = [x.strip() for x in row.split(',')]
        require(len(col) == 3 and col[0].startswith('GPU-'), '无法解析GPU进程状态')
        if col[0] in by_uuid:
            require(col[1].isdigit() and col[2].isdigit(), 'GPU进程显存不可核验')
            by_uuid[col[0]]['apps'].append((int(col[1]), int(col[2])))
    require(bool(devices), '未发现GPU')
    return devices


def eligible(info):
    return ('A100' in info['name'].upper() and info['total'] >= 78000 and info['total'] - info['free'] <= 1024
            and info['util'] == 0 and not info['apps'])


def same_empty(a, b):
    return a['uuid'] == b['uuid'] and eligible(a) and eligible(b)


def own_lock():
    fd = os.open(str(LOCK), os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def write_status(run, phase, detail, **metrics):
    data = {'phase': phase, 'detail': detail, 'time': int(time.time()), 'pid': os.getpid(), **metrics}
    fd, tmp = tempfile.mkstemp(prefix='.status-', dir=str(run))
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, run / 'status.json')
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def holder(parent):
    parent_death(parent)
    import torch
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    require(total >= 78000 * MIB and free >= total - 2 * 1024**3,
            'CUDA初始化后GPU不再空闲')
    amount = free - 1536 * MIB
    require(amount > 70 * 1024**3, '可预留显存不足')
    hold = torch.empty(amount, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    own = int(torch.cuda.memory_allocated(0))
    require(own >= amount, 'CUDA分配值异常')
    print(json.dumps({'holder_pid': os.getpid(), 'reserved_mib': own // MIB}), flush=True)
    while True:
        time.sleep(5)


def controller(parent, gpu_index, gpu_uuid):
    parent_death(parent)
    try:
        lockfd = own_lock()
    except BlockingIOError:
        print(json.dumps({'error': '已有任务持有控制器锁'}), flush=True)
        return 2
    child = None
    try:
        first = snapshot()
        require(gpu_index in first and first[gpu_index]['uuid'] == gpu_uuid and eligible(first[gpu_index]),
                '预留前GPU已被使用')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONDONTWRITEBYTECODE='1')
        child = subprocess.Popen([sys.executable, '-B', str(SCRIPT), '--internal-holder', str(os.getpid())],
                                 env=env, stdout=subprocess.PIPE, stderr=sys.stderr, close_fds=True)
        ready, _, _ = select.select([child.stdout], [], [], 180)
        require(bool(ready), '显存预留启动超时')
        line = child.stdout.readline()
        require(bool(line), '显存占位失败，详见控制器日志')
        data = json.loads(line)
        require(child.poll() is None and data.get('holder_pid') == child.pid,
                '占位进程未保持运行')
        require(isinstance(data.get('reserved_mib'), int) and data['reserved_mib'] > 70 * 1024,
                '占位显存不足')
        print(json.dumps(data), flush=True)
        while True:
            require(child.poll() is None, '占位进程提前退出')
            time.sleep(2)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=5)
        os.close(lockfd)


def lock_probe():
    try:
        fd = own_lock()
    except BlockingIOError:
        return 2
    os.close(fd)
    return 0


def exited_or_zombie(pid):
    try:
        st = Path('/proc', str(pid), 'stat').read_text()
        return st[st.rfind(')') + 2:].split()[0] == 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return True


def ppid_matches(child, parent):
    try:
        lines = Path('/proc', str(child), 'status').read_text().splitlines()
        return any(reprline.split(':', 1)[0] == 'PPid' and int(reprline.split(':', 1)[1]) == parent for reprline in lines if ':' in reprline)
    except (OSError, ValueError):
        return False


def probe_via_child():
    p = subprocess.run([sys.executable, '-B', str(SCRIPT), '--internal-probe'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    require(p.returncode in (0, 2), '锁检查异常')
    return p.returncode


def second_controller_blocked(parent, gpu_index, gpu_uuid):
    p = subprocess.run([sys.executable, '-B', str(SCRIPT), '--internal-controller',
                        str(parent), str(gpu_index), gpu_uuid],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    try:
        data = json.loads(p.stdout)
    except (ValueError, TypeError):
        data = {}
    return p.returncode == 2 and data.get('error') == '已有任务持有控制器锁'


def finish(run, values):
    report = ('# A 阶段自检结果\n\n'
              '- 未加载模型、未训练、未调用外部接口。\n'
              '- 显存占位仅用于验证专属进程意外退出后的回收，不构成系统GPU独占。\n'
              '- 固定版本CUDA环境采用独立占位测试，与论文训练无关。\n'
              '- GPU编号：{gpu}；测试前已用 {before_used} MiB；占位时已用 {during_used} MiB；回收后已用 {after_used} MiB。\n'
              '- 持锁期间第二控制器被阻挡：通过；SIGKILL 后锁释放：通过。\n'
              '- 占位进程退出且其GPU进程记录消失：通过。\n'
              '- 若其他进程同时申请卡、无法匹配唯一GPU进程、或显存未回落，必须报告未通过而非认定释放。\n').format(**values)
    with (run / 'stage-a-report.md').open('x', encoding='utf8') as f: f.write(report)
    write_status(run, '完成', '显存占位释放与控制器锁均已验证；等待用户确认', **values)


def run_test(run):
    require(run.parent == ROOT and run.is_dir() and not run.is_symlink(), '运行目录不属于A组目录')
    require(not (run / 'status.json').exists(), '本次状态已存在，不覆盖')
    require(ROOT.is_dir() and not ROOT.is_symlink(), 'A组目录不可用')
    for name in ('tmp', 'cache', 'cuda_cache'):
        (run / name).mkdir(mode=0o700)
    with (run / 'deviations.md').open('x', encoding='utf8') as f:
        f.write('''# A 阶段与后续计划说明

A 阶段只验证进程、显卡占位及锁回收；未加载模型，未进行论文训练。
以下均为用户确认的 B/C 计划，**本次 A 阶段尚未实施或验证**：

- 教师生成、strip_teacher_controls、原作者最后一轮遗漏路径均保持现状；仅计划记录每轮原始 finish_reason、生成长度、标签状态和聚合指标。原始 finish_reason 及尾轮路径计数的接入尚待实现。
- 原作者裁判重试耗尽记为 REJECT 的奖励逻辑保持不改；B/C 一旦出现技术失败票，计划停止且不得更新参数，不自动重跑批次；技术失败识别与更新前阻断尚待实现。
- 裁判初始全局在途上限计划为 32、许可获取后开始请求计时，服务设置为 gpu_memory_utilization 0.9、max_num_seqs 256、max_model_len 32768；全局限流、票回执及学生调用统计尚待实现。
- Tutor 使用两张空卡，按作者 gpu_memory_utilization 0.58 自动 KV；显存测量断言若复现，再按实测讨论固定 KV。8 GiB 保护只计划用于共享卡；分角色占卡与交接尚待实现。
- B 是功能测试，2 题、每题 4 条对话、后测 4 次、训练 2 步；与论文配置不同，不作为论文结果。C 计划恢复论文口径的 16 题、每题 16 条对话及后测 16 次。

各阶段均须在前一阶段结束并经用户确认后才启动，A 不会触发 B 或 C。
''')
    write_status(run, '准备', '检查现有控制器锁')
    check_existing()
    chosen = None
    period = time.monotonic()
    while chosen is None:
        first = snapshot(); time.sleep(5); second = snapshot()
        for i in sorted(first):
            if i in second and same_empty(first[i], second[i]):
                chosen = (i, second[i]); break
        if chosen is None:
            if time.monotonic() - period >= 45 * 60:
                pause = random.randint(5 * 60, 15 * 60)
                write_status(run, '等待', '45分钟未找到空卡；未占用显卡，进入随机等待', cooldown_minutes=round(pause / 60, 1))
                time.sleep(pause); period = time.monotonic()
            else:
                write_status(run, '等待', '等待一张真正空闲的 A100；目前未占用显卡')
                time.sleep(60)
            check_existing()
    check_existing()
    idx, before = chosen
    before_used = before['total'] - before['free']
    write_status(run, '准备', '启动测试专属控制器和显存占位', gpu=idx)
    env = {k: v for k, v in os.environ.items()
           if not (k.upper().endswith(('_TOKEN', '_API_KEY', '_SECRET')) or
                   k.upper() in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'))}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(run/'tmp'), XDG_CACHE_HOME=str(run/'cache'),
               CUDA_CACHE_PATH=str(run/'cuda_cache'), TORCH_HOME=str(run/'cache'),
               TRITON_CACHE_DIR=str(run/'cache'), HF_HOME=str(run/'cache'))
    # Only this direct Popen child will receive SIGKILL; never select arbitrary PIDs.
    with (run / 'controller.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-B', str(SCRIPT), '--internal-controller',
                              str(os.getpid()), str(idx), before['uuid']],
                             stdout=subprocess.PIPE, stderr=log, env=env, close_fds=True)
    holder_pid = None
    try:
        ready, _, _ = select.select([p.stdout], [], [], 210)
        require(bool(ready), '控制器或显存占位启动超时')
        line = p.stdout.readline()
        require(bool(line), '控制器或显存占位失败，详见控制器日志')
        data = json.loads(line)
        require(not data.get('error') and p.poll() is None, data.get('error') or '控制器提前退出')
        holder_pid = data['holder_pid']; reserved = data['reserved_mib']
        require(isinstance(holder_pid, int) and holder_pid > 1 and ppid_matches(holder_pid, p.pid),
                '无法确认占位进程属于测试控制器')
        # NVML may return host rather than container PIDs.  Identify the sole
        # newly visible GPU process by target UUID and memory, never signal it.
        during = None
        for _ in range(20):
            now = snapshot().get(idx)
            if (now and now['uuid'] == before['uuid'] and len(now['apps']) == 1
                and now['apps'][0][1] >= reserved * .9
                and now['total'] - now['free'] - before_used >= reserved * .9):
                during = now; break
            time.sleep(1)
        require(during is not None and p.poll() is None and ppid_matches(holder_pid, p.pid),
                '无法证明该GPU的唯一新增占位，已取消测试')
        require(second_controller_blocked(os.getpid(), idx, before['uuid']), '第二个控制器未被锁挡住')
        write_status(run, '自检', '已确认占位和锁，准备终止本次测试控制器', gpu=idx)
        p.kill(); p.wait(timeout=15)
        require(p.returncode == -signal.SIGKILL, '控制器未按预期SIGKILL退出')
        after = None
        stable = 0
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            now = snapshot().get(idx)
            holder_stopped = exited_or_zombie(holder_pid)
            if (now and now['uuid'] == before['uuid'] and holder_stopped and
                not now['apps'] and now['total'] - now['free'] <= before_used + 1024):
                stable += 1
                if stable >= 2:
                    after = now; break
            else:
                stable = 0
            time.sleep(1)
        require(after is not None, '占位退出或显存回落无法确认；未触碰其他进程')
        require(probe_via_child() == 0, '控制器退出后锁未释放')
        finish(run, {'gpu': idx, 'before_used': before_used,
                     'during_used': during['total'] - during['free'],
                     'after_used': after['total'] - after['free'],
                     'reserved_mib': reserved, 'lock_blocked': True,
                     'holder_exited': True, 'lock_released': True})
    finally:
        if p.poll() is None:
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=5)
        # Never kill a PID recovered from state or NVML: only Popen above.
        # If the holder's own PID was reported, wait for parent-death cleanup.
        if holder_pid is not None:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not exited_or_zombie(holder_pid):
                time.sleep(.2)
            if not exited_or_zombie(holder_pid):
                raise RuntimeError('占位进程退出未确认；请只读检查该任务状态')


def show(run):
    try:
        st = json.loads((run / 'status.json').read_text(encoding='utf8'))
    except (OSError, ValueError):
        print('A阶段：尚无状态；请检查启动日志。'); return
    alive = False
    pid = st.get('pid')
    if type(pid) is int and pid > 1:
        try:
            argv = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
            alive = (str(SCRIPT).encode() in argv and str(run).encode() in argv and b'--status' not in argv)
        except OSError: pass
    print('+----------------------+------------------------------+')
    for name, value in [('阶段', st.get('phase', '未知')), ('测试进程', '运行中' if alive else '未发现'), ('说明', st.get('detail', '未知')),
                        ('显卡编号', st.get('gpu', '—')), ('占位MiB', st.get('reserved_mib', '—')),
                        ('占位前已用MiB', st.get('before_used', '—')),
                        ('占位时已用MiB', st.get('during_used', '—')),
                        ('回收后已用MiB', st.get('after_used', '—')),
                        ('第二控制器被挡', st.get('lock_blocked', '—')),
                        ('占位进程已退出', st.get('holder_exited', '—')),
                        ('控制器锁已释放', st.get('lock_released', '—'))]:
        print(f'| {name:<20} | {str(value):<28} |')
    print('+----------------------+------------------------------+')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--internal-controller', nargs=3)
    parser.add_argument('--internal-holder', type=int)
    parser.add_argument('--internal-probe', action='store_true')
    a = parser.parse_args()
    if a.internal_probe: return lock_probe()
    if a.internal_holder is not None: holder(a.internal_holder); return 0
    if a.internal_controller:
        parent, gpu, uuid = a.internal_controller
        return controller(int(parent), int(gpu), uuid)
    require(a.run_dir is not None and a.run_dir.is_absolute(), '必须指定绝对运行目录')
    if a.status: show(a.run_dir); return 0
    require(a.run_dir.parent == ROOT and a.run_dir.is_dir() and not a.run_dir.is_symlink(),
            '运行目录不属于A组目录')
    os.umask(0o077)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(RuntimeError('收到停止信号')))
    try:
        run_test(a.run_dir)
        show(a.run_dir)
        return 0
    except Exception as e:
        try:
            write_status(a.run_dir, '失败', str(e))
            report = a.run_dir / 'stage-a-report.md'
            if not report.exists():
                with report.open('x', encoding='utf8') as f:
                    f.write('# A 阶段自检未通过\n\n' + str(e) + '\n\n未加载模型或进行论文训练。\n')
        except Exception: pass
        print('A阶段未通过：', str(e), file=sys.stderr)
        return 1

if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('未完成：', type(exc).__name__, file=sys.stderr)
        sys.exit(1)
