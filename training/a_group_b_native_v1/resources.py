#!/usr/bin/env python3
"""B-stage resource layer. No training is started by importing this module.

The controller must call acquire_lock() once, then use acquire_and_handoff().
The passed callbacks must start only this task's processes and stop only those
processes.  Cross-user GPU allocation is inherently not atomic; a successful
check is not a scheduler reservation or proof of exclusive access.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import select
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable

ROOT = Path('/workspace/A_group_repro_20261117')
STAGE_A = ROOT / 'stage-a-check-v1/stage_a.py'
STAGE_A_SHA256 = 'f504eddbb08699a08f04509496037f9ba1267c904fb024ac64c0098515feb003'
MIB = 1024 * 1024
GIB = 1024 * MIB
ROLE_ORDER = ('judge', 'tutor_1', 'tutor_2', 'student')
FULL_ROLES = frozenset(('judge', 'tutor_1', 'tutor_2'))

class ResourceError(RuntimeError):
    pass

class HandoffError(ResourceError):
    pass

class ServiceError(ResourceError):
    pass


def load_stage_a(path: Path = STAGE_A):
    """Reuse the exact A-stage file verified on A100, including lock and pdeath."""
    if path.is_symlink() or not path.is_file():
        raise ResourceError('A阶段脚本缺失')
    if hashlib.sha256(path.read_bytes()).hexdigest() != STAGE_A_SHA256:
        raise ResourceError('A阶段脚本哈希不一致')
    spec = importlib.util.spec_from_file_location('_stage_a_resource_reuse', path)
    if spec is None or spec.loader is None:
        raise ResourceError('无法加载A阶段脚本')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timestamp(clock: Callable[[], float] = time.time):
    return time.strftime('%Y-%m-%d %H:%M:%S %z', time.localtime(clock()))


@dataclass(frozen=True)
class Binding:
    role: str
    index: int
    uuid: str


class Holder:
    """Handle to a child created by this controller, never to an arbitrary PID."""
    def __init__(self, process, binding: Binding, amount_mib: int, acquired_wall: float,
                 acquired_mono: float):
        self.process, self.binding = process, binding
        self.amount_mib = int(amount_mib)
        self.acquired_wall, self.acquired_mono = acquired_wall, acquired_mono

    def alive(self):
        return self.process.poll() is None

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()  # only our direct Popen child
            self.process.wait(timeout=10)
        if self.process.stdout:
            self.process.stdout.close()


def is_eligible(stage_a, role, info):
    if role in FULL_ROLES:
        return stage_a.eligible(info)
    if role == 'student':
        return ('A100' in str(info.get('name', '')).upper()
                and info.get('total', 0) >= 78000
                and info.get('free', 0) >= 40 * 1024)
    raise ResourceError('未知角色')


def choose_candidates(stage_a, role, first, second, used_uuids):
    """Two read-only snapshots; the holder rechecks immediately before cudaMalloc."""
    result = []
    for i in sorted(first):
        a, b = first[i], second.get(i)
        if not b or a.get('uuid') != b.get('uuid') or b['uuid'] in used_uuids:
            continue
        if is_eligible(stage_a, role, a) and is_eligible(stage_a, role, b):
            result.append(Binding(role, i, b['uuid']))
    if role == 'student':
        result.sort(key=lambda b: (second[b.index]['util'], -second[b.index]['free'], b.index))
    return result


def child_holder(role: str, parent_pid: int, gpu_uuid: str):
    stage_a = load_stage_a()
    stage_a.parent_death(parent_pid)  # actual A-stage verified code
    if role not in ROLE_ORDER:
        raise ResourceError('未知角色')
    info = next((x for x in stage_a.snapshot().values() if x['uuid'] == gpu_uuid), None)
    if info is None or not is_eligible(stage_a, role, info):
        raise ResourceError('占位前显卡条件变化')
    import torch
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    if total < 78000 * MIB:
        raise ResourceError('显存规格不符')
    if role in FULL_ROLES:
        if free < total - 2 * GIB:
            raise ResourceError('CUDA初始化后整卡条件不满足')
        amount = free - 1536 * MIB
        if amount <= 70 * GIB:
            raise ResourceError('整卡预留显存不足')
    else:
        amount = int(total * 0.4)  # student vLLM's 0.4 budget; not a hard total GPU limit
        if free - amount < 8 * GIB:
            raise ResourceError('共享学生卡不足以保留8GiB余量')
    hold = torch.empty(amount, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    allocated = int(torch.cuda.memory_allocated(0))
    if allocated < amount:
        raise ResourceError('CUDA显存预留无法确认')
    print(json.dumps({'state': 'reserved', 'pid': os.getpid(),
                      'amount_mib': allocated // MIB}), flush=True)
    while True:
        time.sleep(5)
        if role == 'student':
            remaining, _ = torch.cuda.mem_get_info(0)
            if remaining < 8 * GIB:
                return 3  # release own allocation; don't act on any external process


def spawn_holder(stage_a, binding: Binding, work: Path, *, timeout=180):
    """Verify own actual allocation and a single new NVML process (PID namespaces may differ)."""
    end = time.monotonic() + timeout
    before_all = stage_a.snapshot()
    before = before_all.get(binding.index)
    if not before or before['uuid'] != binding.uuid or not is_eligible(stage_a, binding.role, before):
        raise ResourceError('预留前显卡条件变化')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=binding.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONDONTWRITEBYTECODE='1')
    for name in list(env):
        if name.upper().endswith(('_TOKEN', '_API_KEY', '_SECRET')):
            env.pop(name, None)
    log = (work / ('holder-' + binding.role + '.log')).open('ab')
    try:
        p = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                              '--holder', binding.role, str(os.getpid()), binding.uuid],
                             env=env, stdout=subprocess.PIPE, stderr=log,
                             close_fds=True, start_new_session=False)
    finally:
        log.close()
    try:
        ready, _, _ = select.select([p.stdout], [], [], max(0, end - time.monotonic()))
        if not ready:
            raise ResourceError('占位进程启动超时')
        line = p.stdout.readline()
        if not line:
            raise ResourceError("占位进程无就绪回执")
        try:
            data = json.loads(line)
        except (ValueError, TypeError) as e:
            raise ResourceError("占位回执无效") from e
        if data.get('state') != 'reserved' or data.get('pid') != p.pid or p.poll() is not None:
            raise ResourceError('显存占位未就绪')
        amount = int(data['amount_mib'])
        baseline = {pid for pid, _ in before['apps']}
        # Never signal a PID from NVML. It may be a host PID in another namespace.
        after = None
        for _ in range(20):
            if time.monotonic() >= end:
                break
            current = stage_a.snapshot().get(binding.index)
            if current and current['uuid'] == binding.uuid:
                extra = [(pid, used) for pid, used in current['apps'] if pid not in baseline]
                if len(extra) == 1 and extra[0][1] >= amount * .9:
                    after = current
                    break
            if p.poll() is not None:
                break
            time.sleep(1)
        if after is None or p.poll() is not None:
            raise ResourceError('无法核验新增占位进程')
        if binding.role in FULL_ROLES and len(after['apps']) != 1:
            raise ResourceError('整卡预留时发现其他进程')
        if binding.role == 'student' and after['free'] < 8 * 1024:
            raise ResourceError('学生共享卡余量低于8GiB')
        return Holder(p, binding, amount, time.time(), time.monotonic())
    except BaseException:
        if p.poll() is None:
            p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)
        if p.stdout:
            p.stdout.close()
        raise


class GPUResources:
    """Resource-only coordinator, at most four role bindings per trial.

    callbacks.launch_service(role, binding) starts our service and returns a
    process; callbacks.service_ready(role, binding, process) returns bool;
    callbacks.launch_tutor([tutor1,tutor2]) starts our trainer and returns it;
    callbacks.stop_started() stops only processes created by this controller.
    Callbacks must be bounded/nonblocking. No callback may release our lock.
    """
    def __init__(self, work: Path, stage_a=None, *, snapshot=None, spawn=None,
                 monotonic=time.monotonic, wall=time.time, sleep=time.sleep,
                 jitter=random.randint, poll_seconds=60, deadline_seconds=2700,
                 handoff_fail_limit=3):
        self.work = Path(work)
        if self.work.is_symlink() or not self.work.is_dir():
            raise ResourceError('运行目录不存在或是链接')
        self.a = stage_a if stage_a is not None else load_stage_a()
        self.snapshot = snapshot if snapshot is not None else self.a.snapshot
        self.spawn = spawn if spawn is not None else lambda b, timeout: spawn_holder(self.a, b, self.work, timeout=timeout)
        self.mono, self.wall, self.sleep, self.jitter = monotonic, wall, sleep, jitter
        self.poll_seconds, self.deadline = poll_seconds, deadline_seconds
        self.fail_limit = handoff_fail_limit
        self.lock_fd = None
        self.holders: dict[str, Holder] = {}
        self.round_no = 0
        self.handoff_failures = 0
        self.events = []
        self.phase = '准备'
        self.reason = ''
        self.state_path = self.work / 'resources.json'
        if self.state_path.exists():
            raise ResourceError('已有资源状态文件，不覆盖')

    def acquire_lock(self):
        """Uses exactly the verified A-stage own_lock() and the same .pilot.lock."""
        if self.lock_fd is not None:
            raise ResourceError('锁已由本控制器持有')
        try:
            self.lock_fd = self.a.own_lock()
        except BlockingIOError as e:
            raise ResourceError('已有控制器持有锁') from e
        self.update('准备', '已获得A阶段验证的任务锁')

    def event(self, role, action, binding=None):
        self.events.append({'role': role, 'action': action, 'time': _timestamp(self.wall),
                            'gpu': binding.index if binding else None})
        self.update(self.phase, self.reason)

    def update(self, phase, reason):
        self.phase, self.reason = phase, reason
        held = [{'role': role, 'gpu': h.binding.index, 'holder_pid': h.process.pid,
                 'amount_mib': h.amount_mib, 'acquired_at': _timestamp(lambda: h.acquired_wall),
                 'acquired_epoch': h.acquired_wall,
                 'held_seconds': max(0, int(self.mono() - h.acquired_mono)),
                 'alive': h.alive()} for role, h in sorted(self.holders.items())]
        data = {'phase': phase, 'reason': reason, 'round': self.round_no,
                'handoff_failures': self.handoff_failures, 'held': held,
                'events': self.events, 'updated_at': _timestamp(self.wall)}
        fd, tmp = tempfile.mkstemp(prefix='.resources-', dir=str(self.work))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, self.state_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def release_role(self, role):
        h = self.holders[role]
        h.release()
        del self.holders[role]
        self.event(role, '释放占位', h.binding)
        return h.binding

    def release_all(self):
        errors = []
        # Signal every own holder before waiting for any single CUDA process.
        for h in list(self.holders.values()):
            try:
                if h.process.poll() is None:
                    h.process.terminate()
            except ProcessLookupError:
                pass
            except Exception as e:
                errors.append(type(e).__name__)
        for role in tuple(self.holders):
            try:
                self.release_role(role)
            except Exception as e:
                errors.append(type(e).__name__)
        if errors:
            raise ResourceError('无法确认本任务占位释放')

    def _all_alive(self):
        return all(h.alive() for h in self.holders.values())

    def acquire(self):
        if self.lock_fd is None:
            raise ResourceError('未持有控制器锁')
        if self.holders:
            raise ResourceError('已有残余占位')
        while True:
            self.round_no += 1
            deadline = self.mono() + self.deadline
            self.update('凑卡', '逐卡预留中')
            while self.mono() < deadline:
                if not self._all_alive():
                    self.release_all()
                    break
                if deadline - self.mono() < 100:
                    self.sleep(max(0, deadline - self.mono()))
                    break
                first = self.snapshot(); self.sleep(5); second = self.snapshot()
                for role in ROLE_ORDER:
                    if role in self.holders:
                        continue
                    used = {h.binding.uuid for h in self.holders.values()}
                    for binding in choose_candidates(self.a, role, first, second, used):
                        remaining = deadline - self.mono()
                        if remaining <= 90:
                            break
                        try:
                            holder = self.spawn(binding, min(180, remaining - 90))
                            self.holders[role] = holder
                            self.event(role, '已预留', binding)
                            break
                        except ResourceError:
                            # Race, pressure, or CUDA failure: no external process is touched.
                            continue
                if len(self.holders) == 4 and self._all_alive():
                    self.update('凑齐', '四个角色均已预留')
                    return {role: h.binding for role, h in self.holders.items()}
                self.update('凑卡', '逐卡预留中')
                self.sleep(min(self.poll_seconds, max(0, deadline - self.mono())))
            self.release_all()
            pause = self.jitter(300, 900)
            self.update('冷却', f'本轮超时或占位失效；全部释放，等待{pause}秒')
            self.sleep(pause)

    def _after_release_ready(self, binding, *, timeout=60):
        end = self.mono() + timeout
        while self.mono() < end:
            current = self.snapshot().get(binding.index)
            if current and current.get('uuid') == binding.uuid and is_eligible(self.a, binding.role, current):
                return True
            self.sleep(2)
        return False

    def _wait_health(self, callbacks, role, binding, proc, timeout):
        end = self.mono() + timeout
        while self.mono() < end:
            if not self._all_alive() or proc.poll() is not None:
                raise ServiceError('占位或服务提前退出')
            if callbacks.service_ready(role, binding, proc):
                return
            self.sleep(5)
        raise ServiceError('服务健康检查超时')

    def acquire_and_handoff(self, callbacks, *, health_timeout=1200):
        """Keep tutor allocations until judge AND student are healthy.

        After three consecutive handoff failures, release own allocations, stop
        own launched services via callback, then raise. No automatic training retry.
        """
        if self.lock_fd is None:
            raise ResourceError('未持有控制器锁')
        while self.handoff_failures < self.fail_limit:
            self.acquire()
            self.update('交接', '先启动裁判和学生服务')
            try:
                services = {}
                for role in ('judge', 'student'):
                    binding = self.release_role(role)
                    if not self._after_release_ready(binding):
                        raise HandoffError('服务卡交接复核失败')
                    self.event(role, '启动服务', binding)
                    proc = callbacks.launch_service(role, binding)
                    if proc is None:
                        raise ServiceError('服务启动失败')
                    services[role] = (binding, proc)
                    self._wait_health(callbacks, role, binding, proc, health_timeout)
                    self.event(role, '服务健康', binding)
                if not self._all_alive():
                    raise HandoffError('Tutor占位提前退出')
                for role, (binding, proc) in services.items():
                    if proc.poll() is not None or not callbacks.service_ready(role, binding, proc):
                        raise ServiceError('启动训练前服务健康复核失败')
                tutors = [self.release_role(role) for role in ('tutor_1', 'tutor_2')]
                for binding in tutors:
                    if not self._after_release_ready(binding):
                        raise HandoffError('Tutor卡交接复核失败')
                self.event('tutor', '启动训练', tutors[0])
                trainer = callbacks.launch_tutor(tutors)
                if trainer is None:
                    raise ServiceError('Tutor启动失败')
                self.update('运行', '角色交接完成；训练由调用方监督')
                return trainer
            except Exception as exc:
                try:
                    callbacks.stop_started()
                finally:
                    self.release_all()
                if not isinstance(exc, HandoffError):
                    self.update('失败', '服务启动或健康检查失败：' + type(exc).__name__)
                    raise
                self.handoff_failures += 1
                self.update('交接失败', f'第{self.handoff_failures}次交接失败：{type(exc).__name__}')
                if self.handoff_failures >= self.fail_limit:
                    raise HandoffError('连续3次交接失败，停止并报告') from exc
                self.sleep(30)

    def close(self):
        # Retain the lock if an own holder cannot be confirmed dead.
        self.release_all()
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None



def status_rows(work: Path):
    """Aggregate only. Durations reflect current wall time; liveness is last snapshot."""
    path = Path(work) / 'resources.json'
    if not path.is_file():
        return {'phase': '尚无资源状态', 'round': 0, 'held': [], 'events': []}
    state = json.loads(path.read_text(encoding='utf-8'))
    for entry in state.get('held', []):
        epoch = entry.get('acquired_epoch')
        entry['held_seconds'] = max(0, int(time.time()-epoch)) if epoch else None
    return state


def print_status(work: Path):
    data = status_rows(work)
    print('阶段:', data.get('phase'), ' | 凑卡轮次:', data.get('round'),
          ' | 交接失败次数:', data.get('handoff_failures', 0))
    print('| 角色 | 显卡 | 开始占位时间 | 占用秒数 | 进程状态（最近快照） |')
    print('|------|------|--------------|----------|----------------------|')
    for h in data.get('held', []):
        print('| {} | {} | {} | {} | {} |'.format(h['role'], h['gpu'], h['acquired_at'],
              h.get('held_seconds', '未知'), '存活' if h['alive'] else '已退出'))
    print('| 角色 | 显卡 | 交接动作 | 时间 |')
    print('|------|------|----------|------|')
    for event in data.get('events', []):
        if event['action'] in ('释放占位', '启动服务', '服务健康', '启动训练'):
            print('| {} | {} | {} | {} |'.format(event['role'], event.get('gpu'), event['action'], event['time']))
    print('说明：显存峰值按本任务进程归属核验；未实现时显示未知。')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holder', nargs=3, metavar=('ROLE', 'PARENT_PID', 'GPU_UUID'))
    args = parser.parse_args()
    if args.holder:
        role, parent, uuid = args.holder
        return child_holder(role, int(parent), uuid)
    parser.error('仅供B阶段控制器调用')


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print('预留未完成：' + type(exc).__name__, file=sys.stderr)
        sys.exit(1)
