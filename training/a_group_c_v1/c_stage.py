"""Run exactly one training step and its native local validation on the A100 host."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a_group_b_native_v1'))
import b_stage as pilot
import verify_checkpoints as ckpt
import c_config
import c_metrics
import c_status

preflight_config = c_config.preflight_config
deviations = HERE / 'DEVIATIONS.md'


class Callbacks(pilot.Callbacks):
    experiment = c_config.EXPERIMENT
    trainer_entry = HERE / 'c_trainer_entry.py'
    overrides = staticmethod(c_config.training_overrides)

    def __init__(self, *args):
        super().__init__(*args)
        self.base.update(EDUARDO_CONV_DIR=str(self.work / 'eduardo_conversations'),
                         RAY_USAGE_STATS_ENABLED='0', NO_PROXY='127.0.0.1,localhost,::1',
                         no_proxy='127.0.0.1,localhost,::1', VLLM_LOGGING_LEVEL='INFO')
        self.access_config = c_metrics.write_access_config(self.work)
        self.generations = {}

    def spawn(self, name, args, env):
        # A handoff may retry before the trainer starts. Keep receipts/logs for
        # earlier launches without adding their calls to the actual training run.
        attempt = self.generations.get(name, 0)
        receipt = self.work / f'exit-{name}.json'
        if attempt:
            for path in (self.work / f'{name}.log', receipt):
                if path.exists():
                    path.rename(path.with_name(f'handoff-{attempt}-{path.name}'))
        self.generations[name] = attempt + 1
        if name in pilot.PORTS:
            args = [*args, '--log-config-file', str(self.access_config)]
        return super().spawn(name, args, dict(env, EDU_ROLE_EXIT_RECEIPT=str(receipt)))

    def stop_started(self):
        # Orderly completion: trainer exits naturally. On errors, stop it first.
        for name in ('trainer', 'student', 'judge', 'cpu_merge'):
            proc = self.process_by_role.get(name)
            if proc is None or proc.poll() is not None:
                continue
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Guard can spend 17 s waiting for the app + 15 s clearing descendants.
            proc.wait(timeout=55)


class CFailure(RuntimeError):
    """Only fixed diagnostic labels may be returned to the terminal."""


def reason(exc):
    if isinstance(exc, CFailure) or str(exc) in ('检测到 fallback，C 无效',
            'C Hydra 配置或迁移后教师预算预检失败', 'C 学生交互配置异常'):
        return str(exc)
    if str(exc).startswith('C 配置不符: '):
        return str(exc)
    return pilot.safe_reason(exc)


def acceptance(work, **fields):
    path = work / 'c_acceptance.json'
    values = json.loads(path.read_text()) if path.exists() else {}
    values.update(fields)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(values) + '\n')
    os.replace(temp, path)


def confirm_exit(work, role, natural):
    try:
        record = json.loads((work / f'exit-{role}.json').read_text())
        valid = (type(record.get('exit_code')) is int and type(record.get('observed_ns')) is int
                 and record.get('requested') is (not natural))
        return valid and (not natural or record['exit_code'] == 0)
    except (OSError, ValueError):
        return False


def finish(cb, scanner, update):
    work = cb.work
    update('服务收尾')
    cb.stop_started()
    if scanner.scan():
        raise CFailure('检测到 fallback，C 无效')
    if not all(confirm_exit(work, role, role == 'trainer') for role in ('trainer','student','judge')):
        raise CFailure('训练器或服务退出回执不完整或意外退出')
    step = work / 'verl_ckpts' / c_config.EXPERIMENT / 'global_step_1'
    out = work / 'eval_checkpoints' / c_config.EXPERIMENT / 'step_1'
    update('CPU合并')
    acceptance(work, merge_verified=False)
    if out.exists() or (step.parent / 'latest_checkpointed_iteration.txt').read_text().strip() != '1':
        raise CFailure('第1步检查点步号或合并目录异常')
    source, n = ckpt.inspect_input(step)
    if n != 5:
        raise CFailure('第1步检查点训练状态缺项')
    started = time.monotonic()
    merge = cb.spawn('cpu_merge', [str(pilot.PYTHON), '-B', '-m', 'verl.model_merger', 'merge',
                     '--backend', 'fsdp', '--local_dir', str(step / 'actor'), '--target_dir', str(out)],
                     dict(cb.base, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(pilot.REPO)))
    if merge.wait() != 0 or not confirm_exit(work, 'cpu_merge', True):
        raise CFailure('作者 CPU 合并工具失败')
    shards, keys = ckpt.inspect_export(out, source)
    acceptance(work, merge_verified=True, merge_shards=shards, merge_keys=keys,
               merge_seconds=round(time.monotonic()-started, 2))
    update('统计核对')
    with (work / 'C统计.md').open('x') as report, redirect_stdout(report):
        metrics_code = c_metrics.report(pilot.ROOT, work)
    acceptance(work, metrics_exit=metrics_code)
    if metrics_code == 2:
        raise CFailure('事后发现技术失败，C 无效')
    if metrics_code:
        raise CFailure('单步已结束，汇总证据不足；请审阅 C统计.md')
    update('待用户确认', '单步、末尾验证、CPU合并及统计检查已完成')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='c_paper_v1')
    parser.add_argument('--run', action='store_true', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    work = pilot.ROOT / 'pilots' / args.name
    try:
        pilot.run(args.name, sys.modules[__name__])
        exit_code = 0
    except BaseException as exc:
        print('C 未完成：' + reason(exc), file=sys.stderr)
        exit_code = 1
    if (work / 'orchestration.json').is_file():
        state = c_status.base.read_obj(work / 'orchestration.json')
        if state.get('controller_pid') == os.getpid():
            if exit_code and (work / 'trainer.log').exists() and not (work / 'C统计.md').exists():
                with (work / 'C统计.md').open('x', encoding='utf-8') as out, redirect_stdout(out):
                    try:
                        metrics_code = c_metrics.report(pilot.ROOT, work)
                    except Exception as exc:
                        print('中断时统计未齐，错误类型', type(exc).__name__)
                        metrics_code = 3
                acceptance(work, metrics_exit=metrics_code)
            with (work / 'C阶段报告.md').open('x', encoding='utf-8') as out, redirect_stdout(out):
                c_status.report(work)
                print()
                if (work / 'C统计.md').is_file():
                    print((work / 'C统计.md').read_text())
                print('运行偏离与冻结配置见本目录 deviations.md。')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
