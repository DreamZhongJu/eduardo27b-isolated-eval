"""Standalone Hydra trainer entry for RTX PRO 5000 Stage 2 & Stage 3."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from monitor import MonitoringFailure, RunMonitor, excess_parse_failures
import pro5000_config as pcfg
import queue_metrics
from serving_proxy import summarize_proxy_window
from step_review_v4 import PHASES, TOKEN_KEY, compare_steps, per_token
from telemetry import summarize as summarize_telemetry

ACTOR_KEYS = ('actor/grad_norm', 'actor/pg_loss', 'actor/lr')
TIMING_KEYS = (
    'timing_s/step', 'timing_s/gen', 'timing_s/old_log_prob', 'timing_s/update_actor',
    'timing_s/reward', 'timing_s/save_checkpoint', 'timing_s/update_weights', 'timing_s/testing',
)
EXTRA_NUMERIC_PREFIXES = (
    'actor/', 'critic/', 'reward/', 'response_length/', 'eduardo/', 'perf/', 'training/',
)
MIN_EXTRA_FREE_BYTES = 20_000_000_000
ESTIMATED_MODEL_EXTRA_CKPT_BYTES = 20_000_000_000
STAGE2_STEP1_SECONDS = 1178.46
STEP1_JUDGE_MAX_SERVICE_P95_MS = 300_000.0


def assess_step1_judge(row: dict, judge_concurrency: int) -> dict:
    """Evaluate Stage 3 step-1 high-concurrency Judge acceptance criteria."""
    jp = row.get('judge_proxy', {})
    vote = row.get('vote_log_counts', {})
    jq = row.get('queues', {}).get('groups', {}).get('judge', {})
    judge_req = jp.get('total_chat_requests') or vote.get('requests', 0)
    non_2xx = jp.get('non_2xx', 0)
    proxy_err = jp.get('errors', 0)
    attempt_err = jq.get('attempt_errors', 0)
    call_err = jq.get('call_errors', 0)
    judge_parse = vote.get('parse', 0)
    parse_rate = (judge_parse / judge_req) if judge_req else 1.0
    service_p95_ms = jq.get('service_p95_ms')
    step_s = row.get('metrics', {}).get('timing_s/step')

    failures = []
    if judge_req <= 0:
        failures.append('judge_requests=0')
    if non_2xx != 0 or proxy_err != 0:
        failures.append(f'judge_non_2xx={non_2xx},proxy_errors={proxy_err}')
    if attempt_err != 0 or call_err != 0:
        failures.append(f'judge_attempt_errors={attempt_err},call_errors={call_err}')
    if judge_req > 0 and excess_parse_failures(judge_parse, judge_req):
        failures.append(f'judge_parse_fail={judge_parse}/{judge_req}>1%')
    if service_p95_ms is None or service_p95_ms >= STEP1_JUDGE_MAX_SERVICE_P95_MS:
        failures.append(f'judge_service_p95_ms={service_p95_ms}>={STEP1_JUDGE_MAX_SERVICE_P95_MS}')

    return {
        'judge_concurrency': judge_concurrency,
        'step_seconds': step_s,
        'stage2_step1_seconds': STAGE2_STEP1_SECONDS,
        'judge_calls': judge_req,
        'judge_non_2xx': non_2xx,
        'proxy_errors': proxy_err,
        'attempt_errors': attempt_err,
        'call_errors': call_err,
        'parse_failures': judge_parse,
        'parse_failure_rate': round(parse_rate, 6),
        'service_p95_ms': service_p95_ms,
        'service_p95_s': round(service_p95_ms / 1000.0, 3) if service_p95_ms is not None else None,
        'passed': not failures,
        'failures': failures,
    }


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
    temp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def dataloader_position(state) -> int:
    if isinstance(state, dict):
        num = state.get('_num_yielded')
        if type(num) is int and num >= 0:
            return num
        snap = state.get('_snapshot')
        if isinstance(snap, dict):
            parts = snap.get('_snapshot_step'), state.get('_steps_since_snapshot')
            if all(type(p) is int and p >= 0 for p in parts):
                return sum(parts)
    raise RuntimeError('原生数据加载位置不可核对')


def verify_model_extra_checkpoint(step_dir: Path, world_size: int = 4) -> dict:
    actor = step_dir / 'actor'
    if not (actor / 'fsdp_config.json').is_file():
        raise RuntimeError('检查点缺少 fsdp_config.json')
    fsdp_cfg = json.loads((actor / 'fsdp_config.json').read_text(encoding='utf-8'))
    if fsdp_cfg.get('world_size') != world_size:
        raise RuntimeError(f'检查点 world_size={fsdp_cfg.get("world_size")} != {world_size}')
    optim_files = list(actor.glob('optim_world_size_*'))
    if optim_files:
        raise RuntimeError(f'试跑检查点不应含优化器文件: {len(optim_files)}')
    for rank in range(world_size):
        for prefix in ('model', 'extra_state'):
            p = actor / f'{prefix}_world_size_{world_size}_rank_{rank}.pt'
            if not p.is_file() or p.stat().st_size <= 0:
                raise RuntimeError(f'检查点缺少 {p.name}')
    data_pt = step_dir / 'data.pt'
    if not data_pt.is_file() or data_pt.stat().st_size <= 0:
        raise RuntimeError('检查点缺少 data.pt')
    if not (actor / 'huggingface' / 'config.json').is_file():
        raise RuntimeError('检查点缺少 huggingface/config.json')
    total_bytes = sum(f.stat().st_size for f in step_dir.rglob('*') if f.is_file())
    return {'world_size': world_size, 'optim_files': 0, 'size_bytes': total_bytes, 'size_gb': round(total_bytes / 1e9, 2)}


def keep_teacher_budgets(config):
    """Preserve 150 + 512 = 662 on legacy custom_reward_function after veRL migration."""
    from omegaconf import open_dict

    reward = config.reward.custom_reward_function
    if reward.path != 'pkg://eduardo.reward_function' or reward.name != 'eduardo_reward_func':
        raise ValueError('奖励函数未绑定作者实现')
    keys = ('max_tokens_per_turn', 'max_thinking_tokens_per_turn')
    budgets = {k: reward.reward_kwargs[k] for k in keys}
    if tuple(budgets.values()) != (150, 512):
        raise ValueError('迁移后的教师预算不符 150/512')
    previous = config.get('custom_reward_function')
    if previous is None:
        with open_dict(config):
            config.custom_reward_function = {'reward_kwargs': budgets}
    elif any(previous.get('reward_kwargs', {}).get(k) != v for k, v in budgets.items()):
        raise ValueError('教师与奖励函数的预算不一致')
    if tuple(config.custom_reward_function.reward_kwargs[k] for k in keys) != (150, 512):
        raise ValueError('教师预算交接失败')


def extract_numeric_metrics(data: dict) -> dict[str, float]:
    out = {}
    for k, v in data.items():
        if (k in ACTOR_KEYS or k in TIMING_KEYS or k == TOKEN_KEY
                or any(k.startswith(pref) for pref in EXTRA_NUMERIC_PREFIXES)):
            if not isinstance(v, bool) and isinstance(v, Real) and math.isfinite(float(v)):
                out[k] = float(v)
    return out


def bound_trainer(native):
    class ObservedTrainer(native):
        def fit(self):
            import verl.utils.tracking as tracking

            self.formal_work = Path(os.environ['WORK_DIR'])
            self.formal_raytmp = Path(os.environ['RAY_TMPDIR'])
            self.formal_monitor = RunMonitor(self.formal_work, self.formal_raytmp)
            self.formal_start = int(os.environ.get('PRO5000_START_STEP', '0'))
            self.formal_tutor_gpus = int(os.environ.get('PRO5000_TUTOR_GPUS', '4'))
            self.formal_all_gpus = tuple(
                int(x) for x in os.environ.get('PRO5000_ALL_GPUS', '0,1,2,3,4,5,6,7').split(',') if x.strip()
            )
            self.formal_num_student_replicas = int(os.environ.get('PRO5000_STUDENT_REPLICAS', '2'))
            self.formal_judge_conc = int(os.environ.get('PRO5000_JUDGE_CONC', '16'))
            self.formal_check_step1_judge = os.environ.get('PRO5000_CHECK_STEP1_JUDGE', '0') == '1'
            self.formal_check_step2 = os.environ.get('PRO5000_CHECK_STEP2_RATIO', '0') == '1'
            self.formal_saved = None
            self.formal_validation = False
            self.formal_ckpt_bytes_hint = int(os.environ.get('PRO5000_CKPT_HINT_BYTES', str(ESTIMATED_MODEL_EXTRA_CKPT_BYTES)))

            replicas = len(self.async_rollout_manager.rollout_replicas)
            write_json(self.formal_work / 'rollout-layout.json', {'replicas': replicas, 'tensor_parallel': 1})
            if replicas != self.formal_tutor_gpus:
                raise MonitoringFailure(f'Tutor rollout 副本数 {replicas} != {self.formal_tutor_gpus}')

            original_tracking = tracking.Tracking
            observed = self
            from verl.trainer.ppo import ray_trainer
            native_timer = ray_trainer.marked_timer

            @contextmanager
            def observed_timer(name, *args, **kwargs):
                windows = getattr(self, 'formal_phase_windows', {})
                track = name in windows and not self.formal_validation
                started = time.time() if track else None
                with native_timer(name, *args, **kwargs) as timer:
                    yield timer
                if track:
                    windows[name].append((started, time.time()))

            ray_trainer.marked_timer = observed_timer

            class Tracking(original_tracking):
                def log(self, data, step):
                    super().log(data, step)
                    observed.report_step(data, step)

            tracking.Tracking = Tracking
            try:
                return super().fit()
            finally:
                tracking.Tracking = original_tracking
                ray_trainer.marked_timer = native_timer

        def _load_checkpoint(self):
            result = super()._load_checkpoint()
            wanted = self.formal_start
            if self.global_steps != wanted:
                raise RuntimeError(f'恢复步号 {self.global_steps} != 期望 {wanted}')
            data_step = dataloader_position(self.train_dataloader.state_dict()) if wanted else 0
            if data_step != wanted:
                raise RuntimeError(f'恢复数据位置 {data_step} != 期望 {wanted}')
            if wanted > 0:
                write_json(self.formal_work / f'restore_step_{wanted}.json', {
                    'global_step': self.global_steps,
                    'data_batches': data_step,
                    'restored_at': time.time(),
                })
            return result

        def _get_gen_batch(self, batch):
            # _validate() also builds gen batches via _get_gen_batch; never clobber training step state during validation.
            if not self.formal_validation:
                self.formal_queue_start = time.monotonic_ns()
                self.formal_step_wall = time.time()
                self.formal_phase_windows = {phase: [] for phase in PHASES}
                self.formal_vote_snap = self.formal_monitor.begin()
            return super()._get_gen_batch(batch)

        def _update_actor(self, batch):
            self.formal_queue_end = time.monotonic_ns()
            vote = self.formal_monitor.end(self.formal_vote_snap, enforce=True)
            data_batches = dataloader_position(self.train_dataloader.state_dict())
            if data_batches != self.global_steps:
                raise RuntimeError(f'训练批次位置 {data_batches} != {self.global_steps}')
            self.formal_vote = vote
            self.formal_data_batches = data_batches
            return super()._update_actor(batch)

        def _validate(self, merged=False):
            self.formal_validation = True
            try:
                return super()._validate(merged=merged)
            finally:
                self.formal_validation = False

        def _save_checkpoint(self):
            free = shutil.disk_usage(self.formal_work).free
            needed = self.formal_ckpt_bytes_hint + MIN_EXTRA_FREE_BYTES
            if free < needed:
                gen_spans = self.formal_phase_windows.get('gen', [])
                write_json(self.formal_work / f'save_blocked_step_{self.global_steps}.json', {
                    'step': self.global_steps,
                    'disk_free_gb': round(free / 1e9, 2),
                    'needed_gb': round(needed / 1e9, 2),
                    'vote_log_counts': getattr(self, 'formal_vote', {}),
                    'phase_windows': self.formal_phase_windows,
                    'queues': queue_metrics.summarize(
                        self.formal_work / 'queue-metrics', self.formal_queue_start, self.formal_queue_end
                    ),
                    'student_proxy': summarize_proxy_window(
                        self.formal_work / 'proxy-student.jsonl', gen_spans, self.formal_num_student_replicas
                    ),
                    'judge_proxy': summarize_proxy_window(self.formal_work / 'proxy-judge.jsonl', gen_spans, 1),
                    'telemetry': summarize_telemetry(
                        self.formal_work / 'tutor-telemetry.jsonl', self.formal_phase_windows, self.formal_all_gpus
                    ),
                })
                raise RuntimeError(f'存盘前 /data 剩余 {free / 1e9:.2f} GB < 一份检查点+20G ({needed / 1e9:.2f} GB)')
            result = super()._save_checkpoint()
            self.formal_saved = self.global_steps
            return result

        def report_step(self, data, step):
            if step < 1 or step != self.global_steps or int(data.get('training/global_step', -1)) != step:
                raise RuntimeError(f'veRL 指标步号不一致: step={step}, global_steps={self.global_steps}')
            self.formal_monitor.poll()
            metrics = extract_numeric_metrics(data)
            try:
                phase_timings = per_token(metrics)
            except ValueError:
                phase_timings = {}

            gen_spans = self.formal_phase_windows.get('gen', [])
            student_proxy = summarize_proxy_window(
                self.formal_work / 'proxy-student.jsonl', gen_spans, self.formal_num_student_replicas
            )
            judge_proxy = summarize_proxy_window(self.formal_work / 'proxy-judge.jsonl', gen_spans, 1)
            queues = queue_metrics.summarize(
                self.formal_work / 'queue-metrics', self.formal_queue_start, self.formal_queue_end
            )
            telemetry = summarize_telemetry(
                self.formal_work / 'tutor-telemetry.jsonl', self.formal_phase_windows, self.formal_all_gpus
            )

            row = {
                'step': step,
                'data_batches': getattr(self, 'formal_data_batches', step),
                'vote_log_counts': getattr(self, 'formal_vote', {}),
                'metrics': metrics,
                'phase_timings': phase_timings,
                'phase_windows': self.formal_phase_windows,
                'queues': queues,
                'student_proxy': student_proxy,
                'judge_proxy': judge_proxy,
                'telemetry': telemetry,
                'disk_free_gb': round(shutil.disk_usage(self.formal_work).free / 1e9, 2),
                'wall_start': self.formal_step_wall,
                'wall_end': time.time(),
            }

            if self.formal_saved == step:
                import torch

                ckpt_root = Path(self.config.trainer.default_local_dir) / f'global_step_{step}'
                info = verify_model_extra_checkpoint(ckpt_root, world_size=self.formal_tutor_gpus)
                data_state = torch.load(ckpt_root / 'data.pt', map_location='cpu', weights_only=False)
                if dataloader_position(data_state) != step:
                    raise RuntimeError('落盘 data.pt 步号不符')
                self.formal_ckpt_bytes_hint = max(self.formal_ckpt_bytes_hint, info['size_bytes'])
                row['checkpoint_saved'] = info
                write_json(self.formal_work / f'ckpt-verified-{step}.json', {'step': step, **info})
                print(
                    f"[CKPT step={step}] step_s={metrics.get('timing_s/step', 0.0):.1f}s "
                    f"gen={phase_timings.get('gen', {}).get('minutes', 0.0):.2f}m "
                    f"old={phase_timings.get('old_log_prob', {}).get('minutes', 0.0):.2f}m "
                    f"upd={phase_timings.get('update_actor', {}).get('minutes', 0.0):.2f}m "
                    f"ckpt_gb={info['size_gb']} disk_free_gb={row['disk_free_gb']}",
                    flush=True,
                )

            steps_dir = self.formal_work / 'step-statistics'
            write_json(steps_dir / f'{step}.json', row)

            if step == 1:
                s1_test = assess_step1_judge(row, self.formal_judge_conc)
                write_json(self.formal_work / 'step1_judge_test.json', s1_test)
                print(
                    f"[STEP1] step_s={s1_test['step_seconds']} (vs stage2 {STAGE2_STEP1_SECONDS}s) "
                    f"judge_conc={s1_test['judge_concurrency']} non_2xx={s1_test['judge_non_2xx']} "
                    f"attempt_errors={s1_test['attempt_errors']} parse_fail={s1_test['parse_failure_rate']:.4%} "
                    f"service_p95_s={s1_test['service_p95_s']} passed={s1_test['passed']}",
                    flush=True,
                )
                if self.formal_check_step1_judge and not s1_test['passed']:
                    raise RuntimeError(
                        f"第 1 步高并发裁判测试未通过: {s1_test['failures']}；请改回 --judge-concurrency 16 从第 0 步重跑"
                    )

            if step == 2:
                first_row = json.loads((steps_dir / '1.json').read_text(encoding='utf-8'))
                try:
                    cmp = compare_steps(first_row, row)
                except (KeyError, ValueError):
                    cmp = {'passed': False, 'missing': True}
                write_json(self.formal_work / 'step2-comparison.json', cmp)
                if self.formal_check_step2 and not cmp.get('passed'):
                    raise RuntimeError(f'第 2 步性能超 1.3 倍上限: {cmp}')

    return ObservedTrainer


def run(check_only: bool):
    import hydra
    import ray
    from omegaconf import OmegaConf
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer import main_ppo
    from verl.utils.device import auto_set_device

    class TaskRunner(main_ppo.TaskRunner):
        def run(self, cfg):
            native = main_ppo.RayPPOTrainer
            main_ppo.RayPPOTrainer = bound_trainer(native)
            try:
                return super().run(cfg)
            finally:
                main_ppo.RayPPOTrainer = native

    @hydra.main(config_path=None, config_name=None, version_base=None)
    def execute(cfg):
        base = Path(os.environ['PRO5000_BASE'])
        work = Path(os.environ['WORK_DIR'])
        ckpt_dir = Path(os.environ['PRO5000_CKPT_DIR'])
        start = int(os.environ.get('PRO5000_START_STEP', '0')) or None
        total_steps = int(os.environ['PRO5000_TOTAL_STEPS'])
        save_freq = int(os.environ['PRO5000_SAVE_FREQ'])
        test_freq = int(os.environ['PRO5000_TEST_FREQ'])
        tutor_gpus = int(os.environ.get('PRO5000_TUTOR_GPUS', '4'))
        offload = os.environ.get('PRO5000_OFFLOAD', 'true') == 'true'
        student_conc = int(os.environ.get('PRO5000_STUDENT_CONC', '32'))
        judge_conc = int(os.environ.get('PRO5000_JUDGE_CONC', '16'))
        max_ckpt_raw = os.environ.get('PRO5000_MAX_CKPT_KEEP', '').strip()
        max_ckpt_keep = int(max_ckpt_raw) if max_ckpt_raw and max_ckpt_raw.lower() != 'none' else None

        pcfg.pin_composed(
            OmegaConf.to_container(cfg, resolve=True),
            base=base,
            work=work,
            ckpt_dir=ckpt_dir,
            student_name=cfg.vars.student_model_name,
            interaction=Path(cfg.actor_rollout_ref.rollout.multi_turn.interaction_config_path),
            total_steps=total_steps,
            save_freq=save_freq,
            test_freq=test_freq,
            tutor_gpus=tutor_gpus,
            optimizer_offload=offload,
            student_concurrency=student_conc,
            judge_concurrency=judge_conc,
            max_actor_ckpt_to_keep=max_ckpt_keep,
            resume_step=start,
        )
        auto_set_device(cfg)
        cfg = migrate_legacy_reward_impl(cfg)
        keep_teacher_budgets(cfg)
        if check_only:
            print('PRO5000_PRECHECK_OK', flush=True)
            return
        print('PRO5000_TUTOR_BUDGET 150+512=662', flush=True)
        main_ppo.run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(TaskRunner))

    def stop(_signal, _frame):
        raise SystemExit('收到停止信号')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        execute()
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    check = '--check-only' in sys.argv
    if check:
        sys.argv.remove('--check-only')
    run(check)
