"""Standalone Hydra trainer entry for RTX PRO 5000 Stage 2 & Stage 3."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import importlib
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
ALLOWED_REWARD_FUNCS = frozenset({
    ('pkg://eduardo.reward_function', 'eduardo_reward_func'),
    ('pkg://training.routeA_reward.reward_function', 'route_a_reward_func'),
})
ROUTEA_TECH_FAIL_MAX_PERCENT = 1
ROUTEA_GATED_REBOUND_MAX = 0.15
ROUTEA_GATED_ABS_MAX = 0.35
T017_VAL_GATED_REF = {25: 0.23, 50: 0.09, 75: 0.15, 100: 0.08}


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
    if (reward.path, reward.name) not in ALLOWED_REWARD_FUNCS:
        raise ValueError(f'奖励函数未绑定允许实现: path={reward.path!r} name={reward.name!r}')
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


def _load_t017_baseline_steps(path_str: str) -> dict[int, dict]:
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding='utf-8'))
    raw_steps = data.get('steps', {})
    return {int(k): v for k, v in raw_steps.items()}


def assess_routea_stage3_gates(
    *,
    step: int,
    steps_dir: Path,
    current_row: dict,
    t017_steps: dict[int, dict] | None = None,
) -> dict:
    """Evaluate Route A Stage 3 stop rules (hybrid tech fail >1%, self-curve gated ma5 rebound/cap)."""
    del t017_steps  # T-017 training logs lack per-step gated_rate; validation refs are reported separately.
    ra = current_row.get('routeA_metrics', {}).get('metrics', {})
    llm_calls = int(round(ra.get('eduardo/routeA_grader_llm_calls_total', 0.0)))
    tech_failed = int(round(ra.get('eduardo/routeA_grader_tech_failed_total', 0.0)))
    tech_fail_rate = (tech_failed / llm_calls) if llm_calls > 0 else 0.0
    tech_fail_exceeded = llm_calls > 0 and (100 * tech_failed > ROUTEA_TECH_FAIL_MAX_PERCENT * llm_calls)

    step_gated: dict[int, float] = {}
    for s in range(1, step):
        p = steps_dir / f'{s}.json'
        if p.is_file():
            try:
                prev = json.loads(p.read_text(encoding='utf-8'))
                val = prev.get('metrics', {}).get('eduardo/routeA_gated_rate')
                if isinstance(val, (int, float)):
                    step_gated[s] = float(val)
            except ValueError:
                pass
    cur_gated = ra.get('eduardo/routeA_gated_rate')
    if isinstance(cur_gated, (int, float)):
        step_gated[step] = float(cur_gated)

    def _ma5_at(target_step: int) -> float | None:
        vals = [step_gated[k] for k in range(max(1, target_step - 4), target_step + 1) if k in step_gated]
        return (sum(vals) / len(vals)) if vals else None

    gated_ma5 = _ma5_at(step)
    prior_ma5s = [m for s in range(1, step) if (m := _ma5_at(s)) is not None]
    min_prior_ma5 = min(prior_ma5s) if prior_ma5s else None
    rebound = (gated_ma5 - min_prior_ma5) if (gated_ma5 is not None and min_prior_ma5 is not None) else None

    gated_rebound_exceeded = step >= 10 and rebound is not None and rebound > ROUTEA_GATED_REBOUND_MAX
    gated_abs_exceeded = step >= 25 and gated_ma5 is not None and gated_ma5 > ROUTEA_GATED_ABS_MAX

    failures = []
    if tech_fail_exceeded:
        failures.append(f'hybrid_tech_fail={tech_failed}/{llm_calls}>1%')
    if gated_rebound_exceeded:
        failures.append(
            f'step>={10} gated_ma5={gated_ma5:.4f} exceeds min_prior_ma5={min_prior_ma5:.4f} by {rebound:.4f}>0.15'
        )
    if gated_abs_exceeded:
        failures.append(f'step>={25} gated_ma5={gated_ma5:.4f}>0.35')

    return {
        'llm_calls': llm_calls,
        'tech_failed': tech_failed,
        'tech_fail_rate': round(tech_fail_rate, 6),
        'tech_fail_exceeded': tech_fail_exceeded,
        'gated_rate': cur_gated,
        'gated_ma5': round(gated_ma5, 6) if gated_ma5 is not None else None,
        'min_gated_ma5_prior': round(min_prior_ma5, 6) if min_prior_ma5 is not None else None,
        'gated_ma5_rebound': round(rebound, 6) if rebound is not None else None,
        'gated_rebound_exceeded': gated_rebound_exceeded,
        'gated_abs_exceeded': gated_abs_exceeded,
        't017_val_gated_ref': T017_VAL_GATED_REF.get(step),
        'passed': not failures,
        'failures': failures,
    }


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
            self.formal_routea_metrics_dir = os.environ.get('ROUTEA_STEP_METRICS_DIR', '').strip()
            if not self.formal_routea_metrics_dir:
                try:
                    cfg_dir = self.config.reward.custom_reward_function.reward_kwargs.get('routeA_metrics_dir', '')
                    if cfg_dir:
                        self.formal_routea_metrics_dir = str(cfg_dir).strip()
                except Exception:
                    pass
            if not self.formal_routea_metrics_dir and (self.formal_work / 'routeA-metrics').is_dir():
                self.formal_routea_metrics_dir = str(self.formal_work / 'routeA-metrics')
            self.formal_routea_enforce = os.environ.get('PRO5000_ROUTEA_ENFORCE_GATES', '0') == '1'
            self.formal_t017_steps = _load_t017_baseline_steps(os.environ.get('PRO5000_T017_BASELINE_JSON', '').strip())
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
            val_start_ns = time.monotonic_ns()
            try:
                out = super()._validate(merged=merged)
            finally:
                self.formal_validation = False
            val_end_ns = time.monotonic_ns()
            routea_metrics_dir = getattr(self, 'formal_routea_metrics_dir', None)
            if routea_metrics_dir:
                try:
                    from training.routeA_reward.reward_function import summarize_metrics_window

                    val_ra = summarize_metrics_window(
                        routea_metrics_dir,
                        val_start_ns,
                        val_end_ns,
                        split='val',
                    )
                    val_metrics = val_ra.get('metrics', {})
                    t017_ref = T017_VAL_GATED_REF.get(int(self.global_steps))
                    payload = {
                        'step': int(self.global_steps),
                        'wall_time': time.time(),
                        'n_rollouts': int(val_ra.get('n_rollouts', 0)),
                        'malformed_rows': int(val_ra.get('malformed_rows', 0)),
                        'metrics': val_metrics,
                        't017_val_gated_ref': t017_ref,
                    }
                    val_dir = self.formal_work / 'val-statistics'
                    val_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                    write_json(val_dir / f'{int(self.global_steps)}.json', payload)
                    t017_str = f'{float(t017_ref):.2f}' if t017_ref is not None else 'N/A'
                    print(
                        f"[VAL step={self.global_steps}] n={payload['n_rollouts']} "
                        f"reward={val_metrics.get('eduardo/routeA_reward_mean', 0.0):.3f} "
                        f"theta={val_metrics.get('eduardo/routeA_theta_mean', 0.0):.3f} "
                        f"gated={val_metrics.get('eduardo/routeA_gated_rate', 0.0):.3f} "
                        f"(t017_ref={t017_str})",
                        flush=True,
                    )
                except ImportError:
                    pass
            return out

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

            if self.formal_routea_metrics_dir:
                try:
                    from training.routeA_reward.reward_function import summarize_metrics_window

                    ra_summary = summarize_metrics_window(
                        self.formal_routea_metrics_dir,
                        self.formal_queue_start,
                        self.formal_queue_end,
                        split='train',
                    )
                    row['routeA_metrics'] = ra_summary
                    metrics.update(ra_summary.get('metrics', {}))
                except ImportError:
                    pass

            steps_dir = self.formal_work / 'step-statistics'
            if 'routeA_metrics' in row and row['routeA_metrics'].get('n_rollouts', 0) > 0:
                gates = assess_routea_stage3_gates(
                    step=step,
                    steps_dir=steps_dir,
                    current_row=row,
                    t017_steps=self.formal_t017_steps,
                )
                row['routeA_gates'] = gates
                if self.formal_routea_enforce and not gates['passed']:
                    write_json(steps_dir / f'{step}.json', row)
                    raise RuntimeError(f'路线 A 阶段 3 停机规则触发 (step={step}): {gates["failures"]}')

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
                extra_ra = ''
                if 'routeA_metrics' in row and row['routeA_metrics'].get('n_rollouts', 0) > 0:
                    ram = row['routeA_metrics']['metrics']
                    recent_secs = []
                    for s in range(max(1, step - 4), step):
                        sp = steps_dir / f'{s}.json'
                        if sp.is_file():
                            try:
                                prev_s = json.loads(sp.read_text(encoding='utf-8')).get('metrics', {}).get('timing_s/step')
                                if isinstance(prev_s, (int, float)):
                                    recent_secs.append(float(prev_s))
                            except ValueError:
                                pass
                    if isinstance(metrics.get('timing_s/step'), (int, float)):
                        recent_secs.append(float(metrics['timing_s/step']))
                    avg5_s = (sum(recent_secs) / len(recent_secs)) if recent_secs else 0.0
                    rgates = row.get('routeA_gates', {})
                    ma5_g = rgates.get('gated_ma5')
                    ma5_str = f'{float(ma5_g):.3f}' if ma5_g is not None else 'N/A'
                    t017_ref = rgates.get('t017_val_gated_ref')
                    t017_str = f'{float(t017_ref):.2f}' if t017_ref is not None else 'N/A'
                    extra_ra = (
                        f" avg5_s={avg5_s:.1f}s reward={ram.get('eduardo/routeA_reward_mean', 0.0):.3f} "
                        f"theta={ram.get('eduardo/routeA_theta_mean', 0.0):.3f} "
                        f"gated={ram.get('eduardo/routeA_gated_rate', 0.0):.3f} (ma5={ma5_str}, t017_val={t017_str})"
                    )
                print(
                    f"[CKPT step={step}] step_s={metrics.get('timing_s/step', 0.0):.1f}s "
                    f"gen={phase_timings.get('gen', {}).get('minutes', 0.0):.2f}m "
                    f"old={phase_timings.get('old_log_prob', {}).get('minutes', 0.0):.2f}m "
                    f"upd={phase_timings.get('update_actor', {}).get('minutes', 0.0):.2f}m "
                    f"ckpt_gb={info['size_gb']} disk_free_gb={row['disk_free_gb']}{extra_ra}",
                    flush=True,
                )

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
        reward_mode = os.environ.get('PRO5000_REWARD_MODE', 'author').strip() or 'author'
        data_dir_raw = os.environ.get('PRO5000_DATA_DIR', '').strip()
        data_dir = Path(data_dir_raw) if data_dir_raw else None
        theta_scale = float(os.environ.get('PRO5000_THETA_SCALE', '0.25'))
        ladder_climbs = int(os.environ.get('PRO5000_LADDER_CLIMBS', '8'))
        grader_conc = int(os.environ.get('PRO5000_GRADER_CONC', '32'))

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
            reward_mode=reward_mode,
            data_dir=data_dir,
            theta_scale=theta_scale,
            ladder_num_climbs=ladder_climbs,
            grader_max_concurrency=grader_conc,
        )
        auto_set_device(cfg)
        cfg = migrate_legacy_reward_impl(cfg)
        keep_teacher_budgets(cfg)
        if check_only:
            rf_path = cfg.reward.custom_reward_function.path
            rf_name = cfg.reward.custom_reward_function.name
            if rf_path == 'pkg://training.routeA_reward.reward_function':
                mod = importlib.import_module('training.routeA_reward.reward_function')
                fn = getattr(mod, rf_name, None)
                if not callable(fn):
                    raise ValueError(f'Route A 奖励函数不可调用: {rf_path}:{rf_name}')
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
