"""Frozen recipe overrides and Hydra preflight checks for RTX PRO 5000."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

TEACHER_MODEL_DEFAULT = Path('/data/models/Qwen3.5-4B')
AUTHOR_COMMIT = 'e8c89b7293e65e632149571b16a1c2d4e33c77fe'
SPLIT_ROWS = {'train.parquet': 4712, 'val.parquet': 100}


def build_overrides(
    *,
    base: Path,
    work: Path,
    ckpt_dir: Path,
    raytmp: Path,
    student_name: str,
    interaction: Path,
    total_steps: int,
    save_freq: int,
    test_freq: int,
    tutor_gpus: int = 4,
    optimizer_offload: bool = True,
    student_concurrency: int = 32,
    judge_concurrency: int = 16,
    resume_step: int | None = None,
) -> list[str]:
    data_dir = base / 'data' / 'author_prepared'
    overrides = [
        'trainer.nnodes=1',
        f'trainer.n_gpus_per_node={tutor_gpus}',
        f'actor_rollout_ref.actor.fsdp_config.fsdp_size={tutor_gpus}',
        'data.train_batch_size=16',
        'actor_rollout_ref.actor.ppo_mini_batch_size=8',
        'actor_rollout_ref.actor.ppo_epochs=1',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.actor.use_dynamic_bsz=false',
        'actor_rollout_ref.actor.optim.lr=1e-6',
        'actor_rollout_ref.actor.policy_loss.loss_mode=dppo_tv',
        'actor_rollout_ref.actor.fsdp_config.param_offload=false',
        f'actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(optimizer_offload).lower()}',
        'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]',
        'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]',
        'actor_rollout_ref.rollout.n=16',
        'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.58',
        'actor_rollout_ref.rollout.free_cache_engine=true',
        'actor_rollout_ref.rollout.temperature=1.15',
        'actor_rollout_ref.rollout.calculate_log_probs=true',
        'actor_rollout_ref.rollout.agent.num_workers=2',
        'reward.num_workers=2',
        'reward_model.num_workers=null',
        'custom_reward_function.reward_kwargs.num_student_attempts=16',
        f'custom_reward_function.reward_kwargs.judge_concurrency={judge_concurrency}',
        f'custom_reward_function.reward_kwargs.student_concurrency={student_concurrency}',
        f'actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction}',
        f'vars.student_model_name={student_name}',
        'vars.judge_model_name=judge-local',
        f'data.train_files={data_dir}/train.parquet',
        f'data.val_files={data_dir}/val.parquet',
        f'trainer.total_training_steps={total_steps}',
        f'trainer.save_freq={save_freq}',
        f'trainer.test_freq={test_freq}',
        'trainer.val_before_train=false',
        'trainer.max_actor_ckpt_to_keep=null',
        'trainer.del_local_ckpt_after_load=false',
        f'trainer.default_local_dir={ckpt_dir}',
        'trainer.logger=[console]',
        "custom_reward_function.reward_kwargs.eval_judge_model=''",
        '+ray_kwargs.ray_init.address=local',
        f'+ray_kwargs.ray_init._temp_dir={raytmp}',
        '+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=queue_metrics.install',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.EDU_QUEUE_METRICS_DIR={work}/queue-metrics',
        '+ray_kwargs.ray_init.runtime_env.env_vars.EDUARDO_CONV_DIR=/dev/null',
    ]
    if resume_step is not None:
        overrides.extend([
            'trainer.resume_mode=resume_path',
            f'trainer.resume_from_path={ckpt_dir}/global_step_{resume_step}',
        ])
    else:
        overrides.append('trainer.resume_mode=disable')
    return overrides


def _expect(mapping, expected):
    for key, want in expected.items():
        got = mapping.get(key)
        if isinstance(want, float) and isinstance(got, str):
            try:
                got = float(got)
            except ValueError:
                pass
        if key not in mapping or got != want:
            raise ValueError(f'配置核对不符: {key}')


def pin_composed(
    config: dict,
    *,
    base: Path,
    work: Path,
    ckpt_dir: Path,
    student_name: str,
    interaction: Path,
    total_steps: int,
    save_freq: int,
    test_freq: int,
    tutor_gpus: int = 4,
    optimizer_offload: bool = True,
    student_concurrency: int = 32,
    judge_concurrency: int = 16,
    resume_step: int | None = None,
):
    trainer = config['trainer']
    ar = config['actor_rollout_ref']
    roll = ar['rollout']
    act = ar['actor']
    reward = config['custom_reward_function']['reward_kwargs']
    data_dir = base / 'data' / 'author_prepared'

    _expect(trainer, {
        'total_training_steps': total_steps,
        'save_freq': save_freq,
        'test_freq': test_freq,
        'val_before_train': False,
        'val_only': False,
        'logger': ['console'],
        'nnodes': 1,
        'n_gpus_per_node': tutor_gpus,
        'max_actor_ckpt_to_keep': None,
        'del_local_ckpt_after_load': False,
        'default_local_dir': str(ckpt_dir),
        'resume_mode': 'disable' if resume_step is None else 'resume_path',
        'resume_from_path': None if resume_step is None else str(ckpt_dir / f'global_step_{resume_step}'),
    })
    _expect(config['data'], {
        'train_batch_size': 16,
        'train_files': str(data_dir / 'train.parquet'),
        'val_files': str(data_dir / 'val.parquet'),
        'shuffle': True,
        'max_prompt_length': 2048,
        'max_response_length': 8192,
    })
    _expect(act, {
        'ppo_mini_batch_size': 8,
        'ppo_epochs': 1,
        'ppo_micro_batch_size_per_gpu': 1,
        'use_dynamic_bsz': False,
        'loss_agg_mode': 'token-mean',
        'use_kl_loss': False,
    })
    _expect(act['optim'], {'lr': 1e-6})
    _expect(act['policy_loss'], {'loss_mode': 'dppo_tv'})
    _expect(act['fsdp_config'], {
        'fsdp_size': tutor_gpus,
        'param_offload': False,
        'optimizer_offload': optimizer_offload,
    })
    _expect(act['checkpoint'], {
        'save_contents': ['model', 'extra'],
        'load_contents': ['model', 'extra'],
    })
    _expect(roll, {
        'n': 16,
        'temperature': 1.15,
        'gpu_memory_utilization': 0.58,
        'free_cache_engine': True,
        'tensor_model_parallel_size': 1,
        'calculate_log_probs': True,
        'max_model_len': 10240,
    })
    _expect(roll['multi_turn'], {'enable': True, 'interaction_config_path': str(interaction)})
    _expect(roll['agent'], {'num_workers': 2})
    _expect(config['reward'], {'num_workers': 2})
    _expect(config['reward_model'], {'num_workers': None})
    _expect(reward, {
        'num_student_attempts': 16,
        'max_thinking_tokens_per_turn': 512,
        'max_tokens_per_turn': 150,
        'judge_votes': 3,
        'judge_hard_gate': True,
        'omit_teacher_turns': True,
        'test_on_transfer': True,
        'enable_thinking': True,
        'judge_concurrency': judge_concurrency,
        'student_concurrency': student_concurrency,
        'student_temperature': 0.6,
        'judge_temperature': 0.6,
        'eval_judge_model': '',
        'judge_model': 'judge-local',
        'student_model': student_name,
    })
    ray_env = config['ray_kwargs']['ray_init']['runtime_env']
    _expect(ray_env, {'worker_process_setup_hook': 'queue_metrics.install'})
    _expect(ray_env['env_vars'], {'EDU_QUEUE_METRICS_DIR': str(work / 'queue-metrics')})


def run_preflight(
    *,
    python_exe: Path,
    trainer_entry: Path,
    repo: Path,
    work: Path,
    tutor_model: Path,
    student_api_base: str,
    judge_api_base: str,
    overrides: list[str],
    check_env: dict[str, str],
):
    cmd = [
        str(python_exe), '-B', str(trainer_entry), '--check-only',
        '--config-path', str(repo / 'eduardo' / 'configs'),
        '--config-name', 'eduardo_base',
        *overrides,
        f'hydra.run.dir={work}/hydra-preflight',
        'hydra.output_subdir=null',
    ]
    env = dict(
        check_env,
        CUDA_VISIBLE_DEVICES='',
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONPATH=f"{trainer_entry.parent}:{repo}",
        WORK_DIR=str(work),
        EDUARDO_RECIPE_DIR=str(repo / 'eduardo'),
        MODEL_PATH=str(tutor_model),
        EXPERIMENT='pro5000-stage2',
        SERVING_API_KEY='preflight-placeholder',
        STUDENT_API_BASE=student_api_base,
        JUDGE_API_BASE=judge_api_base,
        WANDB_MODE='disabled',
        HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        RAY_USAGE_STATS_ENABLED='0',
    )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    if proc.returncode or 'PRO5000_PRECHECK_OK' not in proc.stdout.splitlines():
        raise RuntimeError(f'Hydra 预检失败 rc={proc.returncode}: {proc.stderr[-600:]}')
