"""One-step C reproduction recipe, composed and checked on CPU before allocation."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/workspace/A_group_repro_20261117")
DATA = ROOT / "data/author_prepared"
REPO = ROOT / "source/EduardoRL"
TUTOR = ROOT / "models/Qwen3.5-4B"
PYTHON = ROOT / "envs/env-candidate-vllm0191/bin/python"
EXPERIMENT = "c-paper-one-step"
STUDENT_API_BASE = "http://127.0.0.1:18861/v1"
JUDGE_API_BASE = "http://127.0.0.1:18862/v1"

# Frozen public input and selections from training/a_group/author_pool.zip.
AUTHOR_COMMIT = "e8c89b7293e65e632149571b16a1c2d4e33c77fe"
DATA_REVISION = "5d7529f06359f31b692c92634f5cc11c8b7953af"
AUTHOR_TRAIN_SHA = "5171c9891b502f9792b14ac2b15684ce85b8ceb2593689ed3e88b9f349b25d61"
SELECTION_SHA = {
    "training_pool.jsonl": "1bdea34377d617ce54ccfe59ae353cdd39849b209a961ddfa5c1c7ff7e30fe19",
    "validation.jsonl": "16c8c2fe842a0d5853eff8d63ed1c793d9e44e749cbe4f7a0f1a2775a98d6305",
}
SPLIT_ROWS = {"train.parquet": 4712, "val.parquet": 100}


def training_overrides(work, raytmp, student_name, interaction):
    """Author base + user.yaml; keep the positive native save/test intervals."""
    return [
        "trainer.nnodes=1", "trainer.n_gpus_per_node=2",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
        "data.train_batch_size=16", "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.optim.lr=1e-6", "actor_rollout_ref.rollout.n=16",
        "custom_reward_function.reward_kwargs.num_student_attempts=16",
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "reward.num_workers=2", "reward_model.num_workers=null",
        "custom_reward_function.reward_kwargs.judge_concurrency=16",
        "custom_reward_function.reward_kwargs.student_concurrency=16",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction}",
        f"vars.student_model_name={student_name}", "vars.judge_model_name=judge-local",
        f"data.train_files={DATA}/train.parquet", f"data.val_files={DATA}/val.parquet",
        "trainer.total_training_steps=1", "trainer.resume_mode=disable",
        "trainer.logger=[console]", "custom_reward_function.reward_kwargs.eval_judge_model=''",
        "+ray_kwargs.ray_init.address=local", f"+ray_kwargs.ray_init._temp_dir={raytmp}",
    ]


def _expect(mapping, expected):
    for key, value in expected.items():
        actual = mapping.get(key)
        # PyYAML SafeLoader can read plain 1e-06 in a Hydra dump as a string.
        if isinstance(value, float) and isinstance(actual, str):
            try:
                actual = float(actual)
            except ValueError:
                pass
        if key not in mapping or actual != value:
            # Never include the actual setting: an endpoint or an environment
            # interpolation can carry sensitive information.
            raise ValueError(f"C 配置不符: {key}")


def check_composed(config, work, student_name, interaction):
    trainer = config["trainer"]
    ar = config["actor_rollout_ref"]
    roll = ar["rollout"]
    act = ar["actor"]
    reward = config["custom_reward_function"]["reward_kwargs"]
    _expect(trainer, {"total_training_steps": 1, "total_epochs": 5, "save_freq": 25,
                      "test_freq": 25, "val_before_train": False, "val_only": False,
                      "logger": ["console"], "resume_mode": "disable", "nnodes": 1,
                      "n_gpus_per_node": 2, "max_actor_ckpt_to_keep": 5,
                      "experiment_name": EXPERIMENT,
                      "default_local_dir": str(Path(work) / "verl_ckpts" / EXPERIMENT)})
    _expect(config["data"], {"train_batch_size": 16, "train_files": str(DATA / "train.parquet"),
                             "val_files": str(DATA / "val.parquet"), "train_max_samples": -1,
                             "val_max_samples": -1, "val_batch_size": None, "shuffle": True,
                             "max_response_length": 8192, "max_prompt_length": 2048})
    _expect(ar["model"], {"path": str(TUTOR)})
    _expect(act, {"ppo_mini_batch_size": 8, "loss_agg_mode": "token-mean", "use_kl_loss": False})
    _expect(act["optim"], {"lr": 1e-6})
    _expect(act["policy_loss"], {"loss_mode": "dppo_tv"})
    _expect(act["fsdp_config"], {"fsdp_size": 2})
    _expect(roll, {"n": 16, "temperature": 1.15, "gpu_memory_utilization": 0.58,
                   "free_cache_engine": True, "tensor_model_parallel_size": 2,
                   "calculate_log_probs": True, "max_model_len": 10240})
    _expect(roll["multi_turn"], {"enable": True, "interaction_config_path": str(interaction)})
    _expect(roll["val_kwargs"], {"n": 1, "temperature": 0.6, "top_p": 0.95, "do_sample": True})
    _expect(roll["agent"], {"num_workers": 2})
    _expect(config["reward"], {"num_workers": 2})
    _expect(config["reward_model"], {"num_workers": None})
    _expect(reward, {"num_student_attempts": 16, "max_thinking_tokens_per_turn": 512,
                     "max_tokens_per_turn": 150, "judge_votes": 3, "judge_hard_gate": True,
                     "omit_teacher_turns": True, "test_on_transfer": True, "enable_thinking": True,
                     "judge_concurrency": 16, "student_concurrency": 16, "student_temperature": 0.6,
                     "judge_temperature": 0.6, "eval_judge_model": "", "judge_model": "judge-local",
                     "student_model": student_name, "student_api_base": STUDENT_API_BASE,
                     "judge_api_base": JUDGE_API_BASE})


def check_pool(manifest, row_counts):
    # b_stage.validate() already verifies both actual parquet byte hashes.
    _expect(manifest, {"author_repository_commit": AUTHOR_COMMIT, "source_revision": DATA_REVISION,
                       "source_train_sha256": AUTHOR_TRAIN_SHA, "selection_sha256": SELECTION_SHA})
    for filename, expected in SPLIT_ROWS.items():
        _expect(manifest["output"][filename], {"rows": expected})
    _expect(row_counts, SPLIT_ROWS)


def preflight_config(work, raytmp, student_name, interaction):
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf

    manifest = json.loads((DATA / "materialization_manifest.json").read_text())
    check_pool(manifest, {name: pq.ParquetFile(DATA / name).metadata.num_rows for name in SPLIT_ROWS})
    # This interaction copy is read by Ray agent workers separately from Hydra.
    entries = OmegaConf.to_container(OmegaConf.load(interaction), resolve=True)["interaction"]
    if len(entries) != 1 or entries[0]["name"] != "eduardo":
        raise ValueError("C 学生交互配置异常")
    _expect(entries[0]["config"], {"student_model": student_name, "student_api_base": STUDENT_API_BASE,
                                  "student_api_key_env": "SERVING_API_KEY", "student_concurrency": 16})

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONPATH=str(REPO), WORK_DIR=str(work), EDUARDO_RECIPE_DIR=str(REPO / "eduardo"),
               MODEL_PATH=str(TUTOR), EXPERIMENT=EXPERIMENT, USER=os.environ.get("USER") or "root",
               SERVING_API_KEY="placeholder", STUDENT_API_BASE=STUDENT_API_BASE, JUDGE_API_BASE=JUDGE_API_BASE,
               WANDB_MODE="disabled", RAY_USAGE_STATS_ENABLED="0", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [str(PYTHON), "-B", str(Path(__file__).with_name("c_trainer_entry.py")), "--check-only",
               "--config-path", str(REPO / "eduardo/configs"), "--config-name", "eduardo_base",
               *training_overrides(work, raytmp, student_name, interaction),
               f"hydra.run.dir={Path(work) / 'hydra-preflight'}", "hydra.output_subdir=null"]
    proc = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True, timeout=180)
    if proc.returncode or "C_PRECHECK_662_OK" not in proc.stdout.splitlines():
        raise RuntimeError("C Hydra 配置或迁移后教师预算预检失败")
