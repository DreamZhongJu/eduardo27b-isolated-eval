"""Native veRL startup for C, keeping the two budgets visible to the author loop."""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from c_config import check_composed


def keep_teacher_budgets(config):
    """The pinned author reads the legacy name after veRL's native migration."""
    from omegaconf import open_dict
    reward = config.reward.custom_reward_function
    if reward.path != "pkg://eduardo.reward_function" or reward.name != "eduardo_reward_func":
        raise ValueError("C 奖励函数未绑定作者实现")
    keys = ("max_tokens_per_turn", "max_thinking_tokens_per_turn")
    budgets = {key: reward.reward_kwargs[key] for key in keys}
    if tuple(budgets.values()) != (150, 512):
        raise ValueError("C 迁移后的教师预算不符")
    previous = config.get("custom_reward_function")
    if previous is None:
        with open_dict(config):
            config.custom_reward_function = {"reward_kwargs": budgets}
    elif any(previous.get("reward_kwargs", {}).get(key) != value for key, value in budgets.items()):
        raise ValueError("C 教师与奖励函数的预算不一致")
    if tuple(config.custom_reward_function.reward_kwargs[k] for k in keys) != (150, 512):
        raise ValueError("C 教师预算交接失败")


def main(check_only):
    import hydra
    import ray
    from omegaconf import OmegaConf
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    @hydra.main(config_path=None, config_name=None, version_base=None)
    def execute(cfg):
        work = Path(os.environ["WORK_DIR"])
        check_composed(OmegaConf.to_container(cfg, resolve=True), work, cfg.vars.student_model_name,
                       cfg.actor_rollout_ref.rollout.multi_turn.interaction_config_path)
        auto_set_device(cfg)
        cfg = migrate_legacy_reward_impl(cfg)
        keep_teacher_budgets(cfg)
        if check_only:
            print("C_PRECHECK_662_OK")
            return
        print("C_TUTOR_BUDGET 150+512=662", flush=True)
        run_ppo(cfg)

    def stop(_signal, _frame):
        raise SystemExit("收到停止信号")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        execute()
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    check_only = "--check-only" in sys.argv
    if check_only:
        sys.argv.remove("--check-only")
    main(check_only)
