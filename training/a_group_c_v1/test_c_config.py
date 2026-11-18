import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import c_config as c


class SelectionTest(unittest.TestCase):
    def test_inputs_bound_to_known_original_indices(self):
        manifest = dict(author_repository_commit=c.AUTHOR_COMMIT, source_revision=c.DATA_REVISION,
                        source_train_sha256=c.AUTHOR_TRAIN_SHA, selection_sha256=dict(c.SELECTION_SHA),
                        output={name: {"rows": value} for name, value in c.SPLIT_ROWS.items()})
        c.check_pool(manifest, dict(c.SPLIT_ROWS))
        wrong_rows = dict(c.SPLIT_ROWS)
        wrong_rows["val.parquet"] = 99
        with self.assertRaises(ValueError):
            c.check_pool(manifest, wrong_rows)
        manifest["selection_sha256"]["validation.jsonl"] = "unrelated"
        with self.assertRaises(ValueError):
            c.check_pool(manifest, dict(c.SPLIT_ROWS))


@unittest.skipUnless(os.environ.get("EDUARDO_CONFIG_DIR") and os.environ.get("VERL_CONFIG_DIR"),
                     "set both pinned checkout config paths for Hydra integration")
class PinnedHydraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        import yaml
        cls.work = Path("/tmp/local-test-C")
        cls.student = "Llama-3.1-8B-Instruct-local"
        cls.interaction = cls.work / "interaction.yaml"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            # Use the actual pinned YAMLs without importing torch or Ray on CPU.
            for directory in ("VERL_CONFIG_DIR", "EDUARDO_CONFIG_DIR"):
                for path in Path(os.environ[directory]).iterdir():
                    if path.suffix == ".yaml" or path.is_dir():
                        (root / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
            env = dict(WORK_DIR=str(cls.work), EXPERIMENT=c.EXPERIMENT, MODEL_PATH=str(c.TUTOR),
                       EDUARDO_RECIPE_DIR="/tmp/author/eduardo", SERVING_API_KEY="placeholder",
                       STUDENT_API_BASE=c.STUDENT_API_BASE, JUDGE_API_BASE=c.JUDGE_API_BASE, USER="local")
            with patch.dict(os.environ, env), initialize_config_dir(version_base=None, config_dir=str(root)):
                cfg = compose(config_name="eduardo_base", overrides=c.training_overrides(
                    cls.work, cls.work/"ray", cls.student, cls.interaction))
                cls.config = yaml.safe_load(OmegaConf.to_yaml(cfg, resolve=True))

    def check(self, config):
        c.check_composed(config, self.work, self.student, self.interaction)

    def test_actual_hydra_composition(self):
        self.check(self.config)

    def test_actual_migration_and_author_generation_receive_662(self):
        # Execute the original pinned source function and author loop class.
        # Heavy cluster imports are replaced by a passive fake superclass.
        import ast
        import asyncio
        from types import SimpleNamespace
        from omegaconf import DictConfig, OmegaConf, open_dict
        from c_trainer_entry import keep_teacher_budgets

        def node(path, name):
            tree = ast.parse(path.read_text())
            return next(n for n in tree.body if getattr(n, "name", "") == name)

        def execute_node(n, path, namespace):
            import __future__
            tree = ast.Module(body=[n], type_ignores=[])
            exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)

        migration_path = Path(os.environ["VERL_CONFIG_DIR"]).parents[1] / "experimental/reward_loop/reward_loop.py"
        ns = {"DictConfig": DictConfig, "open_dict": open_dict}
        execute_node(node(migration_path, "migrate_legacy_reward_impl"), migration_path, ns)
        config = ns["migrate_legacy_reward_impl"](OmegaConf.create(self.config))
        class ToolAgentLoop:
            def __init__(self, cfg):
                self.config = cfg

            async def _handle_generating_state(self, agent_data, params, ignore_termination=False):
                return params["max_tokens"]

        loop_path = Path(os.environ["EDUARDO_CONFIG_DIR"]).parent / "eduardo_agent_loop.py"
        loop_class = node(loop_path, "EduardoAgentLoop")
        loop_class.decorator_list = []
        scope = {"ToolAgentLoop": ToolAgentLoop, "_copy_messages": list}
        execute_node(loop_class, loop_path, scope)

        def active_limit():
            loop = scope["EduardoAgentLoop"](config)
            agent = SimpleNamespace(messages=[], extra_fields={}, assistant_turns=0, user_turns=0)
            return asyncio.run(loop._handle_generating_state(agent, {}))

        self.assertEqual(active_limit(), 800)
        before = OmegaConf.to_container(config.reward, resolve=True)
        keep_teacher_budgets(config)
        self.assertEqual(active_limit(), 662)
        self.assertEqual(OmegaConf.to_container(config.reward, resolve=True), before)
        self.assertEqual(OmegaConf.to_container(config.custom_reward_function), {
            "reward_kwargs": {"max_tokens_per_turn": 150, "max_thinking_tokens_per_turn": 512}})
        config.custom_reward_function.reward_kwargs.max_tokens_per_turn = 400
        with self.assertRaises(ValueError):
            keep_teacher_budgets(config)

    def test_validation_has_local_reward_and_no_external_judge(self):
        for key, value in (("eval_judge_model", "gemini"), ("judge_api_base", "https://unexpected"),
                           ("student_api_base", "http://unexpected"), ("judge_concurrency", 32)):
            cfg = copy.deepcopy(self.config)
            cfg["custom_reward_function"]["reward_kwargs"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(cfg)

    def test_cannot_run_b_or_formal_in_c(self):
        for key, value in (("total_training_steps", 2), ("total_training_steps", 100),
                           ("val_before_train", True), ("test_freq", 0), ("save_freq", 0)):
            cfg = copy.deepcopy(self.config)
            cfg["trainer"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(cfg)


if __name__ == "__main__":
    unittest.main()
