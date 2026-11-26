"""Unit tests for T-026 Stage 1 Route A training integration (fake data only)."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
PRO5000_DIR = CODE_ROOT / "training" / "a_group_pro5000"
for p in (CODE_ROOT, PRO5000_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pro5000_config as pcfg
import pro5000_trainer_entry as pentry
import queue_metrics
from training.routeA_reward.hybrid_grader import HybridGrader, verify_hybrid_grader_sha256
from training.routeA_reward.ladder import extract_ladder_variants
from training.routeA_reward.reward_function import (
    _TimedGraderSemaphore,
    close_step_metric_sinks,
    route_a_reward_func,
    summarize_metrics_window,
    summarize_step_metrics,
)
from training.routeA_train.extract_t017_control import extract_t017_control
from training.routeA_train.prepare_parquet import (
    is_ceiling_group,
    prepare_routea_parquets,
)
from training.routeA_train.stage2_routea_pilot import summarize_stage2_routea

EXPECTED_HYBRID_SHA256 = (
    "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
)


def _run(coro):
    return asyncio.run(coro)


class _FakeCompletion:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]


class _FakeOpenAIClient:
    def __init__(self, responder):
        self.calls = []

        async def _create(**kwargs):
            idx = len(self.calls)
            self.calls.append(kwargs)
            return responder(kwargs, idx)

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )


def _make_dialogue(teacher_think="plan", teacher_visible="try factoring"):
    return [
        {"role": "system", "content": "Tutor system prompt"},
        {"role": "user", "content": "This is the problem I am trying to understand: ORIG_X"},
        {"role": "assistant", "content": f"<think>{teacher_think}</think>\n{teacher_visible}"},
        {"role": "user", "content": "I get x = 2."},
    ]


def _make_fake_t023_row(
    source_id: str,
    *,
    split: str,
    direction: str,
    counts: list[int],
    sentinel: str,
) -> dict:
    variants = []
    for idx, cnt in enumerate(counts):
        p_val = round((cnt + 0.5) / 17.0, 8)
        b_val = round(float(__import__("math").log((1.0 - p_val) / p_val)), 8)
        variants.append(
            {
                "variant_id": f"{source_id}_v{idx}",
                "problem": f"FAKE_VARIANT_{sentinel}_{source_id}_{idx}",
                "answer": str(10 + idx),
                "n_correct_hybrid": cnt,
                "n_scored": 16,
                "p_new": p_val,
                "b": b_val,
                "ladder_rank": idx + 1,
            }
        )
    return {
        "source_id": source_id,
        "split": split,
        "direction": direction,
        "source_problem": f"FAKE_SOURCE_{sentinel}_{source_id}",
        "source_answer": "42",
        "source_pre_solve_rate_old": 0.25,
        "variants": variants,
    }


class RouteATrainStage1Tests(unittest.TestCase):
    def test_hybrid_grader_sha256_untouched(self):
        hg_path = CODE_ROOT / "training" / "routeA_reward" / "hybrid_grader.py"
        self.assertEqual(
            verify_hybrid_grader_sha256(hg_path, EXPECTED_HYBRID_SHA256),
            EXPECTED_HYBRID_SHA256,
        )

    def test_keep_teacher_budgets_allows_route_a_and_preserves_150_512(self):
        class _Node(dict):
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc

            def __setattr__(self, key, value):
                self[key] = _Node({k: (_Node(v) if isinstance(v, dict) else v) for k, v in value.items()}) if isinstance(value, dict) else value

        from contextlib import contextmanager

        @contextmanager
        def _fake_open_dict(_cfg):
            yield

        fake_omegaconf = SimpleNamespace(open_dict=_fake_open_dict)

        def _make_cfg(path, name, vis=150, think=512):
            return _Node(
                {
                    "reward": _Node(
                        {
                            "custom_reward_function": _Node(
                                {
                                    "path": path,
                                    "name": name,
                                    "reward_kwargs": _Node(
                                        {
                                            "max_tokens_per_turn": vis,
                                            "max_thinking_tokens_per_turn": think,
                                        }
                                    ),
                                }
                            )
                        }
                    )
                }
            )

        with mock.patch.dict(sys.modules, {"omegaconf": fake_omegaconf}):
            cfg_ra = _make_cfg(
                "pkg://training.routeA_reward.reward_function",
                "route_a_reward_func",
            )
            pentry.keep_teacher_budgets(cfg_ra)
            self.assertEqual(
                cfg_ra["custom_reward_function"]["reward_kwargs"]["max_tokens_per_turn"],
                150,
            )
            self.assertEqual(
                cfg_ra["custom_reward_function"]["reward_kwargs"]["max_thinking_tokens_per_turn"],
                512,
            )

            cfg_auth = _make_cfg("pkg://eduardo.reward_function", "eduardo_reward_func")
            pentry.keep_teacher_budgets(cfg_auth)

            with self.assertRaises(ValueError):
                pentry.keep_teacher_budgets(_make_cfg("pkg://unknown.mod", "bad_fn"))
            with self.assertRaises(ValueError):
                pentry.keep_teacher_budgets(
                    _make_cfg(
                        "pkg://training.routeA_reward.reward_function",
                        "route_a_reward_func",
                        vis=200,
                    )
                )

    def test_config_pin_routea_and_concurrency_cap(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            work = base / "w"
            ckpt = base / "c"
            raytmp = base / "r"
            data_dir = base / "data" / "routeA_t026"
            inter = work / "inter.yaml"
            work.mkdir()
            data_dir.mkdir(parents=True)

            ov = pcfg.build_overrides(
                base=base,
                work=work,
                ckpt_dir=ckpt,
                raytmp=raytmp,
                student_name="s",
                interaction=inter,
                total_steps=100,
                save_freq=5,
                test_freq=25,
                judge_concurrency=64,
                grader_max_concurrency=32,
                max_actor_ckpt_to_keep=1,
                reward_mode="ladder_rasch",
                data_dir=data_dir,
                theta_scale=0.25,
                ladder_num_climbs=8,
                code_root=CODE_ROOT,
            )
            self.assertIn("custom_reward_function.path=pkg://training.routeA_reward.reward_function", ov)
            self.assertIn("custom_reward_function.name=route_a_reward_func", ov)
            self.assertIn("+custom_reward_function.reward_kwargs.theta_scale=0.25", ov)
            self.assertIn("+custom_reward_function.reward_kwargs.ladder_num_climbs=8", ov)
            self.assertIn("+custom_reward_function.reward_kwargs.grader_max_concurrency=32", ov)
            self.assertIn(f"data.train_files={data_dir}/train.parquet", ov)
            self.assertIn(f"data.val_files={data_dir}/val.parquet", ov)

            cfg = {
                "trainer": {
                    "total_training_steps": 100,
                    "save_freq": 5,
                    "test_freq": 25,
                    "val_before_train": False,
                    "val_only": False,
                    "logger": ["console"],
                    "nnodes": 1,
                    "n_gpus_per_node": 4,
                    "max_actor_ckpt_to_keep": 1,
                    "del_local_ckpt_after_load": False,
                    "default_local_dir": str(ckpt),
                    "resume_mode": "disable",
                    "resume_from_path": None,
                },
                "data": {
                    "train_batch_size": 16,
                    "train_files": str(data_dir / "train.parquet"),
                    "val_files": str(data_dir / "val.parquet"),
                    "shuffle": True,
                    "max_prompt_length": 2048,
                    "max_response_length": 8192,
                },
                "actor_rollout_ref": {
                    "actor": {
                        "ppo_mini_batch_size": 8,
                        "ppo_epochs": 1,
                        "ppo_micro_batch_size_per_gpu": 1,
                        "use_dynamic_bsz": False,
                        "loss_agg_mode": "token-mean",
                        "use_kl_loss": False,
                        "optim": {"lr": 1e-6},
                        "policy_loss": {"loss_mode": "dppo_tv"},
                        "fsdp_config": {"fsdp_size": 4, "param_offload": False, "optimizer_offload": True},
                        "checkpoint": {"save_contents": ["model", "extra"], "load_contents": ["model", "extra"]},
                    },
                    "rollout": {
                        "n": 16,
                        "temperature": 1.15,
                        "gpu_memory_utilization": 0.58,
                        "free_cache_engine": True,
                        "tensor_model_parallel_size": 1,
                        "calculate_log_probs": True,
                        "max_model_len": 10240,
                        "multi_turn": {"enable": True, "interaction_config_path": str(inter)},
                        "agent": {"num_workers": 2},
                    },
                },
                "reward": {"num_workers": 2},
                "reward_model": {"num_workers": None},
                "custom_reward_function": {
                    "path": "pkg://training.routeA_reward.reward_function",
                    "name": "route_a_reward_func",
                    "reward_kwargs": {
                        "num_student_attempts": 16,
                        "max_thinking_tokens_per_turn": 512,
                        "max_tokens_per_turn": 150,
                        "judge_votes": 3,
                        "judge_hard_gate": True,
                        "omit_teacher_turns": True,
                        "test_on_transfer": True,
                        "enable_thinking": True,
                        "judge_concurrency": 64,
                        "student_concurrency": 32,
                        "student_temperature": 0.6,
                        "judge_temperature": 0.6,
                        "eval_judge_model": "",
                        "judge_model": "judge-local",
                        "student_model": "s",
                        "mode": "ladder_rasch",
                        "theta_scale": 0.25,
                        "ladder_num_climbs": 8,
                        "grader_max_concurrency": 32,
                    },
                },
                "ray_kwargs": {
                    "ray_init": {
                        "runtime_env": {
                            "worker_process_setup_hook": "queue_metrics.install",
                            "env_vars": {"EDU_QUEUE_METRICS_DIR": str(work / "queue-metrics")},
                        }
                    }
                },
            }
            pcfg.pin_composed(
                cfg,
                base=base,
                work=work,
                ckpt_dir=ckpt,
                student_name="s",
                interaction=inter,
                total_steps=100,
                save_freq=5,
                test_freq=25,
                judge_concurrency=64,
                max_actor_ckpt_to_keep=1,
                reward_mode="ladder_rasch",
                data_dir=data_dir,
                theta_scale=0.25,
                ladder_num_climbs=8,
                grader_max_concurrency=32,
            )

            with self.assertRaises(ValueError):
                pcfg.build_overrides(
                    base=base,
                    work=work,
                    ckpt_dir=ckpt,
                    raytmp=raytmp,
                    student_name="s",
                    interaction=inter,
                    total_steps=100,
                    save_freq=5,
                    test_freq=25,
                    judge_concurrency=64,
                    grader_max_concurrency=65,
                    reward_mode="ladder_rasch",
                    data_dir=data_dir,
                )

    def test_ceiling_filter_boundary(self):
        self.assertTrue(is_ceiling_group([{"n_correct_hybrid": 15}]))
        self.assertTrue(is_ceiling_group([{"n_correct_hybrid": 15}, {"n_correct_hybrid": 16}]))
        self.assertFalse(is_ceiling_group([{"n_correct_hybrid": 14}]))
        self.assertFalse(is_ceiling_group([{"n_correct_hybrid": 14}, {"n_correct_hybrid": 15}]))
        self.assertFalse(is_ceiling_group([]))

    def test_parquet_roundtrip_reward_and_no_text_leak(self):
        import pandas as pd

        sentinel = "SENTINEL_SECRET_PROBLEM_TEXT_XYZ"
        fake_rows = [
            _make_fake_t023_row("src_ceil_1", split="train", direction="easy", counts=[15], sentinel=sentinel),
            _make_fake_t023_row("src_ceil_2", split="train", direction="mix", counts=[15, 16], sentinel=sentinel),
            _make_fake_t023_row("src_keep_edge", split="train", direction="hard", counts=[14, 15], sentinel=sentinel),
            _make_fake_t023_row("src_keep_3", split="train", direction="easy_to_hard", counts=[2, 6, 12], sentinel=sentinel),
            _make_fake_t023_row("src_dev_1", split="dev", direction="easy", counts=[16], sentinel=sentinel),
            _make_fake_t023_row("src_dev_2", split="dev", direction="mix", counts=[3, 9], sentinel=sentinel),
        ]

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jsonl_path = tmp / "fake_measured.jsonl"
            out_dir = tmp / "parquet_out"
            metrics_dir = tmp / "routeA-metrics"
            jsonl_path.write_text(
                "\n".join(json.dumps(r) for r in fake_rows) + "\n",
                encoding="utf-8",
            )
            summary = prepare_routea_parquets(
                input_jsonl=jsonl_path,
                out_dir=out_dir,
                expected_sha256=None,
                teacher_system_prompt="System tutor for: {problem}",
            )
            self.assertEqual(summary["train_before_filter"]["n_sources"], 4)
            self.assertEqual(summary["train_removed_ceiling"]["n_sources"], 2)
            self.assertEqual(summary["train_after_filter"]["n_sources"], 2)
            self.assertEqual(summary["val_dev"]["n_sources"], 2)
            self.assertEqual(summary["val_dev"]["ceiling_groups_kept"], 1)

            summary_text = (out_dir / "dataset_summary.json").read_text(encoding="utf-8")
            self.assertNotIn(sentinel, summary_text)

            df_train = pd.read_parquet(out_dir / "train.parquet")
            self.assertEqual(len(df_train), 2)
            row_dict = df_train.iloc[1].to_dict()
            extra_info = row_dict["extra_info"]
            variants = extract_ladder_variants(extra_info)
            self.assertEqual(len(variants), 3)
            self.assertEqual([v.ladder_rank for v in variants], [1, 2, 3])

            extra_info["messages"] = _make_dialogue()
            judge_client = _FakeOpenAIClient(
                lambda _kw, _idx: _FakeCompletion('{"reasoning": "ok", "decision": "OK"}')
            )
            student_client = _FakeOpenAIClient(
                lambda _kw, _idx: _FakeCompletion(r"\boxed{10}")
            )

            async def _llm_false(_msgs):
                return '{"correct": false}', "stop"

            grader = HybridGrader(llm_caller=_llm_false, backoff_base=0.0)
            log_records: list[str] = []

            class _Capture(logging.Handler):
                def emit(self, record):
                    log_records.append(self.format(record))

            handler = _Capture()
            root_logger = logging.getLogger()
            old_level = root_logger.level
            root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(handler)
            try:
                t_start = queue_metrics.time.monotonic_ns()
                with mock.patch(
                    "training.routeA_reward.reward_function.get_async_openai_client",
                    side_effect=lambda base_u, _env: judge_client if "judge" in base_u else student_client,
                ):
                    res = _run(
                        route_a_reward_func(
                            data_source=row_dict["data_source"],
                            ground_truth=row_dict["reward_model"]["ground_truth"],
                            extra_info=extra_info,
                            mode="ladder_rasch",
                            judge_api_base="http://judge.local/v1",
                            student_api_base="http://student.local/v1",
                            judge_model="judge-local",
                            student_model="student-local",
                            hybrid_grader=grader,
                            ladder_num_climbs=8,
                            routeA_metrics_dir=str(metrics_dir),
                        )
                    )
                t_end = queue_metrics.time.monotonic_ns()
            finally:
                root_logger.removeHandler(handler)
                root_logger.setLevel(old_level)
                close_step_metric_sinks()

            self.assertEqual(res["rasch_valid"], 1)
            self.assertAlmostEqual(res["rasch_theta_scale"], 0.25, places=12)
            self.assertAlmostEqual(res["rasch_gain_g"], res["rasch_theta"] * 0.25, places=12)
            self.assertAlmostEqual(res["score"], res["rasch_gain_g"], places=12)
            self.assertEqual(res["ladder_reach_l1_rate"], 1.0)
            self.assertEqual(res["ladder_pass_l1_rate"], 1.0)
            self.assertEqual(res["ladder_pass_l2_rate"], 0.0)

            with mock.patch(
                "training.routeA_reward.reward_function.get_async_openai_client",
                side_effect=lambda base_u, _env: judge_client if "judge" in base_u else student_client,
            ):
                res_half = _run(
                    route_a_reward_func(
                        data_source=row_dict["data_source"],
                        ground_truth=row_dict["reward_model"]["ground_truth"],
                        extra_info=extra_info,
                        mode="ladder_rasch",
                        theta_scale=0.5,
                        judge_api_base="http://judge.local/v1",
                        student_api_base="http://student.local/v1",
                        judge_model="judge-local",
                        student_model="student-local",
                        hybrid_grader=grader,
                        ladder_num_climbs=8,
                        routeA_metrics_dir="",
                    )
                )
            self.assertAlmostEqual(res_half["rasch_theta_scale"], 0.5, places=12)
            self.assertAlmostEqual(res_half["rasch_gain_g"], res_half["rasch_theta"] * 0.5, places=12)

            all_logs = "\n".join(log_records)
            self.assertNotIn(sentinel, all_logs)
            for mf in metrics_dir.glob("routeA-metrics-*.jsonl"):
                self.assertNotIn(sentinel, mf.read_text(encoding="utf-8"))

            agg = summarize_metrics_window(metrics_dir, t_start, t_end, split="train")
            self.assertEqual(agg["n_rollouts"], 1)
            self.assertAlmostEqual(agg["metrics"]["eduardo/routeA_g_mean"], res["rasch_gain_g"], places=12)
            self.assertAlmostEqual(agg["metrics"]["eduardo/routeA_theta_mean"], res["rasch_theta"], places=12)

    def test_author_mode_bit_for_bit_identical_with_routea_kwargs(self):
        from eduardo import reward_function as author_rf

        judge_client = _FakeOpenAIClient(
            lambda _kw, _idx: _FakeCompletion('{"reasoning": "ok", "decision": "OK"}')
        )
        student_client = _FakeOpenAIClient(
            lambda _kw, _idx: _FakeCompletion(r"\boxed{99}")
        )
        extra = {
            "problem": "ORIG",
            "answer": "42",
            "transfer_problem": "TRANS",
            "transfer_answer": "99",
            "pre_solve_rate": 0.2,
            "pre_solve_rate_transfer_problem": 0.25,
            "messages": _make_dialogue(),
        }
        common = dict(
            data_source="bigmath",
            ground_truth="42",
            extra_info=extra,
            judge_api_base="http://judge.local/v1",
            judge_api_key_env="SERVING_API_KEY",
            student_api_base="http://student.local/v1",
            student_api_key_env="SERVING_API_KEY",
            judge_model="judge-local",
            student_model="student-local",
            num_student_attempts=4,
            max_tokens_per_turn=150,
            max_thinking_tokens_per_turn=512,
            enable_batch_logging=False,
        )
        with ExitStack() as st:
            st.enter_context(
                mock.patch(
                    "eduardo.reward_function.get_async_openai_client",
                    side_effect=lambda u, _e: judge_client if "judge" in u else student_client,
                )
            )
            r_ref = _run(author_rf.eduardo_reward_func(**common))
            r_wrap = _run(
                route_a_reward_func(
                    **common,
                    mode="author",
                    theta_scale=0.25,
                    ladder_num_climbs=8,
                    grader_max_concurrency=32,
                    routeA_metrics_dir="",
                )
            )
        self.assertEqual(r_ref, r_wrap)

    def test_step_metrics_and_t017_control_and_stage3_gates(self):
        step_rows = [
            {
                "score": 0.5,
                "gated": 0,
                "rasch_valid": 1,
                "rasch_theta": 2.0,
                "rasch_gain_g": 0.5,
                "rasch_prior_pulled": 1,
                "rasch_all_correct": 1,
                "rasch_all_wrong": 0,
                "ladder_reach_l1_rate": 1.0,
                "ladder_reach_l2_rate": 1.0,
                "ladder_reach_l3_rate": 0.0,
                "ladder_pass_l1_rate": 1.0,
                "ladder_pass_l2_rate": 0.0,
                "ladder_pass_l3_rate": 0.0,
                "ladder_student_calls": 16,
                "ladder_student_failed": 0,
                "grader_llm_calls": 10,
                "grader_llm_unknown": 1,
                "grader_tech_failed": 0,
            },
            {
                "score": -0.3,
                "gated": 0,
                "rasch_valid": 1,
                "rasch_theta": -1.2,
                "rasch_gain_g": -0.3,
                "rasch_prior_pulled": 1,
                "rasch_all_correct": 0,
                "rasch_all_wrong": 1,
                "ladder_reach_l1_rate": 1.0,
                "ladder_reach_l2_rate": 0.0,
                "ladder_reach_l3_rate": 0.0,
                "ladder_pass_l1_rate": 0.0,
                "ladder_pass_l2_rate": 0.0,
                "ladder_pass_l3_rate": 0.0,
                "ladder_student_calls": 8,
                "ladder_student_failed": 0,
                "grader_llm_calls": 10,
                "grader_llm_unknown": 0,
                "grader_tech_failed": 1,
            },
            {
                "score": -0.2,
                "gated": 1,
                "rasch_valid": 0,
                "rasch_theta": 0.0,
                "rasch_gain_g": 0.0,
                "rasch_prior_pulled": 0,
                "rasch_all_correct": 0,
                "rasch_all_wrong": 0,
                "ladder_student_calls": 0,
                "ladder_student_failed": 0,
                "grader_llm_calls": 0,
                "grader_llm_unknown": 0,
                "grader_tech_failed": 0,
            },
        ]
        sm = summarize_step_metrics(step_rows)
        self.assertAlmostEqual(sm["eduardo/routeA_theta_mean"], 0.4, places=12)
        self.assertAlmostEqual(sm["eduardo/routeA_g_mean"], 0.1, places=12)
        self.assertAlmostEqual(sm["eduardo/routeA_compliant_g_lt_neg02_rate"], 0.5, places=12)
        self.assertAlmostEqual(sm["eduardo/routeA_prior_pulled_all_correct_rate"], 0.5, places=12)
        self.assertAlmostEqual(sm["eduardo/routeA_prior_pulled_all_wrong_rate"], 0.5, places=12)
        self.assertAlmostEqual(sm["eduardo/routeA_gated_rate"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(sm["eduardo/routeA_grader_tech_failed_rate"], 0.05, places=12)

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            steps_dir = run_dir / "step-statistics"
            steps_dir.mkdir(parents=True)
            for s in range(1, 6):
                (steps_dir / f"{s}.json").write_text(
                    json.dumps(
                        {
                            "step": s,
                            "metrics": {
                                "critic/score/mean": -0.1 + 0.02 * s,
                                "critic/score/min": -0.2,
                                "reward/gated_rate": 0.10,
                                "timing_s/step": 1200.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            baseline = extract_t017_control(run_dir)
            self.assertEqual(baseline["n_steps"], 5)
            self.assertAlmostEqual(baseline["steps"]["5"]["gated_rate_ma5"], 0.10, places=6)

            t017_steps = {int(k): v for k, v in baseline["steps"].items()}
            ra_steps = run_dir / "ra_steps"
            ra_steps.mkdir()
            # Steps 1..9: gated_rate drops to 0.10 so min_prior_ma5 = 0.10.
            for s, rate in [(1, 0.40), (2, 0.30), (3, 0.20), (4, 0.10), (5, 0.10), (6, 0.10), (7, 0.10), (8, 0.10), (9, 0.10)]:
                (ra_steps / f"{s}.json").write_text(
                    json.dumps({"metrics": {"eduardo/routeA_gated_rate": rate}}),
                    encoding="utf-8",
                )
            # Step 9 high rate without rebound >= 10: does not trigger rebound/abs gates (only tech fail).
            gates_s9 = pentry.assess_routea_stage3_gates(
                step=9,
                steps_dir=ra_steps,
                current_row={"routeA_metrics": {"metrics": sm}},
                t017_steps=t017_steps,
            )
            self.assertTrue(gates_s9["tech_fail_exceeded"])
            self.assertFalse(gates_s9["gated_rebound_exceeded"])
            self.assertFalse(gates_s9["gated_abs_exceeded"])

            # Step 10 with 0.90 gated jumps ma5 from 0.10 -> 0.26 (+0.16 > 0.15): triggers rebound gate.
            gates_s10 = pentry.assess_routea_stage3_gates(
                step=10,
                steps_dir=ra_steps,
                current_row={
                    "routeA_metrics": {
                        "metrics": {
                            "eduardo/routeA_grader_llm_calls_total": 100.0,
                            "eduardo/routeA_grader_tech_failed_total": 0.0,
                            "eduardo/routeA_gated_rate": 0.90,
                        }
                    }
                },
                t017_steps=t017_steps,
            )
            self.assertFalse(gates_s10["passed"])
            self.assertFalse(gates_s10["tech_fail_exceeded"])
            self.assertTrue(gates_s10["gated_rebound_exceeded"])
            self.assertFalse(gates_s10["gated_abs_exceeded"])

            # Step 25 with steady 0.36 (> 0.35, no rebound): triggers abs gate and carries T-017 val ref 0.23.
            ra_steps2 = run_dir / "ra_steps2"
            ra_steps2.mkdir()
            for s in range(1, 25):
                (ra_steps2 / f"{s}.json").write_text(
                    json.dumps({"metrics": {"eduardo/routeA_gated_rate": 0.36}}),
                    encoding="utf-8",
                )
            gates_s25 = pentry.assess_routea_stage3_gates(
                step=25,
                steps_dir=ra_steps2,
                current_row={
                    "routeA_metrics": {
                        "metrics": {
                            "eduardo/routeA_grader_llm_calls_total": 100.0,
                            "eduardo/routeA_grader_tech_failed_total": 1.0,
                            "eduardo/routeA_gated_rate": 0.36,
                        }
                    }
                },
            )
            self.assertFalse(gates_s25["passed"])
            self.assertFalse(gates_s25["tech_fail_exceeded"])
            self.assertFalse(gates_s25["gated_rebound_exceeded"])
            self.assertTrue(gates_s25["gated_abs_exceeded"])
            self.assertEqual(gates_s25["t017_val_gated_ref"], 0.23)

    def test_timed_grader_semaphore_contextvar_isolation(self):
        from training.routeA_reward import reward_function as rf

        inner = asyncio.Semaphore(2)
        timed = _TimedGraderSemaphore(inner)
        observed = []

        async def _worker(delay: float):
            async with timed:
                await asyncio.sleep(delay)
                q_ns, acq_ns = rf._GRADER_SEM_TIMING.get()
                observed.append((delay, q_ns, acq_ns))

        async def _driver():
            await asyncio.gather(_worker(0.03), _worker(0.01), _worker(0.02))

        _run(_driver())
        self.assertEqual(len(observed), 3)
        acq_times = [x[2] for x in observed]
        self.assertTrue(all(t > 0 for t in acq_times))
        self.assertEqual(len(set(acq_times)), 3)
        self.assertEqual(rf._GRADER_SEM_TIMING.get(), (0, 0))

    def test_stage2_routea_summary_and_acceptance_gates(self):
        layout = {
            "mode": "8gpu",
            "tutor_gpus": [0, 1, 2, 3],
            "student_gpus": [4, 6],
            "judge_gpus": [5, 7],
            "judge_concurrency": 16,
            "grader_max_concurrency": 32,
        }

        def _make_step(step_num: int, *, step_s: float, vram_mib: float, non2xx: int = 0, gate_p95_ms: float = 45000.0, grader_tech_fail: float = 0.0):
            return {
                "step": step_num,
                "vote_log_counts": {"requests": 1536, "parse": 2, "recovery": 10},
                "metrics": {
                    "timing_s/step": step_s,
                    "timing_s/gen": step_s * 0.85,
                    "perf/total_num_tokens": 1_800_000,
                    "actor/grad_norm": 0.1,
                    "actor/pg_loss": -0.003,
                },
                "phase_timings": {
                    "gen": {"minutes": step_s * 0.85 / 60.0, "ms_per_token": 0.5},
                    "old_log_prob": {"minutes": 0.6, "ms_per_token": 0.02},
                    "update_actor": {"minutes": 2.4, "ms_per_token": 0.08},
                },
                "telemetry": {
                    "gen": {
                        "gpus": {
                            str(g): {"used_mib": {"max": vram_mib}, "util_pct": {"mean": 20.0 if g < 4 else 80.0}}
                            for g in range(8)
                        }
                    },
                    "old_log_prob": {"gpus": {}},
                    "update_actor": {"gpus": {}},
                },
                "judge_proxy": {"total_chat_requests": 3000, "non_2xx": non2xx, "errors": 0, "combined_busy_wall_s": 800.0},
                "student_proxy": {"total_chat_requests": 2600, "non_2xx": 0, "errors": 0, "combined_busy_wall_s": 400.0, "backends": {}},
                "queues": {
                    "groups": {
                        "judge": {"calls": 1536, "wait_p50_ms": 100.0, "wait_p95_ms": 500.0, "service_p50_ms": 10000.0, "service_p95_ms": gate_p95_ms, "attempt_errors": 0},
                        "hybrid_grader": {"calls": 1500, "wait_p50_ms": 50.0, "wait_p95_ms": 200.0, "service_p50_ms": 1200.0, "service_p95_ms": 3000.0, "attempt_errors": 0},
                        "student_interaction": {"calls": 2000, "attempt_errors": 0},
                        "student_final": {"calls": 600, "attempt_errors": 0},
                    }
                },
                "routeA_metrics": {
                    "n_rollouts": 256,
                    "metrics": {
                        "eduardo/routeA_theta_mean": 0.1,
                        "eduardo/routeA_g_mean": 0.025,
                        "eduardo/routeA_gated_rate": 0.15,
                        "eduardo/routeA_grader_llm_calls_total": 1500.0,
                        "eduardo/routeA_grader_llm_unknown_total": 0.0,
                        "eduardo/routeA_grader_llm_unknown_rate": 0.0,
                        "eduardo/routeA_grader_tech_failed_total": grader_tech_fail,
                        "eduardo/routeA_grader_tech_failed_rate": grader_tech_fail / 1500.0,
                        "eduardo/routeA_student_failed_total": 0.0,
                    },
                },
            }

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            (work / "step-statistics").mkdir()
            (work / "restore_step_2.json").write_text(json.dumps({"global_step": 2, "data_batches": 2}))
            (work / "step2-comparison.json").write_text(json.dumps({"passed": True}))
            for s_num, sec in ((1, 1300.0), (2, 1350.0), (3, 1320.0)):
                (work / "step-statistics" / f"{s_num}.json").write_text(
                    json.dumps(_make_step(s_num, step_s=sec, vram_mib=67000.0))
                )
            ok_sum = summarize_stage2_routea(work, layout, 20_720_000_000)
            self.assertTrue(ok_sum["gates_passed"], ok_sum["gate_failures"])
            self.assertEqual(ok_sum["rc"], 0)
            self.assertEqual(ok_sum["steps"][0]["gate_calls"], 1536)
            self.assertEqual(ok_sum["steps"][0]["hybrid_grader_qm_calls"], 1500)

            # Fail every gate cleanly
            (work / "restore_step_2.json").unlink()
            (work / "step-statistics" / "2.json").write_text(
                json.dumps(_make_step(2, step_s=2100.0, vram_mib=70000.0, non2xx=1, gate_p95_ms=305000.0, grader_tech_fail=20.0))
            )
            bad_sum = summarize_stage2_routea(work, layout, 20_720_000_000)
            self.assertFalse(bad_sum["gates_passed"])
            self.assertEqual(bad_sum["rc"], 4)
            self.assertFalse(any(bad_sum["per_gate"].values()))


if __name__ == "__main__":
    unittest.main()
