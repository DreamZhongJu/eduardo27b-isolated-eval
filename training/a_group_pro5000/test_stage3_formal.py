"""Unit tests for Stage 3 formal orchestrator, step-1 judge gate, resume, and status."""
from contextlib import ExitStack
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pro5000_config as pcfg
from pro5000_trainer_entry import assess_step1_judge
from stage3_formal import (
    format_stage3_status,
    prepare_resume_checkpoint,
    verify_stage3_gpus_clean,
)
from telemetry import Sampler


def _write_fake_ckpt(work: Path, ckpt_dir: Path, step: int, *, verified: bool = True, complete: bool = True):
    step_dir = ckpt_dir / f'global_step_{step}'
    actor = step_dir / 'actor'
    (actor / 'huggingface').mkdir(parents=True, exist_ok=True)
    (actor / 'fsdp_config.json').write_text(json.dumps({'world_size': 4}))
    (actor / 'huggingface' / 'config.json').write_text('{}')
    for rank in range(4):
        (actor / f'model_world_size_4_rank_{rank}.pt').write_bytes(b'm')
        if complete or rank < 3:
            (actor / f'extra_state_world_size_4_rank_{rank}.pt').write_bytes(b'e')
    (step_dir / 'data.pt').write_bytes(b'd')
    if verified:
        (work / f'ckpt-verified-{step}.json').write_text(json.dumps({'step': step, 'size_gb': 20.72}))


class Stage3FormalTest(unittest.TestCase):
    def test_stage3_config_pin_accepts_keep1_and_judge64(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            work = base / 'w'
            ckpt = base / 'c'
            raytmp = base / 'r'
            inter = work / 'inter.yaml'
            work.mkdir()
            (base / 'data' / 'author_prepared').mkdir(parents=True)
            cfg = {
                'trainer': {
                    'total_training_steps': 100, 'save_freq': 5, 'test_freq': 25,
                    'val_before_train': False, 'val_only': False, 'logger': ['console'],
                    'nnodes': 1, 'n_gpus_per_node': 4, 'max_actor_ckpt_to_keep': 1,
                    'del_local_ckpt_after_load': False, 'default_local_dir': str(ckpt),
                    'resume_mode': 'disable', 'resume_from_path': None,
                },
                'data': {
                    'train_batch_size': 16,
                    'train_files': str(base / 'data' / 'author_prepared' / 'train.parquet'),
                    'val_files': str(base / 'data' / 'author_prepared' / 'val.parquet'),
                    'shuffle': True, 'max_prompt_length': 2048, 'max_response_length': 8192,
                },
                'actor_rollout_ref': {
                    'actor': {
                        'ppo_mini_batch_size': 8, 'ppo_epochs': 1, 'ppo_micro_batch_size_per_gpu': 1,
                        'use_dynamic_bsz': False, 'loss_agg_mode': 'token-mean', 'use_kl_loss': False,
                        'optim': {'lr': 1e-6}, 'policy_loss': {'loss_mode': 'dppo_tv'},
                        'fsdp_config': {'fsdp_size': 4, 'param_offload': False, 'optimizer_offload': True},
                        'checkpoint': {'save_contents': ['model', 'extra'], 'load_contents': ['model', 'extra']},
                    },
                    'rollout': {
                        'n': 16, 'temperature': 1.15, 'gpu_memory_utilization': 0.58,
                        'free_cache_engine': True, 'tensor_model_parallel_size': 1,
                        'calculate_log_probs': True, 'max_model_len': 10240,
                        'multi_turn': {'enable': True, 'interaction_config_path': str(inter)},
                        'agent': {'num_workers': 2},
                    },
                },
                'reward': {'num_workers': 2},
                'reward_model': {'num_workers': None},
                'custom_reward_function': {'reward_kwargs': {
                    'num_student_attempts': 16, 'max_thinking_tokens_per_turn': 512,
                    'max_tokens_per_turn': 150, 'judge_votes': 3, 'judge_hard_gate': True,
                    'omit_teacher_turns': True, 'test_on_transfer': True, 'enable_thinking': True,
                    'judge_concurrency': 64, 'student_concurrency': 32,
                    'student_temperature': 0.6, 'judge_temperature': 0.6,
                    'eval_judge_model': '', 'judge_model': 'judge-local', 'student_model': 's',
                }},
                'ray_kwargs': {'ray_init': {
                    'runtime_env': {
                        'worker_process_setup_hook': 'queue_metrics.install',
                        'env_vars': {'EDU_QUEUE_METRICS_DIR': str(work / 'queue-metrics')},
                    }
                }},
            }
            ov = pcfg.build_overrides(
                base=base, work=work, ckpt_dir=ckpt, raytmp=raytmp, student_name='s',
                interaction=inter, total_steps=100, save_freq=5, test_freq=25,
                judge_concurrency=64, max_actor_ckpt_to_keep=1,
            )
            self.assertIn('trainer.max_actor_ckpt_to_keep=1', ov)
            self.assertIn('custom_reward_function.reward_kwargs.judge_concurrency=64', ov)
            pcfg.pin_composed(
                cfg, base=base, work=work, ckpt_dir=ckpt, student_name='s',
                interaction=inter, total_steps=100, save_freq=5, test_freq=25,
                judge_concurrency=64, max_actor_ckpt_to_keep=1,
            )

    def test_step1_judge_gate_accepts_and_rejects(self):
        good_row = {
            'metrics': {'timing_s/step': 900.0},
            'judge_proxy': {'total_chat_requests': 1536, 'non_2xx': 0, 'errors': 0},
            'vote_log_counts': {'requests': 1536, 'parse': 4, 'recovery': 50},
            'queues': {'groups': {'judge': {'attempt_errors': 0, 'call_errors': 0, 'service_p95_ms': 45000.0}}},
        }
        res_ok = assess_step1_judge(good_row, 64)
        self.assertTrue(res_ok['passed'])

        for mutate in (
            lambda r: r['judge_proxy'].update({'non_2xx': 1}),
            lambda r: r['queues']['groups']['judge'].update({'attempt_errors': 2}),
            lambda r: r['vote_log_counts'].update({'parse': 20}),
            lambda r: r['queues']['groups']['judge'].update({'service_p95_ms': 300000.0}),
        ):
            bad = json.loads(json.dumps(good_row))
            mutate(bad)
            self.assertFalse(assess_step1_judge(bad, 64)['passed'])

    def test_resume_quarantines_unverified_higher_ckpt(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            ckpt_dir = work / 'checkpoints'
            _write_fake_ckpt(work, ckpt_dir, 10, verified=True, complete=True)
            _write_fake_ckpt(work, ckpt_dir, 15, verified=False, complete=False)

            step = prepare_resume_checkpoint(work, ckpt_dir, world_size=4)
            self.assertEqual(step, 10)
            self.assertEqual((ckpt_dir / 'latest_checkpointed_iteration.txt').read_text().strip(), '10')
            self.assertFalse((ckpt_dir / 'global_step_15').exists())
            self.assertTrue(any((work / 'uncommitted').glob('global_step_15_*')))

    def test_sampler_append_mode_across_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'tutor-telemetry.jsonl'
            p.write_text('{"time": 1}\n', encoding='utf-8')
            s = Sampler(p, (0,), lambda: (), interval=10, append=True)
            self.assertTrue(s.append)
            with p.open('a', encoding='utf-8') as f:
                f.write('{"time": 2}\n')
            self.assertEqual(len(p.read_text(encoding='utf-8').splitlines()), 2)

    def test_verify_stage3_gpus_tolerates_gpu6_idle_util_only(self):
        snaps = {
            i: {'index': i, 'apps': [], 'used_mib': 0, 'util': 100 if i == 6 else 0}
            for i in range(8)
        }
        verify_stage3_gpus_clean(snaps, snaps)
        snaps_busy6 = {k: dict(v) for k, v in snaps.items()}
        snaps_busy6[6]['used_mib'] = 1000
        with self.assertRaises(RuntimeError):
            verify_stage3_gpus_clean(snaps_busy6, snaps_busy6)

    def test_format_stage3_status_outputs_all_fields(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            work = base / 'runs' / 'stage3_formal_20261121_150000'
            (work / 'step-statistics').mkdir(parents=True)
            (work / 'run_meta.json').write_text(json.dumps({'total_steps': 100}))
            for step, sec in ((1, 950.0), (2, 920.0), (5, 900.0)):
                (work / 'step-statistics' / f'{step}.json').write_text(json.dumps({
                    'step': step,
                    'metrics': {'timing_s/step': sec},
                    'phase_timings': {
                        'gen': {'minutes': 12.0},
                        'old_log_prob': {'minutes': 0.6},
                        'update_actor': {'minutes': 2.4},
                    },
                }))
            (work / 'ckpt-verified-5.json').write_text(json.dumps({'step': 5}))
            out = format_stage3_status(
                base,
                run_dir=work,
                gpu_lines=['0, 48000 MiB, 73415 MiB, 90 %'],
                proc_lines=['12345 python stage3_formal.py'],
            )
            self.assertIn('5 / 100', out)
            self.assertIn('global_step_5', out)
            self.assertIn('900.0s', out)
            self.assertIn('/data 剩余:', out)
            self.assertIn('GPU 0, 48000 MiB', out)

    def test_main_check_only_enforces_70gb_floor_and_cleans_preflight_dir(self):
        import stage3_formal as s3

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            models = base / 'models'
            for dname in ('Meta-Llama-3.1-8B-Instruct', 'Qwen3.6-27B'):
                mdir = models / dname
                mdir.mkdir(parents=True)
                (mdir / 'config.json').write_text('{}')
                (mdir / 'model.safetensors').write_bytes(b'w')
            (base / 'tmp').mkdir()
            (base / 'runs').mkdir()
            fake_repo = base / 'source' / 'EduardoRL'
            (fake_repo / 'eduardo').mkdir(parents=True)
            (fake_repo / 'eduardo' / 'eduardo_interaction_config.yaml').write_text(
                'interaction:\n- name: eduardo\n  config:\n    student_model: x\n'
            )

            class FakeUsage:
                def __init__(self, free):
                    self.total = 2000 * 10**9
                    self.used = self.total - free
                    self.free = free

            with ExitStack() as st:
                st.enter_context(mock.patch.object(sys, 'argv', ['stage3_formal.py', '--base', str(base), '--check-only']))
                st.enter_context(mock.patch.object(s3, 'verify_stage1_prereqs'))
                st.enter_context(mock.patch.object(s3, 'ensure_eduardo_repo', return_value=fake_repo))
                st.enter_context(mock.patch.object(s3, 'write_interaction_yaml', return_value=base / 'tmp' / 'inter.yaml'))
                st.enter_context(mock.patch.object(s3, 'write_access_log_config', return_value=base / 'tmp' / 'acc.json'))
                mock_pf = st.enter_context(mock.patch.object(s3.pcfg, 'run_preflight'))
                st.enter_context(mock.patch.object(s3.shutil, 'disk_usage', return_value=FakeUsage(69 * 10**9)))
                self.assertEqual(s3.main(), 3)
                mock_pf.assert_not_called()

            with ExitStack() as st:
                st.enter_context(mock.patch.object(sys, 'argv', ['stage3_formal.py', '--base', str(base), '--check-only']))
                st.enter_context(mock.patch.object(s3, 'verify_stage1_prereqs'))
                st.enter_context(mock.patch.object(s3, 'ensure_eduardo_repo', return_value=fake_repo))
                st.enter_context(mock.patch.object(s3, 'write_interaction_yaml', return_value=base / 'tmp' / 'inter.yaml'))
                st.enter_context(mock.patch.object(s3, 'write_access_log_config', return_value=base / 'tmp' / 'acc.json'))
                mock_pf = st.enter_context(mock.patch.object(s3.pcfg, 'run_preflight'))
                st.enter_context(mock.patch.object(s3.shutil, 'disk_usage', return_value=FakeUsage(110 * 10**9)))
                self.assertEqual(s3.main(), 0)
                mock_pf.assert_called_once()
                kwargs = mock_pf.call_args.kwargs
                self.assertEqual(kwargs['check_env']['PRO5000_JUDGE_CONC'], '64')
                self.assertEqual(kwargs['check_env']['PRO5000_MAX_CKPT_KEEP'], '1')
                self.assertEqual(kwargs['check_env']['PRO5000_CHECK_STEP1_JUDGE'], '1')
                self.assertEqual(kwargs['check_env']['PRO5000_CHECK_STEP2_RATIO'], '1')
                self.assertIn('trainer.max_actor_ckpt_to_keep=1', kwargs['overrides'])
                self.assertFalse(any((base / 'runs').iterdir()))


if __name__ == '__main__':
    unittest.main()
