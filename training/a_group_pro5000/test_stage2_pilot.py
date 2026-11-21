"""Unit tests for Stage 2 serving proxy, config pinning, checkpoint audit, and report math."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pro5000_config as pcfg
from pro5000_trainer_entry import bound_trainer, verify_model_extra_checkpoint
from serving_proxy import ProxyGroup, merge_busy_seconds, summarize_proxy_window
from stage2_pilot import build_trainer_env, make_ray_tmp, verify_model_dir


class Stage2PilotTest(unittest.TestCase):
    def test_merge_busy_seconds_and_proxy_window(self):
        self.assertAlmostEqual(merge_busy_seconds([(1.0, 3.0), (2.0, 5.0), (6.0, 7.5)]), 5.5)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'proxy.jsonl'
            rows = [
                [10.0, 12.0, 0, 200, 1, 100, 500],
                [11.0, 14.0, 1, 200, 1, 200, 800],
                [15.0, 16.0, 0, 502, 1, None, 500],
                [30.0, 31.0, 0, 200, 1, 50, 500],  # outside span
            ]
            p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
            out = summarize_proxy_window(p, [(10.0, 20.0)], 2)
            self.assertEqual(out['total_chat_requests'], 3)
            self.assertEqual(out['non_2xx'], 1)
            self.assertAlmostEqual(out['combined_busy_wall_s'], 5.0)
            self.assertAlmostEqual(out['avg_completion_tokens'], 150.0)
            self.assertAlmostEqual(out['avg_interaction_tokens'], 100.0)
            self.assertAlmostEqual(out['avg_final_tokens'], 200.0)
            self.assertEqual(out['backends']['0']['requests'], 2)
            self.assertEqual(out['backends']['1']['requests'], 1)

    def test_least_inflight_pick_backend(self):
        with tempfile.TemporaryDirectory() as td:
            pg = ProxyGroup(['http://127.0.0.1:1', 'http://127.0.0.1:2'], Path(td) / 'p.jsonl')
            try:
                self.assertEqual(pg.pick_backend(), 0)
                pg.inflight[0] = 3
                pg.inflight[1] = 1
                self.assertEqual(pg.pick_backend(), 1)
            finally:
                pg.log_file.close()

    def test_verify_model_extra_checkpoint_rejects_optimizer(self):
        with tempfile.TemporaryDirectory() as td:
            step = Path(td) / 'global_step_2'
            actor = step / 'actor'
            (actor / 'huggingface').mkdir(parents=True)
            (actor / 'huggingface' / 'config.json').write_text('{}')
            (actor / 'fsdp_config.json').write_text('{"world_size": 4}')
            (step / 'data.pt').write_bytes(b'x')
            for r in range(4):
                (actor / f'model_world_size_4_rank_{r}.pt').write_bytes(b'm')
                (actor / f'extra_state_world_size_4_rank_{r}.pt').write_bytes(b'e')
            info = verify_model_extra_checkpoint(step, world_size=4)
            self.assertEqual(info['optim_files'], 0)

            (actor / 'optim_world_size_4_rank_0.pt').write_bytes(b'o')
            with self.assertRaises(RuntimeError):
                verify_model_extra_checkpoint(step, world_size=4)

    def test_config_pin_accepts_expected_and_rejects_lr_drift(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            work = base / 'work'
            ckpt = work / 'checkpoints'
            inter = work / 'interaction.yaml'
            cfg = {
                'trainer': {
                    'total_training_steps': 2, 'save_freq': 2, 'test_freq': 25,
                    'val_before_train': False, 'val_only': False, 'logger': ['console'],
                    'nnodes': 1, 'n_gpus_per_node': 4, 'max_actor_ckpt_to_keep': None,
                    'del_local_ckpt_after_load': False, 'default_local_dir': str(ckpt),
                    'resume_mode': 'disable', 'resume_from_path': None,
                },
                'data': {
                    'train_batch_size': 16,
                    'train_files': str(base / 'data/author_prepared/train.parquet'),
                    'val_files': str(base / 'data/author_prepared/val.parquet'),
                    'shuffle': True, 'max_prompt_length': 2048, 'max_response_length': 8192,
                },
                'actor_rollout_ref': {
                    'actor': {
                        'ppo_mini_batch_size': 8, 'ppo_epochs': 1,
                        'ppo_micro_batch_size_per_gpu': 1, 'use_dynamic_bsz': False,
                        'loss_agg_mode': 'token-mean', 'use_kl_loss': False,
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
                'custom_reward_function': {
                    'reward_kwargs': {
                        'num_student_attempts': 16, 'max_thinking_tokens_per_turn': 512,
                        'max_tokens_per_turn': 150, 'judge_votes': 3, 'judge_hard_gate': True,
                        'omit_teacher_turns': True, 'test_on_transfer': True, 'enable_thinking': True,
                        'judge_concurrency': 16, 'student_concurrency': 32, 'student_temperature': 0.6,
                        'judge_temperature': 0.6, 'eval_judge_model': '', 'judge_model': 'judge-local',
                        'student_model': 's',
                    }
                },
                'ray_kwargs': {
                    'ray_init': {
                        'runtime_env': {
                            'worker_process_setup_hook': 'queue_metrics.install',
                            'env_vars': {'EDU_QUEUE_METRICS_DIR': str(work / 'queue-metrics')},
                        }
                    }
                },
            }
            pcfg.pin_composed(
                cfg, base=base, work=work, ckpt_dir=ckpt, student_name='s',
                interaction=inter, total_steps=2, save_freq=2, test_freq=25,
            )
            cfg['actor_rollout_ref']['actor']['optim']['lr'] = 1e-5
            with self.assertRaises(ValueError):
                pcfg.pin_composed(
                    cfg, base=base, work=work, ckpt_dir=ckpt, student_name='s',
                    interaction=inter, total_steps=2, save_freq=2, test_freq=25,
                )

    def test_make_ray_tmp_fits_unix_socket_limit(self):
        sock_suffix = '/session_2099-12-31_23-59-59_999999_4194304/sockets/plasma_store'
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            long_base = parent / 'a_group_pro5000_very_long_subdir_name_here'
            long_base.mkdir()
            # Simulate Pro 5000 where base.parent fits the socket budget while base itself does not.
            parent_budget = len((str(parent / 'r0000') + sock_suffix).encode('utf-8'))
            rt = make_ray_tmp(long_base, max_socket_bytes=parent_budget)
            try:
                self.assertEqual(rt.parent, parent)
                self.assertLessEqual(len((str(rt) + sock_suffix).encode('utf-8')), parent_budget)
            finally:
                rt.rmdir()

            # Verify default 107-byte budget raises cleanly only if no writable ancestor/tmp fits,
            # and otherwise succeeds within 107 bytes.
            try:
                rt_real = make_ray_tmp(long_base)
            except RuntimeError as exc:
                self.assertIn('107 bytes', str(exc))
            else:
                try:
                    self.assertLessEqual(len((str(rt_real) + sock_suffix).encode('utf-8')), 107)
                finally:
                    rt_real.rmdir()

    def test_build_trainer_env_has_preflight_keys(self):
        env = build_trainer_env(
            base=Path('/b'), work=Path('/b/w'), ckpt_dir=Path('/b/c'), raytmp=Path('/tmp/r1234'),
            tutor_gpus=[0, 1, 2, 3], all_gpus=[0, 1, 2, 3, 4, 5, 6, 7], student_gpus=[4, 6],
            layout={'student_concurrency': 32, 'judge_concurrency': 16},
            repo=Path('/b/repo'), tutor_model=Path('/m/t'),
            student_api_base='http://127.0.0.1:18861/v1', judge_api_base='http://127.0.0.1:18862/v1',
            start_step=0, total_steps=2, save_freq=2,
        )
        for key in (
            'PRO5000_BASE', 'WORK_DIR', 'PRO5000_CKPT_DIR', 'RAY_TMPDIR',
            'PRO5000_START_STEP', 'PRO5000_TOTAL_STEPS', 'PRO5000_SAVE_FREQ', 'PRO5000_TEST_FREQ',
        ):
            self.assertIn(key, env)
        self.assertEqual(env['PRO5000_TOTAL_STEPS'], '2')

    def test_validate_does_not_clobber_step_queue_window(self):
        class FakeMonitor:
            def begin(self):
                return 'snap'

            def end(self, snap, enforce=True):
                return {'requests': 1, 'parse': 0, 'recovery': 0}

        class FakeLoader:
            def state_dict(self):
                return {'_num_yielded': 2}

        class DummyBase:
            def _get_gen_batch(self, batch):
                return batch

            def _update_actor(self, batch):
                return 'updated'

            def _validate(self, merged=False):
                self._get_gen_batch('val_batch')
                self.saw_validation_flag = self.formal_validation
                return {'val': 1}

        Trainer = bound_trainer(DummyBase)
        t = Trainer()
        t.global_steps = 2
        t.train_dataloader = FakeLoader()
        t.formal_monitor = FakeMonitor()
        t.formal_validation = False

        t._get_gen_batch('train_batch')
        start_ns = t.formal_queue_start
        t.formal_phase_windows['gen'].append((10.0, 20.0))
        t._update_actor('train_batch')
        end_ns = t.formal_queue_end

        t._validate()
        self.assertTrue(t.saw_validation_flag)
        self.assertFalse(t.formal_validation)
        self.assertEqual(t.formal_queue_start, start_ns)
        self.assertEqual(t.formal_queue_end, end_ns)
        self.assertLessEqual(t.formal_queue_start, t.formal_queue_end)
        self.assertEqual(t.formal_phase_windows['gen'], [(10.0, 20.0)])

    def test_verify_model_dir_symlink_and_missing_shard(self):
        with tempfile.TemporaryDirectory() as td:
            shm = Path(td) / 'shm' / 'Qwen3.6-27B'
            shm.mkdir(parents=True)
            (shm / 'config.json').write_text('{"model_type":"qwen3"}')
            (shm / 'model-00001-of-00002.safetensors').write_bytes(b'shard1')
            (shm / 'model-00002-of-00002.safetensors').write_bytes(b'shard2')
            (shm / 'model.safetensors.index.json').write_text(
                json.dumps({'weight_map': {'a': 'model-00001-of-00002.safetensors', 'b': 'model-00002-of-00002.safetensors'}})
            )
            link = Path(td) / 'models' / 'Qwen3.6-27B'
            link.parent.mkdir(parents=True)
            link.symlink_to(shm)

            self.assertEqual(verify_model_dir('Judge', link), link)

            (shm / 'model-00002-of-00002.safetensors').unlink()
            with self.assertRaises(RuntimeError):
                verify_model_dir('Judge', link)

            import shutil as _shutil
            _shutil.rmtree(shm)
            with self.assertRaises(RuntimeError):
                verify_model_dir('Judge', link)


if __name__ == '__main__':
    unittest.main()
