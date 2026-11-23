"""Unit tests for T-017 wrap-up inventory, export sha256 verification, and guarded cleanup."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cleanup_checkpoints as cc


class CleanupCheckpointsTest(unittest.TestCase):
    def _build_tree(self, base: Path):
        runs = base / 'runs'
        formal = runs / cc.FORMAL_RUN_DEFAULT
        s2 = runs / 'stage2_pilot_20261121_120603'
        t64 = runs / 'stage3_formal_20261121_140431'
        (formal / 'checkpoints' / 'global_step_100' / 'actor').mkdir(parents=True)
        (formal / 'checkpoints' / 'global_step_100' / 'actor' / 'model.pt').write_bytes(b'ckpt100')
        (formal / 'checkpoints' / 'global_step_95').mkdir(parents=True)
        (formal / 'checkpoints' / 'latest_checkpointed_iteration.txt').write_text('100\n')
        (formal / 'exports' / 'step_100').mkdir(parents=True)
        (formal / 'exports' / 'step_100' / 'config.json').write_text('{"a": 1}')
        (formal / 'exports' / 'step_100' / 'model.safetensors').write_bytes(b'hfweights')
        (formal / 'trainer.log').write_text('ok')
        (s2 / 'uncommitted' / 'global_step_2').mkdir(parents=True)
        (s2 / 'uncommitted' / 'global_step_2' / 'x').write_bytes(b'old')
        (t64 / 'checkpoints').mkdir(parents=True)
        (base / 'envs' / 'pro5000-fast').mkdir(parents=True)
        return formal, s2

    def test_dry_run_default_preserves_all_and_skips_step100(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            formal, s2 = self._build_tree(base)
            rc = cc.run_prune(base, confirm=False, include_step100=False)
            self.assertEqual(rc, 0)
            self.assertTrue((formal / 'checkpoints' / 'global_step_95').exists())
            self.assertTrue((formal / 'checkpoints' / 'global_step_100').exists())
            self.assertTrue((s2 / 'uncommitted' / 'global_step_2').exists())

    def test_rejects_symlinks_and_non_step_names(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            formal, _ = self._build_tree(base)
            runs = base / 'runs'
            outside = Path(td) / 'outside_dir'
            outside.mkdir()
            (formal / 'checkpoints' / 'global_step_77').symlink_to(outside)
            with self.assertRaises(RuntimeError):
                cc.collect_checkpoint_targets(runs, include_step100=False)

            (formal / 'checkpoints' / 'global_step_77').unlink()
            with self.assertRaises(RuntimeError):
                cc.validate_cleanup_target(runs, formal / 'exports')
            with self.assertRaises(RuntimeError):
                cc.validate_cleanup_target(runs, formal / 'checkpoints' / 'global_step_abc')

    def test_include_step100_requires_step2_receipt_and_preserves_exports_logs(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            formal, s2 = self._build_tree(base)

            # Without Step 2 receipt, --include-step100 refuses even in dry-run.
            with self.assertRaises(RuntimeError):
                cc.run_prune(base, confirm=False, include_step100=True)

            # Without --include-step100, --confirm deletes only old/empty ckpts and leaves step 100.
            cc.run_prune(base, confirm=True, include_step100=False)
            self.assertFalse((formal / 'checkpoints' / 'global_step_95').exists())
            self.assertFalse((s2 / 'uncommitted' / 'global_step_2').exists())
            self.assertTrue((formal / 'checkpoints' / 'global_step_100').exists())

            # Write valid Step 2 receipt matching exports/step_100 sha256s.
            hashes = cc.compute_export_sha256(formal / 'exports' / 'step_100')
            (formal / 'export_step100_verified.json').write_text(json.dumps({
                'passed': True,
                'files_sha256': hashes,
            }))
            cc.run_prune(base, confirm=True, include_step100=True)
            self.assertFalse((formal / 'checkpoints' / 'global_step_100').exists())
            # Exports, logs, and envs remain untouched.
            self.assertTrue((formal / 'exports' / 'step_100' / 'model.safetensors').is_file())
            self.assertTrue((formal / 'trainer.log').is_file())
            self.assertTrue((base / 'envs' / 'pro5000-fast').is_dir())

    def test_compare_tokenizer_reports_functional_equivalence_despite_resave_diff(self):
        with tempfile.TemporaryDirectory() as td:
            exp = Path(td) / 'exp'
            base = Path(td) / 'base'
            exp.mkdir()
            base.mkdir()
            for fn, content in (
                ('chat_template.jinja', 'tpl'),
                ('tokenizer.json', '{"v": 1}'),
                ('tokenizer_config.json', '{"k": 1}'),
            ):
                (base / fn).write_text(content)
            (exp / 'chat_template.jinja').write_text('tpl')
            (exp / 'tokenizer.json').write_text('{"v": 2}')  # v5 re-serialized
            (exp / 'tokenizer_config.json').write_text('{"k": 2}')

            class FakeTok:
                chat_template = 'tpl'
                eos_token_id = 248046
                pad_token_id = 248044
                all_special_ids = [248044, 248046]
                added_tokens_decoder = {248046: types.SimpleNamespace(content='<|im_end|>')}

                def get_vocab(self):
                    return {'a': 0}

                def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True):
                    return [1, 2, 3 if enable_thinking else 4]

                def encode(self, s):
                    return [10, 11]

                def decode(self, ids, skip_special_tokens=True):
                    return 'x' if skip_special_tokens else '<|im_end|>x'

            fake_tf = types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTok())
            )
            with mock.patch.dict(sys.modules, {'transformers': fake_tf}):
                res = cc.compare_tokenizer_and_template(exp, base)
            self.assertTrue(res['chat_template_same'])
            self.assertTrue(res['tokenizer_same'])
            self.assertFalse(res['tokenizer_files_same'])
            self.assertEqual(res['tokenizer_behavior'], '一样')
            self.assertEqual(res['tokenizer_files'], '不一样')


if __name__ == '__main__':
    unittest.main()
