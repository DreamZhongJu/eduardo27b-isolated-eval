import contextlib
import io
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import verify_checkpoints as v


class CheckpointTests(unittest.TestCase):
    def test_model_and_training_shards_have_distinct_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / 'actor'
            (actor/'huggingface').mkdir(parents=True)
            (actor/'huggingface/config.json').write_text('{"model_type":"fixture"}')
            (actor/'fsdp_config.json').write_text('{"world_size":2}')
            (actor/'model_world_size_2_rank_0.pt').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'model rank'):
                v.inspect_input(root)
            (actor/'model_world_size_2_rank_1.pt').write_bytes(b'x')
            (root/'data.pt').write_bytes(b'x')
            self.assertEqual(v.inspect_input(root)[1], 1)

    def test_refuses_existing_output_or_output_under_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work = base/'work'
            work.mkdir()
            with self.assertRaises(ValueError):
                v.verify(work, work/'new')
            with self.assertRaises(FileExistsError):
                v.verify(work, base)

    def test_two_sequential_merges_and_early_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work = base/'work'
            work.mkdir()
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(v, 'inspect_input', return_value=({}, 5)), \
                 patch.object(v, 'inspect_export', return_value=(1, 2)), \
                 patch.object(v.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as merger:
                self.assertEqual(v.verify(work, base/'first'), 0)
                self.assertEqual(merger.call_count, 2)
                for step, call in enumerate(merger.call_args_list, 1):
                    self.assertIn(str(work/'verl_ckpts/b-functional'/f'global_step_{step}'/'actor'), call.args[0])
                    self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                merger.reset_mock()
                merger.return_value = SimpleNamespace(returncode=1)
                self.assertEqual(v.verify(work, base/'second'), 1)
                self.assertEqual(merger.call_count, 1)

    @unittest.skipUnless(find_spec('torch') and find_spec('safetensors'), 'CPU tensor libraries unavailable')
    def test_actual_cpu_shards_and_bad_index(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            config = {'model_type':'fixture','architectures':['Fixture']}
            (out/'config.json').write_text(json.dumps(config))
            save_file({'one':torch.ones((3, 2), dtype=torch.bfloat16)}, out/'a.safetensors')
            self.assertEqual(v.inspect_export(out, config), (1, 1))
            save_file({'two':torch.tensor(2.0)}, out/'b.safetensors')
            index = out/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'one':'a.safetensors','two':'b.safetensors'}}))
            self.assertEqual(v.inspect_export(out, config), (2, 2))
            index.write_text(json.dumps({'weight_map':{'one':'b.safetensors','two':'a.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'weight map'):
                v.inspect_export(out, config)
