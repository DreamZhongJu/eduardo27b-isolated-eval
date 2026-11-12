import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('inventory', Path(__file__).with_name('unresolved_inventory.py'))
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)
from test_agnes_topbool_recovery import setup_source, snapshot


class InventoryTests(unittest.TestCase):
    def make_root(self, tmp):
        root, overlay = Path(tmp)/'root', Path(tmp)/'overlay'
        root.mkdir()
        overlay.mkdir()
        key, report = setup_source(root, ['true', 'false', 'null', '{"correct":true}', '"true"'])
        return root, overlay, key, report

    def test_overlay_checked_and_no_mutation_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, overlay, key, _ = self.make_root(tmp)
            data = inventory.source_overlay(root, 'EASY', key)
            out = overlay/'EASY'/key/'OVERLAY.json'
            out.parent.mkdir(parents=True)
            out.write_text(json.dumps(data))
            before = snapshot(root), snapshot(overlay)
            with patch('socket.socket.connect', side_effect=AssertionError('no network')):
                counts = inventory.run(root, overlay)['EASY']['判分']
            self.assertEqual((snapshot(root), snapshot(overlay)), before)
            self.assertEqual(counts['报告中的未决作答'], 4)
            self.assertEqual(counts['可由原始回执重新提取'], 2)
            self.assertEqual(counts['已有叠层已核验候选'], 2)
            self.assertEqual(counts['未决减去可提取候选'], 2)

    def test_modified_overlay_never_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, overlay, key, _ = self.make_root(tmp)
            out = overlay/'EASY'/key/'OVERLAY.json'
            out.parent.mkdir(parents=True)
            out.write_text('{}')
            counts = inventory.run(root, overlay)['EASY']['判分']
            self.assertEqual(counts.get('已有叠层已核验候选', 0), 0)
            self.assertEqual(counts['叠层不符但原始回执可提取'], 2)
            self.assertEqual(counts['叠层不符或无法读取来源'], 1)

    def test_corrupt_input_is_unknown_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, overlay, key, _ = self.make_root(tmp)
            score = root/'EASY'/'sources'/key/'SCORE_INPUT.json'
            score.write_bytes(score.read_bytes() + b' ')
            counts = inventory.run(root, overlay)['EASY']['判分']
            self.assertEqual(counts['无法完整核验的未决作答'], 4)
            self.assertEqual(counts.get('可由原始回执重新提取', 0), 0)
            self.assertEqual(counts['未决减去可提取候选'], 4)

    def test_incomplete_only_structural_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, overlay, _, _ = self.make_root(tmp)
            source_key = hashlib.sha256(b'synthetic-partial').hexdigest()[:24]
            source = root/'HARD'/'sources'/source_key
            sample = source/'samples'/'source'/'pre'/'00'
            sample.mkdir(parents=True)
            (sample/'INTENT.json').write_text('{}')
            turn = source/'dialogue'/'teacher'/'00'
            turn.mkdir(parents=True)
            (turn/'INTENT.json').write_text('{}')
            (source/'dialogue'/'ROLLOUT_INTENT.json').write_text('{}')
            counts = inventory.run(root, overlay)['HARD']['来源']
            self.assertEqual(counts['不完整来源'], 1)
            self.assertEqual(counts['学生请求有意图无回答数'], 1)
            self.assertEqual(counts['对话请求有意图无回执数'], 1)
            self.assertEqual(counts['对话整体无完成回执的来源'], 1)

if __name__ == '__main__':
    unittest.main()
