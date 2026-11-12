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

    def test_table_combines_pilot_and_displays_errors(self):
        source = {'有完成标记来源': 496, '暂停来源': 1, '不完整来源': 1,
                  '学生请求有意图无回答数': 1, '对话请求有意图无回执数': 1,
                  '来源检查异常': 2, '暂停原因类型:RuntimeError': 1}
        data = {
            'EASY': {'来源': source, '判分': {'报告中的未决作答': 1003,
                     '可由原始回执重新提取': 413, '已有叠层已核验候选': 403,
                     '未决减去可提取候选': 590, '无法完整核验的未决作答': 3}},
            'PILOT': {'来源': {'有完成标记来源': 1},
                      '判分': {'报告中的未决作答': 12, '可由原始回执重新提取': 4,
                               '已有叠层已核验候选': 4, '未决减去可提取候选': 8}},
            'MEDIUM': {'来源': {}, '判分': {'报告中的未决作答': 1138,
                       '可由原始回执重新提取': 463, '未决减去可提取候选': 675}},
            'HARD': {'来源': {}, '判分': {'报告中的未决作答': 443,
                     '可由原始回执重新提取': 0, '未决减去可提取候选': 443}},
        }
        text = inventory.render_report(data)
        line = next(line for line in text.splitlines() if '未决作答' in line and '无法' not in line)
        self.assertEqual([s.strip() for s in line.strip('|').split('|')][1:],
                         ['1015', '1138', '443', '2596'])
        line = next(line for line in text.splitlines() if '原回执可提取候选' in line)
        self.assertEqual([s.strip() for s in line.strip('|').split('|')][1:],
                         ['417', '463', '0', '880'])
        self.assertIn('无法完整核验的未决作答', text)
        self.assertIn('RuntimeError', text)
        self.assertIn('不能等同可直接补判数', text)
        self.assertNotIn('source_id', text)
        self.assertNotIn('reference_answer', text)
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith('|'):
                previous_border = next(x for x in reversed(lines[:i]) if x.startswith('+'))
                self.assertEqual(inventory.display_width(line), inventory.display_width(previous_border))

    def test_report_uses_whitelist_of_keys(self):
        text = inventory.render_report({'EASY': {'来源': {'任意错误:私密内容': 9}, '判分': {}}})
        self.assertNotIn('私密内容', text)
        self.assertIn('来源结构异常', text)

if __name__ == '__main__':
    unittest.main()
