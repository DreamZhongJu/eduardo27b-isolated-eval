from copy import deepcopy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_pro5000 as c


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return [ord(w[0]) % 250 + 1 for w in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        words = []
        for m in messages:
            words.extend(m['content'].split())
        return {'input_ids': [ord(w[0]) % 250 + 1 for w in words] + [251, 252]}


class ComparePro5000Test(unittest.TestCase):
    def sample_result(self, *, repeat_same=False):
        slow1_vals = [-2.0, -3.0, -4.0, -5.0]
        slow2_vals = list(slow1_vals) if repeat_same else [-2.2, -3.0, -4.0, -5.0]
        fast_vals = [-2.005, -3.005, -4.0, -5.0]
        runs = [
            {
                'mode': 'slow', 'values': slow1_vals, 'loss': 3.5, 'grad_norm': 2.0,
                'seconds_per_token': 0.02, 'peak_alloc_mib': 40000, 'peak_reserved_mib': 42000,
                'kernel_call_counts': {'slow_conv': 24, 'slow_chunk': 24, 'slow_norm': 24},
            },
            {
                'mode': 'slow', 'values': slow2_vals,
                'loss': 3.5 if repeat_same else 3.6,
                'grad_norm': 2.0 if repeat_same else 2.1,
                'seconds_per_token': 0.02, 'peak_alloc_mib': 40000, 'peak_reserved_mib': 42000,
                'kernel_call_counts': {'slow_conv': 24, 'slow_chunk': 24, 'slow_norm': 24},
            },
            {
                'mode': 'fast', 'values': fast_vals, 'loss': 3.51, 'grad_norm': 2.02,
                'seconds_per_token': 0.01, 'peak_alloc_mib': 39000, 'peak_reserved_mib': 41000,
                'kernel_call_counts': {'fast_conv': 24, 'fast_chunk': 24, 'fast_norm': 24},
            },
        ]
        endurance = {
            mode: {
                'elapsed_seconds': dur + 1.0, 'cycles': 10, 'seconds_per_token': 0.01,
                'peak_alloc_mib': 40000, 'peak_reserved_mib': 42000,
                'round_trend': c.round_trend([{'round_seconds': 2.0, 'ms_per_token': 10.0, 'loadavg_1': 1.0}]),
            }
            for mode, dur in c.SOAK_SECONDS.items()
        }
        return {
            'fast_kernel_verified': True,
            'sequence_lengths': [1200, 4200, 8800],
            'teacher_tokens': [1, 2, 1],
            'runs': runs,
            'endurance': endurance,
            'endurance_skipped': False,
        }

    def test_synthetic_messages_and_mask(self):
        base = [{'role': 'system', 'content': 'sys prompt'}, {'role': 'user', 'content': 'solve x'}]
        msgs = c._build_synthetic_messages(base, turns=3, variant=0, scale=2)
        seq = c.reconstruct(msgs, FakeTokenizer(), 'short')
        self.assertIsNotNone(seq)
        self.assertTrue(any(seq.mask))
        self.assertFalse(seq.mask[0])

    def test_within_repeat_passes(self):
        res = self.sample_result(repeat_same=False)
        self.assertTrue(c.score_result(res, max_mem_mib=73415))
        self.assertEqual(res['criterion_used'], 'within_repeat')

    def test_section14_passes_when_slow_repeat_is_zero(self):
        res = self.sample_result(repeat_same=True)
        self.assertTrue(c.score_result(res, max_mem_mib=73415))
        self.assertEqual(res['criterion_used'], 'section14_tolerance')

    def test_section14_rejects_large_diff_or_missing_fast_kernel(self):
        bad_loss = self.sample_result(repeat_same=True)
        bad_loss['runs'][2]['loss'] = 3.60  # > 1%
        bad_loss['endurance'] = {}
        bad_loss['endurance_skipped'] = True
        self.assertFalse(c.score_result(bad_loss, max_mem_mib=73415))
        self.assertEqual(bad_loss['criterion_used'], 'failed')

        bad_tok = self.sample_result(repeat_same=True)
        bad_tok['runs'][2]['values'][0] = -3.5  # > 1.0 nat
        bad_tok['endurance'] = {}
        bad_tok['endurance_skipped'] = True
        self.assertFalse(c.score_result(bad_tok, max_mem_mib=73415))

        no_fast = self.sample_result(repeat_same=True)
        no_fast['runs'][2]['kernel_call_counts']['fast_conv'] = 0
        no_fast['endurance'] = {}
        no_fast['endurance_skipped'] = True
        self.assertFalse(c.score_result(no_fast, max_mem_mib=73415))


if __name__ == '__main__':
    unittest.main()
