import importlib.util
import pathlib
import unittest
from collections import Counter

p = pathlib.Path(__file__).with_name('source_gain.py')
spec = importlib.util.spec_from_file_location('gain_tested', p)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def counts(pre, control, post, unknown=0):
    out = {}
    for target in ('原题', '迁移题'):
        for phase, n in (('pre', pre), ('control', control), ('post', post)):
            out[(target, phase)] = Counter(correct=n, incorrect=16-n-unknown, unknown=unknown)
    return out


class GainTest(unittest.TestCase):
    def test_end_to_end_real_verified_fixture_readonly(self):
        import hashlib
        import socket
        from unittest.mock import patch
        fixture_path = p.with_name('test_summary.py')
        fspec = importlib.util.spec_from_file_location('summary_fixture', fixture_path)
        fixture = importlib.util.module_from_spec(fspec)
        fspec.loader.exec_module(fixture)
        f = fixture.SummaryTest()
        f.setUp()
        try:
            before = {x: hashlib.sha256(x.read_bytes()).hexdigest() for d in (f.root, f.overlay, f.out)
                      for x in d.rglob('*') if x.is_file()}
            with patch.object(socket, 'socket', side_effect=AssertionError('network called')):
                records, errors = g.collect_sources(fixture.s, fixture.inv, f.root, f.overlay, f.out)
                result = g.render(fixture.s, records, errors, 10000, 42)
            self.assertEqual(len(records), 1)
            self.assertNotIn(f.sid, result)
            self.assertNotIn('synthetic question', result)
            self.assertIn('原题合并', result)
            after = {x: hashlib.sha256(x.read_bytes()).hexdigest() for d in (f.root, f.overlay, f.out)
                     for x in d.rglob('*') if x.is_file()}
            self.assertEqual(before, after)
        finally:
            f.tearDown()

    def test_formula(self):
        self.assertAlmostEqual(g.normalized_gain(.5, .75), .5)
        self.assertAlmostEqual(g.normalized_gain(.75, .5), -.25)
        self.assertEqual(g.normalized_gain(1, 1), 0)
        self.assertEqual(g.normalized_gain(0, 1), 1)

    def test_unknown(self):
        c = Counter(correct=8, incorrect=4, unknown=4)
        self.assertEqual(g.p_value(c, True), .5)
        self.assertAlmostEqual(g.p_value(c, False), 2/3)
        self.assertIsNone(g.p_value(Counter(unknown=16), False))
        with self.assertRaises(ValueError):
            g.p_value(Counter(correct=3), True)

    def test_cluster_weighting(self):
        # Source A appears twice but receives same total weight as source B.
        data = [('简单', 'A', counts(0,0,8)), ('中等','A', counts(0,0,16)),
                ('简单','B',counts(0,0,0))]
        pairs, _, _ = g.estimate(data,'汇总','原题','pre',True)
        self.assertEqual(pairs, [('A', .75), ('B', 0.)])
        self.assertEqual(sum(v for _, v in pairs)/len(pairs), .375)

    def test_filter_fixed_pre_not_control(self):
        data = [('简单','A',counts(15,0,16)), ('简单','B',counts(14,16,16))]
        for baseline in ('pre','control'):
            pairs, n, _ = g.estimate(data, '简单', '迁移题', baseline, True, True)
            self.assertEqual([x[0] for x in pairs], ['B'])
            self.assertEqual(n, 1)

    def test_bootstrap_reproducible_and_degenerate(self):
        self.assertEqual(g.bootstrap([.1, .2, .3], 42, 100), g.bootstrap([.1, .2, .3], 42, 100))
        self.assertEqual(g.bootstrap([.5]*5, 9, 100), (.5,.5,.5,5))
        self.assertIsNone(g.bootstrap([], 42, 100))

    def test_all_unknown_excluded(self):
        row = counts(1, 1, 1)
        row[('迁移题', 'post')] = Counter(unknown=16)
        pairs, _, n = g.estimate([('简单','A',row)], '简单','迁移题','pre', False)
        self.assertEqual(pairs, [])
        self.assertEqual(n, 1)
        pairs, _, n = g.estimate([('简单','A',row)], '简单','迁移题','pre', True)
        self.assertEqual(pairs, [('A', -1/16)])

if __name__ == '__main__':
    unittest.main()
