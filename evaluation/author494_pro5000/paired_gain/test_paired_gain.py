import csv
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import paired_gain as pg


def _load_source_gain():
    env_path = os.environ.get("SOURCE_GAIN_PY", "").strip()
    if env_path:
        sg_path = pathlib.Path(env_path)
        if not sg_path.is_file():
            raise AssertionError(
                f"SOURCE_GAIN_PY points to a missing file: {sg_path}. "
                "Set SOURCE_GAIN_PY to the full path of source_gain.py."
            )
    else:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        sg_path = (
            repo_root
            / "代码"
            / "eduardo-a-group-prep"
            / "evaluation"
            / "agnes_rejudge"
            / "source_gain.py"
        )
        if not sg_path.is_file():
            raise AssertionError(
                f"Default source_gain.py not found at {sg_path}. "
                "Please set environment variable SOURCE_GAIN_PY to the full path of source_gain.py."
            )

    spec = importlib.util.spec_from_file_location("ref_source_gain", sg_path)
    if spec is None or spec.loader is None:
        raise AssertionError(
            f"Failed to load module spec from {sg_path} (SOURCE_GAIN_PY)."
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPairedGain(unittest.TestCase):
    def test_formula4_hand_examples_and_boundaries(self):
        # Positive gain: (0.75 - 0.5) / (1 - 0.5) = 0.5
        self.assertAlmostEqual(pg.normalized_gain(0.5, 0.75), 0.5)
        # Negative gain: 0.5 - 0.75 = -0.25
        self.assertAlmostEqual(pg.normalized_gain(0.75, 0.5), -0.25)
        # Equal pre and post: 0.0
        self.assertAlmostEqual(pg.normalized_gain(0.4, 0.4), 0.0)
        # Boundary pre = 0
        self.assertAlmostEqual(pg.normalized_gain(0.0, 0.0), 0.0)
        self.assertAlmostEqual(pg.normalized_gain(0.0, 0.6), 0.6)
        self.assertAlmostEqual(pg.normalized_gain(0.0, 1.0), 1.0)
        # Boundary pre = 1
        self.assertAlmostEqual(pg.normalized_gain(1.0, 1.0), 0.0)
        self.assertAlmostEqual(pg.normalized_gain(1.0, 0.25), -0.75)

        with self.assertRaises(ValueError):
            pg.normalized_gain(-0.1, 0.5)
        with self.assertRaises(ValueError):
            pg.normalized_gain(0.5, 1.1)

    def test_matches_source_gain_on_grid(self):
        sg = _load_source_gain()
        grid = [i / 20.0 for i in range(21)] + [i / 16.0 for i in range(17)]
        for pre in grid:
            for post in grid:
                self.assertEqual(
                    pg.normalized_gain(pre, post),
                    sg.normalized_gain(pre, post),
                    msg=f"Mismatch at pre={pre}, post={post}",
                )

    def test_hand_calculated_small_dataset(self):
        # Hand-crafted dataset:
        # unit_1 (easy):
        #   base: 2 rows -> pre=0.5, post=(12/16=0.75 => gain=0.5) and (16/16=1.0 => gain=1.0); mean=0.75
        #   a_group: 1 row -> pre=0.5, post=16/16=1.0 => gain=1.0
        #   diff = 1.0 - 0.75 = 0.25
        # unit_2 (easy):
        #   base: pre=0.0, post=4/16=0.25 => gain=0.25
        #   a_group: pre=0.0, post=8/16=0.5 => gain=0.5
        #   diff = 0.5 - 0.25 = 0.25
        # unit_3 (easy, high pre=15/16=0.9375):
        #   base: post=15/16 => gain=0.0
        #   a_group: post=16/16 => gain=(1 - 15/16)/(1 - 15/16) = 1.0
        #   diff = 1.0
        # unit_4 (easy, unpaired: only in base):
        #   base: pre=0.5, post=0.5 => gain=0.0
        # unit_5 (easy, unpaired: only in a_group):
        #   a_group: pre=0.5, post=1.0 => gain=1.0
        # unit_6 (easy, all unknown in a_group):
        #   base: pre=0.5, post=8/16=0.5 => gain=0.0
        #   a_group: pre=0.5, post_correct=0, post_total=16, post_unknown=16
        #     under unknown_as_wrong=True: post=0.0 => gain=-0.5; diff = -0.5
        #     under unknown_as_wrong=False: skipped => unit_6 becomes unpaired (only in base)!
        rows = [
            {"unit_id": "unit_1", "condition": "base", "group": "easy", "pre": 0.5, "post_correct": 12, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_1", "condition": "base", "group": "easy", "pre": 0.5, "post_correct": 16, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_1", "condition": "a_group", "group": "easy", "pre": 0.5, "post_correct": 16, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_2", "condition": "base", "group": "easy", "pre": 0.0, "post_correct": 4, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_2", "condition": "a_group", "group": "easy", "pre": 0.0, "post_correct": 8, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_3", "condition": "base", "group": "easy", "pre": 15 / 16, "post_correct": 15, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_3", "condition": "a_group", "group": "easy", "pre": 15 / 16, "post_correct": 16, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_4", "condition": "base", "group": "easy", "pre": 0.5, "post_correct": 8, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_5", "condition": "a_group", "group": "easy", "pre": 0.5, "post_correct": 16, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_6", "condition": "base", "group": "easy", "pre": 0.5, "post_correct": 8, "post_total": 16, "post_unknown": 0},
            {"unit_id": "unit_6", "condition": "a_group", "group": "easy", "pre": 0.5, "post_correct": 0, "post_total": 16, "post_unknown": 16},
        ]

        report = pg.evaluate_dataset(rows, draws=500, seed=123, include_high_pre_filtered=True)
        by_key = {
            (r["source_filter"], r["unknown_policy"], r["group"]): r
            for r in report["results"]
        }

        # 1) Primary (all), unknown_as_wrong: paired={unit_1, unit_2, unit_3, unit_6}
        r_all_uaw = by_key[("all", "unknown_as_wrong", "easy")]
        self.assertEqual(r_all_uaw["n_paired"], 4)
        self.assertEqual(r_all_uaw["n_unpaired_base"], 1)
        self.assertEqual(r_all_uaw["n_unpaired_target"], 1)
        self.assertAlmostEqual(r_all_uaw["stats"]["base_mean"], (0.75 + 0.25 + 0.0 + 0.0) / 4)
        self.assertAlmostEqual(r_all_uaw["stats"]["target_mean"], (1.0 + 0.5 + 1.0 - 0.5) / 4)
        self.assertAlmostEqual(r_all_uaw["stats"]["diff_mean"], (0.25 + 0.25 + 1.0 - 0.5) / 4)

        # 2) Primary (all), exclude_unknown: unit_6 excluded from a_group => paired={unit_1, unit_2, unit_3}
        r_all_ex = by_key[("all", "exclude_unknown", "easy")]
        self.assertEqual(r_all_ex["n_paired"], 3)
        self.assertEqual(r_all_ex["n_unpaired_base"], 2)  # unit_4 and unit_6
        self.assertEqual(r_all_ex["n_unpaired_target"], 1)  # unit_5
        self.assertEqual(r_all_ex["n_skipped_rows_target"], 1)
        self.assertEqual(r_all_ex["n_excluded_zero_target"], 1)
        self.assertAlmostEqual(r_all_ex["stats"]["base_mean"], (0.75 + 0.25 + 0.0) / 3)
        self.assertAlmostEqual(r_all_ex["stats"]["target_mean"], (1.0 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(r_all_ex["stats"]["diff_mean"], (0.25 + 0.25 + 1.0) / 3)

        # 3) Secondary (exclude_pre_ge_15_16), unknown_as_wrong: unit_3 filtered out
        r_filt_uaw = by_key[("exclude_pre_ge_15_16", "unknown_as_wrong", "easy")]
        self.assertEqual(r_filt_uaw["n_paired"], 3)
        self.assertEqual(r_filt_uaw["n_filtered_high_pre"], 1)
        self.assertAlmostEqual(r_filt_uaw["stats"]["diff_mean"], (0.25 + 0.25 - 0.5) / 3)

    def test_reproducibility_same_seed(self):
        rows = [
            {
                "unit_id": f"u{i}",
                "condition": cond,
                "group": "author494",
                "pre": 0.25,
                "post_correct": 6 + (1 if cond == "a_group" else 0) + (i % 4),
                "post_total": 16,
                "post_unknown": 0,
            }
            for i in range(30)
            for cond in ("base", "a_group")
        ]
        rep1 = pg.evaluate_dataset(rows, draws=1000, seed=42)
        rep2 = pg.evaluate_dataset(rows, draws=1000, seed=42)
        self.assertEqual(rep1, rep2)

    def test_bootstrap_coverage_simulation(self):
        # 250 synthetic datasets, n=80 paired sources each, true diff = 0.12
        rng = np.random.default_rng(20261118)
        n_trials = 250
        n_units = 80
        true_diff = 0.12
        covered = 0

        for trial in range(n_trials):
            base_vals = rng.normal(loc=0.10, scale=0.20, size=n_units)
            diff_vals = rng.normal(loc=true_diff, scale=0.15, size=n_units)
            target_vals = base_vals + diff_vals
            stats = pg._bootstrap_stats_from_arrays(
                base_vals, target_vals, seed=10000 + trial, draws=400
            )
            if stats["diff_ci_low"] <= true_diff <= stats["diff_ci_high"]:
                covered += 1

        coverage = covered / n_trials
        self.assertGreaterEqual(coverage, 0.92, msg=f"Coverage too low: {coverage}")
        self.assertLessEqual(coverage, 0.98, msg=f"Coverage too high: {coverage}")

    def test_paired_ci_narrower_than_unpaired_when_correlated(self):
        rng = np.random.default_rng(777)
        n_units = 100
        shared_difficulty = rng.normal(loc=0.0, scale=0.35, size=n_units)
        base_vals = shared_difficulty + rng.normal(loc=0.10, scale=0.05, size=n_units)
        target_vals = shared_difficulty + rng.normal(loc=0.22, scale=0.05, size=n_units)

        paired_stats = pg._bootstrap_stats_from_arrays(
            base_vals, target_vals, seed=99, draws=2000
        )
        paired_width = paired_stats["diff_ci_high"] - paired_stats["diff_ci_low"]

        # Unpaired bootstrap: resample base and target independently
        boot_rng = np.random.default_rng(99)
        idx_b = boot_rng.integers(0, n_units, size=(2000, n_units))
        idx_t = boot_rng.integers(0, n_units, size=(2000, n_units))
        unpaired_diffs = target_vals[idx_t].mean(axis=1) - base_vals[idx_b].mean(axis=1)
        unp_lo, unp_hi = np.percentile(unpaired_diffs, [2.5, 97.5])
        unpaired_width = float(unp_hi - unp_lo)

        self.assertLess(paired_width, unpaired_width * 0.5)

    def test_no_unit_id_in_outputs_and_cli_e2e(self):
        secret_ids = [f"SECRET_SOURCE_ID_{i:04d}" for i in range(15)]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            csv_path = tmp / "input.csv"
            json_path = tmp / "out.json"
            md_path = tmp / "out.md"

            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(pg.REQUIRED_COLUMNS)
                for i, uid in enumerate(secret_ids):
                    writer.writerow([uid, "base", "author494", 0.25, 6 + (i % 3), 16, 1])
                    writer.writerow([uid, "a_group", "author494", 0.25, 9 + (i % 3), 16, 1])

            rc = pg.main(
                [
                    "--input",
                    str(csv_path),
                    "--draws",
                    "500",
                    "--seed",
                    "42",
                    "--json-out",
                    str(json_path),
                    "--md-out",
                    str(md_path),
                ]
            )
            self.assertEqual(rc, 0)

            json_text = json_path.read_text(encoding="utf-8")
            md_text = md_path.read_text(encoding="utf-8")
            parsed = json.loads(json_text)
            self.assertEqual(parsed["meta"]["draws"], 500)
            self.assertEqual(parsed["meta"]["seed"], 42)

            for uid in secret_ids:
                self.assertNotIn(uid, json_text)
                self.assertNotIn(uid, md_text)

    def test_pre_inconsistency_across_models_raises_without_unit_id(self):
        secret_uid = "SECRET_UNIT_CROSS_MODEL_99"
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = pathlib.Path(tmpdir) / "bad_cross_model.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(pg.REQUIRED_COLUMNS)
                writer.writerow([secret_uid, "base", "easy", 0.5, 8, 16, 0])
                writer.writerow([secret_uid, "a_group", "easy", 0.75, 12, 16, 0])

            with self.assertRaises(ValueError) as ctx:
                pg.load_csv(csv_path)
            msg = str(ctx.exception)
            self.assertNotIn(secret_uid, msg)
            self.assertIn("easy", msg)
            self.assertIn("1", msg)
            self.assertIn("3", msg)

    def test_pre_inconsistency_within_same_model_raises_without_unit_id(self):
        secret_uid = "SECRET_UNIT_SAME_MODEL_77"
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = pathlib.Path(tmpdir) / "bad_same_model.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(pg.REQUIRED_COLUMNS)
                writer.writerow([secret_uid, "base", "hard", 0.25, 4, 16, 0])
                writer.writerow([secret_uid, "base", "hard", 0.5, 8, 16, 0])
                writer.writerow([secret_uid, "a_group", "hard", 0.25, 10, 16, 0])

            with self.assertRaises(ValueError) as ctx:
                pg.load_csv(csv_path)
            msg = str(ctx.exception)
            self.assertNotIn(secret_uid, msg)
            self.assertIn("hard", msg)
            self.assertIn("3", msg)

        # Also verify direct evaluate_dataset call without load_csv
        rows = [
            {"unit_id": secret_uid, "condition": "base", "group": "hard", "pre": 0.25, "post_correct": 4, "post_total": 16, "post_unknown": 0},
            {"unit_id": secret_uid, "condition": "base", "group": "hard", "pre": 0.50, "post_correct": 8, "post_total": 16, "post_unknown": 0},
            {"unit_id": secret_uid, "condition": "a_group", "group": "hard", "pre": 0.25, "post_correct": 10, "post_total": 16, "post_unknown": 0},
        ]
        with self.assertRaises(ValueError) as ctx2:
            pg.evaluate_dataset(rows, draws=100, seed=42)
        self.assertNotIn(secret_uid, str(ctx2.exception))

    def test_pre_difference_within_1e_12_allowed(self):
        rows = [
            {"unit_id": "u1", "condition": "base", "group": "orig", "pre": 0.5, "post_correct": 8, "post_total": 16, "post_unknown": 0},
            {"unit_id": "u1", "condition": "base", "group": "orig", "pre": 0.5 + 1e-12, "post_correct": 10, "post_total": 16, "post_unknown": 0},
            {"unit_id": "u1", "condition": "a_group", "group": "orig", "pre": 0.5 - 1e-12, "post_correct": 12, "post_total": 16, "post_unknown": 0},
        ]
        report = pg.evaluate_dataset(rows, draws=100, seed=42)
        self.assertEqual(report["results"][0]["n_paired"], 1)

    def test_source_gain_py_env_missing_file_fails_with_env_name(self):
        missing_path = str(pathlib.Path(tempfile.gettempdir()) / "non_existent_source_gain_12345.py")
        with mock.patch.dict(os.environ, {"SOURCE_GAIN_PY": missing_path}):
            with self.assertRaises(AssertionError) as ctx:
                _load_source_gain()
            self.assertIn("SOURCE_GAIN_PY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
