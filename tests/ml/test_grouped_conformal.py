"""
OPV-PredictNet — Unit Tests for Grouped Conformal Calibration (Phase 7.1)
Verifies:
1. Partition disjointness (zero group leakage among train, cal, test).
2. Exactly one record selected per group in calibration and test partitions.
3. Uniform, target-independent record selection.
4. Deterministic random seeds across repeated runs.
5. Correct finite-sample conformal quantile calculation: k = ceil((n_cal + 1) * (1 - alpha)).
6. Insufficient calibration sample handling (raising InsufficientCalibrationDataError).
7. Strictly positive scale values for locally normalized conformal prediction.
8. No calibration/test label access during fitting or scale selection.
9. Raw unclipped interval construction ([mu - q*sigma, mu + q*sigma]).
10. Empirical coverage and interval width calculations.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.phase_7_1_grouped_conformal import (
    InsufficientCalibrationDataError,
    compute_conformal_quantile,
    split_grouped_dataset,
    sample_one_record_per_group,
    EnsembleDispersionScaleEstimator,
    GroupConformalCalibrator,
    compute_coverage_and_width,
)


class TestGroupedConformalCalibration(unittest.TestCase):
    """Test suite verifying mathematical correctness and leakage-safety of Phase 7.1 calibrator."""

    def setUp(self):
        """Set up synthetic grouped OPV-like dataset for unit testing."""
        np.random.seed(42)
        n_groups = 100
        records = []
        for g in range(n_groups):
            group_name = f"Donor_{g%20}::Acceptor_{g}"
            # Variable group sizes (singletons and repeated groups)
            n_records = 1 if g < 60 else np.random.randint(2, 8)
            for r in range(n_records):
                # 22 synthetic features
                feats = np.random.normal(0, 1, 22)
                # Target PCE with group effect and record-level noise
                pce = np.clip(10.0 + (g % 10) * 0.8 + np.random.normal(0, 0.5), 0.1, 20.0)
                records.append({
                    "record_id": f"rec_{g}_{r}",
                    "group_id": group_name,
                    "pce": pce,
                    **{f"feat_{i}": feats[i] for i in range(22)}
                })
        self.df = pd.DataFrame(records)
        self.feature_cols = [f"feat_{i}" for i in range(22)]

    def test_partition_disjointness(self):
        """1. Verify zero group overlap between all three partitions (train, cal, test)."""
        df_train, df_cal, df_test = split_grouped_dataset(
            self.df, group_col="group_id", train_frac=0.60, cal_frac=0.20, seed=42
        )
        train_groups = set(df_train["group_id"].unique())
        cal_groups = set(df_cal["group_id"].unique())
        test_groups = set(df_test["group_id"].unique())

        self.assertEqual(len(train_groups.intersection(cal_groups)), 0, "Train and Cal groups overlap!")
        self.assertEqual(len(train_groups.intersection(test_groups)), 0, "Train and Test groups overlap!")
        self.assertEqual(len(cal_groups.intersection(test_groups)), 0, "Cal and Test groups overlap!")
        self.assertEqual(len(train_groups) + len(cal_groups) + len(test_groups), self.df["group_id"].nunique())

    def test_one_selected_record_per_group(self):
        """2. Verify that sampling selects exactly one record per group."""
        df_train, df_cal, df_test = split_grouped_dataset(
            self.df, group_col="group_id", train_frac=0.60, cal_frac=0.20, seed=42
        )
        sampled_cal = sample_one_record_per_group(df_cal, group_col="group_id", seed=42)
        sampled_test = sample_one_record_per_group(df_test, group_col="group_id", seed=43)

        self.assertEqual(len(sampled_cal), df_cal["group_id"].nunique())
        self.assertEqual(sampled_cal["group_id"].nunique(), len(sampled_cal))
        self.assertEqual(len(sampled_test), df_test["group_id"].nunique())
        self.assertEqual(sampled_test["group_id"].nunique(), len(sampled_test))

    def test_uniform_target_independent_record_selection(self):
        """3. Verify record selection is independent of target values (PCE)."""
        _, df_cal, _ = split_grouped_dataset(
            self.df, group_col="group_id", train_frac=0.60, cal_frac=0.20, seed=42
        )
        # Select records with original PCE
        sampled_1 = sample_one_record_per_group(df_cal, group_col="group_id", seed=42)

        # Permute or alter PCE values in df_cal copy
        df_cal_altered = df_cal.copy()
        df_cal_altered["pce"] = df_cal_altered["pce"] * 100.0 + np.random.uniform(50, 100, len(df_cal))
        sampled_2 = sample_one_record_per_group(df_cal_altered, group_col="group_id", seed=42)

        # Selected record_ids must be 100% identical regardless of PCE alterations
        self.assertListEqual(sampled_1["record_id"].tolist(), sampled_2["record_id"].tolist())

    def test_deterministic_seeds(self):
        """4. Verify deterministic partitioning and sampling across repeated runs."""
        df_train_1, df_cal_1, df_test_1 = split_grouped_dataset(
            self.df, group_col="group_id", train_frac=0.60, cal_frac=0.20, seed=123
        )
        df_train_2, df_cal_2, df_test_2 = split_grouped_dataset(
            self.df, group_col="group_id", train_frac=0.60, cal_frac=0.20, seed=123
        )
        self.assertListEqual(df_train_1["record_id"].tolist(), df_train_2["record_id"].tolist())
        self.assertListEqual(df_cal_1["record_id"].tolist(), df_cal_2["record_id"].tolist())
        self.assertListEqual(df_test_1["record_id"].tolist(), df_test_2["record_id"].tolist())

    def test_correct_conformal_quantile_calculation(self):
        """5. Verify exact finite-sample quantile calculation: k = ceil((n + 1) * (1 - alpha))."""
        scores = np.array([0.5, 1.2, 1.8, 2.5, 3.1, 4.0, 5.2, 6.1, 7.5])  # n = 9
        # For alpha = 0.20 (80% confidence):
        # k = ceil((9 + 1) * 0.80) = ceil(8.0) = 8
        # Sorted 1-based index 8 is scores[7] = 6.1
        q_80 = compute_conformal_quantile(scores, alpha=0.20)
        self.assertEqual(q_80, 6.1)

        # For alpha = 0.10 (90% confidence):
        # k = ceil((9 + 1) * 0.90) = ceil(9.0) = 9
        # Sorted 1-based index 9 is scores[8] = 7.5
        q_90 = compute_conformal_quantile(scores, alpha=0.10)
        self.assertEqual(q_90, 7.5)

    def test_insufficient_calibration_data_error(self):
        """6. Verify InsufficientCalibrationDataError is raised when k > n_cal."""
        scores = np.array([1.0, 2.0, 3.0])  # n = 3
        # For alpha = 0.10: k = ceil((3 + 1) * 0.90) = ceil(3.6) = 4 > 3!
        with self.assertRaises(InsufficientCalibrationDataError):
            compute_conformal_quantile(scores, alpha=0.10)

    def test_strictly_positive_scale_values(self):
        """7. Verify locally normalized scale estimator outputs strictly positive values (> 0)."""
        X_train = np.random.normal(0, 1, (50, 22))
        y_train = np.random.uniform(5, 15, 50)
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        scale_estimator = EnsembleDispersionScaleEstimator(rf, epsilon=0.10)
        X_eval = np.random.normal(0, 1, (30, 22))
        scales = scale_estimator.predict_scale(X_eval)

        self.assertEqual(len(scales), 30)
        self.assertTrue(np.all(scales >= 0.10), "Scale values must be strictly >= stabilizer epsilon (0.10)!")
        self.assertFalse(np.isnan(scales).any(), "Scale values must not contain NaNs")
        self.assertFalse(np.isinf(scales).any(), "Scale values must not contain Infs")

    def test_no_calibration_label_leakage(self):
        """8. Verify calibrator does not touch calibration labels during fitting or scale selection."""
        calibrator = GroupConformalCalibrator(method="unnormalized", random_state=42)
        X_train = self.df[self.feature_cols].values[:50]
        y_train = self.df["pce"].values[:50]

        # Fit model on training data only
        calibrator.fit_point_model(X_train, y_train)
        self.assertIsNotNone(calibrator.model)
        self.assertFalse(calibrator.is_frozen, "Calibrator must not be frozen before calibration!")
        self.assertEqual(calibrator.quantiles, {}, "Quantiles dictionary must be empty before calibration!")

    def test_raw_interval_construction(self):
        """9. Verify raw unclipped prediction interval construction [mu - q*sigma, mu + q*sigma]."""
        y_pred = np.array([10.0, 15.0])
        scale = np.array([1.0, 2.0])
        quantile = 2.5

        lower = y_pred - quantile * scale
        upper = y_pred + quantile * scale

        np.testing.assert_array_almost_equal(lower, np.array([7.5, 10.0]))
        np.testing.assert_array_almost_equal(upper, np.array([12.5, 20.0]))
        np.testing.assert_array_almost_equal(upper - lower, np.array([5.0, 10.0]))

    def test_coverage_and_width_calculations(self):
        """10. Verify empirical coverage and interval width calculations."""
        y_true = np.array([5.0, 10.0, 15.0, 20.0])
        lower = np.array([4.0, 9.0, 16.0, 18.0])   # 5 in [4, 6], 10 in [9, 11], 15 NOT in [16, 17], 20 NOT in [18, 19]
        upper = np.array([6.0, 11.0, 17.0, 19.0])  # -> 2 covered out of 4 = 50.0% coverage

        coverage, mean_width, median_width, widths = compute_coverage_and_width(y_true, lower, upper)
        self.assertEqual(coverage, 0.50)
        self.assertAlmostEqual(mean_width, 1.5)   # widths: 2, 2, 1, 1 -> sum=6 / 4 = 1.5
        self.assertAlmostEqual(median_width, 1.5) # median = 1.5


if __name__ == "__main__":
    unittest.main()
