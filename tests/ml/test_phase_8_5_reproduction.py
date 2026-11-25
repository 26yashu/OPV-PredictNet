import json
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TestPhase85Reproduction(unittest.TestCase):
    def test_01_deliverables_and_artifacts_present(self):
        required_docs = [
            "ml/PHASE_8_5_CORRECTED_BENCHMARK_REPORT.md",
            "ml/PHASE_8_5_HISTORICAL_COMPARISON.md",
            "ml/PHASE_8_5_REPRODUCIBILITY_AUDIT.md",
        ]
        for path in required_docs:
            self.assertTrue(os.path.isfile(path), f"Missing Phase 8.5 doc: {path}")

        required_results = [
            "ml/results/phase_8_5/phase_8_5_config.json",
            "ml/results/phase_8_5/phase_8_5_split_metadata.json",
            "ml/results/phase_8_5/phase_8_5_scaffold_metrics.csv",
            "ml/results/phase_8_5/phase_8_5_interval_metrics.csv",
            "ml/results/phase_8_5/phase_8_5_paired_comparisons.csv",
            "ml/results/phase_8_5/phase_8_5_subgroup_metrics.csv",
            "ml/results/phase_8_5/phase_8_5_scaffold_pair_subgroups.csv",
            "ml/results/phase_8_5/phase_8_5_test_predictions.csv",
            "ml/results/phase_8_5/phase_8_5_historical_comparison.csv",
            "ml/results/phase_8_5/phase_8_5_m3_attribution_decomposition.csv",
        ]
        for path in required_results:
            self.assertTrue(os.path.isfile(path), f"Missing Phase 8.5 artifact: {path}")

    def test_02_frozen_partitions_zero_leakage_and_m0_parity(self):
        with open("ml/results/phase_8_2_config.json", "r", encoding="utf-8") as f:
            cfg82 = json.load(f)
        with open("ml/results/phase_8_5/phase_8_5_config.json", "r", encoding="utf-8") as f:
            cfg85 = json.load(f)
        with open("ml/results/phase_8_5/phase_8_5_split_metadata.json", "r", encoding="utf-8") as f:
            split_meta = json.load(f)

        self.assertEqual(cfg85["dataset_sha256"], cfg82["dataset_sha256"])
        self.assertEqual(cfg85["seeds"], cfg82["seeds"])
        self.assertEqual(cfg85["partition_counts"], cfg82["partition_counts"])
        self.assertEqual(cfg85["feature_counts"], {"M0": 22, "M1": 50, "M2A": 37, "M3": 44})
        self.assertNotIn("M2", cfg85["feature_counts"])

        # Zero leakage on held-out structural axes
        for s_id in ["Strategy_A", "Strategy_B", "Strategy_C1", "Strategy_C2"]:
            leak = split_meta["strategies"][s_id]["leakage_audit"]
            self.assertEqual(leak["chem_pair"]["tr_cal_overlap"], 0)
            self.assertEqual(leak["chem_pair"]["tr_test_overlap"], 0)
            self.assertEqual(leak["chem_pair"]["cal_test_overlap"], 0)
            self.assertEqual(leak["scaffold_pair"]["tr_test_overlap"], 0)

        self.assertEqual(split_meta["strategies"]["Strategy_A"]["leakage_audit"]["donor_scaffold"]["tr_test_overlap"], 0)
        self.assertEqual(split_meta["strategies"]["Strategy_B"]["leakage_audit"]["acceptor_scaffold"]["tr_test_overlap"], 0)
        self.assertEqual(split_meta["strategies"]["Strategy_C1"]["leakage_audit"]["donor_scaffold"]["tr_test_overlap"], 0)
        self.assertEqual(split_meta["strategies"]["Strategy_C1"]["leakage_audit"]["acceptor_scaffold"]["tr_test_overlap"], 0)

        # Exact M0 reproduction vs frozen Phase 8.2 CSV
        comp = pd.read_csv("ml/results/phase_8_5/phase_8_5_historical_comparison.csv")
        m0 = comp[comp["model_id"] == "M0"]
        self.assertEqual(len(m0), 4)
        self.assertTrue((m0["delta_mae"] == 0.0).all())
        self.assertTrue((m0["delta_rmse"] == 0.0).all())
        self.assertTrue((m0["delta_r2"] == 0.0).all())
        self.assertTrue((m0["delta_cov_90"] == 0.0).all())

    def test_03_predictions_reproduce_reported_metrics_and_attribution(self):
        preds = pd.read_csv("ml/results/phase_8_5/phase_8_5_test_predictions.csv")
        metrics = pd.read_csv("ml/results/phase_8_5/phase_8_5_scaffold_metrics.csv")
        intervals = pd.read_csv("ml/results/phase_8_5/phase_8_5_interval_metrics.csv")
        paired = pd.read_csv("ml/results/phase_8_5/phase_8_5_paired_comparisons.csv")
        attr = pd.read_csv("ml/results/phase_8_5/phase_8_5_m3_attribution_decomposition.csv")

        self.assertEqual(len(preds), 1481 + 3797 + 284 + 2067)

        for _, row in metrics.iterrows():
            s_id = row["strategy_id"]
            m_id = row["model_id"]
            sub = preds[preds["strategy_id"] == s_id]
            y = sub["pce"].values
            y_hat = sub[f"pred_{m_id}"].values
            q90 = sub[f"q90_{m_id}"].values[0]

            self.assertEqual(len(sub), int(row["n_test"]))
            self.assertAlmostEqual(mean_absolute_error(y, y_hat), row["mae"], places=4)
            self.assertAlmostEqual(np.sqrt(mean_squared_error(y, y_hat)), row["rmse"], places=4)
            self.assertAlmostEqual(r2_score(y, y_hat), row["r2"], places=4)
            self.assertAlmostEqual(float(np.mean(y_hat - y)), row["bias"], places=4)

            int_row = intervals[
                (intervals["strategy_id"] == s_id)
                & (intervals["model_id"] == m_id)
                & (intervals["confidence_level"] == "90%")
            ].iloc[0]
            cov90 = float(np.mean((y >= y_hat - q90) & (y <= y_hat + q90)))
            self.assertAlmostEqual(cov90, int_row["empirical_coverage"], places=4)
            self.assertAlmostEqual(2.0 * q90, int_row["median_width"], places=4)

        # Verify joint crossed dependence is explicitly False across all paired rows
        self.assertEqual(len(paired), 12)
        self.assertFalse(paired["joint_crossed_dependence_tested"].any())

        # Verify thermal annealing fix alone caused 0.0000 delta in M3
        for s_id in ["Strategy_A", "Strategy_B", "Strategy_C1", "Strategy_C2"]:
            v0 = attr[(attr["strategy_id"] == s_id) & (attr["variant"] == "v0_phase_8_2_baseline")].iloc[0]
            v1 = attr[(attr["strategy_id"] == s_id) & (attr["variant"] == "v1_thermal_annealing_fix_only")].iloc[0]
            v3 = attr[(attr["strategy_id"] == s_id) & (attr["variant"] == "v3_lookups_plus_missing_da_fix")].iloc[0]
            v4 = attr[(attr["strategy_id"] == s_id) & (attr["variant"] == "v4_phase_8_5_unified_m3")].iloc[0]
            self.assertEqual(v0["mae"], v1["mae"])
            self.assertEqual(v3["mae"], v4["mae"])

        # Verify zero loose pre-Phase-9 .joblib saved models at top level (phase9_v1 is Phase 9)
        saved_files = [f for f in os.listdir("ml/models/saved_models") if f != "phase9_v1"]
        self.assertEqual(saved_files, ["README.md"])


if __name__ == "__main__":
    unittest.main()
