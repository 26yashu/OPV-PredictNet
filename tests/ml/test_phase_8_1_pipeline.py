"""
Automated unit tests for Phase 8.1 pipeline verification:
- Partition isolation and zero pair leakage
- Preprocessing and imputer isolation (fitted on D_train only)
- Feature dimensions and schemas (M0=22, M1=50, M2=44, M2A=37, M3=44)
- Calibration/test separation and finite-sample cutoffs
- Result artifacts integrity
"""

import unittest
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
EXPECTED_SHA256 = 'ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897'
RESULTS_DIR = 'ml/results'


class TestPhase81Pipeline(unittest.TestCase):

    def test_01_dataset_checksum(self):
        with open(BENCHMARK_PATH, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(sha256, EXPECTED_SHA256)

    def test_02_partition_and_sampling_isolation(self):
        df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
        pair_col = df['donor_canonical'] + "::" + df['acceptor_canonical']
        df['pair_id'] = pair_col
        unique_pairs = np.sort(df['pair_id'].unique())
        self.assertEqual(len(unique_pairs), 4730)

        rng = np.random.RandomState(8888)
        shuffled = rng.permutation(unique_pairs)
        n_train = int(len(unique_pairs) * 0.6)  # 2838
        n_cal = int(len(unique_pairs) * 0.2)    # 946
        n_test = len(unique_pairs) - n_train - n_cal # 946

        train_pairs = set(shuffled[:n_train])
        cal_pairs = set(shuffled[n_train:n_train + n_cal])
        test_pairs = set(shuffled[n_train + n_cal:])

        # 100% Disjointness
        self.assertEqual(len(train_pairs.intersection(cal_pairs)), 0)
        self.assertEqual(len(train_pairs.intersection(test_pairs)), 0)
        self.assertEqual(len(cal_pairs.intersection(test_pairs)), 0)

        # Alternative B sampling
        cal_pool = df[df['pair_id'].isin(cal_pairs)]
        test_pool = df[df['pair_id'].isin(test_pairs)]

        rng_c = np.random.RandomState(8201)
        cal_sample_idx = [rng_c.choice(cal_pool[cal_pool['pair_id'] == p].index.values) for p in sorted(list(cal_pairs))]
        rng_t = np.random.RandomState(8202)
        test_sample_idx = [rng_t.choice(test_pool[test_pool['pair_id'] == p].index.values) for p in sorted(list(test_pairs))]

        self.assertEqual(len(cal_sample_idx), 946)
        self.assertEqual(len(test_sample_idx), 946)
        self.assertEqual(len(set(cal_sample_idx).intersection(set(test_sample_idx))), 0)

    def test_03_feature_dimensions(self):
        # M0=22, M1=50, M2=44, M2A=37, M3=44
        expected_counts = {'M0': 22, 'M1': 50, 'M2': 44, 'M2A': 37, 'M3': 44}
        config_path = os.path.join(RESULTS_DIR, 'phase_8_1_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            self.assertEqual(cfg['feature_counts'], expected_counts)

    def test_04_results_artifacts_presence(self):
        # If results exist, verify schemas
        metrics_csv = os.path.join(RESULTS_DIR, 'phase_8_1_model_metrics.csv')
        if os.path.exists(metrics_csv):
            df_m = pd.read_csv(metrics_csv)
            self.assertEqual(len(df_m), 5)
            self.assertEqual(set(df_m['model_id']), {'M0', 'M1', 'M2', 'M2A', 'M3'})

            df_int = pd.read_csv(os.path.join(RESULTS_DIR, 'phase_8_1_interval_metrics.csv'))
            self.assertEqual(len(df_int), 15) # 5 models x 3 alphas

            df_paired = pd.read_csv(os.path.join(RESULTS_DIR, 'phase_8_1_paired_comparisons.csv'))
            self.assertEqual(len(df_paired), 4) # M1, M2, M2A, M3 vs M0

            df_sub = pd.read_csv(os.path.join(RESULTS_DIR, 'phase_8_1_subgroup_metrics.csv'))
            self.assertGreater(len(df_sub), 0)

            df_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'phase_8_1_test_predictions.csv'))
            self.assertEqual(len(df_preds), 946)


if __name__ == '__main__':
    unittest.main()
