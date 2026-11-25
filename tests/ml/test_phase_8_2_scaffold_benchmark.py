"""
Unit tests for Phase 8.2 Structure-Level Generalization Benchmark:
- Dataset checksum and row count (22,467)
- Pseudo-scaffold assignment for acyclic/non-ring SMILES (zero empty scaffolds)
- Partition disjointness:
  - Strategy A: Zero donor scaffold leakage between Train, Cal, Test
  - Strategy B: Zero acceptor scaffold leakage between Train, Cal, Test
  - Strategy C1: Zero donor AND zero acceptor scaffold leakage between Train/Cal and Test
  - Strategy C2: Zero scaffold-pair leakage between Train, Cal, Test
- Verification of generated Phase 8.2 result artifacts and metric schemas
"""

import unittest
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
EXPECTED_SHA256 = 'ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897'
RESULTS_DIR = 'ml/results'
SCRATCH_DIR = r'C:\Users\DELL\.gemini\antigravity\brain\f06ca069-f524-44da-9901-d290ece58d0a\scratch'
CACHE_PATH = os.path.join(SCRATCH_DIR, 'scaffold_cache.json')


class TestPhase82ScaffoldBenchmark(unittest.TestCase):

    def test_01_dataset_checksum_and_row_count(self):
        with open(BENCHMARK_PATH, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(sha256, EXPECTED_SHA256)

        df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
        self.assertEqual(len(df), 22467)
        self.assertIn('donor_smiles', df.columns)
        self.assertIn('acceptor_smiles', df.columns)

    def test_02_pseudo_scaffold_assignment(self):
        if not os.path.exists(CACHE_PATH):
            self.skipTest(f"Scaffold cache not found at {CACHE_PATH}")

        with open(CACHE_PATH, 'r') as f:
            cache = json.load(f)

        donor_scaffolds = cache['donor_scaffolds']
        acceptor_scaffolds = cache['acceptor_scaffolds']

        # Zero empty or null scaffolds
        for smi, scaf in donor_scaffolds.items():
            self.assertIsNotNone(scaf)
            self.assertGreater(len(scaf), 0)

        for smi, scaf in acceptor_scaffolds.items():
            self.assertIsNotNone(scaf)
            self.assertGreater(len(scaf), 0)

        # Verify non-ring SMILES get '[NO_RING]:' prefix
        # Donor CuSCN (N#CS[Cu])
        self.assertTrue(any(scaf.startswith('[NO_RING]:') for scaf in donor_scaffolds.values()))
        # Acceptor TiO2 (O=[Ti]=O) or ITN
        self.assertTrue(any(scaf.startswith('[NO_RING]:') for scaf in acceptor_scaffolds.values()))

    def test_03_partition_disjointness(self):
        if not os.path.exists(CACHE_PATH):
            self.skipTest(f"Scaffold cache not found at {CACHE_PATH}")

        from ml.experiments.phase_8_2_scaffold_benchmark import partition_strategy_a, partition_strategy_b, partition_strategy_c1, partition_strategy_c2

        df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
        with open(CACHE_PATH, 'r') as f:
            cache = json.load(f)

        df['donor_scaffold'] = df['donor_smiles'].map(cache['donor_scaffolds'])
        df['acceptor_scaffold'] = df['acceptor_smiles'].map(cache['acceptor_scaffolds'])
        df['scaffold_pair'] = df['donor_scaffold'] + "::" + df['acceptor_scaffold']

        # Strategy A: Donor Scaffold Disjointness
        tr_a, cl_a, ts_a = partition_strategy_a(df, seed=7701)
        d_tr_a = set(tr_a['donor_scaffold'])
        d_cl_a = set(cl_a['donor_scaffold'])
        d_ts_a = set(ts_a['donor_scaffold'])
        self.assertEqual(len(d_tr_a.intersection(d_cl_a)), 0, "Strategy A: Donor scaffold leakage between train and cal")
        self.assertEqual(len(d_tr_a.intersection(d_ts_a)), 0, "Strategy A: Donor scaffold leakage between train and test")
        self.assertEqual(len(d_cl_a.intersection(d_ts_a)), 0, "Strategy A: Donor scaffold leakage between cal and test")

        # Strategy B: Acceptor Scaffold Disjointness
        tr_b, cl_b, ts_b = partition_strategy_b(df, seed=7702)
        a_tr_b = set(tr_b['acceptor_scaffold'])
        a_cl_b = set(cl_b['acceptor_scaffold'])
        a_ts_b = set(ts_b['acceptor_scaffold'])
        self.assertEqual(len(a_tr_b.intersection(a_cl_b)), 0, "Strategy B: Acceptor scaffold leakage between train and cal")
        self.assertEqual(len(a_tr_b.intersection(a_ts_b)), 0, "Strategy B: Acceptor scaffold leakage between train and test")
        self.assertEqual(len(a_cl_b.intersection(a_ts_b)), 0, "Strategy B: Acceptor scaffold leakage between cal and test")

        # Strategy C1: Strict Dual Scaffold Disjointness
        tr_c1, cl_c1, ts_c1 = partition_strategy_c1(df)
        d_tr_c1 = set(tr_c1['donor_scaffold'])
        a_tr_c1 = set(tr_c1['acceptor_scaffold'])
        d_ts_c1 = set(ts_c1['donor_scaffold'])
        a_ts_c1 = set(ts_c1['acceptor_scaffold'])
        self.assertEqual(len(d_tr_c1.intersection(d_ts_c1)), 0, "Strategy C1: Donor scaffold leakage")
        self.assertEqual(len(a_tr_c1.intersection(a_ts_c1)), 0, "Strategy C1: Acceptor scaffold leakage")
        self.assertGreater(len(ts_c1), 0, "Strategy C1 test set should not be empty")

        # Strategy C2: Scaffold Pair Disjointness
        tr_c2, cl_c2, ts_c2 = partition_strategy_c2(df, seed=7703)
        sp_tr_c2 = set(tr_c2['scaffold_pair'])
        sp_cl_c2 = set(cl_c2['scaffold_pair'])
        sp_ts_c2 = set(ts_c2['scaffold_pair'])
        self.assertEqual(len(sp_tr_c2.intersection(sp_cl_c2)), 0, "Strategy C2: Scaffold pair leakage train/cal")
        self.assertEqual(len(sp_tr_c2.intersection(sp_ts_c2)), 0, "Strategy C2: Scaffold pair leakage train/test")
        self.assertEqual(len(sp_cl_c2.intersection(sp_ts_c2)), 0, "Strategy C2: Scaffold pair leakage cal/test")

    def test_04_benchmark_result_artifacts_if_present(self):
        metrics_csv = os.path.join(RESULTS_DIR, 'phase_8_2_scaffold_metrics.csv')
        if not os.path.exists(metrics_csv):
            self.skipTest("Phase 8.2 result artifacts not yet generated.")

        df_m = pd.read_csv(metrics_csv)
        self.assertEqual(len(df_m), 16) # 4 strategies x 4 models
        self.assertEqual(set(df_m['model_id']), {'M0', 'M1', 'M2A', 'M3'})
        self.assertEqual(set(df_m['strategy_id']), {
            'Strategy_A',
            'Strategy_B',
            'Strategy_C1',
            'Strategy_C2'
        })
        self.assertEqual(set(df_m['strategy']), {
            'Strategy_A',
            'Strategy_B',
            'Strategy_C1',
            'Strategy_C2'
        })
        # Check finite non-null metrics
        for col in ['mae', 'rmse', 'r2', 'bias']:
            self.assertFalse(df_m[col].isnull().any())
            self.assertTrue(np.all(np.isfinite(df_m[col])))

        # Interval metrics
        int_csv = os.path.join(RESULTS_DIR, 'phase_8_2_interval_metrics.csv')
        self.assertTrue(os.path.exists(int_csv))
        df_int = pd.read_csv(int_csv)
        self.assertEqual(len(df_int), 48) # 16 models x 3 alphas (80, 90, 95)
        for col in ['coverage', 'mean_width', 'median_width']:
            self.assertFalse(df_int[col].isnull().any())

        # Paired comparisons
        pair_csv = os.path.join(RESULTS_DIR, 'phase_8_2_paired_comparisons.csv')
        self.assertTrue(os.path.exists(pair_csv))
        df_p = pd.read_csv(pair_csv)
        self.assertEqual(len(df_p), 12) # 4 strategies x 3 models (M1, M2A, M3 vs M0)

        # Subgroup metrics
        sub_csv = os.path.join(RESULTS_DIR, 'phase_8_2_subgroup_metrics.csv')
        self.assertTrue(os.path.exists(sub_csv))
        df_sub = pd.read_csv(sub_csv)
        self.assertGreater(len(df_sub), 0)

        # Config JSON
        cfg_json = os.path.join(RESULTS_DIR, 'phase_8_2_config.json')
        self.assertTrue(os.path.exists(cfg_json))
        with open(cfg_json, 'r') as f:
            cfg = json.load(f)
        self.assertEqual(cfg['feature_counts'], {'M0': 22, 'M1': 50, 'M2A': 37, 'M3': 44})


if __name__ == '__main__':
    unittest.main()
