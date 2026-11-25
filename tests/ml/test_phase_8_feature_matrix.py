"""
Unit tests and automated assertions for Phase 8 feature matrices, schemas,
categorical encodings, and partition disjointness.
"""

import unittest
import os
import sys
import hashlib
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
EXPECTED_SHA256 = 'ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897'

M0_COLUMNS = list(FEATURE_NAMES)

# M1 nominal categories one-hot expanded (23 columns fitted on D_train)
M1_OHE_COLUMNS = [
    'solvent_chlorobenzene', 'solvent_chloroform', 'solvent_missing', 'solvent_o-dcb', 'solvent_other',
    'additive_cn', 'additive_dio', 'additive_missing', 'additive_other',
    'architecture_conventional', 'architecture_inverted', 'architecture_missing', 'architecture_other',
    'htl_missing', 'htl_moo3', 'htl_other', 'htl_pedot_pss',
    'etl_ca', 'etl_missing', 'etl_other', 'etl_pdinn', 'etl_pfn_br', 'etl_zno'
]

M1_COLUMNS = M0_COLUMNS + [
    'da_ratio_numeric',
    'thickness_nm',
    'annealing_temp_c',
    'has_additive',
    'has_thermal_annealing'
] + M1_OHE_COLUMNS

M2_COLUMNS = M0_COLUMNS + [
    # 4 Continuous normalized processing features
    'donor_blend_fraction',
    'thickness_log10',
    'annealing_temp_norm',
    'additive_concentration_pct',
    # 7 Explicit missingness indicators
    'missing_da_ratio',
    'missing_thickness',
    'missing_annealing',
    'missing_solvent',
    'missing_additive',
    'missing_architecture',
    'missing_interlayers',
    # 7 Domain-aware physical regimes
    'regime_as_cast_room_temp',
    'regime_thermal_annealed',
    'regime_solvent_additive_used',
    'regime_chlorinated_solvent',
    'regime_green_solvent',
    'regime_inverted_stack',
    'regime_conventional_stack',
    # 4 Physical interaction / interface constants
    'thickness_per_donor_mw',
    'solvent_boiling_point_c',
    'etl_workfunction_ev',
    'htl_workfunction_ev'
]

M2A_COLUMNS = [c for c in M2_COLUMNS if not c.startswith('missing_')]

M3_COLUMNS = list(M2_COLUMNS)


class TestPhase8FeatureMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert os.path.exists(BENCHMARK_PATH), f"File not found: {BENCHMARK_PATH}"
        with open(BENCHMARK_PATH, "rb") as f:
            cls.file_sha256 = hashlib.sha256(f.read()).hexdigest()
        cls.df = pd.read_csv(BENCHMARK_PATH, low_memory=False)

    def test_01_dataset_checksum_and_pairs(self):
        self.assertEqual(self.file_sha256, EXPECTED_SHA256, "Dataset SHA-256 checksum mismatch!")
        self.assertEqual(len(self.df), 22467, f"Expected 22,467 rows, got {len(self.df)}")
        
        pair_col = self.df['donor_canonical'] + "::" + self.df['acceptor_canonical']
        unique_pairs = pair_col.unique()
        self.assertEqual(len(unique_pairs), 4730, f"Expected 4,730 unique pairs, got {len(unique_pairs)}")

    def test_02_column_counts_and_order(self):
        self.assertEqual(len(M0_COLUMNS), 22, f"M0 count {len(M0_COLUMNS)} != 22")
        self.assertEqual(len(M1_COLUMNS), 50, f"M1 count {len(M1_COLUMNS)} != 50")
        self.assertEqual(len(M2_COLUMNS), 44, f"M2 count {len(M2_COLUMNS)} != 44")
        self.assertEqual(len(M2A_COLUMNS), 37, f"M2A count {len(M2A_COLUMNS)} != 37")
        self.assertEqual(len(M3_COLUMNS), 44, f"M3 count {len(M3_COLUMNS)} != 44")

        # Verify exact subset relationships
        self.assertEqual(M1_COLUMNS[:22], M0_COLUMNS)
        self.assertEqual(M2_COLUMNS[:22], M0_COLUMNS)
        self.assertEqual(M3_COLUMNS, M2_COLUMNS)
        self.assertEqual(len(set(M2_COLUMNS) - set(M2A_COLUMNS)), 7)

    def test_03_partition_disjointness_seed_8888(self):
        pair_col = self.df['donor_canonical'] + "::" + self.df['acceptor_canonical']
        self.df['pair_id'] = pair_col
        unique_pairs = np.sort(self.df['pair_id'].unique())

        rng = np.random.RandomState(8888)
        shuffled = rng.permutation(unique_pairs)

        n_train = int(len(unique_pairs) * 0.6)  # 2838
        n_cal = int(len(unique_pairs) * 0.2)    # 946
        n_test = len(unique_pairs) - n_train - n_cal # 946

        train_pairs = set(shuffled[:n_train])
        cal_pairs = set(shuffled[n_train:n_train + n_cal])
        test_pairs = set(shuffled[n_train + n_cal:])

        self.assertEqual(len(train_pairs), 2838)
        self.assertEqual(len(cal_pairs), 946)
        self.assertEqual(len(test_pairs), 946)

        # Zero pair overlap
        self.assertEqual(len(train_pairs.intersection(cal_pairs)), 0)
        self.assertEqual(len(train_pairs.intersection(test_pairs)), 0)
        self.assertEqual(len(cal_pairs.intersection(test_pairs)), 0)

        # Record counts
        train_records = self.df[self.df['pair_id'].isin(train_pairs)]
        cal_pool = self.df[self.df['pair_id'].isin(cal_pairs)]
        test_pool = self.df[self.df['pair_id'].isin(test_pairs)]

        self.assertEqual(len(train_records), 14704)
        self.assertEqual(len(cal_pool), 3349)
        self.assertEqual(len(test_pool), 4414)

    def test_04_alternative_b_sampling_determinism(self):
        pair_col = self.df['donor_canonical'] + "::" + self.df['acceptor_canonical']
        self.df['pair_id'] = pair_col
        unique_pairs = np.sort(self.df['pair_id'].unique())

        rng = np.random.RandomState(8888)
        shuffled = rng.permutation(unique_pairs)
        n_train = int(len(unique_pairs) * 0.6)
        n_cal = int(len(unique_pairs) * 0.2)

        cal_pairs = sorted(list(shuffled[n_train:n_train + n_cal]))
        test_pairs = sorted(list(shuffled[n_train + n_cal:]))

        def sample_records(pool_df, pairs, seed):
            rng_s = np.random.RandomState(seed)
            sampled = []
            for p in pairs:
                idxs = pool_df[pool_df['pair_id'] == p].index.values
                sampled.append(rng_s.choice(idxs))
            return np.array(sampled)

        cal_idx = sample_records(self.df, cal_pairs, seed=8201)
        test_idx = sample_records(self.df, test_pairs, seed=8202)

        self.assertEqual(len(cal_idx), 946)
        self.assertEqual(len(test_idx), 946)
        self.assertEqual(len(set(cal_idx).intersection(set(test_idx))), 0)

    def test_05_m1_one_hot_encoding_fit_on_train_only(self):
        pair_col = self.df['donor_canonical'] + "::" + self.df['acceptor_canonical']
        self.df['pair_id'] = pair_col
        unique_pairs = np.sort(self.df['pair_id'].unique())
        rng = np.random.RandomState(8888)
        shuffled = rng.permutation(unique_pairs)
        n_train = int(len(unique_pairs) * 0.6)
        train_pairs = set(shuffled[:n_train])
        df_train = self.df[self.df['pair_id'].isin(train_pairs)]

        def map_nominal(df_sub):
            return pd.DataFrame({
                'solvent': df_sub['solvent_canonical'].apply(lambda s: 'chloroform' if 'chloroform' in str(s).lower() else ('chlorobenzene' if 'chlorobenzene' in str(s).lower() and 'di' not in str(s).lower() else ('o-dcb' if 'dcb' in str(s).lower() else ('missing' if pd.isna(s) else 'other')))),
                'additive': df_sub['additive_canonical'].apply(lambda s: 'dio' if 'dio' in str(s).lower() else ('cn' if 'cn' in str(s).lower() else ('missing' if (pd.isna(s) or str(s).lower() in ['none','neat','0','no']) else 'other'))),
                'architecture': df_sub['device_type'].apply(lambda s: 'conventional' if ('conventional' in str(s).lower() or 'regular' in str(s).lower()) else ('inverted' if 'inverted' in str(s).lower() else ('missing' if pd.isna(s) else 'other'))),
                'htl': df_sub['htl_canonical'].apply(lambda s: 'pedot_pss' if 'pedot' in str(s).lower() else ('moo3' if ('moo3' in str(s).lower() or 'mox' in str(s).lower()) else ('missing' if pd.isna(s) else 'other'))),
                'etl': df_sub['etl_canonical'].apply(lambda s: 'zno' if 'zno' in str(s).lower() else ('pdinn' if 'pdinn' in str(s).lower() else ('ca' if (str(s).lower() == 'ca' or 'calcium' in str(s).lower()) else ('pfn_br' if 'pfn' in str(s).lower() else ('missing' if pd.isna(s) else 'other'))))),
            })

        cats_train = map_nominal(df_train)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(cats_train)
        names = list(ohe.get_feature_names_out())
        self.assertEqual(len(names), 23)
        self.assertEqual(names, M1_OHE_COLUMNS)


if __name__ == '__main__':
    unittest.main()
