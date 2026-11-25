"""
Unit tests verifying Phase 8.3 ML Integration Contract invariants and artifact consistency:
1. Presence of all five Phase 8.3 governance, selection, contract, limitations, and readiness deliverables.
2. Exact feature matrix schemas and ordering (M0=22, M1=50, M2A=37, M3=44) matching the locked contract.
3. Missing-value and unknown-category behavior across M1, M2A, and M3 (zero NaNs in M0/M1/M2A; controlled NaNs in M3).
4. Integration safety bounds and artifact provenance rejection rules.
5. Numerical consistency with frozen Phase 5, 6.1, 7.1, 8.1, and 8.2 result artifacts.
"""

import json
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.abspath('.'))

from ml.src.features.descriptors import FEATURE_NAMES
from ml.src.evaluation.sanity import validate_prediction_sanity
from ml.src.models.predictor import validate_scientific_artifact
from ml.experiments.phase_8_2_scaffold_benchmark import FeaturePipeline, M1_OHE_COLUMNS


EXPECTED_M1_COLS = (
    list(FEATURE_NAMES)
    + ['da_ratio_numeric', 'thickness_nm', 'annealing_temp_c', 'has_additive', 'has_thermal_annealing']
    + M1_OHE_COLUMNS
)

EXPECTED_M2A_COLS = list(FEATURE_NAMES) + [
    'donor_blend_fraction', 'thickness_log10', 'annealing_temp_norm', 'additive_concentration_pct',
    'regime_as_cast_room_temp', 'regime_thermal_annealed', 'regime_solvent_additive_used',
    'regime_chlorinated_solvent', 'regime_green_solvent', 'regime_inverted_stack', 'regime_conventional_stack',
    'thickness_per_donor_mw', 'solvent_boiling_point_c', 'etl_workfunction_ev', 'htl_workfunction_ev'
]

EXPECTED_M3_COLS = list(FEATURE_NAMES) + [
    'donor_blend_fraction', 'thickness_log10', 'annealing_temp_norm', 'additive_concentration_pct',
    'missing_da_ratio', 'missing_thickness', 'missing_annealing', 'missing_solvent',
    'missing_additive', 'missing_architecture', 'missing_interlayers',
    'regime_as_cast_room_temp', 'regime_thermal_annealed', 'regime_solvent_additive_used',
    'regime_chlorinated_solvent', 'regime_green_solvent', 'regime_inverted_stack', 'regime_conventional_stack',
    'thickness_per_donor_mw', 'solvent_boiling_point_c', 'etl_workfunction_ev', 'htl_workfunction_ev'
]

M3_NAN_ALLOWED_COLS = {
    'donor_blend_fraction', 'thickness_log10', 'annealing_temp_norm', 'additive_concentration_pct',
    'thickness_per_donor_mw', 'solvent_boiling_point_c', 'etl_workfunction_ev', 'htl_workfunction_ev'
}


class TestPhase83IntegrationContract(unittest.TestCase):

    def test_01_deliverables_present(self):
        required_docs = [
            'ml/PHASE_8_3_FINAL_MODEL_SELECTION.md',
            'ml/PHASE_8_3_MODEL_GOVERNANCE.md',
            'ml/PHASE_8_3_INTEGRATION_CONTRACT.md',
            'ml/PHASE_8_3_LIMITATIONS_REGISTRY.md',
            'ml/PHASE_8_3_FINAL_READINESS_REPORT.md',
        ]
        for path in required_docs:
            self.assertTrue(os.path.exists(path), f"Missing Phase 8.3 deliverable: {path}")
            self.assertGreater(os.path.getsize(path), 1000, f"Deliverable unexpectedly small: {path}")

    def test_02_locked_feature_schemas_and_missingness_contract(self):
        # Minimal deterministic training fixture covering standard vocabularies + missing values
        df_train = pd.DataFrame([
            {
                'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1.2', 'active_layer_thickness': '100', 'annealing_temp': 110.0,
                'additive_ratio': '0.5', 'solvent_canonical': 'chloroform', 'additive_canonical': 'DIO',
                'device_type': 'inverted', 'htl_canonical': 'MoO3', 'etl_canonical': 'ZnO'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '120', 'annealing_temp': 25.0,
                'additive_ratio': np.nan, 'solvent_canonical': 'chlorobenzene', 'additive_canonical': 'CN',
                'device_type': 'conventional', 'htl_canonical': 'PEDOT:PSS', 'etl_canonical': 'PFN-Br'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
                'additive_ratio': np.nan, 'solvent_canonical': 'o-DCB', 'additive_canonical': 'other_add',
                'device_type': 'other_arch', 'htl_canonical': 'other_htl', 'etl_canonical': 'PDINN'
            },
            {
                'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:2', 'active_layer_thickness': '90', 'annealing_temp': 100.0,
                'additive_ratio': '1.0', 'solvent_canonical': 'toluene', 'additive_canonical': np.nan,
                'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': 'Ca'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
                'additive_ratio': np.nan, 'solvent_canonical': np.nan, 'additive_canonical': np.nan,
                'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': np.nan
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': 100.0,
                'additive_ratio': '0.5', 'solvent_canonical': 'toluene', 'additive_canonical': 'other_add',
                'device_type': 'other_arch', 'htl_canonical': 'other_htl', 'etl_canonical': 'other_etl'
            },
        ])

        # Query rows: 1 fully missing, 1 unseen/exotic categories
        df_query = pd.DataFrame([
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
                'additive_ratio': np.nan, 'solvent_canonical': np.nan, 'additive_canonical': np.nan,
                'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': np.nan
            },
            {
                'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': 100.0,
                'additive_ratio': '0.5', 'solvent_canonical': 'novel_exotic_solvent',
                'additive_canonical': 'novel_exotic_additive', 'device_type': 'tandem_unknown',
                'htl_canonical': 'novel_htl', 'etl_canonical': 'novel_etl'
            }
        ])

        pipe = FeaturePipeline()
        pipe.fit_descriptors(pd.concat([df_train, df_query], ignore_index=True))
        pipe.fit_training(df_train)

        X_m0 = pipe.extract_m0(df_query)
        X_m1 = pipe.extract_m1(df_query, X_m0)
        X_m2a = pipe.extract_m2a(df_query, X_m0)
        X_m3 = pipe.extract_m3(df_query, X_m0)

        # 1. Exact dimensions and column ordering
        self.assertEqual(X_m0.shape, (2, 22))
        self.assertEqual(list(X_m0.columns), list(FEATURE_NAMES))

        self.assertEqual(X_m1.shape, (2, 50))
        self.assertEqual(list(X_m1.columns), EXPECTED_M1_COLS)

        self.assertEqual(X_m2a.shape, (2, 37))
        self.assertEqual(list(X_m2a.columns), EXPECTED_M2A_COLS)

        self.assertEqual(X_m3.shape, (2, 44))
        self.assertEqual(list(X_m3.columns), EXPECTED_M3_COLS)

        # 2. Strict finiteness on M0, M1, M2A
        self.assertFalse(X_m0.isna().any().any())
        self.assertFalse(X_m1.isna().any().any())
        self.assertFalse(X_m2a.isna().any().any())

        # 3. M1 unknown category routing: row 0 -> missing, row 1 -> other
        self.assertEqual(X_m1.iloc[0]['solvent_missing'], 1.0)
        self.assertEqual(X_m1.iloc[1]['solvent_other'], 1.0)
        self.assertEqual(X_m1.iloc[1]['etl_other'], 1.0)

        # 4. Controlled NaN & binary checks on M3
        for col in X_m3.columns:
            if col in M3_NAN_ALLOWED_COLS:
                self.assertTrue(pd.isna(X_m3.iloc[0][col]), f"Row 0 {col} should be NaN in M3")
            else:
                self.assertFalse(X_m3[col].isna().any(), f"Column {col} must not contain NaN in M3")

            if col.startswith('missing_') or col.startswith('regime_'):
                self.assertTrue(set(X_m3[col].unique()).issubset({0, 1}), f"{col} must be binary")

        # Row 0 has all 7 missingness flags = 1 and all 7 regimes = 0
        for col in [c for c in X_m3.columns if c.startswith('missing_')]:
            self.assertEqual(X_m3.iloc[0][col], 1)
        for col in [c for c in X_m3.columns if c.startswith('regime_')]:
            self.assertEqual(X_m3.iloc[0][col], 0)

    def test_03_safety_and_artifact_validation_guards(self):
        # Unphysical output checks
        sane_ok, _ = validate_prediction_sanity(pce=14.2)
        self.assertTrue(sane_ok)

        sane_high, errs_high = validate_prediction_sanity(pce=36.0)
        self.assertFalse(sane_high)
        self.assertGreater(len(errs_high), 0)

        sane_neg, _ = validate_prediction_sanity(pce=-0.1)
        self.assertFalse(sane_neg)

        # Artifact fixture rejection
        rf = RandomForestRegressor(n_estimators=10, n_jobs=1)
        bad_meta = {
            "is_scientifically_validated": True,
            "training_provenance": "Synthetic benchmark dummy",
            "dataset_source": "opv_devices_strict_molecular_benchmark.csv",
            "evaluation_metrics": {"r2": 0.65, "mae": 2.05, "rmse": 2.75},
            "features": list(FEATURE_NAMES),
        }
        is_valid, errs = validate_scientific_artifact(rf, bad_meta)
        self.assertFalse(is_valid)
        self.assertTrue(any("synthetic" in e.lower() or "dummy" in e.lower() for e in errs))

    def test_04_frozen_source_of_truth_artifacts_reconciled(self):
        # Verify exact Phase 8.1 and Phase 8.2 numbers referenced in Phase 8.3 exist intact
        p81_m = pd.read_csv('ml/results/phase_8_1_model_metrics.csv').set_index('model_id')
        self.assertAlmostEqual(float(p81_m.loc['M0', 'test_mae_pce']), 2.1727, places=4)
        self.assertAlmostEqual(float(p81_m.loc['M1', 'test_mae_pce']), 2.0321, places=4)
        self.assertAlmostEqual(float(p81_m.loc['M2A', 'test_mae_pce']), 2.0689, places=4)
        self.assertAlmostEqual(float(p81_m.loc['M3', 'test_mae_pce']), 2.0540, places=4)
        self.assertAlmostEqual(float(p81_m.loc['M3', 'test_r2']), 0.6502, places=4)

        p82_m = pd.read_csv('ml/results/phase_8_2_scaffold_metrics.csv')
        c1_m0 = p82_m[(p82_m['strategy_id'] == 'Strategy_C1') & (p82_m['model_id'] == 'M0')].iloc[0]
        c1_m3 = p82_m[(p82_m['strategy_id'] == 'Strategy_C1') & (p82_m['model_id'] == 'M3')].iloc[0]
        self.assertAlmostEqual(float(c1_m0['r2']), 0.0786, places=4)
        self.assertAlmostEqual(float(c1_m3['r2']), 0.2385, places=4)

        with open('ml/results/phase_5_summary.json', 'r') as f:
            p5 = json.load(f)
        self.assertAlmostEqual(p5['splits_summary']['Pair_Grouped_5Fold']['RandomForestRegressor']['mae_mean'], 2.0628, places=4)


if __name__ == '__main__':
    unittest.main()
