"""
Focused verification tests for Phase 8.3.1 Integration Contract Audit:
1. Confirms Phase 8.3.1 audit deliverables exist.
2. Verifies current non-existence of serialized M0/M1/M2A/M3 artifacts in ml/models/saved_models/.
3. Verifies ml/src/models/predictor.py currently rejects >22 features (confirming multi-schema update is NOT YET IMPLEMENTED).
4. Verifies exact M1 nominal mapper + OneHotEncoder behavior (neat->missing, unknown->other, empty->other).
5. Verifies RDKit SMILES/scaffold edge cases (invalid SMILES, BigSMILES, empty Murcko, wildcard *, disconnected . salts).
6. Verifies Phase 8.1 extract_m2 vs Phase 8.2 extract_m3 reg_ann discrepancy and Phase 8.1 90% conformal quantiles.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.abspath('.'))

from ml.src.features.descriptors import FEATURE_NAMES
from ml.src.models.predictor import validate_scientific_artifact
from ml.experiments.phase_8_1_processing_aware_models import (
    FeaturePipeline as P81Pipeline,
    map_solvent_nominal as p81_map_solvent,
    map_additive_nominal as p81_map_additive,
)
from ml.experiments.phase_8_2_scaffold_benchmark import (
    FeaturePipeline as P82Pipeline,
)


class TestPhase831ContractVerification(unittest.TestCase):

    def test_01_phase_8_3_1_deliverables_present(self):
        for p in [
            'ml/PHASE_8_3_1_INTEGRATION_AUDIT.md',
            'ml/PHASE_8_3_1_FINAL_CONTRACT_CORRECTIONS.md',
        ]:
            self.assertTrue(os.path.exists(p), f"Missing deliverable: {p}")
            self.assertGreater(os.path.getsize(p), 1000)

    def test_02_saved_models_not_yet_implemented(self):
        models_dir = Path('ml/models/saved_models')
        self.assertTrue(models_dir.exists())
        artifacts = [
            p for p in models_dir.glob('*')
            if p.is_file() and p.suffix in {'.joblib', '.pkl'}
        ]
        self.assertEqual(len(artifacts), 0, "No loose pre-Phase-9 binary model artifacts should exist at top level")

        # Confirm predictor.py currently rejects 50-feature M1 model
        rf_50 = RandomForestRegressor(n_estimators=5, random_state=8888, n_jobs=1)
        rf_50.fit(np.zeros((3, 50)), np.array([1.0, 2.0, 3.0]))
        meta = {
            "is_scientifically_validated": True,
            "training_provenance": "OPV-DB v1.1.0 experimental literature dataset",
            "dataset_source": "opv_devices_strict_molecular_benchmark.csv",
            "evaluation_metrics": {"r2": 0.64, "mae": 2.03, "rmse": 2.78},
            "features": list(FEATURE_NAMES),
        }
        is_valid, errs = validate_scientific_artifact(rf_50, meta)
        self.assertFalse(is_valid)
        self.assertTrue(any("50" in e and "22" in e for e in errs))

    def test_03_m1_exact_categorical_pipeline_semantics(self):
        # Verify exact documented behavior of map_*_nominal
        self.assertEqual(p81_map_solvent('chloroform'), 'chloroform')
        self.assertEqual(p81_map_solvent(np.nan), 'missing')
        self.assertEqual(p81_map_solvent('brand_new_exotic_solvent'), 'other')
        self.assertEqual(p81_map_solvent(''), 'other')  # empty string is NOT pd.isna -> collapses to other

        # Additive conflation: 'neat'/'none' map to 'missing'
        self.assertEqual(p81_map_additive('neat'), 'missing')
        self.assertEqual(p81_map_additive('none'), 'missing')
        self.assertEqual(p81_map_additive(np.nan), 'missing')

    def test_04_scaffold_ood_edge_cases(self):
        # 1. Invalid SMILES & BigSMILES -> None
        self.assertIsNone(Chem.MolFromSmiles('NOT_A_VALID_SMILES'))
        self.assertIsNone(Chem.MolFromSmiles('{[]CC(C)[]}'))

        # 2. Empty Murcko scaffold on acyclic molecule -> pseudo-scaffold
        acyclic = Chem.MolFromSmiles('C=C(CC(=O)O)C(=O)O')
        self.assertIsNotNone(acyclic)
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=acyclic)
        self.assertEqual(scaff, '')
        pseudo = f"[NO_RING]:{Chem.MolToSmiles(acyclic, canonical=True)}"
        self.assertTrue(pseudo.startswith('[NO_RING]:'))

        # 3. Polymer wildcard * and disconnected salt/mixture . detection
        poly_mol = Chem.MolFromSmiles('*/C(C#N)=C1\\C(=O)c2ccccc2C1=C\\c1cccs1')
        self.assertIsNotNone(poly_mol)
        self.assertTrue(any(a.GetAtomicNum() == 0 for a in poly_mol.GetAtoms()))

        salt_smiles = 'N#C[S-].[Cu+]'
        self.assertIn('.', salt_smiles)
        self.assertEqual(len(Chem.GetMolFrags(Chem.MolFromSmiles(salt_smiles))), 2)

    def test_05_phase_8_1_vs_8_2_m3_reg_ann_and_quantiles(self):
        # Confirm Phase 8.1 extract_m2(is_imputed=False) sets regime_thermal_annealed=1 for 110 C,
        # whereas Phase 8.2 extract_m3 silently sets 0 due to NameError on v.
        df_row = pd.DataFrame([
            {
                'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': 110.0,
                'additive_ratio': '0.5', 'solvent_canonical': 'chloroform', 'additive_canonical': 'DIO',
                'device_type': 'inverted', 'htl_canonical': 'MoO3', 'etl_canonical': 'ZnO'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': 25.0,
                'additive_ratio': '0.5', 'solvent_canonical': 'chlorobenzene', 'additive_canonical': 'CN',
                'device_type': 'conventional', 'htl_canonical': 'PEDOT:PSS', 'etl_canonical': 'PFN-Br'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': np.nan,
                'additive_ratio': np.nan, 'solvent_canonical': 'o-DCB', 'additive_canonical': 'other',
                'device_type': 'other', 'htl_canonical': 'other', 'etl_canonical': 'PDINN'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
                'additive_ratio': np.nan, 'solvent_canonical': np.nan, 'additive_canonical': np.nan,
                'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': 'Ca'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': 100.0,
                'additive_ratio': '0.5', 'solvent_canonical': 'toluene', 'additive_canonical': 'other',
                'device_type': 'other', 'htl_canonical': 'other', 'etl_canonical': 'other'
            },
            {
                'donor_smiles': 'c1ccsc1',
                'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
                'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
                'additive_ratio': np.nan, 'solvent_canonical': np.nan, 'additive_canonical': np.nan,
                'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': np.nan
            }
        ])

        p81 = P81Pipeline()
        p81.fit_descriptors(df_row)
        p81.fit_training(df_row)
        X81_m3 = p81.extract_m2(df_row.iloc[[0]], p81.extract_m0(df_row.iloc[[0]]), is_imputed=False)
        self.assertEqual(int(X81_m3.iloc[0]['regime_thermal_annealed']), 1)

        p82 = P82Pipeline()
        p82.fit_descriptors(df_row)
        p82.fit_training(df_row)
        X82_m3 = p82.extract_m3(df_row.iloc[[0]], p82.extract_m0(df_row.iloc[[0]]))
        self.assertEqual(int(X82_m3.iloc[0]['regime_thermal_annealed']), 0)

        # Verify exact Phase 8.1 90% conformal quantiles cited in contract
        p81_int = pd.read_csv('ml/results/phase_8_1_interval_metrics.csv')
        q90 = p81_int[p81_int['confidence_level'] == '90%'].set_index('model_id')
        self.assertAlmostEqual(float(q90.loc['M0', 'conformal_quantile_q']), 4.4836, places=4)
        self.assertAlmostEqual(float(q90.loc['M1', 'conformal_quantile_q']), 4.3244, places=4)
        self.assertAlmostEqual(float(q90.loc['M2A', 'conformal_quantile_q']), 4.3300, places=4)
        self.assertAlmostEqual(float(q90.loc['M3', 'conformal_quantile_q']), 4.2141, places=4)


if __name__ == '__main__':
    unittest.main()
