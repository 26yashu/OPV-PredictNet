"""
Phase 8.4 Regression & Reproducibility Tests for UnifiedFeaturePipeline:
1. Exact schema ordering, dimensions (M0=22, M1=50, M2=44, M2A=37, M3=44), and types.
2. M3 thermal annealing regression fix (numeric vs missing vs Phase 8.2 defect non-retention).
3. Bit-for-bit agreement between Phase 8.1 FeaturePipeline and UnifiedFeaturePipeline on equivalent inputs.
4. State serialization (to_state_dict / from_state_dict) deterministic reproducibility.
5. Additive semantics audit (locked M1 conflation preserved vs v2 absent_explicit specification).
6. Edge-case & validation coverage: unknown categories, missing metadata, invalid SMILES,
   non-ring [NO_RING]: scaffolds, wildcard (*) & mixture (.) flags/rejection, and unphysical rejections.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from ml.src.features.processing import (
    M0_FEATURE_NAMES,
    M1_FEATURE_NAMES,
    M2_FEATURE_NAMES,
    M2A_FEATURE_NAMES,
    M3_FEATURE_NAMES,
    M3_NAN_ALLOWED_COLUMNS,
    UnifiedFeaturePipeline,
    MissingMolecularIdentityError,
    InvalidMolecularInputError,
    UnsupportedPolymerOrMixtureError,
    PhysicalInputValidationError,
    map_additive_nominal,
    map_additive_nominal_v2_spec,
    validate_and_parse_smiles,
    validate_processing_record,
)
from ml.experiments.phase_8_1_processing_aware_models import FeaturePipeline as P81Pipeline
from ml.experiments.phase_8_2_scaffold_benchmark import FeaturePipeline as P82Pipeline


def _make_training_fixture() -> pd.DataFrame:
    """Fixture covering all 23 M1 OHE columns and canonical lookup materials."""
    return pd.DataFrame([
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
            'd_a_ratio': '1:1.5', 'active_layer_thickness': '90', 'annealing_temp': np.nan,
            'additive_ratio': np.nan, 'solvent_canonical': 'o-DCB', 'additive_canonical': 'other_add',
            'device_type': 'other_arch', 'htl_canonical': '2PACz', 'etl_canonical': 'PDINN'
        },
        {
            'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': '1:2', 'active_layer_thickness': '80', 'annealing_temp': 150.0,
            'additive_ratio': '1.0', 'solvent_canonical': 'o-xylene', 'additive_canonical': 'neat',
            'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': 'Ca'
        },
        {
            'donor_smiles': 'c1ccsc1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
            'additive_ratio': np.nan, 'solvent_canonical': np.nan, 'additive_canonical': np.nan,
            'device_type': np.nan, 'htl_canonical': 'other_htl', 'etl_canonical': 'ZrAcac'
        },
        {
            'donor_smiles': 'c1ccsc1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
            'additive_ratio': np.nan, 'solvent_canonical': 'toluene', 'additive_canonical': np.nan,
            'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': np.nan
        },
    ])


def _make_representative_queries() -> pd.DataFrame:
    """Representative test cases covering all 8 required Phase 8.4 reproducibility scenarios."""
    return pd.DataFrame([
        # 0: Familiar structure + thermal annealing 110 C + Phase 8.1 materials (o-xylene, 2PACz, PFN)
        {
            'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': '1:1.2', 'active_layer_thickness': '105nm', 'annealing_temp': 110.0,
            'additive_ratio': '0.5 vol%', 'solvent_canonical': 'o-xylene', 'additive_canonical': 'DIO',
            'device_type': 'inverted', 'htl_canonical': '2PACz', 'etl_canonical': 'PFN'
        },
        # 1: As-cast room temperature 25 C + explicit additive absence ('neat') + ZrAcac (must NOT match Ca)
        {
            'donor_smiles': 'c1ccsc1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': '1:1', 'active_layer_thickness': '95', 'annealing_temp': 25.0,
            'additive_ratio': np.nan, 'solvent_canonical': 'chloroform', 'additive_canonical': 'neat',
            'device_type': 'conventional', 'htl_canonical': 'PEDOT:PSS', 'etl_canonical': 'ZrAcac'
        },
        # 2: Unknown exotic solvent and additive categories + PDINN (4.00 eV)
        {
            'donor_smiles': 'c1ccsc1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': '2:1', 'active_layer_thickness': '110', 'annealing_temp': 140.0,
            'additive_ratio': '2.0', 'solvent_canonical': 'exotic_green_solvent', 'additive_canonical': 'novel_additive',
            'device_type': 'conventional', 'htl_canonical': 'MoO3', 'etl_canonical': 'PDINN'
        },
        # 3: Completely missing processing metadata (k=0)
        {
            'donor_smiles': 'Cc1cc(C)sc1-c1ccc(C)s1',
            'acceptor_smiles': 'O=C1c2ccccc2C(=O)C1=Cc1cccs1',
            'd_a_ratio': np.nan, 'active_layer_thickness': np.nan, 'annealing_temp': np.nan,
            'additive_ratio': np.nan, 'solvent_canonical': np.nan, 'additive_canonical': np.nan,
            'device_type': np.nan, 'htl_canonical': np.nan, 'etl_canonical': np.nan
        },
        # 4: Non-ring donor/acceptor structures (CuSCN salt + itaconic acid)
        {
            'donor_smiles': 'N#C[S-].[Cu+]',
            'acceptor_smiles': 'C=C(CC(=O)O)C(=O)O',
            'd_a_ratio': '1:1', 'active_layer_thickness': '100', 'annealing_temp': 80.0,
            'additive_ratio': np.nan, 'solvent_canonical': 'chlorobenzene', 'additive_canonical': 'none',
            'device_type': 'inverted', 'htl_canonical': 'MoO3', 'etl_canonical': 'ZnO'
        },
    ])


class TestPhase84UnifiedPipeline(unittest.TestCase):

    def test_01_m3_thermal_annealing_regression_and_phase_8_1_parity(self):
        df_tr = _make_training_fixture()
        df_q = _make_representative_queries()

        unified = UnifiedFeaturePipeline()
        unified.fit_descriptors(pd.concat([df_tr, df_q], ignore_index=True))
        unified.fit_training(df_tr)

        p81 = P81Pipeline()
        p81.fit_descriptors(pd.concat([df_tr, df_q], ignore_index=True))
        p81.fit_training(df_tr)

        p82 = P82Pipeline()
        p82.fit_descriptors(pd.concat([df_tr, df_q], ignore_index=True))
        p82.fit_training(df_tr)

        # Extract all models via unified and Phase 8.1
        u_m0 = unified.extract_m0(df_q)
        u_m1 = unified.extract_m1(df_q, u_m0)
        u_m2 = unified.extract_m2(df_q, u_m0)
        u_m2a = unified.extract_m2a(df_q, u_m0)
        u_m3 = unified.extract_m3(df_q, u_m0)

        p81_m0 = p81.extract_m0(df_q)
        p81_m1 = p81.extract_m1(df_q, p81_m0)
        p81_m2 = p81.extract_m2(df_q, p81_m0, is_imputed=True)
        p81_m2a = p81.extract_m2a(p81_m2)
        p81_m3 = p81.extract_m2(df_q, p81_m0, is_imputed=False)

        # 1. Exact agreement with Phase 8.1 across all models and rows
        pd.testing.assert_frame_equal(u_m0, p81_m0)
        pd.testing.assert_frame_equal(u_m1, p81_m1)
        pd.testing.assert_frame_equal(u_m2, p81_m2)
        pd.testing.assert_frame_equal(u_m2a, p81_m2a)
        pd.testing.assert_frame_equal(u_m3, p81_m3)

        # 2. Thermal annealing regression assertions in M3
        # Row 0 (110 C): thermal annealed = 1, as_cast = 0, norm = (110-100)/50 = 0.2, missing = 0
        self.assertEqual(int(u_m3.iloc[0]['regime_thermal_annealed']), 1)
        self.assertEqual(int(u_m3.iloc[0]['regime_as_cast_room_temp']), 0)
        self.assertEqual(int(u_m3.iloc[0]['missing_annealing']), 0)
        self.assertAlmostEqual(float(u_m3.iloc[0]['annealing_temp_norm']), 0.2, places=6)

        # Row 1 (25 C): thermal annealed = 0, as_cast = 1, norm = -1.3, missing = 0
        self.assertEqual(int(u_m3.iloc[1]['regime_thermal_annealed']), 0)
        self.assertEqual(int(u_m3.iloc[1]['regime_as_cast_room_temp']), 1)
        self.assertAlmostEqual(float(u_m3.iloc[1]['annealing_temp_norm']), -1.3, places=6)

        # Row 3 (NaN): thermal annealed = 0, as_cast = 0, missing = 1, norm = NaN
        self.assertEqual(int(u_m3.iloc[3]['regime_thermal_annealed']), 0)
        self.assertEqual(int(u_m3.iloc[3]['regime_as_cast_room_temp']), 0)
        self.assertEqual(int(u_m3.iloc[3]['missing_annealing']), 1)
        self.assertTrue(np.isnan(u_m3.iloc[3]['annealing_temp_norm']))

        # 3. Confirm Phase 8.2 defective extract_m3 is NOT retained (Phase 8.2 produced 0 on 110 C)
        p82_m3 = p82.extract_m3(df_q, p82.extract_m0(df_q))
        self.assertEqual(int(p82_m3.iloc[0]['regime_thermal_annealed']), 0)
        self.assertNotEqual(
            int(u_m3.iloc[0]['regime_thermal_annealed']),
            int(p82_m3.iloc[0]['regime_thermal_annealed']),
        )

        # 4. Confirm ZrAcac in row 1 is mapped to etl_other (NOT etl_ca) and o-xylene/2PACz match Phase 8.1
        self.assertEqual(float(u_m1.iloc[1]['etl_ca']), 0.0)
        self.assertEqual(float(u_m1.iloc[1]['etl_other']), 1.0)
        self.assertAlmostEqual(float(u_m3.iloc[0]['solvent_boiling_point_c']), 144.4, places=4)
        self.assertAlmostEqual(float(u_m3.iloc[0]['htl_workfunction_ev']), 5.20, places=4)
        self.assertAlmostEqual(float(u_m3.iloc[2]['etl_workfunction_ev']), 4.00, places=4)

    def test_02_schema_determinism_and_state_serialization(self):
        df_tr = _make_training_fixture()
        df_q = _make_representative_queries()

        pipe1 = UnifiedFeaturePipeline()
        pipe1.fit_descriptors(pd.concat([df_tr, df_q], ignore_index=True))
        pipe1.fit_training(df_tr)

        state = pipe1.to_state_dict()
        pipe2 = UnifiedFeaturePipeline.from_state_dict(state)

        for model_id, expected_cols in [
            ('M0', M0_FEATURE_NAMES),
            ('M1', M1_FEATURE_NAMES),
            ('M2', M2_FEATURE_NAMES),
            ('M2A', M2A_FEATURE_NAMES),
            ('M3', M3_FEATURE_NAMES),
        ]:
            fn1 = getattr(pipe1, f'extract_{model_id.lower()}')
            fn2 = getattr(pipe2, f'extract_{model_id.lower()}')
            df1 = fn1(df_q)
            df2 = fn2(df_q)
            self.assertEqual(list(df1.columns), list(expected_cols))
            self.assertEqual(df1.shape, (len(df_q), len(expected_cols)))
            pd.testing.assert_frame_equal(df1, df2)

            if model_id != 'M3':
                self.assertFalse(df1.isna().any().any())
            else:
                for col in df1.columns:
                    if col not in M3_NAN_ALLOWED_COLUMNS:
                        self.assertFalse(df1[col].isna().any())

    def test_03_additive_semantics_locked_vs_v2_spec(self):
        # Locked M1 schema preserves Phase 8.1 conflation ('neat' -> 'missing')
        self.assertEqual(map_additive_nominal('neat'), 'missing')
        self.assertEqual(map_additive_nominal('none'), 'missing')
        self.assertEqual(map_additive_nominal(np.nan), 'missing')

        # Proposed v2 separates explicit absence from missing metadata
        self.assertEqual(map_additive_nominal_v2_spec('neat'), 'absent_explicit')
        self.assertEqual(map_additive_nominal_v2_spec('none'), 'absent_explicit')
        self.assertEqual(map_additive_nominal_v2_spec('additive-free'), 'absent_explicit')
        self.assertEqual(map_additive_nominal_v2_spec(np.nan), 'missing')
        self.assertEqual(map_additive_nominal_v2_spec('DIO'), 'dio')
        with self.assertRaises(PhysicalInputValidationError):
            map_additive_nominal_v2_spec('')

    def test_04_molecular_and_processing_edge_case_validation(self):
        # Missing identity
        with self.assertRaises(MissingMolecularIdentityError):
            validate_and_parse_smiles(None, role='donor')

        # Invalid & BigSMILES syntax
        with self.assertRaises(InvalidMolecularInputError):
            validate_and_parse_smiles('', role='donor')
        with self.assertRaises(InvalidMolecularInputError):
            validate_and_parse_smiles('{[]CC(C)[]}', role='acceptor')

        # Non-ring pseudo-scaffold
        _, can_smi, scaff_id, warns_nr = validate_and_parse_smiles('C=C(CC(=O)O)C(=O)O', role='acceptor')
        self.assertEqual(scaff_id, f'[NO_RING]:{can_smi}')
        self.assertIn('ACCEPTOR_NON_RING_PSEUDO_SCAFFOLD', warns_nr)

        # Wildcard polymer & disconnected mixture detection + strict rejection
        _, _, _, warns_w = validate_and_parse_smiles('*/C(C#N)=C1\\C(=O)c2ccccc2C1=C\\c1cccs1', role='acceptor')
        self.assertIn('ACCEPTOR_POLYMER_WILDCARD_PRESENT', warns_w)
        with self.assertRaises(UnsupportedPolymerOrMixtureError):
            validate_and_parse_smiles('*/C(C#N)=C1\\C(=O)c2ccccc2C1=C\\c1cccs1', role='acceptor', reject_wildcards_or_mixtures=True)

        _, _, _, warns_m = validate_and_parse_smiles('N#C[S-].[Cu+]', role='donor')
        self.assertIn('DONOR_MULTI_COMPONENT_MIXTURE_PRESENT', warns_m)
        with self.assertRaises(UnsupportedPolymerOrMixtureError):
            validate_and_parse_smiles('N#C[S-].[Cu+]', role='donor', reject_wildcards_or_mixtures=True)

        # Physical validation rejections (no silent clipping)
        with self.assertRaises(PhysicalInputValidationError):
            validate_processing_record({'active_layer_thickness': -10.0})
        with self.assertRaises(PhysicalInputValidationError):
            validate_processing_record({'annealing_temp': 400.0})
        with self.assertRaises(PhysicalInputValidationError):
            validate_processing_record({'d_a_ratio': '0:1'})
        with self.assertRaises(PhysicalInputValidationError):
            validate_processing_record({'additive_ratio': '120'})
        with self.assertRaises(PhysicalInputValidationError):
            validate_processing_record({'solvent_canonical': ''})

        # Unusual valid values emit warnings without clipping
        warns_dom = validate_processing_record({'active_layer_thickness': 800.0, 'annealing_temp': 280.0})
        self.assertIn('THICKNESS_OUT_OF_TRAINING_RANGE_WARNING', warns_dom)
        self.assertIn('ANNEALING_TEMP_OUT_OF_TRAINING_RANGE_WARNING', warns_dom)


if __name__ == '__main__':
    unittest.main()
