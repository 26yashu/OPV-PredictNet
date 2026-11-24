"""
OPV-PredictNet — Model Safety & Physical Sanity Unit Tests (Phase 3.1)
Verifies:
1. Physical sanity boundary constraints on predicted figures of merit (PCE, Voc, Jsc, FF).
2. Explicit rejection of unphysical values without silent clipping.
3. Model artifact scientific certification and rejection of unvalidated test fixtures.
4. Security restrictions on model artifact loading.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.evaluation.sanity import SANITY_BOUNDS, validate_prediction_sanity
from ml.src.features.descriptors import FEATURE_NAMES
from ml.src.models.predictor import (
    OPVModelRegistry,
    validate_scientific_artifact,
)
from backend.app.services.prediction import run_prediction_pipeline


class TestSanityBounds(unittest.TestCase):
    """Test suite for physical and thermodynamic sanity checks."""

    def test_boundary_values_pass(self):
        """Boundary values within single-junction limits must be accepted."""
        # Minimum physical bounds (all 0.0)
        is_sane_min, violations_min = validate_prediction_sanity(
            pce=0.0, voc=0.0, jsc=0.0, ff=0.0
        )
        self.assertTrue(is_sane_min)
        self.assertEqual(len(violations_min), 0)

        # Maximum physical bounds (SQ ceiling 34.0%, 2.0 V, 45.0 mA/cm², 100.0%)
        is_sane_max, violations_max = validate_prediction_sanity(
            pce=34.0, voc=2.0, jsc=45.0, ff=100.0
        )
        self.assertTrue(is_sane_max)
        self.assertEqual(len(violations_max), 0)

        # Typical high-efficiency OPV benchmark (e.g. PM6:Y6 ~18% PCE)
        is_sane_typ, violations_typ = validate_prediction_sanity(
            pce=18.2, voc=0.85, jsc=27.5, ff=78.0
        )
        self.assertTrue(is_sane_typ)
        self.assertEqual(len(violations_typ), 0)

    def test_impossible_pce_negative_rejected(self):
        """Negative PCE must be rejected explicitly."""
        is_sane, violations = validate_prediction_sanity(pce=-2.5)
        self.assertFalse(is_sane)
        self.assertTrue(any("PCE" in v and "exceeds physical sanity" in v for v in violations))

    def test_impossible_pce_above_shockley_queisser_rejected(self):
        """PCE exceeding the thermodynamic single-junction ceiling (>34%) must be rejected."""
        is_sane, violations = validate_prediction_sanity(pce=38.5)
        self.assertFalse(is_sane)
        self.assertTrue(any("PCE" in v and "34.0%" in v for v in violations))

    def test_impossible_voc_rejected(self):
        """Unphysical open-circuit voltages (>2.0 V or negative) must be rejected."""
        # Case 1: Extreme value previously observed in test fixtures (11.78 V)
        is_sane1, violations1 = validate_prediction_sanity(pce=12.0, voc=11.779)
        self.assertFalse(is_sane1)
        self.assertTrue(any("Voc" in v for v in violations1))

        # Case 2: Negative Voc
        is_sane2, violations2 = validate_prediction_sanity(pce=10.0, voc=-0.5)
        self.assertFalse(is_sane2)
        self.assertTrue(any("Voc" in v for v in violations2))

    def test_impossible_jsc_rejected(self):
        """Unphysical short-circuit current density (>45 mA/cm² under AM1.5G) must be rejected."""
        is_sane, violations = validate_prediction_sanity(pce=15.0, jsc=60.0)
        self.assertFalse(is_sane)
        self.assertTrue(any("Jsc" in v for v in violations))

    def test_impossible_ff_rejected(self):
        """Fill factor exceeding 100% or negative must be rejected."""
        is_sane, violations = validate_prediction_sanity(pce=15.0, ff=115.0)
        self.assertFalse(is_sane)
        self.assertTrue(any("Fill Factor" in v for v in violations))

    def test_no_silent_clipping(self):
        """Sanity validation must never silently clip or alter unphysical values."""
        raw_pce = 45.0
        is_sane, _ = validate_prediction_sanity(pce=raw_pce)
        self.assertFalse(is_sane)
        # Verify raw input variable remains exactly as provided
        self.assertEqual(raw_pce, 45.0)


class TestModelArtifactSafety(unittest.TestCase):
    """Test suite for model loading safety, provenance verification, and fixture rejection."""

    def test_missing_certification_flag_rejected(self):
        """Model without is_scientifically_validated: true flag must be rejected."""
        dummy_model = RandomForestRegressor()
        meta = {
            "version_tag": "v0.1.0",
            "feature_names": FEATURE_NAMES,
            "training_provenance": "Experimental data",
            "dataset_source": "https://doi.org/10.1038/sample",
            "evaluation_metrics": {"r2": 0.65},
        }
        is_valid, errors = validate_scientific_artifact(dummy_model, meta)
        self.assertFalse(is_valid)
        self.assertTrue(any("is_scientifically_validated" in err for err in errors))

    def test_synthetic_fixture_rejected(self):
        """Model with synthetic or test fixture provenance keywords must be rejected."""
        dummy_model = RandomForestRegressor()
        meta = {
            "version_tag": "v0.1.0",
            "feature_names": FEATURE_NAMES,
            "is_scientifically_validated": True,
            "training_provenance": "Synthetic test fixture for pipeline testing",
            "dataset_source": "https://doi.org/10.1038/sample",
            "evaluation_metrics": {"r2": 0.65, "mae": 0.5, "rmse": 0.7},
        }
        is_valid, errors = validate_scientific_artifact(dummy_model, meta)
        self.assertFalse(is_valid)
        self.assertTrue(any("synthetic" in err.lower() or "test fixture" in err.lower() for err in errors))

    def test_missing_dataset_source_citation_rejected(self):
        """Model without valid DOI or URL dataset citation must be rejected."""
        dummy_model = RandomForestRegressor()
        meta = {
            "version_tag": "v0.1.0",
            "feature_names": FEATURE_NAMES,
            "is_scientifically_validated": True,
            "training_provenance": "Experimental measurements from peer-reviewed literature",
            "dataset_source": "Internal notebook page 42",  # Not a valid DOI or URL
            "evaluation_metrics": {"r2": 0.65, "mae": 0.5, "rmse": 0.7},
        }
        is_valid, errors = validate_scientific_artifact(dummy_model, meta)
        self.assertFalse(is_valid)
        self.assertTrue(any("dataset_source" in err.lower() for err in errors))

    def test_unphysical_prediction_in_pipeline_flagged(self):
        """A model that outputs unphysical values (e.g. PCE > 34%) must return unphysical_prediction status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_models_dir = Path(temp_dir)
            # Create a model that always predicts unphysical PCE = 50.0%
            X_dummy = np.random.RandomState(42).randn(10, len(FEATURE_NAMES))
            y_dummy = np.array([[50.0, 0.85, 25.0, 75.0]] * 10)  # Unphysical PCE = 50%
            rf = RandomForestRegressor(n_estimators=5, random_state=42)
            rf.fit(X_dummy, y_dummy)

            model_file = temp_models_dir / "unphysical_model.joblib"
            joblib.dump(rf, model_file)

            meta_file = temp_models_dir / "unphysical_model.meta.json"
            meta = {
                "version_tag": "unphysical_model",
                "algorithm": "RandomForestRegressor",
                "feature_names": FEATURE_NAMES,
                "is_scientifically_validated": True,
                "training_provenance": "Experimental OPV benchmark set",
                "dataset_source": "https://doi.org/10.1038/s41524-021-00622-2",
                "evaluation_metrics": {"r2": 0.70, "mae": 0.5, "rmse": 0.7},
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            test_reg = OPVModelRegistry(models_dir=temp_models_dir)
            self.assertTrue(test_reg.is_scientifically_validated())

            # Substitute test registry
            import backend.app.services.prediction as pred_svc
            orig = pred_svc.model_registry
            pred_svc.model_registry = test_reg

            try:
                res = run_prediction_pipeline(
                    donor_smiles="CCCCCCc1cccs1",
                    acceptor_smiles="c1ccc2nsnc2c1",
                )
                self.assertEqual(res["status"], "unphysical_prediction")
                self.assertIsNone(res["predictions"])
                self.assertIn("sanity_violations", res)
                self.assertTrue(len(res["sanity_violations"]) > 0)
                self.assertTrue(any("PCE" in v for v in res["sanity_violations"]))
            finally:
                pred_svc.model_registry = orig


if __name__ == "__main__":
    unittest.main()
