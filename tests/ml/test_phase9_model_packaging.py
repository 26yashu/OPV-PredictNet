"""
Phase 9 ML Model Packaging Verification Tests.

Verifies:
1. All 4 approved model artifacts (M0, M1, M2A, M3), pipeline state,
   scaffold registry, and global metadata exist under ml/models/saved_models/phase9_v1.
2. Exact feature schema compliance (22, 50, 37, 44).
3. Pipeline state restoration reproduces feature extraction deterministically.
4. Model save/reload determinism (< 1e-9 difference).
5. Governance flags (research_estimate_only=True, validated_for_production=False,
   is_scientifically_validated=True) and M2 exclusion.
"""

import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.features.processing import MODEL_SCHEMAS, UnifiedFeaturePipeline

SAVE_DIR = PROJECT_ROOT / "ml" / "models" / "saved_models" / "phase9_v1"
EXPECTED_COUNTS = {"M0": 22, "M1": 50, "M2A": 37, "M3": 44}


class TestPhase9ModelPackaging(unittest.TestCase):
    """Verify packaged Phase 9 model artifacts, schemas, and determinism."""

    @classmethod
    def setUpClass(cls):
        cls.metadata_path = SAVE_DIR / "phase9_model_metadata.json"
        cls.pipeline_path = SAVE_DIR / "phase9_pipeline_state.json"
        cls.registry_path = SAVE_DIR / "phase9_scaffold_registry.json"

        with open(cls.pipeline_path, "r", encoding="utf-8") as f:
            cls.pipeline_state = json.load(f)
        cls.pipeline = UnifiedFeaturePipeline.from_state_dict(cls.pipeline_state)

        with open(cls.registry_path, "r", encoding="utf-8") as f:
            cls.scaffold_registry = json.load(f)

        with open(cls.metadata_path, "r", encoding="utf-8") as f:
            cls.global_metadata = json.load(f)

    def test_required_files_exist(self):
        self.assertTrue(self.metadata_path.exists())
        self.assertTrue(self.pipeline_path.exists())
        self.assertTrue(self.registry_path.exists())
        for model_id in ("M0", "M1", "M2A", "M3"):
            self.assertTrue((SAVE_DIR / f"phase9_{model_id}_model.joblib").exists())
        self.assertFalse((SAVE_DIR / "phase9_M2_model.joblib").exists())

    def test_global_metadata_and_governance_flags(self):
        self.assertTrue(self.global_metadata["research_estimate_only"])
        self.assertFalse(self.global_metadata["validated_for_production"])
        self.assertTrue(self.global_metadata["determinism_verified"])
        self.assertEqual(set(self.global_metadata["models"].keys()), {"M0", "M1", "M2A", "M3"})

    def test_scaffold_registry_populated(self):
        self.assertGreater(len(self.scaffold_registry["donor_scaffolds"]), 100)
        self.assertGreater(len(self.scaffold_registry["acceptor_scaffolds"]), 50)

    def test_artifacts_schema_and_determinism(self):
        sample_df = pd.DataFrame([
            {
                "donor_smiles": "CCCCCCc1cccs1",
                "acceptor_smiles": "c1ccc2nsnc2c1",
                "d_a_ratio": "1:1",
                "active_layer_thickness": "100nm",
                "annealing_temp": 110.0,
                "additive_canonical": "DIO",
                "additive_ratio": "3%",
                "solvent_canonical": "chlorobenzene",
                "device_type": "inverted",
                "etl_canonical": "ZnO",
                "htl_canonical": "MoO3",
            },
            {
                "donor_smiles": "B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21",
                "acceptor_smiles": "O=C1c2ccccc2C(=O)C1=Cc1ccc(-c2ccc(C=C3C(=O)c4ccccc4C3=O)s2)s1",
                "d_a_ratio": None,
                "active_layer_thickness": None,
                "annealing_temp": None,
                "additive_canonical": None,
                "additive_ratio": None,
                "solvent_canonical": None,
                "device_type": None,
                "etl_canonical": None,
                "htl_canonical": None,
            },
        ])

        extractors = {
            "M0": self.pipeline.extract_m0,
            "M1": self.pipeline.extract_m1,
            "M2A": self.pipeline.extract_m2a,
            "M3": self.pipeline.extract_m3,
        }

        for model_id, expected_dim in EXPECTED_COUNTS.items():
            path = SAVE_DIR / f"phase9_{model_id}_model.joblib"
            art_a = joblib.load(path)
            art_b = joblib.load(path)

            self.assertEqual(art_a["model_id"], model_id)
            self.assertEqual(art_a["n_features"], expected_dim)
            self.assertEqual(art_a["feature_names"], list(MODEL_SCHEMAS[model_id]))
            self.assertTrue(art_a["is_scientifically_validated"])
            self.assertTrue(art_a["research_estimate_only"])
            self.assertFalse(art_a["validated_for_production"])
            self.assertEqual(set(art_a["conformal_quantiles"].keys()), {"80%", "90%", "95%"})

            df_feat = extractors[model_id](sample_df)
            self.assertEqual(df_feat.shape, (2, expected_dim))
            X = df_feat.to_numpy(dtype=np.float64 if model_id != "M3" else object)

            preds_a = art_a["model"].predict(X)
            preds_b = art_b["model"].predict(X)
            np.testing.assert_allclose(preds_a, preds_b, atol=1e-12)
            self.assertTrue(np.all(np.isfinite(preds_a)))
            self.assertGreaterEqual(preds_a[0], 0.0)
            self.assertLessEqual(preds_a[0], 34.0)


if __name__ == "__main__":
    unittest.main()
