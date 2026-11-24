import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.base import Base
import backend.app.models
from backend.app.models.user import User
from backend.app.models.model_version import ModelVersion
from backend.app.models.prediction import Prediction
from backend.app.schemas.prediction import (
    MolecularAnalysisRequest,
    PredictionRequest,
)
from backend.app.services.molecular import (
    validate_and_canonicalize_smiles,
    compute_molecular_descriptors,
    analyze_blend,
)
from backend.app.services.prediction import run_prediction_pipeline
from backend.app.api.v1.predictions import (
    analyze_molecules,
    create_prediction,
    get_prediction_history,
    get_prediction_by_id,
)
from ml.src.models.predictor import OPVModelRegistry, model_registry
from ml.src.features.descriptors import FEATURE_NAMES


class TestPhase3MolecularAndInference(unittest.TestCase):
    """Automated test suite for Phase 3: RDKit Analysis and ML Inference."""

    def setUp(self):
        # Create an isolated temporary SQLite database
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        # Create test users
        self.user_a = User(
            email="researcher.a@solar.org",
            hashed_password="hash",
            full_name="Researcher A",
            institution="Lab A"
        )
        self.user_b = User(
            email="researcher.b@solar.org",
            hashed_password="hash",
            full_name="Researcher B",
            institution="Lab B"
        )
        self.db.add_all([self.user_a, self.user_b])
        self.db.commit()

        # Real OPV benchmark SMILES fragments
        self.valid_donor = "c1cc(sc1)CCCCCC"  # Hexylthiophene
        self.valid_acceptor = "c1ccc2nsnc2c1"  # Benzothiadiazole

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_smiles_validation_valid(self):
        """Verify real RDKit validation and canonicalization of valid SMILES."""
        is_valid, canonical, error = validate_and_canonicalize_smiles(self.valid_donor)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertEqual(canonical, "CCCCCCc1cccs1")

    def test_smiles_validation_invalid(self):
        """Verify RDKit returns clear validation errors for unphysical or invalid SMILES."""
        # Case 1: unclosed ring
        is_valid1, _, err1 = validate_and_canonicalize_smiles("C1=CC=CC=C")
        self.assertFalse(is_valid1)
        self.assertIn("Invalid chemical structure", err1)

        # Case 2: hypervalent carbon
        is_valid2, _, err2 = validate_and_canonicalize_smiles("C(=O)(=O)(=O)(=O)")
        self.assertFalse(is_valid2)
        self.assertIsNotNone(err2)

        # Case 3: empty string
        is_valid3, _, err3 = validate_and_canonicalize_smiles("   ")
        self.assertFalse(is_valid3)
        self.assertIn("cannot be empty", err3)

    def test_descriptor_calculation(self):
        """Verify scientific RDKit physicochemical descriptors are computed accurately."""
        desc = compute_molecular_descriptors(self.valid_donor, role="donor")
        self.assertEqual(desc["role"], "donor")
        self.assertEqual(desc["formula"], "C10H16S")
        self.assertAlmostEqual(desc["molecular_weight"], 168.30, delta=0.5)
        self.assertGreater(desc["logp"], 0)
        self.assertEqual(desc["aromatic_rings"], 1)
        self.assertEqual(desc["heavy_atom_count"], 11)

    def test_distinct_donor_acceptor_roles(self):
        """Verify donor and acceptor molecules retain distinct roles and independent descriptors."""
        blend = analyze_blend(
            donor_smiles=self.valid_donor,
            acceptor_smiles=self.valid_acceptor,
            donor_name="Hexylthiophene",
            acceptor_name="Benzothiadiazole"
        )
        self.assertEqual(blend["donor"]["role"], "donor")
        self.assertEqual(blend["donor"]["name"], "Hexylthiophene")
        self.assertEqual(blend["acceptor"]["role"], "acceptor")
        self.assertEqual(blend["acceptor"]["name"], "Benzothiadiazole")
        self.assertNotEqual(blend["donor"]["formula"], blend["acceptor"]["formula"])
        self.assertIn("molecular_weight_diff", blend["blend_features"])

    def test_endpoint_analyze_molecules(self):
        """Verify POST /api/predictions/analyze returns HTTP 200 with descriptor payload."""
        req = MolecularAnalysisRequest(
            donor_smiles=self.valid_donor,
            acceptor_smiles=self.valid_acceptor,
            donor_name="D1",
            acceptor_name="A1"
        )
        res = analyze_molecules(req)
        self.assertIn("donor", res)
        self.assertIn("acceptor", res)
        self.assertIn("blend_features", res)

    def test_endpoint_analyze_invalid_raises_422(self):
        """Verify POST /api/predictions/analyze raises HTTP 422 for unparseable chemistry."""
        req = MolecularAnalysisRequest(
            donor_smiles="NOT_A_VALID_SMILES",
            acceptor_smiles=self.valid_acceptor
        )
        with self.assertRaises(HTTPException) as ctx:
            analyze_molecules(req)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_model_unavailable_response_without_fabrication(self):
        """Verify when no trained model exists, the pipeline returns explicit model_unavailable without fake PCE."""
        res = run_prediction_pipeline(
            donor_smiles=self.valid_donor,
            acceptor_smiles=self.valid_acceptor
        )
        self.assertEqual(res["status"], "model_unavailable")
        self.assertIsNone(res["predictions"])
        self.assertIn("No validated machine learning model artifact found", res["message"])
        self.assertIn("donor", res)
        self.assertIn("acceptor", res)

    def test_prediction_with_validated_model_and_persistence(self):
        """Verify genuine inference and persistence when a validated model artifact is present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_models_dir = Path(temp_dir)
            # Create a genuine scikit-learn regressor for testing pipeline execution
            X_dummy = np.random.RandomState(42).randn(10, len(FEATURE_NAMES))
            y_dummy = np.array([[12.5, 0.82, 21.0, 70.0]] * 10)
            rf = RandomForestRegressor(n_estimators=5, random_state=42)
            rf.fit(X_dummy, y_dummy)

            model_file = temp_models_dir / "test_v1.0.0.joblib"
            joblib.dump(rf, model_file)

            # Accompany with valid scientific certification metadata
            meta_content = {
                "version_tag": "test_v1.0.0",
                "algorithm": "RandomForestRegressor",
                "feature_names": FEATURE_NAMES,
                "is_scientifically_validated": True,
                "training_provenance": "Validated experimental peer-reviewed benchmark dataset",
                "dataset_source": "https://doi.org/10.1038/s41524-021-00622-2",
                "evaluation_metrics": {
                    "r2": 0.75,
                    "mae": 0.58,
                    "rmse": 0.81,
                    "pce_r2": 0.75,
                    "pce_mae": 0.58,
                    "pce_rmse": 0.81,
                },
            }
            with open(temp_models_dir / "test_v1.0.0.meta.json", "w", encoding="utf-8") as f:
                json.dump(meta_content, f)

            # Point a test registry to this temp directory
            test_registry = OPVModelRegistry(models_dir=temp_models_dir)
            self.assertTrue(test_registry.is_model_available())
            self.assertTrue(test_registry.is_scientifically_validated())

            # Seed model_version row to satisfy FK
            mv = ModelVersion(
                version_tag="test_v1.0.0",
                model_name="TestRF",
                algorithm="RandomForestRegressor",
                target_metrics="PCE,Voc,Jsc,FF",
                artifact_path=str(model_file),
                is_active=True
            )
            self.db.add(mv)
            self.db.commit()

            # Temporarily substitute registry in service
            import backend.app.services.prediction as pred_svc
            original_reg = pred_svc.model_registry
            pred_svc.model_registry = test_registry

            try:
                req = PredictionRequest(
                    donor_smiles=self.valid_donor,
                    acceptor_smiles=self.valid_acceptor,
                    donor_name="DonorTest",
                    acceptor_name="AccTest"
                )
                pred_resp = create_prediction(req, current_user=self.user_a, db=self.db)
                self.assertEqual(pred_resp["status"], "success")
                self.assertIsNotNone(pred_resp["predictions"])
                self.assertIn("pce_percent", pred_resp["predictions"])
                self.assertIn("id", pred_resp)

                # Verify saved in SQLite database
                saved_record = self.db.query(Prediction).filter(Prediction.id == pred_resp["id"]).first()
                self.assertIsNotNone(saved_record)
                self.assertEqual(saved_record.user_id, self.user_a.id)
                self.assertEqual(saved_record.donor_name, "DonorTest")
                self.assertAlmostEqual(saved_record.predicted_pce, 12.5, delta=0.5)
            finally:
                pred_svc.model_registry = original_reg

    def test_unvalidated_model_artifact_is_rejected(self):
        """Verify unvalidated test fixtures or models without certification are rejected and write 0 DB records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_models_dir = Path(temp_dir)
            X_dummy = np.random.RandomState(42).randn(10, len(FEATURE_NAMES))
            y_dummy = np.array([[12.5, 0.82, 21.0, 70.0]] * 10)
            rf = RandomForestRegressor(n_estimators=5, random_state=42)
            rf.fit(X_dummy, y_dummy)

            # Dump model WITHOUT valid scientific certification
            model_file = temp_models_dir / "unvalidated_fixture.joblib"
            joblib.dump(rf, model_file)

            test_registry = OPVModelRegistry(models_dir=temp_models_dir)
            self.assertTrue(test_registry.is_model_available())
            self.assertFalse(test_registry.is_scientifically_validated())

            import backend.app.services.prediction as pred_svc
            original_reg = pred_svc.model_registry
            pred_svc.model_registry = test_registry

            try:
                initial_count = self.db.query(Prediction).count()
                req = PredictionRequest(
                    donor_smiles=self.valid_donor,
                    acceptor_smiles=self.valid_acceptor,
                    donor_name="DonorTest",
                    acceptor_name="AccTest",
                )
                pred_resp = create_prediction(req, current_user=self.user_a, db=self.db)
                self.assertEqual(pred_resp["status"], "model_unvalidated")
                self.assertIsNone(pred_resp["predictions"])
                self.assertIn("failed scientific validation", pred_resp["message"])
                # Zero records written to database
                final_count = self.db.query(Prediction).count()
                self.assertEqual(final_count, initial_count)
            finally:
                pred_svc.model_registry = original_reg

    def test_prediction_history_and_user_isolation(self):
        """Verify users can only retrieve their own prediction records."""
        # Insert prediction for User A
        pred_a = Prediction(
            id="pred_user_a_1",
            user_id=self.user_a.id,
            input_method="smiles",
            donor_name="DonorA",
            donor_smiles=self.valid_donor,
            acceptor_name="AccA",
            acceptor_smiles=self.valid_acceptor,
            predicted_pce=14.2
        )
        # Insert prediction for User B
        pred_b = Prediction(
            id="pred_user_b_1",
            user_id=self.user_b.id,
            input_method="smiles",
            donor_name="DonorB",
            donor_smiles=self.valid_donor,
            acceptor_name="AccB",
            acceptor_smiles=self.valid_acceptor,
            predicted_pce=16.8
        )
        self.db.add_all([pred_a, pred_b])
        self.db.commit()

        # Query history as User A
        history_a = get_prediction_history(current_user=self.user_a, db=self.db)
        self.assertEqual(history_a.total, 1)
        self.assertEqual(history_a.items[0].id, "pred_user_a_1")
        self.assertEqual(history_a.items[0].donor_name, "DonorA")

        # Query history as User B
        history_b = get_prediction_history(current_user=self.user_b, db=self.db)
        self.assertEqual(history_b.total, 1)
        self.assertEqual(history_b.items[0].id, "pred_user_b_1")
        self.assertEqual(history_b.items[0].donor_name, "DonorB")

        # User A attempting to fetch User B's prediction by ID must raise 404
        with self.assertRaises(HTTPException) as ctx:
            get_prediction_by_id(prediction_id="pred_user_b_1", current_user=self.user_a, db=self.db)
        self.assertEqual(ctx.exception.status_code, 404)

        # User A fetching User A's prediction succeeds
        fetched_a = get_prediction_by_id(prediction_id="pred_user_a_1", current_user=self.user_a, db=self.db)
        self.assertEqual(fetched_a.id, "pred_user_a_1")


if __name__ == "__main__":
    unittest.main()
