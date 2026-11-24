"""
OPV-PredictNet — Phase 3 End-to-End HTTP API Integration Tests
Tests real HTTP request/response lifecycles using Starlette/FastAPI TestClient.
"""

import os
import unittest
import uuid
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.base import Base
from backend.app.db.session import engine
from backend.app.core.config import settings
from ml.src.models.predictor import model_registry


class TestPhase3HttpApiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        Base.metadata.create_all(bind=engine)
        # Clean any stray test models from previous test runs
        if os.path.exists(model_registry.models_dir):
            for f in os.listdir(model_registry.models_dir):
                if f.endswith(".joblib") or f.endswith(".meta.json"):
                    os.remove(os.path.join(model_registry.models_dir, f))
        model_registry.refresh()

    def test_01_health_endpoint_http(self):
        """Verify GET /api/health returns HTTP 200 with service metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], settings.APP_NAME)
        self.assertIn("timestamp", data)

    def test_02_analyze_molecules_valid_http(self):
        """Verify POST /api/predictions/analyze validates SMILES and returns real descriptors."""
        payload = {
            "donor_smiles": "CCCCCCc1cccs1",
            "acceptor_smiles": "c1ccc2nsnc2c1",
            "donor_name": "P3HT Subunit",
            "acceptor_name": "Benzothiadiazole",
        }
        response = self.client.post("/api/predictions/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check donor descriptors
        donor = data["donor"]
        self.assertEqual(donor["name"], "P3HT Subunit")
        self.assertEqual(donor["role"], "donor")
        self.assertEqual(donor["canonical_smiles"], "CCCCCCc1cccs1")
        self.assertAlmostEqual(donor["molecular_weight"], 168.3, places=1)
        self.assertEqual(donor["formula"], "C10H16S")
        self.assertGreater(donor["logp"], 0)

        # Check acceptor descriptors
        acceptor = data["acceptor"]
        self.assertEqual(acceptor["name"], "Benzothiadiazole")
        self.assertEqual(acceptor["role"], "acceptor")
        self.assertEqual(acceptor["canonical_smiles"], "c1ccc2nsnc2c1")
        self.assertAlmostEqual(acceptor["molecular_weight"], 136.17, places=1)
        self.assertEqual(acceptor["formula"], "C6H4N2S")

        # Check composite blend features
        blend = data["blend_features"]
        self.assertIn("molecular_weight_diff", blend)
        self.assertIn("tpsa_total", blend)
        self.assertIn("aromatic_rings_total", blend)

    def test_03_analyze_molecules_invalid_http(self):
        """Verify POST /api/predictions/analyze returns HTTP 422 for unphysical or invalid SMILES."""
        payload = {
            "donor_smiles": "INVALID_CHEMISTRY_XYZ",
            "acceptor_smiles": "c1ccc2nsnc2c1",
        }
        response = self.client.post("/api/predictions/analyze", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("Invalid donor SMILES", data["detail"])

    def test_04_prediction_auth_and_model_unavailable_http(self):
        """Verify POST /api/predictions requires authentication and returns model_unavailable without fake metrics."""
        pred_payload = {
            "donor_smiles": "CCCCCCc1cccs1",
            "acceptor_smiles": "c1ccc2nsnc2c1",
            "device_parameters": {
                "active_layer_thickness_nm": 100,
                "d_a_ratio": "1:1",
                "solvent": "Chlorobenzene"
            }
        }

        # 1. Without Authorization header -> HTTP 401
        anon_response = self.client.post("/api/predictions", json=pred_payload)
        self.assertEqual(anon_response.status_code, 401)

        # 2. With valid user credentials -> HTTP 200 with model_unavailable
        uid = str(uuid.uuid4())[:8]
        email = f"user_{uid}@opv.org"
        self.client.post("/api/auth/register", json={
            "email": email,
            "password": "SecurePassword123!",
            "full_name": "Test User",
        })
        login_res = self.client.post("/api/auth/login", json={
            "email": email,
            "password": "SecurePassword123!",
        })
        token = login_res.json()["access_token"]

        auth_response = self.client.post(
            "/api/predictions",
            json=pred_payload,
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(auth_response.status_code, 200)
        data = auth_response.json()
        self.assertEqual(data["status"], "model_unavailable")
        self.assertIsNone(data["predictions"])
        self.assertIn("No validated machine learning model artifact found", data["message"])
        self.assertIn("donor", data)
        self.assertIn("acceptor", data)

    def test_05_prediction_persistence_and_isolation_http(self):
        """Verify genuine inference persistence and strict isolation between User A and User B."""
        # Setup temporary valid model artifact
        os.makedirs(model_registry.models_dir, exist_ok=True)
        temp_model_path = os.path.join(model_registry.models_dir, "test_integration_rf.joblib")

        from ml.src.features.descriptors import FEATURE_NAMES
        X_dummy = np.random.RandomState(42).rand(25, len(FEATURE_NAMES))
        y_dummy = np.random.RandomState(42).uniform(4.0, 14.0, size=(25, 4))
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X_dummy, y_dummy)

        artifact = {
            "model": rf,
            "version_tag": "test_integration_v1",
            "algorithm": "RandomForestRegressor",
            "features": FEATURE_NAMES,
            "training_provenance": "Integration test harness",
            "evaluation_metrics": {"mae": 0.35, "rmse": 0.48, "r2": 0.81},
            "disclaimer": "Test artifact only",
            "limitations": ["Integration testing synthetic artifact"]
        }
        joblib.dump(artifact, temp_model_path)
        model_registry.refresh()

        try:
            uid_a = str(uuid.uuid4())[:8]
            uid_b = str(uuid.uuid4())[:8]

            # Register & login User A
            email_a = f"researcher_{uid_a}@opv.org"
            pwd = "SecurePassword123!"
            self.client.post("/api/auth/register", json={
                "email": email_a,
                "password": pwd,
                "full_name": "Dr. Researcher A",
                "institution": "Solar Labs A"
            })
            login_a = self.client.post("/api/auth/login", json={"email": email_a, "password": pwd})
            token_a = login_a.json()["access_token"]
            headers_a = {"Authorization": f"Bearer {token_a}"}

            # Register & login User B
            email_b = f"researcher_{uid_b}@opv.org"
            self.client.post("/api/auth/register", json={
                "email": email_b,
                "password": pwd,
                "full_name": "Dr. Researcher B",
                "institution": "Solar Labs B"
            })
            login_b = self.client.post("/api/auth/login", json={"email": email_b, "password": pwd})
            token_b = login_b.json()["access_token"]
            headers_b = {"Authorization": f"Bearer {token_b}"}

            # Submit prediction as User A
            pred_payload = {
                "donor_smiles": "CCCCCCc1cccs1",
                "acceptor_smiles": "c1ccc2nsnc2c1",
                "donor_name": "P3HT Subunit",
                "acceptor_name": "BT Unit",
                "device_parameters": {
                    "active_layer_thickness_nm": 105,
                    "d_a_ratio": "1:1.2",
                }
            }
            pred_res = self.client.post("/api/predictions", json=pred_payload, headers=headers_a)
            self.assertEqual(pred_res.status_code, 200)
            pred_data = pred_res.json()
            self.assertEqual(pred_data["status"], "success")
            self.assertIsNotNone(pred_data["predictions"])
            prediction_id = pred_data["id"]
            self.assertIsNotNone(prediction_id)

            # Retrieve User A history
            hist_a = self.client.get("/api/predictions", headers=headers_a)
            self.assertEqual(hist_a.status_code, 200)
            items_a = hist_a.json()["items"]
            self.assertTrue(any(item["id"] == prediction_id for item in items_a))

            # Retrieve single record as User A
            single_a = self.client.get(f"/api/predictions/{prediction_id}", headers=headers_a)
            self.assertEqual(single_a.status_code, 200)
            self.assertEqual(single_a.json()["id"], prediction_id)

            # Check User B history does not contain User A's prediction
            hist_b = self.client.get("/api/predictions", headers=headers_b)
            self.assertEqual(hist_b.status_code, 200)
            self.assertEqual(hist_b.json()["total"], 0)
            self.assertEqual(len(hist_b.json()["items"]), 0)

            # Check User B cannot retrieve User A's record by ID -> 404
            single_b = self.client.get(f"/api/predictions/{prediction_id}", headers=headers_b)
            self.assertEqual(single_b.status_code, 404)

        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            model_registry.refresh()


if __name__ == "__main__":
    unittest.main()
