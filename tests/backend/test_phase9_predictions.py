"""
Phase 9 Backend API & Prediction Service Tests.

Tests:
1. Valid prediction for all 4 models (M0, M1, M2A, M3) -> 200, separated point prediction,
   qualified interval estimate, OOD status, scientific support status, governance flags.
2. Invalid SMILES, empty SMILES, polymer wildcards (*), and mixtures (.) -> 422.
3. Invalid physical processing parameters -> 422.
4. Unknown donor scaffold OOD -> 200 with OOD warning.
5. Unknown acceptor scaffold OOD -> 200 with OOD warning (Strategy B coverage note).
6. Dual-scaffold OOD -> 422 by default; 200 with CRITICAL warning when explicitly allowed.
7. Prediction history persistence and user isolation in SQLite.
8. Authentication required (401 without token).
9. GET /api/predictions/models returns packaged model metadata.
"""

import json
import sys
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.base import Base
from backend.app.db.session import engine, SessionLocal
from backend.app.main import app
from backend.app.models.prediction import Prediction
from backend.app.services.phase9_prediction import phase9_service


class TestPhase9PredictionsAPI(unittest.TestCase):
    """End-to-end API tests for Phase 9 FastAPI prediction endpoints."""

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

        # Register and login User A
        uid_a = uuid.uuid4().hex[:8]
        cls.email_a = f"phase9_a_{uid_a}@opv.org"
        cls.client.post("/api/auth/register", json={
            "email": cls.email_a,
            "password": "SecurePassword123!",
            "full_name": "Phase9 User A",
        })
        res_a = cls.client.post("/api/auth/login", json={
            "email": cls.email_a,
            "password": "SecurePassword123!",
        })
        cls.token_a = res_a.json()["access_token"]
        cls.headers_a = {"Authorization": f"Bearer {cls.token_a}"}

        # Register and login User B
        uid_b = uuid.uuid4().hex[:8]
        cls.email_b = f"phase9_b_{uid_b}@opv.org"
        cls.client.post("/api/auth/register", json={
            "email": cls.email_b,
            "password": "SecurePassword123!",
            "full_name": "Phase9 User B",
        })
        res_b = cls.client.post("/api/auth/login", json={
            "email": cls.email_b,
            "password": "SecurePassword123!",
        })
        cls.token_b = res_b.json()["access_token"]
        cls.headers_b = {"Authorization": f"Bearer {cls.token_b}"}

        # Pick real known scaffolds from the loaded registry for in-scaffold tests
        cls.known_donor_smiles = "CCCCCCc1cccs1"  # thiophene ring -> c1cccs1
        cls.known_acceptor_smiles = "B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21"  # known SubPc acceptor scaffold

        # Rare / unseen scaffolds guaranteed not to be in OPVdb
        cls.unseen_donor_smiles = "C1CC2CCC1C2"  # norbornane
        cls.unseen_acceptor_smiles = "C12C3C4C1C5C2C3C45"  # cubane

    def test_01_models_endpoint(self):
        resp = self.client.get("/api/predictions/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["research_estimate_only"])
        self.assertFalse(data["validated_for_production"])
        self.assertEqual(set(data["models"].keys()), {"M0", "M1", "M2A", "M3"})

    def test_02_requires_auth(self):
        resp = self.client.post("/api/predictions/pce", json={
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
        })
        self.assertEqual(resp.status_code, 401)

    def test_03_valid_predictions_all_four_models(self):
        for model_id in ("M0", "M1", "M2A", "M3"):
            payload = {
                "donor_smiles": self.known_donor_smiles,
                "acceptor_smiles": self.known_acceptor_smiles,
                "donor_name": "Hexylthiophene",
                "acceptor_name": "Benzothiadiazole",
                "d_a_ratio": "1:1",
                "active_layer_thickness": "100nm",
                "annealing_temp": 110.0,
                "additive_canonical": "DIO",
                "additive_ratio": "3%",
                "solvent_canonical": "chlorobenzene",
                "device_type": "inverted",
                "etl_canonical": "ZnO",
                "htl_canonical": "MoO3",
                "model_id": model_id,
                "confidence_level": "90%",
            }
            resp = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
            self.assertEqual(resp.status_code, 200, f"Failed for {model_id}: {resp.text}")
            data = resp.json()

            self.assertEqual(data["model_id"], model_id)
            self.assertTrue(data["research_estimate_only"])
            self.assertFalse(data["validated_for_production"])
            self.assertTrue(data["scientific_support"]["research_estimate_only"])
            self.assertFalse(data["scientific_support"]["validated_for_production"])

            # Separate point prediction & interval estimate
            pce = data["point_prediction"]["predicted_pce"]
            self.assertGreaterEqual(pce, 0.0)
            self.assertLessEqual(pce, 34.0)

            iv = data["interval_estimate"]
            self.assertEqual(iv["confidence_level"], "90%")
            self.assertAlmostEqual(
                iv["prediction_interval_high"] - iv["prediction_interval_low"],
                iv["interval_width"],
                places=3,
            )

            # History ID returned and persisted
            self.assertIsNotNone(data["id"])

    def test_04_invalid_smiles_and_wildcards_rejected_422(self):
        bad_inputs = [
            {"donor_smiles": "NOT_A_SMILES", "acceptor_smiles": self.known_acceptor_smiles},
            {"donor_smiles": "   ", "acceptor_smiles": self.known_acceptor_smiles},
            {"donor_smiles": "*c1cccs1*", "acceptor_smiles": self.known_acceptor_smiles},
            {"donor_smiles": "CCCCCCc1cccs1.CC", "acceptor_smiles": self.known_acceptor_smiles},
        ]
        for payload in bad_inputs:
            resp = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
            self.assertEqual(resp.status_code, 422, f"Expected 422 for {payload}, got {resp.status_code}")

    def test_05_invalid_processing_bounds_rejected_422(self):
        bad_proc = {
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "active_layer_thickness": "-50nm",
        }
        resp = self.client.post("/api/predictions/pce", json=bad_proc, headers=self.headers_a)
        self.assertEqual(resp.status_code, 422)

    def test_06_ood_donor_scaffold_warns(self):
        # Ensure test scaffold is truly unseen
        self.assertNotIn("C1CC2CCC1C2", phase9_service._donor_scaffolds)
        payload = {
            "donor_smiles": self.unseen_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "model_id": "M3",
        }
        resp = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ood_status"]["ood_donor_scaffold"])
        self.assertFalse(data["ood_status"]["ood_acceptor_scaffold"])
        self.assertFalse(data["ood_status"]["ood_dual_scaffold"])
        self.assertEqual(data["ood_status"]["severity"], "warning")
        self.assertTrue(any("OOD_DONOR_SCAFFOLD" in w for w in data["warnings"]))

    def test_07_ood_acceptor_scaffold_warns(self):
        payload = {
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.unseen_acceptor_smiles,
            "model_id": "M3",
        }
        resp = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["ood_status"]["ood_donor_scaffold"])
        self.assertTrue(data["ood_status"]["ood_acceptor_scaffold"])
        self.assertTrue(any("OOD_ACCEPTOR_SCAFFOLD" in w for w in data["warnings"]))

    def test_08_dual_scaffold_ood_rejected_by_default(self):
        payload = {
            "donor_smiles": self.unseen_donor_smiles,
            "acceptor_smiles": self.unseen_acceptor_smiles,
            "model_id": "M3",
        }
        resp = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
        self.assertEqual(resp.status_code, 422)
        self.assertIn("DUAL_SCAFFOLD_OOD_REJECTED", resp.json()["detail"])

        # Explicit opt-in returns 200 with critical warning
        payload["allow_dual_scaffold_ood"] = True
        resp_override = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
        self.assertEqual(resp_override.status_code, 200)
        data = resp_override.json()
        self.assertTrue(data["ood_status"]["ood_dual_scaffold"])
        self.assertEqual(data["ood_status"]["severity"], "critical")
        self.assertTrue(any("CRITICAL_DUAL_SCAFFOLD_OOD" in w for w in data["warnings"]))

    def test_09_prediction_history_persisted_and_isolated(self):
        payload = {
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "donor_name": "HistoryTestDonor",
            "acceptor_name": "HistoryTestAcceptor",
            "model_id": "M3",
        }
        resp = self.client.post("/api/predictions/pce", json=payload, headers=self.headers_a)
        self.assertEqual(resp.status_code, 200)
        pred_id = resp.json()["id"]

        # Verify in User A history
        hist_a = self.client.get("/api/predictions", headers=self.headers_a)
        self.assertEqual(hist_a.status_code, 200)
        ids_a = [item["id"] for item in hist_a.json()["items"]]
        self.assertIn(pred_id, ids_a)

        # Verify uncertainty_json persisted in DB
        db = SessionLocal()
        try:
            rec = db.query(Prediction).filter(Prediction.id == pred_id).first()
            self.assertIsNotNone(rec)
            unc = json.loads(rec.uncertainty_json)
            self.assertTrue(unc["research_estimate_only"])
            self.assertFalse(unc["validated_for_production"])
            self.assertIn("interval_estimate", unc)
            self.assertIn("ood_status", unc)
        finally:
            db.close()

        # User B cannot access User A's prediction
        resp_b = self.client.get(f"/api/predictions/{pred_id}", headers=self.headers_b)
        self.assertEqual(resp_b.status_code, 404)


if __name__ == "__main__":
    unittest.main()
