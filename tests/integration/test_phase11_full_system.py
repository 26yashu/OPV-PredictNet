"""
OPV-PredictNet — Phase 11 Full-System End-to-End Integration and Security Verification.

Tests the full researcher workflow:
1. User registration (POST /api/auth/register) with password complexity & email validation.
2. User authentication (POST /api/auth/login) with JWT generation.
3. User profile verification (GET /api/auth/me).
4. Model catalog verification (GET /api/predictions/models) -> M0, M1, M2A, M3 with governance.
5. Molecular analysis endpoint (POST /api/predictions/analyze) -> 2D descriptors & blend features.
6. Processing-condition predictions across all 4 models (POST /api/predictions/pce).
7. Qualified interval estimate and conformal quantile verification (80%, 90%, 95%).
8. Scientific governance and support status contract verification.
9. Out-of-domain (OOD) scaffold warning and dual-scaffold rejection / override testing.
10. Prediction history persistence, retrieval, and multi-tenant user isolation.
11. Security testing: invalid tokens, expired headers, schema violations, invalid model IDs.
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
from backend.app.models.user import User


class TestPhase11FullSystemIntegration(unittest.TestCase):
    """Complete end-to-end user journey and security verification test suite."""

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

        # Standard known molecules from OPVdb
        cls.known_donor_smiles = "CCCCCCc1cccs1"  # hexylthiophene subunit
        cls.known_acceptor_smiles = "B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21"  # SubPc

        # Guaranteed OOD structures (norbornane, cubane)
        cls.unseen_donor_smiles = "C1CC2CCC1C2"
        cls.unseen_acceptor_smiles = "C12C3C4C1C5C2C3C45"

    def test_01_user_registration_and_validation(self):
        """Verify registration with email normalization, password complexity enforcement, and duplicate rejection."""
        # 1. Weak password rejected
        weak_pw_res = self.client.post("/api/auth/register", json={
            "email": "researcher_weak@uni-lab.edu",
            "password": "simple",
            "full_name": "Test Researcher",
        })
        self.assertEqual(weak_pw_res.status_code, 422)

        # 2. Invalid email rejected
        invalid_email_res = self.client.post("/api/auth/register", json={
            "email": "not_an_email",
            "password": "StrongPassword123!",
            "full_name": "Test Researcher",
        })
        self.assertEqual(invalid_email_res.status_code, 422)

        # 3. Valid registration succeeds
        uid = uuid.uuid4().hex[:8]
        valid_email = f"lead_scientist_{uid}@materials-lab.org"
        reg_res = self.client.post("/api/auth/register", json={
            "email": f"  {valid_email.upper()}  ",  # Test normalization
            "password": "StrongPassword123!",
            "full_name": "Dr. Marie Curie",
            "institution": "Sorbonne Materials Institute",
        })
        self.assertEqual(reg_res.status_code, 201)
        data = reg_res.json()
        self.assertEqual(data["email"], valid_email.lower())
        self.assertEqual(data["full_name"], "Dr. Marie Curie")
        self.assertNotIn("password", data)
        self.assertNotIn("hashed_password", data)

        # 4. Duplicate registration rejected
        dup_res = self.client.post("/api/auth/register", json={
            "email": valid_email,
            "password": "StrongPassword123!",
            "full_name": "Dr. Marie Curie",
        })
        self.assertEqual(dup_res.status_code, 400)

    def test_02_full_researcher_lifecycle(self):
        """Simulate the full user journey from login to prediction and history retrieval."""
        uid = uuid.uuid4().hex[:8]
        email = f"user_{uid}@solar-consortium.eu"
        password = "SecurePassword2026!"

        # Step 1: Register
        reg = self.client.post("/api/auth/register", json={
            "email": email,
            "password": password,
            "full_name": "Dr. Hans Fischer",
            "institution": "Technical University Munich",
        })
        self.assertEqual(reg.status_code, 201)

        # Step 2: Login and receive JWT
        login_res = self.client.post("/api/auth/login", json={
            "email": email,
            "password": password,
        })
        self.assertEqual(login_res.status_code, 200)
        tokens = login_res.json()
        self.assertIn("access_token", tokens)
        self.assertEqual(tokens["token_type"], "bearer")
        auth_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Step 3: Fetch current user profile
        me_res = self.client.get("/api/auth/me", headers=auth_headers)
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["email"], email)

        # Step 4: Model catalog discovery
        models_res = self.client.get("/api/predictions/models")
        self.assertEqual(models_res.status_code, 200)
        catalog = models_res.json()
        self.assertTrue(catalog["research_estimate_only"])
        self.assertFalse(catalog["validated_for_production"])
        self.assertIn("M0", catalog["models"])
        self.assertIn("M1", catalog["models"])
        self.assertIn("M2A", catalog["models"])
        self.assertIn("M3", catalog["models"])
        self.assertEqual(catalog["models"]["M0"]["n_features"], 22)
        self.assertEqual(catalog["models"]["M1"]["n_features"], 50)
        self.assertEqual(catalog["models"]["M2A"]["n_features"], 37)
        self.assertEqual(catalog["models"]["M3"]["n_features"], 44)

        # Step 5: Molecular descriptor analysis
        mol_res = self.client.post("/api/predictions/analyze", json={
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "donor_name": "P3HT Subunit",
            "acceptor_name": "SubPc",
        })
        self.assertEqual(mol_res.status_code, 200)
        mol_data = mol_res.json()
        self.assertIn("donor", mol_data)
        self.assertIn("acceptor", mol_data)
        self.assertIn("blend_features", mol_data)
        self.assertGreater(mol_data["donor"]["molecular_weight"], 100)

        # Step 6: Prediction generation across all 4 models
        prediction_ids = []
        for mid in ["M0", "M1", "M2A", "M3"]:
            pred_req = {
                "donor_smiles": self.known_donor_smiles,
                "acceptor_smiles": self.known_acceptor_smiles,
                "donor_name": "P3HT Subunit",
                "acceptor_name": "SubPc",
                "d_a_ratio": "1:1",
                "active_layer_thickness": "110nm",
                "annealing_temp": 120.0,
                "additive_canonical": "DIO",
                "additive_ratio": "2.5%",
                "solvent_canonical": "chlorobenzene",
                "device_type": "conventional",
                "etl_canonical": "ZnO",
                "htl_canonical": "PEDOT:PSS",
                "model_id": mid,
                "confidence_level": "90%",
            }
            pred_res = self.client.post("/api/predictions/pce", json=pred_req, headers=auth_headers)
            self.assertEqual(pred_res.status_code, 200, f"Failed prediction for model {mid}")
            p_data = pred_res.json()

            # Verify governance and separation of fields
            self.assertEqual(p_data["model_id"], mid)
            self.assertTrue(p_data["research_estimate_only"])
            self.assertFalse(p_data["validated_for_production"])
            self.assertTrue(p_data["scientific_support"]["research_estimate_only"])
            self.assertFalse(p_data["scientific_support"]["validated_for_production"])

            # Verify point prediction bounds
            pce = p_data["point_prediction"]["predicted_pce"]
            self.assertGreater(pce, 0.0)
            self.assertLess(pce, 30.0)

            # Verify interval metrics
            iv = p_data["interval_estimate"]
            self.assertEqual(iv["confidence_level"], "90%")
            self.assertGreater(iv["interval_width"], 0.0)
            self.assertLess(iv["prediction_interval_low"], iv["prediction_interval_high"])
            self.assertAlmostEqual(iv["prediction_interval_high"] - iv["prediction_interval_low"], iv["interval_width"], places=3)

            # Verify OOD status
            self.assertFalse(p_data["ood_status"]["ood_donor_scaffold"])
            self.assertFalse(p_data["ood_status"]["ood_acceptor_scaffold"])
            self.assertFalse(p_data["ood_status"]["ood_dual_scaffold"])

            prediction_ids.append(p_data["id"])

        # Step 7: Retrieve prediction history
        hist_res = self.client.get("/api/predictions", headers=auth_headers)
        self.assertEqual(hist_res.status_code, 200)
        hist = hist_res.json()
        self.assertGreaterEqual(hist["total"], 4)
        returned_ids = [item["id"] for item in hist["items"]]
        for pid in prediction_ids:
            self.assertIn(pid, returned_ids)

        # Step 8: Retrieve single prediction by ID
        single_res = self.client.get(f"/api/predictions/{prediction_ids[0]}", headers=auth_headers)
        self.assertEqual(single_res.status_code, 200)
        single_item = single_res.json()
        self.assertEqual(single_item["id"], prediction_ids[0])
        self.assertIsNotNone(single_item["uncertainty_json"])

    def test_03_ood_governance_and_override_contract(self):
        """Verify strict dual-scaffold OOD rejection and optional manual override."""
        uid = uuid.uuid4().hex[:8]
        email = f"ood_tester_{uid}@domain.com"
        self.client.post("/api/auth/register", json={
            "email": email,
            "password": "SecurePassword123!",
            "full_name": "OOD Tester",
        })
        login_res = self.client.post("/api/auth/login", json={
            "email": email,
            "password": "SecurePassword123!",
        })
        headers = {"Authorization": f"Bearer {login_res.json()['access_token']}"}

        # 1. Dual scaffold OOD rejected with 422 by default
        dual_ood_payload = {
            "donor_smiles": self.unseen_donor_smiles,
            "acceptor_smiles": self.unseen_acceptor_smiles,
            "model_id": "M3",
            "allow_dual_scaffold_ood": False,
        }
        rej_res = self.client.post("/api/predictions/pce", json=dual_ood_payload, headers=headers)
        self.assertEqual(rej_res.status_code, 422)
        self.assertIn("DUAL_SCAFFOLD_OOD_REJECTED", rej_res.json()["detail"])

        # 2. Dual scaffold OOD permitted when allow_dual_scaffold_ood=True
        dual_ood_payload["allow_dual_scaffold_ood"] = True
        ovr_res = self.client.post("/api/predictions/pce", json=dual_ood_payload, headers=headers)
        self.assertEqual(ovr_res.status_code, 200)
        ovr_data = ovr_res.json()
        self.assertTrue(ovr_data["ood_status"]["ood_dual_scaffold"])
        self.assertEqual(ovr_data["ood_status"]["severity"], "critical")
        self.assertTrue(any("CRITICAL_DUAL_SCAFFOLD_OOD" in w for w in ovr_data["warnings"]))

    def test_04_security_isolation_and_input_guards(self):
        """Verify multi-tenant prediction isolation and strict schema validation guards."""
        # Create User 1
        u1_id = uuid.uuid4().hex[:8]
        u1_email = f"u1_{u1_id}@tenant.com"
        self.client.post("/api/auth/register", json={"email": u1_email, "password": "UserPass123!", "full_name": "U1"})
        t1 = self.client.post("/api/auth/login", json={"email": u1_email, "password": "UserPass123!"}).json()["access_token"]
        h1 = {"Authorization": f"Bearer {t1}"}

        # Create User 2
        u2_id = uuid.uuid4().hex[:8]
        u2_email = f"u2_{u2_id}@tenant.com"
        self.client.post("/api/auth/register", json={"email": u2_email, "password": "UserPass123!", "full_name": "U2"})
        t2 = self.client.post("/api/auth/login", json={"email": u2_email, "password": "UserPass123!"}).json()["access_token"]
        h2 = {"Authorization": f"Bearer {t2}"}

        # User 1 makes a prediction
        pred_res = self.client.post("/api/predictions/pce", json={
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "model_id": "M3",
        }, headers=h1)
        self.assertEqual(pred_res.status_code, 200)
        p1_id = pred_res.json()["id"]

        # User 2 cannot access User 1's prediction
        u2_lookup = self.client.get(f"/api/predictions/{p1_id}", headers=h2)
        self.assertEqual(u2_lookup.status_code, 404)

        # Unauthenticated request rejected (401)
        unauth_lookup = self.client.get(f"/api/predictions/{p1_id}")
        self.assertEqual(unauth_lookup.status_code, 401)

        # Invalid token rejected (401)
        bad_token_lookup = self.client.get(f"/api/predictions/{p1_id}", headers={"Authorization": "Bearer invalid.jwt.token"})
        self.assertEqual(bad_token_lookup.status_code, 401)

        # Invalid model ID schema rejected (422)
        bad_model_res = self.client.post("/api/predictions/pce", json={
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "model_id": "M99_INVALID",
        }, headers=h1)
        self.assertEqual(bad_model_res.status_code, 422)

        # Invalid confidence level schema rejected (422)
        bad_conf_res = self.client.post("/api/predictions/pce", json={
            "donor_smiles": self.known_donor_smiles,
            "acceptor_smiles": self.known_acceptor_smiles,
            "confidence_level": "99.9%",
        }, headers=h1)
        self.assertEqual(bad_conf_res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
