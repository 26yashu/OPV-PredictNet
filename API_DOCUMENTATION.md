# OPV-PredictNet — REST API Specification & Reference Manual

**API Version:** `v1` (`0.1.0`)  
**Base Protocol:** HTTPS / HTTP REST  
**Data Format:** JSON (`application/json`)  
**Standard Authentication:** HTTP Bearer Token (JWT via HMAC-SHA256)  

---

## 1. Global Endpoints & Operational Health

### 1.1 Health Check & Governance Status
Retrieves operational service health, model artifact loading status, and immutable governance flags.

- **Route:** `GET /api/health`
- **Authentication:** None (Public)
- **Response (HTTP 200 OK):**
```json
{
  "status": "healthy",
  "service": "OPV-PredictNet Backend",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-25T07:00:00Z",
  "models_ready": {
    "M0": true,
    "M1": true,
    "M2A": true,
    "M3": true
  },
  "scientific_governance": {
    "research_estimate_only": true,
    "validated_for_production": false
  }
}
```

---

## 2. Authentication & User Management

### 2.1 User Registration
Registers a new researcher account using Argon2id password hashing.

- **Route:** `POST /api/auth/register`
- **Authentication:** None (Public)
- **Request Body:**
```json
{
  "email": "scientist@materials-institute.org",
  "password": "SecurePassword123!",
  "full_name": "Dr. Elena Rostova",
  "institution": "Polytechnic Institute of Solar Energy"
}
```
- **Validation Rules:**
  - `email`: Normalized lowercase; valid RFC 5322 email syntax.
  - `password`: Minimum 8 characters; must contain at least one uppercase letter, one lowercase letter, and one numeric digit.
  - `full_name`: 1–100 characters; non-empty trimmed string.
  - `institution`: Optional; max 150 characters.
- **Response (HTTP 201 Created):**
```json
{
  "id": "usr_9b1a4f2e8c3d",
  "email": "scientist@materials-institute.org",
  "full_name": "Dr. Elena Rostova",
  "institution": "Polytechnic Institute of Solar Energy",
  "created_at": "2026-11-25T07:05:00Z"
}
```
- **Error Codes:** `400 Bad Request` (Email already registered), `422 Unprocessable Content` (Validation failure).

### 2.2 User Login & Token Acquisition
Authenticates credentials and returns a signed JSON Web Token (JWT).

- **Route:** `POST /api/auth/login`
- **Authentication:** None (Public)
- **Request Body:**
```json
{
  "email": "scientist@materials-institute.org",
  "password": "SecurePassword123!"
}
```
- **Response (HTTP 200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```
- **Error Codes:** `401 Unauthorized` (Invalid email or password).

### 2.3 Current User Profile
Retrieves the profile of the currently authenticated researcher.

- **Route:** `GET /api/auth/me`
- **Authentication:** Required (`Authorization: Bearer <access_token>`)
- **Response (HTTP 200 OK):** Returns `UserResponse` object.
- **Error Codes:** `401 Unauthorized` (Missing, malformed, or expired token).

---

## 3. Molecular Analysis & Model Discovery

### 3.1 Model Catalog Discovery
Enumerates all packaged Phase 8.3/9 research models and their feature schemas.

- **Route:** `GET /api/predictions/models`
- **Authentication:** None (Public)
- **Response (HTTP 200 OK):**
```json
{
  "models": {
    "M0": {
      "version_tag": "phase9_M0",
      "algorithm": "RandomForestRegressor",
      "n_features": 22,
      "calibration_strategy": "Phase 8.5 Strategy C2 (N_cal=3863)",
      "research_estimate_only": true,
      "validated_for_production": false
    },
    "M1": {
      "version_tag": "phase9_M1",
      "algorithm": "RandomForestRegressor",
      "n_features": 50,
      "calibration_strategy": "Phase 8.5 Strategy C2 (N_cal=3863)",
      "research_estimate_only": true,
      "validated_for_production": false
    },
    "M2A": {
      "version_tag": "phase9_M2A",
      "algorithm": "RandomForestRegressor",
      "n_features": 37,
      "calibration_strategy": "Phase 8.5 Strategy C2 (N_cal=3863)",
      "research_estimate_only": true,
      "validated_for_production": false
    },
    "M3": {
      "version_tag": "phase9_M3",
      "algorithm": "HistGradientBoostingRegressor",
      "n_features": 44,
      "calibration_strategy": "Phase 8.5 Strategy C2 (N_cal=3863)",
      "research_estimate_only": true,
      "validated_for_production": false
    }
  },
  "n_donor_scaffolds": 430,
  "n_acceptor_scaffolds": 172,
  "research_estimate_only": true,
  "validated_for_production": false
}
```

### 3.2 Chemoinformatics Molecular Analysis
Parses donor and acceptor SMILES via RDKit, calculating 2D physicochemical descriptors and blend contrast metrics.

- **Route:** `POST /api/predictions/analyze`
- **Authentication:** None (Public)
- **Request Body:**
```json
{
  "donor_smiles": "CCCCCCc1cccs1",
  "acceptor_smiles": "B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21",
  "donor_name": "P3HT Subunit",
  "acceptor_name": "SubPc"
}
```
- **Response (HTTP 200 OK):**
```json
{
  "donor": {
    "name": "P3HT Subunit",
    "role": "donor",
    "canonical_smiles": "CCCCCCc1cccs1",
    "formula": "C10H16S",
    "molecular_weight": 168.3,
    "exact_mass": 168.0973,
    "logp": 3.74,
    "tpsa": 0.0,
    "hbd": 0,
    "hba": 0,
    "rotatable_bonds": 5,
    "aromatic_rings": 1,
    "fraction_csp3": 0.6,
    "heavy_atom_count": 11,
    "ring_count": 1
  },
  "acceptor": {
    "name": "SubPc",
    "role": "acceptor",
    "canonical_smiles": "...",
    "molecular_weight": 430.7,
    "logp": 5.12,
    "tpsa": 41.5
  },
  "blend_features": {
    "molecular_weight_diff": 262.4,
    "logp_diff": 1.38,
    "tpsa_total": 41.5,
    "aromatic_rings_total": 5,
    "heavy_atom_count_total": 43
  }
}
```
- **Error Codes:** `422 Unprocessable Content` (Invalid SMILES syntax, BigSMILES, or multi-component mixtures).

---

## 4. PCE Prediction & Uncertainty Estimation

### 4.1 Phase 9 Authoritative Prediction Endpoint
Executes inference using the Unified Feature Pipeline, computes conformal prediction intervals, performs Bemis-Murcko scaffold OOD checks, and logs history.

- **Route:** `POST /api/predictions/pce`
- **Authentication:** Required (`Authorization: Bearer <access_token>`)
- **Request Body:**
```json
{
  "donor_smiles": "CCCCCCc1cccs1",
  "acceptor_smiles": "B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21",
  "donor_name": "P3HT Subunit",
  "acceptor_name": "SubPc",
  "input_method": "smiles",
  "d_a_ratio": "1:1",
  "active_layer_thickness": "110nm",
  "annealing_temp": 120.0,
  "additive_canonical": "DIO",
  "additive_ratio": "2.5%",
  "solvent_canonical": "chlorobenzene",
  "device_type": "conventional",
  "etl_canonical": "ZnO",
  "htl_canonical": "PEDOT:PSS",
  "model_id": "M3",
  "confidence_level": "90%",
  "allow_dual_scaffold_ood": false
}
```
- **Request Constraints:**
  - `model_id`: Must be one of `"M0"`, `"M1"`, `"M2A"`, `"M3"`. Default `"M3"`.
  - `confidence_level`: Must be one of `"80%"`, `"90%"`, `"95%"`. Default `"90%"`.
  - `allow_dual_scaffold_ood`: Boolean flag. Default `false`.
- **Response (HTTP 200 OK):**
```json
{
  "id": "pred_a7f92b3c4d1e",
  "created_at": "2026-11-25T07:10:00Z",
  "model_id": "M3",
  "model_version": "phase9_M3",
  "algorithm": "HistGradientBoostingRegressor",
  "point_prediction": {
    "predicted_pce": 4.82,
    "unit": "%"
  },
  "interval_estimate": {
    "prediction_interval_low": 3.14,
    "prediction_interval_high": 6.50,
    "confidence_level": "90%",
    "conformal_quantile_q": 1.832,
    "interval_width": 3.36,
    "calibration_strategy": "Phase 8.5 Strategy C2 (N_cal=3863)"
  },
  "ood_status": {
    "ood_donor_scaffold": false,
    "ood_acceptor_scaffold": false,
    "ood_dual_scaffold": false,
    "donor_scaffold_known": true,
    "acceptor_scaffold_known": true,
    "donor_scaffold_id": "c1cccs1",
    "acceptor_scaffold_id": "...",
    "processing_ood_flags": [],
    "severity": "normal"
  },
  "scientific_support": {
    "research_estimate_only": true,
    "validated_for_production": false,
    "support_level": "in_scaffold_research_estimate",
    "governance_role": "Phase 8.3 Champion (Native Missingness)",
    "benchmark_metrics": {
      "r2": 0.584,
      "mae": 1.341,
      "rmse": 1.842,
      "evaluation_partition": "Phase 8.5 Held-Out Test"
    },
    "limitations": "Research estimate only; not validated for production device engineering..."
  },
  "warnings": [],
  "research_estimate_only": true,
  "validated_for_production": false
}
```
- **Error Codes:**
  - `401 Unauthorized`: Token missing or invalid.
  - `422 Unprocessable Content`:
    - Chemical syntax error or polymer wildcards (`*`, `{}`).
    - Physical parameter out of bounds (negative thickness or temperature).
    - `DUAL_SCAFFOLD_OOD_REJECTED`: Both donor and acceptor scaffolds are unseen, and `allow_dual_scaffold_ood` was not set to `true`.

---

## 5. Prediction History & User Isolation

### 5.1 List User Prediction History
Retrieves paginated prediction records belonging strictly to the authenticated user.

- **Route:** `GET /api/predictions?skip=0&limit=50`
- **Authentication:** Required (`Authorization: Bearer <access_token>`)
- **Response (HTTP 200 OK):**
```json
{
  "total": 12,
  "items": [
    {
      "id": "pred_a7f92b3c4d1e",
      "input_method": "smiles",
      "donor_name": "P3HT Subunit",
      "donor_smiles": "CCCCCCc1cccs1",
      "acceptor_name": "SubPc",
      "acceptor_smiles": "...",
      "predicted_pce": 4.82,
      "model_version": "phase9_M3",
      "uncertainty_json": "{\"interval_estimate\": {...}, \"ood_status\": {...}}",
      "created_at": "2026-11-25T07:10:00Z"
    }
  ]
}
```

### 5.2 Retrieve Single Prediction
- **Route:** `GET /api/predictions/{id}`
- **Authentication:** Required (`Authorization: Bearer <access_token>`)
- **Response (HTTP 200 OK):** Returns `PredictionHistoryItem`.
- **Error Codes:** `404 Not Found` (Record does not exist or belongs to another user).
