# OPV-PredictNet — REST API Contract Specification

**Version**: 1.0.0 (Phase 1 Specification)  
**Base URL**: `http://localhost:8000/api`  
**Data Format**: JSON (`application/json`) / Multipart Form (`multipart/form-data` for file uploads)  
**Authentication Scheme**: HTTP Bearer Token (JWT, RFC 7519)  

---

## 1. Implementation Status Summary

> **Important Notice on Phase Implementation Status**:  
> In accordance with Phase 3 project milestones, **`/health`, `/auth/register`, `/auth/login`, `/auth/me`, `/predictions/analyze`, `/predictions`, and `/predictions/{id}` are currently ACTIVE and fully implemented**.  
> In strict accordance with scientific integrity guidelines, when no validated empirical model artifact exists in `ml/models/saved_models/`, `/predictions` returns an explicit `model_unavailable` payload with real RDKit descriptors and scientific limitations, rather than fabricating fake figures of merit.
> `/chatbot/*` remains formally specified and contracted for Phase 4.

| Endpoint | Method | Status | Implemented Phase | Auth Required | Description |
|---|---|---|---|---|---|
| `/health` | `GET` | **ACTIVE** | Phase 1 | No | System health check and environment status |
| `/auth/register` | `POST` | **ACTIVE** | Phase 2 | No | User registration with Argon2 password hashing |
| `/auth/login` | `POST` | **ACTIVE** | Phase 2 | No | User authentication and JWT access token issuance |
| `/auth/me` | `GET` | **ACTIVE** | Phase 2 | Yes (Bearer) | Retrieve authenticated user profile |
| `/predictions/analyze` | `POST` | **ACTIVE** | Phase 3 | No | Real RDKit SMILES validation and descriptor calculation |
| `/predictions` | `POST` | **ACTIVE** | Phase 3 | Yes (Bearer) | Run prediction pipeline (genuine inference or model_unavailable) |
| `/predictions` | `GET` | **ACTIVE** | Phase 3 | Yes (Bearer) | Retrieve paginated prediction history for current user |
| `/predictions/{id}`| `GET` | **ACTIVE** | Phase 3 | Yes (Bearer) | Retrieve single detailed prediction record (isolated per user) |
| `/chatbot/message` | `POST` | CONTRACTED | Phase 4 | No / Optional | Query local offline knowledge chatbot |

---

## 2. Standard Response & Error Envelopes

### 2.1 Standard Error Envelope
All error responses adhere to the standard FastAPI/Pydantic structure:

```json
{
  "detail": [
    {
      "loc": ["body", "donor_smiles"],
      "msg": "Invalid chemical SMILES: unparseable valence on atom C at position 4",
      "type": "value_error.smiles"
    }
  ]
}
```

For authentication or permission errors:
```json
{
  "detail": "Could not validate credentials"
}
```

---

## 3. Detailed Endpoint Contracts

### 3.1 GET `/health`
Check the operational status of the backend API service and verify CORS configuration.

- **Authentication**: None
- **Request Parameters**: None
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "service": "OPV-PredictNet Backend",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-24T05:22:00.000000+00:00"
}
```

---

### 3.2 POST `/auth/register`
Register a new scientific user account. Passwords are subjected to Argon2 hashing prior to storage.

- **Authentication**: None
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "email": "researcher@laboratory.org",
  "password": "StrongPassword123!",
  "full_name": "Dr. Eleanor Vance",
  "institution": "Institute of Organic Electronics"
}
```
- **Validation Constraints**:
  - `email`: Valid RFC 5322 email string.
  - `password`: Minimum 8 characters, containing uppercase, lowercase, and numeric digits.
- **Response**: `201 Created`
```json
{
  "id": "usr_9b1deb4d3b7d4e8",
  "email": "researcher@laboratory.org",
  "full_name": "Dr. Eleanor Vance",
  "institution": "Institute of Organic Electronics",
  "created_at": "2026-11-24T05:30:00.000000+00:00"
}
```
- **Error Codes**:
  - `400 Bad Request`: Email already registered.
  - `422 Unprocessable Entity`: Validation failure on email or password rules.

---

### 3.3 POST `/auth/login`
Authenticate existing user credentials and receive a signed JWT bearer token.

- **Authentication**: None
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "email": "researcher@laboratory.org",
  "password": "StrongPassword123!"
}
```
- **Response**: `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```
- **Error Codes**:
  - `401 Unauthorized`: Invalid email or password.

---

### 3.4 GET `/auth/me`
Retrieve user profile and account details for the currently authenticated session.

- **Authentication**: `Bearer <access_token>`
- **Response**: `200 OK`
```json
{
  "id": "usr_9b1deb4d3b7d4e8",
  "email": "researcher@laboratory.org",
  "full_name": "Dr. Eleanor Vance",
  "institution": "Institute of Organic Electronics",
  "is_active": true,
  "created_at": "2026-11-24T05:30:00.000000+00:00"
}
```
- **Error Codes**:
  - `401 Unauthorized`: Missing, expired, or malformed JWT token.

---

### 3.5 Prediction Pipeline Specification & Unified Normalization

The prediction subsystem accepts candidates through three distinct entry points:
1. **Direct SMILES**: Direct text string input of donor and acceptor canonical SMILES.
2. **Molecular File Upload**: Upload of `.sdf`, `.mol`, or `.pdb` coordinate files.
3. **Molecular Library Selection**: Choosing pre-curated benchmark donor and acceptor IDs from the SQLite library (e.g., `PM6`, `PTB7-Th`, `Y6`, `ITIC`, `PC71BM`).

#### Unified Internal Prediction Schema
Regardless of entry point, all requests are translated by the backend adapter into the following normalized prediction request:

```json
{
  "donor": {
    "name": "PM6",
    "canonical_smiles": "C1=C(C2=CC=C(S2)C3=C4C(=O)C5=C(C(=O)N4C(CC)CCCC)SC(=C5)C6=CC=C(S6)..."
  },
  "acceptor": {
    "name": "Y6",
    "canonical_smiles": "CC(CC)CCCCN1C2=C(C3=C1C4=C(S3)C5=C(C=C4)C6=C(C5)C7=C(S6)C8=C(N7CC(CC)CCCC)..."
  },
  "device_parameters": {
    "blend_ratio": "1:1.2",
    "solvent": "chloroform",
    "additive": "1,8-diiodooctane",
    "additive_concentration_vol_pct": 0.5,
    "thermal_annealing_temp_c": 100.0
  }
}
```

---

### 3.6 POST `/predictions`
Submit candidate materials and optional process conditions for performance prediction.

- **Authentication**: `Bearer <access_token>`
- **Request Headers**: `Content-Type: application/json`
- **Request Body (Direct SMILES)**:
```json
{
  "input_method": "smiles",
  "donor_name": "PBDB-T-2F (PM6)",
  "donor_smiles": "C1=C(C2=CC=C(S2)...",
  "acceptor_name": "BTP-4F (Y6)",
  "acceptor_smiles": "CC(CC)CCCCN1...",
  "device_parameters": {
    "blend_ratio": "1:1.2",
    "solvent": "chloroform",
    "additive": "1,8-diiodooctane"
  }
}
```

- **Request Body (Library Selection)**:
```json
{
  "input_method": "library_selection",
  "donor_library_id": "mol_donor_pm6",
  "acceptor_library_id": "mol_acc_y6",
  "device_parameters": {
    "blend_ratio": "1:1.2"
  }
}
```

- **Response**: `201 Created`
```json
{
  "id": "pred_a8e23f9901b",
  "user_id": "usr_9b1deb4d3b7d4e8",
  "input_method": "smiles",
  "donor": {
    "name": "PBDB-T-2F (PM6)",
    "smiles": "C1=C(C2=CC=C(S2)...",
    "molecular_weight": 1137.6,
    "tpsa": 65.2
  },
  "acceptor": {
    "name": "BTP-4F (Y6)",
    "smiles": "CC(CC)CCCCN1...",
    "molecular_weight": 1451.8,
    "tpsa": 78.4
  },
  "device_parameters": {
    "blend_ratio": "1:1.2",
    "solvent": "chloroform",
    "additive": "1,8-diiodooctane"
  },
  "predictions": {
    "pce_percent": 15.82,
    "voc_volts": 0.84,
    "jsc_ma_cm2": 25.10,
    "fill_factor_percent": 75.0,
    "prediction_intervals": {
      "pce_95_ci": [14.60, 17.04]
    }
  },
  "model_metadata": {
    "model_name": "OPV-RF-Ensemble",
    "model_version": "v1.0.0",
    "algorithm": "RandomForestRegressor"
  },
  "created_at": "2026-11-24T05:35:10.000000+00:00"
}
```
- **Error Codes**:
  - `400 Bad Request`: Incompatible blend parameters.
  - `422 Unprocessable Entity`: Invalid chemical structure or SMILES parse failure.

---

### 3.7 GET `/predictions`
Retrieve a paginated history of predictions submitted by the authenticated user.

- **Authentication**: `Bearer <access_token>`
- **Query Parameters**:
  - `limit` (int, default=20, max=100)
  - `offset` (int, default=0)
  - `sort_by` (string, default="created_at", options=["created_at", "pce"])
  - `order` (string, default="desc", options=["asc", "desc"])
- **Response**: `200 OK`
```json
{
  "total": 1,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "id": "pred_a8e23f9901b",
      "donor_name": "PBDB-T-2F (PM6)",
      "acceptor_name": "BTP-4F (Y6)",
      "predicted_pce": 15.82,
      "predicted_voc": 0.84,
      "predicted_jsc": 25.10,
      "predicted_ff": 75.0,
      "model_version": "v1.0.0",
      "created_at": "2026-11-24T05:35:10.000000+00:00"
    }
  ]
}
```

---

### 3.8 GET `/predictions/{prediction_id}`
Retrieve complete technical details for a specific historical prediction.

- **Authentication**: `Bearer <access_token>`
- **Path Parameters**: `prediction_id` (string)
- **Response**: `200 OK` (Full schema identical to `POST /predictions` response)
- **Error Codes**:
  - `404 Not Found`: Prediction does not exist or does not belong to the user.

---

### 3.9 POST `/chatbot/message`
Query the offline knowledge-based assistant for scientific questions regarding organic photovoltaics, feature calculations, or system operations.

- **Authentication**: Optional
- **Request Body**:
```json
{
  "query": "What causes the open circuit voltage in a PM6:Y6 blend to be around 0.84 V?",
  "conversation_context_id": "ctx_77a98b12"
}
```
- **Response**: `200 OK`
```json
{
  "response": "The open-circuit voltage (Voc) in an OPV device is fundamentally governed by the energetic offset between the donor HOMO level and the acceptor LUMO level: Voc ≈ (1/q)(E_LUMO_Acceptor - E_HOMO_Donor) - ΔV_loss. For PM6 (HOMO ≈ -5.45 eV) and Y6 (LUMO ≈ -4.10 eV), the effective bandgap is ~1.35 eV. Radiative and non-radiative recombination losses (typically 0.5 - 0.6 V in modern non-fullerene acceptor blends) reduce the achievable Voc to approximately 0.83 - 0.86 V.",
  "sources": [
    "docs/scientific_basis/opv_physics_and_metrics.md"
  ],
  "tokens_used": 142
}
```
- **Error Codes**:
  - `400 Bad Request`: Empty query string.
