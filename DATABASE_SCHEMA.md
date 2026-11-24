# OPV-PredictNet — SQLite Database Schema Specification

**Database Engine**: SQLite 3 (ACID compliant, zero server process)  
**ORM Target**: SQLAlchemy 2.0 (`backend/app/models/`)  
**Specification Status**: Phase 1 Schema Design  

---

## 1. Schema Design Principles

1. **Local Efficiency**: Designed for high performance and low memory footprint on Windows desktop systems with 8 GB RAM.
2. **Data Integrity & Isolation**: Foreign key constraints enforce relational integrity; user-specific records are strictly partitioned by `user_id`.
3. **Security by Design**: User passwords are encrypted using **Argon2** hashing. Storing plaintext passwords or unsalted cryptographic hashes (e.g., raw MD5, SHA-1, SHA-256) is strictly prohibited.
4. **Flexible Process Metadata**: Device architecture and fabrication conditions vary widely across literature. To avoid sparse relational columns, device and process parameters are stored in standardized, indexed JSON text fields.

---

## 2. Entity-Relationship Diagram

```
+------------------------------------+
|               users                |
+------------------------------------+
| PK  id (TEXT)                      |
|     email (TEXT, UNIQUE)           |
|     hashed_password (TEXT)         |
|     full_name (TEXT)               |
|     institution (TEXT)             |
|     is_active (BOOLEAN)            |
|     created_at (DATETIME)          |
|     updated_at (DATETIME)          |
+-----------------+------------------+
                  |
                  | 1:N
                  v
+-----------------+------------------+
|            predictions             |
+------------------------------------+
| PK  id (TEXT)                      |
| FK  user_id (TEXT)                 |
|     input_method (TEXT)            |
|     donor_name (TEXT)              |
|     donor_smiles (TEXT)            |
|     acceptor_name (TEXT)           |
|     acceptor_smiles (TEXT)         |
|     device_parameters_json (TEXT)  |
|     predicted_pce (REAL)           |
|     predicted_voc (REAL)           |
|     predicted_jsc (REAL)           |
|     predicted_ff (REAL)            |
|     uncertainty_json (TEXT)        |
| FK  model_version (TEXT)           |
|     created_at (DATETIME)          |
|     updated_at (DATETIME)          |
+-----------------+------------------+
                  |
                  | N:1
                  v
+-----------------+------------------+
|           model_versions           |
+------------------------------------+
| PK  version_tag (TEXT)             |
|     model_name (TEXT)              |
|     algorithm (TEXT)               |
|     target_metrics (TEXT)          |
|     r2_score (REAL)                |
|     mae_score (REAL)               |
|     rmse_score (REAL)              |
|     hyperparameters_json (TEXT)    |
|     artifact_path (TEXT)           |
|     is_active (BOOLEAN)            |
|     created_at (DATETIME)          |
+------------------------------------+

+------------------------------------+
|         molecular_library          |
+------------------------------------+
| PK  id (TEXT)                      |
|     name (TEXT, UNIQUE)            |
|     role (TEXT) ['donor','acceptor']|
|     canonical_smiles (TEXT)        |
|     inchi_key (TEXT, UNIQUE)       |
|     molecular_weight (REAL)        |
|     tpsa (REAL)                    |
|     homo_ev (REAL)                 |
|     lumo_ev (REAL)                 |
|     bandgap_ev (REAL)              |
|     literature_reference (TEXT)    |
|     created_at (DATETIME)          |
+------------------------------------+
```

---

## 3. SQL Data Definition Language (DDL)

```sql
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Table: users
-- Stores registered researcher credentials and profile information.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    hashed_password TEXT NOT NULL,
    full_name TEXT NOT NULL,
    institution TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- -----------------------------------------------------------------------------
-- Table: model_versions
-- Tracks trained machine learning model iterations, algorithms, and validation metrics.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    version_tag TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    target_metrics TEXT NOT NULL,  -- e.g., '["PCE", "Voc", "Jsc", "FF"]'
    r2_score REAL,
    mae_score REAL,
    rmse_score REAL,
    hyperparameters_json TEXT,
    artifact_path TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Table: molecular_library
-- Curated benchmark molecular entities (donors and non-fullerene/fullerene acceptors).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS molecular_library (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK(role IN ('donor', 'acceptor', 'additive')),
    canonical_smiles TEXT NOT NULL,
    inchi_key TEXT NOT NULL UNIQUE,
    molecular_weight REAL NOT NULL,
    tpsa REAL,
    homo_ev REAL,
    lumo_ev REAL,
    bandgap_ev REAL,
    literature_reference TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_molecular_library_role ON molecular_library(role);
CREATE INDEX IF NOT EXISTS idx_molecular_library_inchi ON molecular_library(inchi_key);

-- -----------------------------------------------------------------------------
-- Table: predictions
-- Records all prediction runs submitted by users, linking input pairs with outputs.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    input_method TEXT NOT NULL CHECK(input_method IN ('smiles', 'file_upload', 'library_selection')),
    donor_name TEXT NOT NULL,
    donor_smiles TEXT NOT NULL,
    acceptor_name TEXT NOT NULL,
    acceptor_smiles TEXT NOT NULL,
    device_parameters_json TEXT,     -- e.g., '{"blend_ratio": "1:1.2", "solvent": "chloroform"}'
    predicted_pce REAL NOT NULL,
    predicted_voc REAL,
    predicted_jsc REAL,
    predicted_ff REAL,
    uncertainty_json TEXT,           -- e.g., '{"pce_ci_95": [14.6, 17.0]}'
    model_version TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (model_version) REFERENCES model_versions(version_tag) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
```

---

## 4. Column Details & Validation Rules

### 4.1 `users`
- `id`: Unique identifier prefixing `usr_` followed by UUIDv4 or nano-hash.
- `email`: Normalized lowercase unique email.
- `hashed_password`: Argon2id hash with parameters ($m=65536, t=3, p=4$). Plaintext passwords must never touch the database.
- `is_active`: 1 for active accounts, 0 for deactivated accounts.

### 4.2 `predictions`
- `id`: Unique identifier prefixing `pred_`.
- `user_id`: Foreign key enforcing user ownership.
- `input_method`: `smiles` (manual string), `file_upload` (parsed from `.sdf`), or `library_selection` (selected from benchmark library).
- `donor_smiles` / `acceptor_smiles`: Sanitized and canonicalized via RDKit.
- `device_parameters_json`: JSON-encoded device stack details (solvent, additives, annealing temperature).
- `predicted_pce`: Power Conversion Efficiency percentage (e.g., 15.82).
- `predicted_voc`: Open-circuit voltage in Volts (e.g., 0.84).
- `predicted_jsc`: Short-circuit current density in mA/cm² (e.g., 25.10).
- `predicted_ff`: Fill factor percentage (e.g., 75.0).

### 4.3 `molecular_library`
- Standard benchmark donors (`PM6`, `PTB7-Th`, `D18`, `P3HT`) and acceptors (`Y6`, `ITIC`, `PC71BM`, `PC61BM`, `L8-BO`).
- InChIKey ensures cryptographic uniqueness independent of SMILES branch ordering.
