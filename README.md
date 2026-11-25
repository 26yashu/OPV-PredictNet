# OPV-PredictNet

**AI-Driven Molecular Screening, Feature Generation, and Uncertainty-Quantified Performance Optimization for Organic Photovoltaics**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178c6.svg)](https://www.typescriptlang.org/)
[![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-brightgreen.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.4704870-blue.svg)](https://doi.org/10.5281/zenodo.4704870)

---

## 1. Project Overview & Scientific Tenets

**OPV-PredictNet** is an open-source, full-stack scientific application engineered for computational chemists and materials scientists working on bulk-heterojunction (BHJ) organic photovoltaic (OPV) solar cells.

The platform combines chemoinformatics (RDKit) and machine learning (Scikit-Learn) with rigorous conformal uncertainty estimation to provide:
1. **Interactive Chemoinformatics Screening:** Real-time 2D molecular descriptor extraction (MW, LogP, TPSA, ring counts, rotatable bonds) and blend contrast calculation.
2. **Unified Feature Generation:** Standardized, deterministic feature transformation pipelines across 4 research models (M0, M1, M2A, M3).
3. **Conformal Uncertainty Quantification:** Distribution-free prediction intervals calibrated via Phase 8.5 Strategy C2 ($N_{\text{cal}} = 3,863$) across nominal 80%, 90%, and 95% confidence tiers.
4. **Scaffold Generalization & Out-of-Domain (OOD) Governance:** Dynamic Bemis-Murcko generic scaffold extraction with automated detection of novel scaffolds and default rejection of unverified dual-scaffold extrapolation ($R^2 \approx 0.08$).
5. **Research Governance:** Strict enforcement of non-production caveats (`research_estimate_only = true` and `validated_for_production = false`).

### Core Scientific Principles
- **Molecular Candidate vs. Macroscopic Device:** An isolated chemical structure alone does not dictate photovoltaic performance. Efficiency ($PCE$) is an emergent macroscopic property resulting from donor:acceptor blend nanomorphology, phase separation, domain crystallite orientation, transport interlayers, and environmental fabrication conditions.
- **Scientific Honesty & Epistemic Humility:** Conformal prediction intervals communicate genuine statistical uncertainty. We do not claim universal coverage or experimentally certified performance.
- **Zero Paid Dependencies:** Runs completely locally on consumer hardware (Windows 10/11, macOS, Linux) with zero external commercial API dependencies.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                        Frontend (React 19 + TypeScript + Vite)          |
|  - Cinematic Scientific Landing Page                                    |
|  - Interactive BHJ Morphology Canvas & Band Energy Diagrams             |
|  - Molecular Screening Dashboard with Quick-Load Preset Libraries      |
|  - Real-time Conformal Uncertainty Gauge & OOD Alert Banners            |
+------------------------------------+------------------------------------+
                                     | HTTP REST (JSON / Bearer JWT)
                                     v
+-------------------------------------------------------------------------+
|                        Backend Service (FastAPI / ASGI)                 |
|  - Argon2id Password Hashing & Stateless JWT Authentication             |
|  - Chemoinformatics Descriptor Engine (RDKit)                           |
|  - Unified Feature Pipeline (Feature schemas: 22, 50, 37, 44 features)  |
|  - Bemis-Murcko Generic Scaffold Matcher (430 donor, 172 acceptor)      |
|  - Strategy C2 Locally Scaled Conformal Predictor                       |
|  - Multi-Tenant SQLite Database Persistence (ACID compliant)            |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                        Packaged Research Model Suite                    |
|  - M0: Molecular-Only Baseline (RandomForest, 22 features)              |
|  - M1: One-Hot Categorical Processing Model (RandomForest, 50 features) |
|  - M2A: Continuous Processing Model (RandomForest, 37 features)         |
|  - M3: Native-Missingness Champion Model (HistGradientBoosting, 44 feat)|
+-------------------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons, HTML5 Canvas.
- **Backend:** Python 3.11, FastAPI, Pydantic v2 (Settings & Schemas), SQLAlchemy 2.0.
- **Chemoinformatics & ML:** RDKit (2023.9+), Scikit-Learn (1.4+), NumPy, Pandas, Joblib.
- **Security & Database:** SQLite 3 (PRAGMA foreign_keys=ON), Argon2id (`argon2-cffi`), python-jose (JWT).
- **Deployment:** Docker, Docker Compose, Nginx Alpine, PowerShell / Bash demonstration runners.

---

## 4. Dataset Source and CC BY 4.0 Attribution

OPV-PredictNet models are trained and calibrated on experimental literature records curated from **OPVdb**:

- **Dataset Identifier:** OPVdb (Organic Photovoltaic Database)
- **Persistent Identifier (DOI):** [10.5281/zenodo.4704870](https://doi.org/10.5281/zenodo.4704870)
- **Licensing:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Attribution Notice:** 
  > The underlying experimental dataset used for benchmark evaluation, scaffold clustering, and conformal calibration is sourced from OPVdb (Zenodo DOI: `10.5281/zenodo.4704870`), licensed under CC BY 4.0. In accordance with CC BY 4.0, we acknowledge the dataset creators.
  > Modifications performed: Canonical SMILES normalization via RDKit, removal of unphysical entries, extraction of Bemis-Murcko generic scaffolds, deduplication of molecular donor:acceptor pairs, extraction of quantitative fabrication records, and construction of disjoint train/calibration/test partitions for structure-generalization benchmarking.

---

## 5. Quick Start Instructions

### 5.1 Automated Demonstration Launcher (Recommended)

#### Windows (PowerShell):
```powershell
.\scripts\run_demo.ps1
```

#### Linux / macOS (Bash):
```bash
chmod +x scripts/run_demo.sh
./scripts/run_demo.sh
```

The script verifies prerequisites, packages models if absent, starts the FastAPI service (`http://127.0.0.1:8000`), starts the Vite frontend (`http://localhost:5173`), and displays access links.

---

### 5.2 Containerized Docker Deployment

To launch the full reproducible containerized stack:
```bash
# Configure production environment
cp .env.production.example .env

# Start backend and frontend containers
docker compose up -d --build
```
Access points:
- **Web Dashboard:** `http://localhost:3000`
- **Backend API & OpenAPI Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/api/health`

---

### 5.3 Manual Setup

#### Backend Setup:
```bash
# Initialize and activate Python 3.11 venv
python -m venv backend/.venv
# On Windows:
backend\.venv\Scripts\activate.ps1
# On Linux/macOS:
source backend/.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Run backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Frontend Setup:
```bash
cd frontend
npm install
npm run dev
```

---

## 6. Approved Research Models

| Model ID | Input Features | Algorithm | Primary Governance Role |
| :---: | :---: | :---: | :--- |
| **M0** | 22 | `RandomForestRegressor` | Molecular-only baseline; screening without processing records. |
| **M1** | 50 | `RandomForestRegressor` | One-hot categorical processing model for familiar device stacks. |
| **M2A** | 37 | `RandomForestRegressor` | Continuous processing model for numeric fabrication parameter exploration. |
| **M3** | 44 | `HistGradientBoostingRegressor` | **Default Primary Champion**; natively handles missing fabrication metadata. |

---

## 7. Model Limitations and Governance

1. **Research Estimate Only:** All outputs carry `research_estimate_only = true` and `validated_for_production = false`. Predictions are not physical warranties.
2. **Scaffold Generalization Limits:**
   - **Unseen Donor Scaffold:** Evaluated as exploratory candidate (`severity: warning`).
   - **Unseen Acceptor Scaffold:** Evaluated as exploratory candidate; empirical coverage drops to ~45% at 90% nominal confidence for high-efficiency acceptors.
   - **Dual-Scaffold OOD:** Jointly unseen donor and acceptor scaffolds result in predictive collapse ($R^2 \approx 0.08$) and are **strictly rejected with HTTP 422 by default**.
3. **Finite Repeat-Unit Representations:** Polymer donors are modeled as discrete monomeric or oligomeric repeat units; macroscale chain entanglements and molecular weight polydispersity are not captured in 2D SMILES descriptors.
4. **Literature Reporting Bias:** Datasets predominantly reflect successful published devices; devices with $PCE < 2\%$ are under-reported in scientific literature.

For complete scientific and mathematical documentation, consult [MODEL_LIMITATIONS.md](file:///c:/Users/DELL/OPV-PredictNet/MODEL_LIMITATIONS.md).

---

## 8. Verification and Testing

All **97 automated test cases** across backend, ML, and integration suites pass cleanly:

```bash
# Run backend unit tests
python -m unittest discover -s tests/backend -p "test_*.py" -v

# Run ML unit tests
python -m unittest discover -s tests/ml -p "test_*.py" -v

# Run end-to-end integration tests
python -m unittest discover -s tests/integration -p "test_*.py" -v

# Run frontend typecheck, build, and linter
cd frontend
npm run lint
npm run build
```

---

## 9. Comprehensive Documentation Index

- **[DEPLOYMENT_GUIDE.md](file:///c:/Users/DELL/OPV-PredictNet/DEPLOYMENT_GUIDE.md):** Local demonstration, Docker Compose, production security, and database backup/recovery.
- **[API_DOCUMENTATION.md](file:///c:/Users/DELL/OPV-PredictNet/API_DOCUMENTATION.md):** Full REST API reference with endpoints, request/response schemas, and error codes.
- **[MODEL_LIMITATIONS.md](file:///c:/Users/DELL/OPV-PredictNet/MODEL_LIMITATIONS.md):** Scientific governance, scaffold OOD policies, conformal calibration nuances, and finite polymer representation disclaimers.
- **[PHASE_12_DEPLOYMENT_REPORT.md](file:///c:/Users/DELL/OPV-PredictNet/PHASE_12_DEPLOYMENT_REPORT.md):** Detailed Phase 12 verification and deployment audit report.
