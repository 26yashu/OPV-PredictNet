# OPV-PredictNet

**AI-Driven Molecular Screening and Performance Optimization for Organic Photovoltaic Solar Cells**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178c6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

**OPV-PredictNet** is an open-source, full-stack scientific platform engineered to accelerate the discovery, virtual screening, and performance optimization of novel donor and non-fullerene acceptor materials for bulk-heterojunction (BHJ) organic photovoltaic (OPV) solar cells.

By integrating chemoinformatics (RDKit) and machine learning (Random Forest regression), OPV-PredictNet estimates device-level figures of merit—including **Power Conversion Efficiency ($PCE$)**, **Open-Circuit Voltage ($V_{OC}$)**, **Short-Circuit Current Density ($J_{SC}$)**, and **Fill Factor ($FF$)**—under standard test conditions (AM1.5G, 100 mW/cm², 25°C).

### Scientific Tenets
- **Molecular Candidate vs. Device Record**: An isolated molecule alone does not determine solar cell efficiency. Efficiency is an emergent macroscopic property of the donor–acceptor active layer blend, film morphology, and device architecture.
- **Rigor & Transparency**: Predictions represent statistical estimations with confidence intervals derived from empirical experimental datasets, avoiding fabricated claims of absolute quantum certainty.
- **Local & Free**: Operates efficiently on standard desktop hardware (Windows 10/11, 8 GB RAM) with zero paid API dependencies and an offline knowledge assistant.

---

## 2. System Architecture

```
+-------------------------------------------------------------+
|                      Frontend (Vite + React)                |
|  - Interactive 3D molecular viewer (3Dmol.js / Three.js)   |
|  - Real-time performance dashboards & charts (Recharts)     |
|  - Lightweight responsive UI (Tailwind CSS + shadcn/ui)     |
+------------------------------+------------------------------+
                               | REST (JSON / Bearer JWT)
                               v
+-------------------------------------------------------------+
|                      Backend (FastAPI)                      |
|  - REST routing & request normalization                     |
|  - Authentication (Argon2 + JWT tokens)                     |
|  - Relational persistence (SQLAlchemy 2.0 + SQLite)         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Scientific & ML Subsystems                    |
|  - Molecular informatics & SMILES validation (RDKit)        |
|  - Blend feature engineering & energy offsets (ΔHOMO/ΔLUMO) |
|  - Machine learning predictor (Random Forest via Joblib)    |
|  - Offline scientific knowledge chatbot                     |
+-------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Three.js, React Three Fiber, 3Dmol.js, Recharts.
- **Backend**: Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.0.
- **Database & Auth**: SQLite, Argon2 password hashing, JSON Web Tokens (JWT).
- **Scientific Computing**: RDKit, Pandas, NumPy, Scikit-learn, Joblib.
- **Chatbot**: Local Python-based knowledge chatbot (offline, no external paid API).
- **Target OS**: Windows 10/11 (8 GB RAM budget).

---

## 4. Setup Instructions (Windows 10/11)

### 4.1 Prerequisites
- **Python 3.11** installed and added to `PATH`.
- **Node.js** (v18+ LTS recommended) and **npm**.
- **Git** version control.

### 4.2 Repository Setup & Virtual Environment
Clone the repository and enter the project root:
```powershell
git clone https://github.com/your-username/OPV-PredictNet.git
cd OPV-PredictNet
```

Initialize the Python 3.11 virtual environment (if not already created):
```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

Initialize frontend dependencies:
```powershell
cd frontend
npm install
cd ..
```

Configure environment variables:
```powershell
Copy-Item .env.example .env
```

---

## 5. Startup Commands

### 5.1 Backend Startup
Activate the virtual environment and run the FastAPI development server:
```powershell
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API will be accessible at:
- **Health check**: `http://127.0.0.1:8000/api/health`
- **Interactive OpenAPI docs**: `http://127.0.0.1:8000/docs`

### 5.2 Frontend Startup
In a separate terminal, start the Vite development server:
```powershell
cd frontend
npm run dev
```
The frontend interface will launch at:
- **Application URL**: `http://localhost:5173`

---

## 6. Testing Instructions

### 6.1 Backend Tests
Run the unit test suite verifying health endpoints, routing, database models, and authentication:
```powershell
backend\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

### 6.2 Specific Module Tests
```powershell
# Health and CORS test
backend\.venv\Scripts\python.exe -m unittest tests/backend/test_health.py

# Database and Argon2/JWT Authentication test
backend\.venv\Scripts\python.exe -m unittest tests/backend/test_auth.py

# ML Specification and Policy test
backend\.venv\Scripts\python.exe -m unittest tests/ml/test_ml_structure.py
```

### 6.3 Frontend Verification
Verify TypeScript types and production build integrity:
```powershell
cd frontend
npm run build
npm run lint
```

---

## 7. Project Status & Roadmap

### Phase 1: Repository Initialization & Architecture Setup — **[COMPLETED]**
- [x] Fixed technology architecture defined and confirmed.
- [x] Complete modular directory structure for frontend, backend, ML, docs, and tests.
- [x] Full architecture documentation (`ARCHITECTURE.md`).
- [x] API contract specification for all future endpoints (`API_CONTRACT.md`).
- [x] SQLite schema design with Argon2 hashing and prediction records (`DATABASE_SCHEMA.md`).
- [x] Machine learning technical specification (`ML_SPECIFICATION.md`).
- [x] Contribution rules and integration standards (`CONTRIBUTING.md`).
- [x] Minimal FastAPI starter with `GET /api/health` and CORS configuration.
- [x] Preserved Vite React TypeScript starter with real-time backend health detection.
- [x] Environment configuration template (`.env.example`) and comprehensive `.gitignore`.

### Phase 2: Database Models & Authentication — **[COMPLETED]**
- [x] Implemented SQLite connection with PRAGMA foreign_keys enforcement and session management.
- [x] Implemented SQLAlchemy 2.0 ORM models: `User`, `Prediction`, `MolecularLibrary`, `ModelVersion`.
- [x] Implemented `POST /api/auth/register` with input validation and Argon2id password hashing.
- [x] Implemented `POST /api/auth/login` with credential verification and signed JWT bearer tokens.
- [x] Implemented `GET /api/auth/me` protected endpoint returning safe user profiles.
- [x] Built React + TypeScript interface for registration, login, session re-verification, and logout.
- [x] Added automated test suite (`tests/backend/test_auth.py`) covering all authentication security workflows.

### Phase 3: RDKit Chemoinformatics & ML Inference Engine — **[COMPLETED]**
- [x] Real RDKit chemical validation and canonicalization for arbitrary donor and acceptor SMILES.
- [x] Independent 2D physicochemical descriptor extraction (MolWt, LogP, TPSA, HBD, HBA, Rotatable Bonds, Rings, Fraction CSP3, Formula).
- [x] Distinct Donor and Acceptor role preservation and blend-level feature engineering.
- [x] Leakage-aware Random Forest inference pipeline with dynamic model registry (`OPVModelRegistry`).
- [x] Explicit `model_unavailable` payload when no validated model artifact exists, rejecting fabricated performance values.
- [x] Implemented `POST /api/predictions/analyze` (public molecular analysis and descriptor extraction).
- [x] Implemented `POST /api/predictions` (authenticated prediction inference and SQLite persistence).
- [x] Implemented `GET /api/predictions` and `GET /api/predictions/{id}` with strict user isolation.
- [x] Interactive React + TypeScript UI (`PredictionSection.tsx`) with benchmark presets, descriptor comparisons, and live history table.
- [x] 29 comprehensive automated tests passing across backend units and full HTTP API integration suites.

### Phase 4: Offline Scientific Chatbot & 3D Visualization (Upcoming)
- Implement pure-Python knowledge retrieval chatbot over OPV physics documentation.
- Implement interactive 3D molecular canvas (3Dmol.js / Three.js) and Recharts solar cell dashboard.
- End-to-end integration testing and user validation.
