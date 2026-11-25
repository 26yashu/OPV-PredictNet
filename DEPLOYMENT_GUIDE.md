# OPV-PredictNet — Deployment & Operations Guide

**Document Version:** 1.0 (Phase 12 Release)  
**Target Environments:** Academic Demonstration, Local Workstations, Containerized Research Servers  

---

## 1. Deployment Architecture Overview

OPV-PredictNet is structured into three distinct layers designed for zero external API costs and high operational reproducibility:

```
[ User Browser ]
       │
       ▼ (HTTP :3000 / :5173)
┌──────────────────────────────────────────────┐
│  Frontend Presentation Layer                 │
│  - Vite + React 19 + TypeScript              │
│  - In-browser RDKit analysis client          │
│  - Device Morphology & Energy Canvas         │
└──────────────────────┬───────────────────────┘
                       │ REST JSON (Bearer JWT)
                       ▼ (HTTP :8000)
┌──────────────────────────────────────────────┐
│  FastAPI Backend Service Layer               │
│  - Python 3.11 ASGI Server                   │
│  - Unified Feature Pipeline (M0, M1, M2A, M3)│
│  - Scaffold Registry & Conformal Quantiles   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Persistence & Storage Layer                 │
│  - SQLite Database (`opv_predictnet.db`)     │
│  - Packaged Joblib Model Artifacts (v1)      │
│  - Frozen Pipeline State JSON                │
└──────────────────────────────────────────────┘
```

---

## 2. Option A: Local Demonstration Setup (Recommended for Review)

The local demonstration runs directly on a development workstation without requiring Docker.

### 2.1 Prerequisites
- **Python 3.11** installed and on `PATH`.
- **Node.js** (v18+ LTS) and **npm**.
- **Git** version control.

### 2.2 Quick Start via Demonstration Script

#### On Windows (PowerShell):
```powershell
# From the repository root:
.\scripts\run_demo.ps1
```

#### On Linux / macOS (Bash):
```bash
# Make script executable and run:
chmod +x scripts/run_demo.sh
./scripts/run_demo.sh
```

The launcher performs automated prerequisite verification, confirms model artifacts in `ml/models/saved_models/phase9_v1/`, starts the FastAPI server on `http://127.0.0.1:8000`, starts the Vite frontend on `http://localhost:5173`, and opens the demonstration interface.

### 2.3 Manual Step-by-Step Local Setup

#### Step 1: Configure Environment
```bash
# Copy local configuration
cp .env.example .env
cp frontend/.env.example frontend/.env
```

#### Step 2: Install Backend Dependencies
```bash
# Initialize and activate Python 3.11 venv
python -m venv backend/.venv
# On Windows:
backend\.venv\Scripts\activate.ps1
# On Linux/macOS:
source backend/.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Verify or Package Model Artifacts
If `ml/models/saved_models/phase9_v1/phase9_M3_model.joblib` is not already present, run the reproducible packaging script:
```bash
python ml/scripts/package_models_phase9.py
```

#### Step 4: Install Frontend Dependencies & Build
```bash
cd frontend
npm install
npm run build
cd ..
```

#### Step 5: Run Servers
Terminal 1 (Backend):
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

---

## 3. Option B: Containerized Docker Deployment

OPV-PredictNet includes a multi-container Docker Compose configuration suitable for reproducible academic servers.

### 3.1 Prerequisites
- **Docker Engine** (v24.0+)
- **Docker Compose** (v2.0+)

### 3.2 Deployment Instructions

```bash
# Step 1: Configure production environment variables
cp .env.production.example .env

# Generate a cryptographically secure JWT secret key:
# Linux/macOS:
export JWT_SECRET_KEY=$(openssl rand -hex 32)
# Windows PowerShell:
$env:JWT_SECRET_KEY = [System.Guid]::NewGuid().ToString("N") + [System.Guid]::NewGuid().ToString("N")

# Step 2: Build and start containers in detached mode
docker compose up -d --build

# Step 3: Check container logs and health status
docker compose ps
docker compose logs -f backend
```

### 3.3 Containerized Access Points
- **Web Application Interface:** `http://localhost:3000`
- **Backend API & OpenAPI Docs:** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/api/health`
- **Data Persistence Volume:** Named Docker volume `opv_predictnet_data` mounted at `/app/data` inside the backend container.

---

## 4. Database Administration, Persistence & Recovery

OPV-PredictNet uses SQLite with `PRAGMA foreign_keys=ON` for relational persistence of user accounts, model versions, and prediction logs.

### 4.1 Database Status Check
```bash
python scripts/manage_db.py status
```
Outputs file size and table record counts:
```
--- OPV-PredictNet Database Status ---
Path: C:\Users\DELL\OPV-PredictNet\opv_predictnet.db
Size: 200.00 KB

Table Row Counts:
  - model_versions: 6 records
  - molecular_library: 0 records
  - predictions: 55 records
  - users: 66 records
--------------------------------------
```

### 4.2 SQLite Integrity Verification
```bash
python scripts/manage_db.py check
```
Verifies B-tree structure, indices, and table consistency via `PRAGMA integrity_check`.

### 4.3 Database Backup Procedure
Create a non-blocking, ACID-compliant online backup:
```bash
python scripts/manage_db.py backup --dest backups/
```
Generates a timestamped snapshot: `backups/opv_predictnet_backup_YYYYMMDD_HHMMSS.db`.

### 4.4 Database Disaster Recovery
Restore the database from any verified backup:
```bash
python scripts/manage_db.py restore backups/opv_predictnet_backup_YYYYMMDD_HHMMSS.db
```
The utility validates the backup's internal integrity, creates a safety backup of the existing active database (`.pre_restore.bak`), and atomically replaces the active database.

---

## 5. Security & Production Hardening Checklist

When moving from local demonstration to a public or institutional research server:

1. **Secret Rotation:** Replace default `JWT_SECRET_KEY` with a cryptographically generated string of at least 32 characters.
2. **CORS Restrictions:** Set `CORS_ORIGINS` in `.env` strictly to the institutional domain (e.g., `["https://opv.materials-lab.org"]`).
3. **Disable Debug Mode:** Ensure `DEBUG=False` and `ENVIRONMENT=production`.
4. **HTTPS Termination:** Terminate SSL/TLS via a reverse proxy (e.g., Nginx, Caddy, or Cloudflare) with modern TLS 1.3 ciphers.
5. **No Committed Credentials:** Confirm `.env` and `*.db` files are never tracked in Git (enforced via `.gitignore`).
6. **Scientific Disclaimers:** Preserve mandatory governance notices (`research_estimate_only=true` and `validated_for_production=false`).
