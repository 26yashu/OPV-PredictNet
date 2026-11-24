# OPV-PredictNet — Contribution & Integration Guidelines

This document outlines the engineering protocols, code conventions, testing standards, and architectural rules governing the OPV-PredictNet repository.

---

## 1. Code Style and Conventions

### 1.1 Python Conventions (Backend & ML)
- **Standard**: Follow PEP 8 and PEP 484 strictly.
- **Naming**:
  - Modules & packages: `snake_case` (e.g., `molecular.py`, `feature_engineering.py`)
  - Classes: `PascalCase` (e.g., `RandomForestPipeline`, `PredictionRequest`)
  - Functions & variables: `snake_case` (e.g., `compute_molecular_descriptors`, `pce_score`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MODEL_VERSION`, `MAX_RETRIES`)
- **Type Annotations**: Mandatory on all public function signatures, parameters, and return values.
- **Docstrings**: Google or Sphinx style explaining scientific parameters, units, and exceptions.

### 1.2 TypeScript Conventions (Frontend)
- **Standard**: TypeScript strict mode enabled (`noImplicitAny: true`, `strictNullChecks: true`).
- **Naming**:
  - Components & Component files: `PascalCase` (e.g., `MolecularViewer.tsx`, `HealthStatusBadge.tsx`)
  - Utilities & Hooks: `camelCase` (e.g., `useBackendHealth.ts`, `formatEfficiency.ts`)
  - Types & Interfaces: `PascalCase` (e.g., `PredictionRecord`, `HealthResponse`)
- **No `any`**: Explicitly declare interfaces or unions.

---

## 2. API Contract & Architectural Integrity

1. **Strict Specification Adherence**:
   - Every endpoint implementation must conform strictly to `API_CONTRACT.md`.
   - Never alter request or response payload shapes without prior architectural review and documentation updates.
2. **Normalized Predictions**:
   - All three molecular input mechanisms (SMILES, file upload, molecular library selection) must route through the internal normalized prediction adapter.
3. **No Fake or Mock Prediction Outputs**:
   - Never implement fake UI actions, simulated randomized PCE counters, or mock model predictions in production routes. If an endpoint or feature is not yet implemented, return an explicit HTTP 501 or display an honest pending/unavailable state in the UI.

---

## 3. Environment Variables & Security

1. **No Hardcoded Credentials**:
   - Real passwords, cryptographic keys, and database credentials must never be committed to git.
   - All sensitive parameters must be read from environment variables or `.env`.
   - Update `.env.example` whenever new configuration parameters are introduced.
2. **Password Security**:
   - All user passwords must be hashed using Argon2id with recommended memory and iteration factors. Plaintext passwords must never be stored, logged, or returned.

---

## 4. Dependencies & Resource Footprint

1. **8 GB RAM Constraint**:
   - Avoid installing heavyweight, unnecessary libraries or memory-hungry background services.
   - Frontend bundle sizes should remain optimized with tree-shaking.
2. **Free & Open-Source Tools**:
   - All mandatory project dependencies must be free and open source.
   - Do not introduce mandatory dependencies on paid external APIs (e.g., proprietary LLMs, paid chemical databases).

---

## 5. Error Handling Conventions

- **Backend**:
  - Use `HTTPException` with informative `detail` payloads.
  - Return HTTP 400 for bad parameters, HTTP 401 for unauthorized calls, HTTP 404 for missing resources, HTTP 422 for unparseable chemistry, and HTTP 500 for unexpected internal errors.
- **Frontend**:
  - Display clear, non-technical error messages when network calls fail.
  - Handle offline backend states gracefully by notifying the user rather than crashing or hanging.

---

## 6. Testing & Quality Assurance

- **Backend Tests**:
  - Unit and integration tests reside in `tests/backend/`.
  - Execute with: `backend\.venv\Scripts\python.exe -m unittest discover tests/backend`
- **ML Tests**:
  - Pipeline verification tests reside in `tests/ml/`.
  - Execute with: `backend\.venv\Scripts\python.exe -m unittest discover tests/ml`
- **Frontend Verification**:
  - Ensure type checking passes: `npm run build`
  - Run linting: `npm run lint`

---

## 7. Git Workflow

1. **Branching Model**:
   - `master` / `main`: Production-ready, stable codebase.
   - `phase/<N>-<topic>`: Major phase deliverables (e.g., `phase/1-repo-init`, `phase/2-auth-db`).
   - `feat/<feature-name>`: Specific individual features.
   - `fix/<bug-name>`: Bug fixes.
2. **Commit Messages**:
   - Use conventional commits format: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
   - Example: `docs: initialize architecture and api contract specifications for Phase 1`
3. **Clean Commits**:
   - Check `git status` prior to committing to ensure `.venv`, `node_modules`, `dist`, `.env`, and temporary files are not tracked.
