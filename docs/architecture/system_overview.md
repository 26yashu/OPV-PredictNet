# System Architecture Overview

OPV-PredictNet is structured into three primary tiers:

```
+-------------------------------------------------------------+
|                     Client Tier (Frontend)                  |
|  - React 19 + TypeScript + Vite                             |
|  - 3D Molecular Viewer (3Dmol.js / Three.js / R3F)          |
|  - Performance Dashboard & Charts (Recharts)                |
|  - Modular Styling (Tailwind CSS + shadcn/ui components)    |
+------------------------------+------------------------------+
                               | REST / JSON (HTTP)
                               v
+-------------------------------------------------------------+
|                     Application Tier (Backend)              |
|  - FastAPI (Python 3.11 ASGI)                               |
|  - Pydantic v2 (Input validation & Data normalization)      |
|  - SQLAlchemy 2.0 ORM + SQLite                              |
|  - Security: Argon2 password hashing + JWT Bearer Auth      |
|  - Service Layer: Molecular parsing, Prediction, Chatbot   |
+------------------------------+------------------------------+
                               | In-memory / Inter-process
                               v
+-------------------------------------------------------------+
|             Scientific & ML Tier (Prediction Engine)        |
|  - RDKit: Molecular parsing, validation, descriptors        |
|  - Blend Feature Engineering: Energy offsets & interaction  |
|  - Random Forest Regressor (Scikit-Learn / Joblib)          |
|  - Knowledge-Based Local Chatbot (Offline, no paid APIs)    |
+-------------------------------------------------------------+
```

## System Tenets
1. **Low Resource Footprint**: Built to operate comfortably within an 8 GB RAM envelope on Windows 10/11.
2. **Scientific Integrity**: Distinct representations for molecular candidates, donor-acceptor blends, physical devices, and statistical predictions.
3. **Composability**: Decoupled REST contracts and typed interfaces between frontend and backend.
