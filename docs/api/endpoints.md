# API Endpoints Reference

This document provides a summary of all planned API endpoints for OPV-PredictNet.

See [API_CONTRACT.md](../../API_CONTRACT.md) for full request/response schemas, JSON payloads, and validation constraints.

## Status Summary

| Method | Endpoint | Status | Implemented Phase | Function |
|---|---|---|---|---|
| `GET` | `/api/health` | **Implemented** | Phase 1 | Backend health check and CORS verification |
| `POST` | `/api/auth/register` | **Implemented** | Phase 2 | Register new user with Argon2 password hashing |
| `POST` | `/api/auth/login` | **Implemented** | Phase 2 | Authenticate and issue JWT bearer access token |
| `GET` | `/api/auth/me` | **Implemented** | Phase 2 | Retrieve authenticated profile |
| `POST` | `/api/predictions/analyze` | **Implemented** | Phase 3 | RDKit validation & physicochemical descriptor extraction |
| `POST` | `/api/predictions` | **Implemented** | Phase 3 | Run ML inference pipeline (or explicit model_unavailable) |
| `GET` | `/api/predictions` | **Implemented** | Phase 3 | Retrieve user-isolated prediction history |
| `GET` | `/api/predictions/{id}` | **Implemented** | Phase 3 | Retrieve specific prediction record |
| `POST` | `/api/chatbot/message` | Documented | Phase 4 | Offline scientific knowledge chatbot |
