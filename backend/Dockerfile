# ==============================================================================
# OPV-PredictNet — Backend Dockerfile
# Python 3.11 with FastAPI, RDKit, and Scikit-Learn
# ==============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS dependencies for numerical/C extensions (libgomp for scikit-learn/RDKit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code, ML modules, and packaged artifacts
COPY backend /app/backend
COPY ml /app/ml

# Create persistent data directory for SQLite database
RUN mkdir -p /app/data

EXPOSE 8000

# Health check using FastAPI /api/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
