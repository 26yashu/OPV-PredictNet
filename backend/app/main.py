from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.v1 import api_v1_router
from backend.app.core.config import settings
from backend.app.db.session import init_db
from backend.app.services.phase9_prediction import phase9_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("opv_predictnet.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager: initialize database tables and verify model status."""
    logger.info("Initializing OPV-PredictNet database schema...")
    init_db()

    # Verify model artifact readiness
    ready_models = [m for m in ("M0", "M1", "M2A", "M3") if phase9_service.is_ready(m)]
    logger.info(
        f"OPV-PredictNet service initialized. Ready research models: {ready_models}. "
        f"Governance: research_estimate_only=True, validated_for_production=False."
    )
    yield
    logger.info("OPV-PredictNet service shutting down cleanly.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend API for OPV-PredictNet: AI-Driven Molecular Screening, Feature Generation, "
        "and Uncertainty-Quantified PCE Estimation for Organic Photovoltaic Materials."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration allowing local Vite development server and optional configured origins
cors_origins = list(settings.CORS_ORIGINS)
if settings.FRONTEND_URL and settings.FRONTEND_URL not in cors_origins:
    cors_origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
def get_health():
    """Health check endpoint to verify backend service status, model readiness, and governance flags."""
    models_status = {m: phase9_service.is_ready(m) for m in ("M0", "M1", "M2A", "M3")}
    all_ready = all(models_status.values())
    return {
        "status": "healthy" if all_ready else "degraded",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_ready": models_status,
        "scientific_governance": {
            "research_estimate_only": settings.RESEARCH_ESTIMATE_ONLY,
            "validated_for_production": settings.VALIDATED_FOR_PRODUCTION,
        },
    }


# Mount API v1 router under /api
app.include_router(api_v1_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
