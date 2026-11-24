from fastapi import APIRouter
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.predictions import router as predictions_router

api_v1_router = APIRouter()
api_v1_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_v1_router.include_router(predictions_router, prefix="/predictions", tags=["Predictions"])

__all__ = ["api_v1_router"]
