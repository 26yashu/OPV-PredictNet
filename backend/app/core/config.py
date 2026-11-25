import json
import logging
from typing import Any, List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    APP_NAME: str = "OPV-PredictNet Backend"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server binding
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # CORS settings: Local Vite React dev server + optional production origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    FRONTEND_URL: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite:///./opv_predictnet.db"

    # Security
    JWT_SECRET_KEY: str = "change-this-insecure-secret-key-in-production-min32chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Scientific Governance (Non-negotiable system constants)
    RESEARCH_ESTIMATE_ONLY: bool = True
    VALIDATED_FOR_PRODUCTION: bool = False

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    parsed = json.loads(v_stripped)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if item]
                except Exception:
                    pass
            # Comma-separated fallback
            return [origin.strip() for origin in v_stripped.split(",") if origin.strip()]
        elif isinstance(v, (list, tuple)):
            return [str(item).strip() for item in v if item]
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def validate_production_security(self) -> None:
        """Issue operational warnings or guards if running in production mode with default credentials."""
        if self.ENVIRONMENT.lower() == "production":
            if "change-this-insecure" in self.JWT_SECRET_KEY or len(self.JWT_SECRET_KEY) < 32:
                logger.warning(
                    "[SECURITY WARNING] Running in production with an insecure or default JWT_SECRET_KEY. "
                    "Ensure a cryptographically random secret (>=32 chars) is supplied via environment variables."
                )


settings = Settings()
settings.validate_production_security()
