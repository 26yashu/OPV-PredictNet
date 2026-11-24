from datetime import datetime, timezone
from sqlalchemy import String, Boolean, Float, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_tag: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    algorithm: Mapped[str] = mapped_column(String, nullable=False)
    target_metrics: Mapped[str] = mapped_column(String, nullable=False)
    r2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hyperparameters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    predictions = relationship("Prediction", back_populates="model")

    def __repr__(self) -> str:
        return f"<ModelVersion tag={self.version_tag} algorithm={self.algorithm}>"
