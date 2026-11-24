from datetime import datetime, timezone
import uuid
from sqlalchemy import String, Float, Text, DateTime, ForeignKey, CheckConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.db.base import Base


def generate_prediction_id() -> str:
    return f"pred_{uuid.uuid4().hex[:12]}"


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "input_method IN ('smiles', 'file_upload', 'library_selection')",
            name="check_prediction_input_method"
        ),
        Index("idx_predictions_user_id", "user_id"),
        Index("idx_predictions_created_at", "created_at"),
        Index("idx_predictions_model_version", "model_version"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_prediction_id)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    input_method: Mapped[str] = mapped_column(String, nullable=False)
    donor_name: Mapped[str] = mapped_column(String, nullable=False)
    donor_smiles: Mapped[str] = mapped_column(Text, nullable=False)
    acceptor_name: Mapped[str] = mapped_column(String, nullable=False)
    acceptor_smiles: Mapped[str] = mapped_column(Text, nullable=False)
    device_parameters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicted_pce: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_voc: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_jsc: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_ff: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("model_versions.version_tag", ondelete="RESTRICT"),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="predictions")
    model = relationship("ModelVersion", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction id={self.id} pce={self.predicted_pce}>"
