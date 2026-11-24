from datetime import datetime, timezone
from sqlalchemy import String, Float, Text, DateTime, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base


class MolecularLibrary(Base):
    __tablename__ = "molecular_library"
    __table_args__ = (
        CheckConstraint("role IN ('donor', 'acceptor', 'additive')", name="check_molecular_role"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String, index=True, nullable=False)
    canonical_smiles: Mapped[str] = mapped_column(Text, nullable=False)
    inchi_key: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    molecular_weight: Mapped[float] = mapped_column(Float, nullable=False)
    tpsa: Mapped[float | None] = mapped_column(Float, nullable=True)
    homo_ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    lumo_ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    bandgap_ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    literature_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<MolecularLibrary name={self.name} role={self.role}>"
