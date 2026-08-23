from __future__ import annotations

from typing import Optional
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Text, ForeignKey, JSON, func, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class SchemaMetadata(Base):
    __tablename__ = "schema_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    database_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("database_connections.id", ondelete="CASCADE"),
        nullable=False, unique=True
    )
    db_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Full schema stored as JSON
    full_schema: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Summarized overview
    table_names: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of table/collection names
    table_summaries: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # brief per-table summaries
    relationships: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    total_tables: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_relationships: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    database_connection = relationship("DatabaseConnection", back_populates="schema_metadata")

    def __repr__(self):
        return f"<SchemaMetadata db_id={self.database_connection_id} tables={self.total_tables}>"
