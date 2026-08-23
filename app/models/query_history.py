from __future__ import annotations

from typing import Optional
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Text, ForeignKey, JSON, func, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class QueryHistory(Base):
    __tablename__ = "query_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    database_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    query_language: Mapped[str] = mapped_column(String(50), nullable=False)  # sql, mongodb
    generated_query: Mapped[str] = mapped_column(Text, nullable=False)
    was_validated: Mapped[bool] = mapped_column(default=False, nullable=False)
    was_executed: Mapped[bool] = mapped_column(default=False, nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<QueryHistory id={self.id} language={self.query_language}>"
