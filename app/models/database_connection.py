from __future__ import annotations

from typing import Optional
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class DatabaseConnection(Base):
    __tablename__ = "database_connections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    db_type: Mapped[str] = mapped_column(String(50), nullable=False)  # postgresql, mysql, sqlite, mongodb
    encrypted_connection_string: Mapped[str] = mapped_column(Text, nullable=False)
    masked_connection_string: Mapped[str] = mapped_column(Text, nullable=False)  # password replaced with ***
    host: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    port: Mapped[Optional[int]] = mapped_column(nullable=True)
    database_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_connected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_tested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    schema_analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    connection_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    owner = relationship("User", back_populates="database_connections")
    conversations = relationship("Conversation", back_populates="database_connection", cascade="all, delete-orphan")
    schema_metadata = relationship("SchemaMetadata", back_populates="database_connection", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DatabaseConnection id={self.id} name={self.name} type={self.db_type}>"
