from __future__ import annotations

from typing import Optional
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, ForeignKey, JSON, func, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import enum


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    database_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="New Conversation")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    owner = relationship("User", back_populates="conversations")
    database_connection = relationship("DatabaseConnection", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self):
        return f"<Conversation id={self.id} title={self.title}>"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user/assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Structured response data for assistant messages
    structured_response: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Query that was generated (masked)
    generated_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    query_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Visualization spec
    visualization: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Result metadata
    result_row_count: Mapped[Optional[int]] = mapped_column(nullable=True)
    execution_time_ms: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message id={self.id} role={self.role}>"
