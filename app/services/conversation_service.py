"""
Conversation service — manages chat history with context-aware conversations.
"""
import logging
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.conversation import Conversation, Message
from app.core.exceptions import AuthorizationError, DatabaseNotFoundError

logger = logging.getLogger(__name__)


async def get_or_create_conversation(
    db: AsyncSession,
    user_id: str,
    database_id: str,
    conversation_id: Optional[str],
    first_message: str,
) -> Conversation:
    """Get existing conversation or create new one."""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = result.scalar_one_or_none()
        if conv:
            if str(conv.owner_id) != str(user_id):
                raise AuthorizationError("Access denied to this conversation.")
            return conv

    # Create new conversation
    title = _generate_title(first_message)
    conv = Conversation(
        owner_id=user_id,
        database_connection_id=database_id,
        title=title,
    )
    db.add(conv)
    await db.flush()
    await db.refresh(conv)
    return conv


async def get_conversation_history(
    db: AsyncSession,
    conversation_id: str,
    limit: int = 20,
) -> list[dict]:
    """Get conversation history as list of dicts for AI context."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    messages.reverse()  # Chronological order

    return [
        {
            "role": msg.role,
            "content": msg.content,
        }
        for msg in messages
    ]


async def save_user_message(
    db: AsyncSession,
    conversation_id: str,
    content: str,
) -> Message:
    """Save a user message."""
    msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=content,
    )
    db.add(msg)
    await db.flush()
    await db.refresh(msg)
    return msg


async def save_assistant_message(
    db: AsyncSession,
    conversation_id: str,
    answer: str,
    structured_response: dict,
    generated_query: Optional[str],
    query_language: Optional[str],
    visualization: Optional[dict],
    row_count: Optional[int],
    execution_time_ms: Optional[float],
) -> Message:
    """Save an assistant (AI) message."""
    msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        structured_response=structured_response,
        generated_query=generated_query,
        query_language=query_language,
        visualization=visualization,
        result_row_count=row_count,
        execution_time_ms=execution_time_ms,
    )
    db.add(msg)
    await db.flush()
    await db.refresh(msg)
    return msg


async def get_user_conversations(
    db: AsyncSession,
    user_id: str,
    skip: int = 0,
    limit: int = 50,
) -> list[Conversation]:
    """Get all conversations for a user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.owner_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def get_conversation_with_auth(
    db: AsyncSession,
    conversation_id: str,
    user_id: str,
) -> Conversation:
    """Get a conversation with ownership check."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise DatabaseNotFoundError("Conversation not found.")
    if str(conv.owner_id) != str(user_id):
        raise AuthorizationError("Access denied to this conversation.")
    return conv


async def get_conversation_messages(
    db: AsyncSession,
    conversation_id: str,
    user_id: str,
) -> list[Message]:
    """Get all messages in a conversation."""
    await get_conversation_with_auth(db, conversation_id, user_id)

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


async def delete_conversation(
    db: AsyncSession,
    conversation_id: str,
    user_id: str,
) -> None:
    """Delete a single conversation."""
    conv = await get_conversation_with_auth(db, conversation_id, user_id)
    await db.delete(conv)
    await db.flush()


async def delete_all_conversations(
    db: AsyncSession,
    user_id: str,
    database_id: Optional[str] = None,
) -> int:
    """Delete all conversations for a user (optionally filtered by database)."""
    query = select(Conversation).where(Conversation.user_id == user_id)
    if database_id:
        query = query.where(Conversation.database_id == database_id)
    
    result = await db.execute(query)
    convs = result.scalars().all()
    count = len(convs)
    for conv in convs:
        await db.delete(conv)
    await db.flush()
    return count


def _generate_title(first_message: str) -> str:
    """Generate a conversation title from the first message."""
    title = first_message.strip()
    if len(title) > 60:
        title = title[:57] + "..."
    return title or "New Conversation"
