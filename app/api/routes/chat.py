from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging
import time

from app.core.database import get_db
from app.api.deps import get_current_user_dep
from app.models.user import User
from app.schemas.chat import (
    ChatRequest, ChatResponse, ConversationListResponse, ConversationResponse,
    MessageListResponse, MessageResponse, VisualizationSpec, QueryInfo, QueryResultData
)
from app.services.conversation_service import (
    get_or_create_conversation, get_conversation_history,
    save_user_message, save_assistant_message,
    get_user_conversations, get_conversation_messages,
    delete_conversation, delete_all_conversations
)
from app.services.database_service import get_database_by_id
from app.services.query_service import run_query_pipeline
from app.services.ai_provider import get_ai_service
from app.core.exceptions import (
    AuthorizationError, DatabaseNotFoundError, QueryValidationError,
    QueryExecutionError, AIServiceError, WriteOperationError, QueryTimeoutError
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Main chat endpoint — processes a user question against their connected database."""
    req_t0 = time.perf_counter()
    try:
        # Validate database access
        conn = await get_database_by_id(db, request.database_id, str(current_user.id))

        # Get or create conversation
        conversation = await get_or_create_conversation(
            db, str(current_user.id), request.database_id,
            request.conversation_id, request.message
        )

        # Save user message
        await save_user_message(db, str(conversation.id), request.message)

        # Get conversation history for context
        history = await get_conversation_history(db, str(conversation.id), limit=10)

        # Run the AI query pipeline
        ai_service = get_ai_service()
        pipeline_result = await run_query_pipeline(
            user_question=request.message,
            database_connection_id=str(conn.id),
            encrypted_connection_string=conn.encrypted_connection_string,
            db_type=conn.db_type,
            db_session=db,
            ai_service=ai_service,
            conversation_history=history,
        )

        # Build structured response
        structured = {
            "answer": pipeline_result.answer,
            "insights": pipeline_result.insights,
            "warnings": pipeline_result.warnings,
            "data_quality_notes": pipeline_result.data_quality_notes,
            "refused": pipeline_result.refused,
        }

        # Build visualization spec
        viz = None
        if pipeline_result.visualization and pipeline_result.visualization.get("required"):
            v = pipeline_result.visualization
            viz = {
                "required": v.get("required", False),
                "type": v.get("type"),
                "title": v.get("title"),
                "description": v.get("description"),
                "x_key": v.get("x_key"),
                "y_keys": v.get("y_keys"),
                "value_key": v.get("value_key"),
                "label_key": v.get("label_key"),
                "format": v.get("format"),
                "mermaid": v.get("mermaid") or (v.get("value_key") if v.get("type") == "er_diagram" else None),
            }

        # Build query result
        result_data = None
        if pipeline_result.result:
            r = pipeline_result.result
            result_data = {
                "columns": r.columns,
                "rows": r.rows,
                "row_count": r.row_count,
                "truncated": r.truncated,
                "execution_time_ms": r.execution_time_ms,
            }

        # Save assistant message
        assistant_msg = await save_assistant_message(
            db=db,
            conversation_id=str(conversation.id),
            answer=pipeline_result.answer,
            structured_response={
                **structured,
                "intent": pipeline_result.intent,
                "result": result_data,
                "visualization": viz,
            },
            generated_query=pipeline_result.query,
            query_language=pipeline_result.query_language,
            visualization=viz,
            row_count=pipeline_result.result.row_count if pipeline_result.result else None,
            execution_time_ms=pipeline_result.result.execution_time_ms if pipeline_result.result else None,
        )

        total_req_ms = (time.perf_counter() - req_t0) * 1000.0
        logger.info(f"🏁 [CHAT ENDPOINT COMPLETE] Total HTTP handler time: {total_req_ms:.1f}ms ({total_req_ms/1000:.2f}s)")

        return ChatResponse(
            conversation_id=str(conversation.id),
            conversation_title=conversation.title,
            message=MessageResponse(
                id=str(assistant_msg.id),
                role="assistant",
                answer=pipeline_result.answer,
                insights=pipeline_result.insights,
                warnings=pipeline_result.warnings,
                intent=pipeline_result.intent,
                query=QueryInfo(
                    display=bool(pipeline_result.query),
                    language=pipeline_result.query_language or "sql",
                    content=pipeline_result.query or "",
                ) if pipeline_result.query else None,
                result=QueryResultData(**result_data) if result_data else None,
                visualization=VisualizationSpec(**viz) if viz else None,
                created_at=assistant_msg.created_at,
            ),
        )

    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)
    except WriteOperationError as e:
        raise HTTPException(status_code=403, detail=e.message)
    except QueryValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except QueryTimeoutError as e:
        raise HTTPException(status_code=408, detail=e.message)
    except (QueryExecutionError, AIServiceError) as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.exception(f"Unexpected chat error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the current user."""
    from sqlalchemy import select, func
    from app.models.conversation import Conversation, Message
    from app.models.database_connection import DatabaseConnection

    result = await db.execute(
        select(Conversation, DatabaseConnection.name.label("db_name"))
        .join(DatabaseConnection, Conversation.database_connection_id == DatabaseConnection.id)
        .where(Conversation.owner_id == str(current_user.id))
        .order_by(Conversation.updated_at.desc())
        .limit(100)
    )
    rows = result.all()

    conversations = []
    for conv, db_name in rows:
        # Count messages
        count_result = await db.execute(
            select(func.count(Message.id)).where(Message.conversation_id == conv.id)
        )
        msg_count = count_result.scalar() or 0
        conversations.append(ConversationResponse(
            id=str(conv.id),
            title=conv.title,
            database_id=str(conv.database_connection_id),
            database_name=db_name or "Unknown",
            message_count=msg_count,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        ))

    return ConversationListResponse(conversations=conversations, total=len(conversations))


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages in a conversation."""
    try:
        messages = await get_conversation_messages(db, conversation_id, str(current_user.id))
        msg_responses = []
        for msg in messages:
            sr = msg.structured_response or {}
            msg_responses.append(MessageResponse(
                id=str(msg.id),
                role=msg.role,
                answer=msg.content,
                insights=sr.get("insights", []),
                warnings=sr.get("warnings", []),
                intent=sr.get("intent"),
                query=QueryInfo(
                    display=True,
                    language=msg.query_language or "sql",
                    content=msg.generated_query or "",
                ) if msg.generated_query else None,
                result=QueryResultData(**sr["result"]) if sr.get("result") else None,
                visualization=VisualizationSpec(**msg.visualization) if msg.visualization else None,
                created_at=msg.created_at,
            ))
        return MessageListResponse(
            conversation_id=conversation_id,
            messages=msg_responses,
            total=len(msg_responses),
        )
    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    try:
        await delete_conversation(db, conversation_id, str(current_user.id))
    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.delete("/conversations", status_code=status.HTTP_200_OK)
async def remove_all_conversations(
    database_id: Optional[str] = None,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Delete all conversations for the user (optionally filtered by database)."""
    deleted_count = await delete_all_conversations(db, str(current_user.id), database_id)
    return {"deleted": deleted_count}
