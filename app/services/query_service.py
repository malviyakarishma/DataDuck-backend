"""
Main query execution pipeline service.

Orchestrates the full flow:
User question → Schema retrieval → Query generation → Validation → Execution → Analysis
"""
import json
import logging
import time
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ai_provider import AIProvider
from app.services.schema_service import get_relevant_schema
from app.security.query_validator import (
    validate_sql_query, validate_mongodb_operation, is_write_intent
)
from app.database_adapters.base import DatabaseAdapter, QueryResult
from app.database_adapters.factory import get_adapter
from app.security.encryption import decrypt_string
from app.core.config import settings
from app.core.exceptions import (
    QueryValidationError, QueryExecutionError, WriteOperationError,
    AIServiceError, DatabaseConnectionError
)

logger = logging.getLogger(__name__)


class QueryPipelineResult:
    def __init__(
        self,
        answer: str,
        insights: list[str],
        warnings: list[str],
        data_quality_notes: list[str],
        query: Optional[str],
        query_language: Optional[str],
        result: Optional[QueryResult],
        visualization: Optional[dict],
        refused: bool = False,
    ):
        self.answer = answer
        self.insights = insights
        self.warnings = warnings
        self.data_quality_notes = data_quality_notes
        self.query = query
        self.query_language = query_language
        self.result = result
        self.visualization = visualization
        self.refused = refused


async def run_query_pipeline(
    user_question: str,
    database_connection_id: str,
    encrypted_connection_string: str,
    db_type: str,
    db_session: AsyncSession,
    ai_service: AIProvider,
    conversation_history: list[dict],
) -> QueryPipelineResult:
    """
    Full AI query pipeline:
    1. Check for write intent → refuse immediately
    2. Get relevant schema
    3. Generate query via AI
    4. Validate query (security layer)
    5. Execute query
    6. Analyze results via AI
    7. Return structured result
    """

    # Step 1: Early write intent detection
    if is_write_intent(user_question):
        refusal = await ai_service.handle_write_intent(user_question)
        return QueryPipelineResult(
            answer=refusal,
            insights=[],
            warnings=[],
            data_quality_notes=[],
            query=None,
            query_language=None,
            result=None,
            visualization=None,
            refused=True,
        )

    # Step 2: Get relevant schema (no credentials included)
    relevant_schema = await get_relevant_schema(
        db_session, database_connection_id, user_question
    )

    if not relevant_schema:
        return QueryPipelineResult(
            answer="No schema information found. Please re-analyze your database first.",
            insights=[],
            warnings=[],
            data_quality_notes=[],
            query=None,
            query_language=None,
            result=None,
            visualization=None,
        )

    # Step 3: Generate query via AI
    gen_result = await ai_service.generate_query(
        user_question=user_question,
        relevant_schema=relevant_schema,
        db_type=db_type,
        conversation_history=conversation_history,
    )

    # Step 4: Handle AI refusal
    if gen_result.refused:
        return QueryPipelineResult(
            answer=gen_result.refusal_reason or "This operation is not supported.",
            insights=[],
            warnings=[],
            data_quality_notes=[],
            query=None,
            query_language=None,
            result=None,
            visualization=None,
            refused=True,
        )

    # Step 5: Security validation
    generated_query = gen_result.query
    query_language = "mongodb" if db_type == "mongodb" else "sql"

    if db_type == "mongodb":
        # Parse MongoDB operation from JSON string
        try:
            if isinstance(generated_query, str):
                query_dict = json.loads(generated_query)
            else:
                query_dict = generated_query
            query_dict = validate_mongodb_operation(query_dict)
            validated_query = query_dict
        except (json.JSONDecodeError, QueryValidationError, WriteOperationError) as e:
            raise QueryValidationError(f"Invalid MongoDB operation: {e}")
    else:
        # SQL validation via SQLGlot
        validated_query = validate_sql_query(generated_query, db_type)

    # Step 6: Execute query
    # Decrypt connection string and create adapter
    try:
        plain_conn = decrypt_string(encrypted_connection_string)
        adapter = get_adapter(plain_conn)
        await adapter.connect()
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to connect: {e}")

    try:
        query_result = await adapter.execute_read_query(
            validated_query,
            timeout=settings.QUERY_TIMEOUT_SECONDS
        )
    finally:
        await adapter.close()

    # Step 7: Analyze results
    result_for_analysis = {
        "columns": query_result.columns,
        "rows": query_result.rows,
        "row_count": query_result.row_count,
        "truncated": query_result.truncated,
    }

    analysis = await ai_service.analyze_results(
        user_question=user_question,
        query=str(validated_query),
        query_result=result_for_analysis,
        db_type=db_type,
        conversation_history=conversation_history,
    )

    return QueryPipelineResult(
        answer=analysis.answer,
        insights=analysis.insights,
        warnings=analysis.warnings,
        data_quality_notes=analysis.data_quality_notes,
        query=str(validated_query),
        query_language=query_language,
        result=query_result,
        visualization=analysis.visualization,
    )
