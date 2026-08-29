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

from app.services.ai_provider import (
    AIProvider, INTENT_DATA_QUERY, INTENT_SCHEMA_EXPLORATION,
    INTENT_CASUAL_CHAT, INTENT_WRITE_REQUEST
)
from app.services.schema_service import (
    get_relevant_schema, get_full_schema_metadata, generate_mermaid_er_diagram
)
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
        intent: str = INTENT_DATA_QUERY,
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
        self.intent = intent


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
    4-tier intent query pipeline with stage-by-stage timing instrumentation:
    1. WRITE_REQUEST → Refuse immediately (strictly read-only)
    2. CASUAL_CHAT → Respond naturally without querying database
    3. SCHEMA_EXPLORATION → Inspect schema metadata, explain tables/columns/relations or generate ER diagram
    4. DATA_QUERY → Generate read-only query → Validate → Execute → Analyze
    """
    pipeline_t0 = time.perf_counter()
    logger.info(f"🚀 [PIPELINE START] Processing question: '{user_question[:80]}' (DB: {db_type})")

    # ── Step 1: Intent Classification ──────────────────────────────────────
    t_stage_start = time.perf_counter()
    intent = await ai_service.classify_intent(user_question, conversation_history)
    t_intent_ms = (time.perf_counter() - t_stage_start) * 1000.0
    logger.info(f"⏱️ [STAGE 1: Intent Classification] Took {t_intent_ms:.1f}ms | Result: '{intent}'")

    # ── Intent 1: WRITE_REQUEST ───────────────────────────────────────────
    if intent == INTENT_WRITE_REQUEST or is_write_intent(user_question):
        t_stage_start = time.perf_counter()
        refusal = await ai_service.handle_write_intent(user_question)
        t_refusal_ms = (time.perf_counter() - t_stage_start) * 1000.0
        t_total_ms = (time.perf_counter() - pipeline_t0) * 1000.0
        logger.info(f"⏱️ [PIPELINE FINISH] Refused write request in {t_total_ms:.1f}ms")
        return QueryPipelineResult(
            answer=refusal,
            insights=[],
            warnings=["Write operations are strictly prohibited in DataDuck."],
            data_quality_notes=[],
            query=None,
            query_language=None,
            result=None,
            visualization=None,
            refused=True,
            intent=INTENT_WRITE_REQUEST,
        )

    # ── Intent 2: CASUAL_CHAT ─────────────────────────────────────────────
    if intent == INTENT_CASUAL_CHAT:
        t_stage_start = time.perf_counter()
        casual_response = await ai_service.handle_casual_chat(user_question, conversation_history)
        t_chat_ms = (time.perf_counter() - t_stage_start) * 1000.0
        t_total_ms = (time.perf_counter() - pipeline_t0) * 1000.0
        logger.info(f"⏱️ [STAGE 2: Casual Chat LLM] Took {t_chat_ms:.1f}ms | Total Pipeline: {t_total_ms:.1f}ms")
        return QueryPipelineResult(
            answer=casual_response,
            insights=[],
            warnings=[],
            data_quality_notes=[],
            query=None,
            query_language=None,
            result=None,
            visualization=None,
            intent=INTENT_CASUAL_CHAT,
        )

    # ── Intent 3: SCHEMA_EXPLORATION ──────────────────────────────────────
    if intent == INTENT_SCHEMA_EXPLORATION:
        t_stage_start = time.perf_counter()
        full_schema = await get_full_schema_metadata(db_session, database_connection_id)
        if not full_schema:
            full_schema = await get_relevant_schema(db_session, database_connection_id, user_question, max_tables=50)
        t_schema_ms = (time.perf_counter() - t_stage_start) * 1000.0
        logger.info(f"⏱️ [STAGE 2: Schema Metadata Fetch] Took {t_schema_ms:.1f}ms (Found {len(full_schema.get('tables', []))} tables)")

        if not full_schema or not full_schema.get("tables"):
            return QueryPipelineResult(
                answer="No schema metadata found for this database. Please re-analyze the database schema in Settings or the Databases tab.",
                insights=[],
                warnings=["Schema not analyzed."],
                data_quality_notes=[],
                query=None,
                query_language=None,
                result=None,
                visualization=None,
                intent=INTENT_SCHEMA_EXPLORATION,
            )

        # Check if ER diagram is requested
        import re
        q_lower = user_question.lower()
        is_er_diagram_request = bool(
            re.search(r"\b(er[ -]?diagram|erd)\b", q_lower) or
            re.search(r"\b(schema|database|table|entity)\s+(diagram|chart|graph|map|visual)\b", q_lower) or
            re.search(r"\b(show|draw|generate|display|view|render)\b.*\b(diagram|erd)\b", q_lower)
        )

        if is_er_diagram_request:
            t_stage_start = time.perf_counter()
            mermaid_code = generate_mermaid_er_diagram(full_schema)
            t_mermaid_ms = (time.perf_counter() - t_stage_start) * 1000.0
            total_tables = len(full_schema.get("tables", []))
            total_relationships = len(full_schema.get("relationships", []))
            t_total_ms = (time.perf_counter() - pipeline_t0) * 1000.0
            logger.info(f"⏱️ [STAGE 3: Mermaid ER Generation] Took {t_mermaid_ms:.1f}ms | Total Pipeline: {t_total_ms:.1f}ms")

            answer = (
                f"### 📊 Database ER Diagram ({full_schema.get('database_name', 'Database')})\n\n"
                f"Generated interactive Entity-Relationship diagram representing **{total_tables} tables** and **{total_relationships} relationships**.\n\n"
                f"You can explore the interactive diagram below or open the **Schema Explorer** for in-depth column definitions and constraint inspection."
            )
            insights = [
                f"Database contains {total_tables} tables and {total_relationships} foreign key relationships.",
                f"Primary entities: {', '.join(t.get('name', '') for t in full_schema.get('tables', [])[:5])}"
            ]
            viz_spec = {
                "required": True,
                "type": "er_diagram",
                "title": f"{full_schema.get('database_name', 'Database')} ER Diagram",
                "description": f"Entity-Relationship diagram with {total_tables} tables",
                "value_key": mermaid_code,
            }
            return QueryPipelineResult(
                answer=answer,
                insights=insights,
                warnings=[],
                data_quality_notes=[],
                query=None,
                query_language=None,
                result=None,
                visualization=viz_spec,
                intent=INTENT_SCHEMA_EXPLORATION,
            )

        # Schema question answering (e.g. explain table, show columns, relationships)
        t_stage_start = time.perf_counter()
        schema_analysis = await ai_service.answer_schema_question(
            user_question=user_question,
            full_schema=full_schema,
            db_type=db_type,
            conversation_history=conversation_history,
        )
        t_schema_ans_ms = (time.perf_counter() - t_stage_start) * 1000.0
        t_total_ms = (time.perf_counter() - pipeline_t0) * 1000.0
        logger.info(f"⏱️ [STAGE 3: Schema Explanation LLM] Took {t_schema_ans_ms:.1f}ms | Total Pipeline: {t_total_ms:.1f}ms")

        return QueryPipelineResult(
            answer=schema_analysis.answer,
            insights=schema_analysis.insights,
            warnings=schema_analysis.warnings,
            data_quality_notes=schema_analysis.data_quality_notes,
            query=None,
            query_language=None,
            result=None,
            visualization=schema_analysis.visualization,
            intent=INTENT_SCHEMA_EXPLORATION,
        )

    # ── Intent 4: DATA_QUERY (Read-only query execution flow) ──────────────
    
    # Stage 2: Schema Selection
    t_stage_start = time.perf_counter()
    relevant_schema = await get_relevant_schema(
        db_session, database_connection_id, user_question
    )
    t_schema_retrieval_ms = (time.perf_counter() - t_stage_start) * 1000.0
    tables_count = len(relevant_schema.get("tables", [])) if relevant_schema else 0
    logger.info(f"⏱️ [STAGE 2: Relevant Schema Selection] Took {t_schema_retrieval_ms:.1f}ms | Selected {tables_count} relevant tables")

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
            intent=INTENT_DATA_QUERY,
        )

    # Stage 3: Query Generation via AI
    t_stage_start = time.perf_counter()
    gen_result = await ai_service.generate_query(
        user_question=user_question,
        relevant_schema=relevant_schema,
        db_type=db_type,
        conversation_history=conversation_history,
    )
    t_query_gen_ms = (time.perf_counter() - t_stage_start) * 1000.0
    logger.info(f"⏱️ [STAGE 3: AI Query Generation] Took {t_query_gen_ms:.1f}ms ({t_query_gen_ms/1000:.2f}s) | Query: '{str(gen_result.query)[:60]}...'")

    # Handle AI refusal
    if gen_result.refused:
        t_total_ms = (time.perf_counter() - pipeline_t0) * 1000.0
        logger.info(f"⏱️ [PIPELINE FINISH] AI refused query in {t_total_ms:.1f}ms: {gen_result.refusal_reason}")
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
            intent=INTENT_DATA_QUERY,
        )

    # Stage 4: Security & AST Validation
    t_stage_start = time.perf_counter()
    generated_query = gen_result.query
    query_language = "mongodb" if db_type == "mongodb" else "sql"

    if db_type == "mongodb":
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
        validated_query = validate_sql_query(generated_query, db_type)

    t_validation_ms = (time.perf_counter() - t_stage_start) * 1000.0
    logger.info(f"⏱️ [STAGE 4: Security AST Validation] Took {t_validation_ms:.1f}ms | Passed read-only verification")

    # Stage 5: Database Connection & Query Execution
    t_stage_start = time.perf_counter()
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

    t_db_exec_ms = (time.perf_counter() - t_stage_start) * 1000.0
    logger.info(f"⏱️ [STAGE 5: Database Execution] Took {t_db_exec_ms:.1f}ms | Returned {query_result.row_count} rows ({len(query_result.columns)} columns)")

    # Stage 6: Result Analysis via AI
    t_stage_start = time.perf_counter()
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
    t_analysis_ms = (time.perf_counter() - t_stage_start) * 1000.0
    logger.info(f"⏱️ [STAGE 6: AI Result Analysis] Took {t_analysis_ms:.1f}ms ({t_analysis_ms/1000:.2f}s)")

    # End-of-Pipeline Summary
    t_total_pipeline_ms = (time.perf_counter() - pipeline_t0) * 1000.0

    logger.info(
        f"\n"
        f"================================================================================\n"
        f"📊 [DATADUCK PIPELINE TIMING SUMMARY]\n"
        f"├─ Question           : \"{user_question[:60]}\"\n"
        f"├─ Intent Classify    : {t_intent_ms:8.1f} ms  ({t_intent_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ Schema Retrieval   : {t_schema_retrieval_ms:8.1f} ms  ({t_schema_retrieval_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ AI Query Gen (LLM) : {t_query_gen_ms:8.1f} ms  ({t_query_gen_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ AST Validation     : {t_validation_ms:8.1f} ms  ({t_validation_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ DB Execution       : {t_db_exec_ms:8.1f} ms  ({t_db_exec_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ AI Analysis (LLM)  : {t_analysis_ms:8.1f} ms  ({t_analysis_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"────────────────────────────────────────────────────────────────────────────────\n"
        f"⏱️ TOTAL PIPELINE TIME: {t_total_pipeline_ms:8.1f} ms  ({t_total_pipeline_ms/1000.0:.2f}s)\n"
        f"================================================================================"
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
        intent=INTENT_DATA_QUERY,
    )

