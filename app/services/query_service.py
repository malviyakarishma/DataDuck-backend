"""
Main query execution pipeline service.

Orchestrates the full flow:
User question → Schema retrieval → Query generation → Validation → Execution → Analysis
"""
import json
import logging
import re
import time
from typing import Any, Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ai_provider import (
    AIProvider, AnalysisResult, INTENT_DATA_QUERY, INTENT_SCHEMA_EXPLORATION,
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


def _format_column_title(col_name: str) -> str:
    """Turn a database column name into a natural title (e.g. 'total_users' -> 'total users')."""
    clean = col_name.replace("_", " ").strip()
    replacements = {
        r"\bavg\b": "average",
        r"\bpk\b": "primary key",
        r"\bfk\b": "foreign key",
        r"\bcnt\b": "count",
        r"\bnum\b": "number of",
        r"\bqty\b": "quantity",
    }
    for pat, rep in replacements.items():
        clean = re.sub(pat, rep, clean, flags=re.IGNORECASE)
    return clean.strip()


def _format_scalar_value(val: Any) -> str:
    """Format scalar number or string with clean decimals/commas."""
    if val is None:
        return "None"
    if isinstance(val, float):
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def should_use_llm_for_result_analysis(
    rows: list[dict],
    columns: list[str],
    query: str,
    user_question: str,
) -> bool:
    """
    Determines if database query results are simple enough to format deterministically
    without spending ~14+ seconds on an unnecessary LLM inference call.

    Returns False (SKIP LLM) for:
    - Empty results (0 rows)
    - Single scalar aggregation (1 row, 1 column: COUNT, SUM, AVG, MIN, MAX, etc.)
    - Simple 1-row summary aggregation (1 row, <= 3 scalar columns: total_orders, total_revenue)

    Returns True (USE LLM) for:
    - Multiple rows (> 1 row) representing lists, trends, comparisons, breakdowns
    - Complex multi-category analytical tables
    """
    row_count = len(rows)

    # 1. 0 rows -> Deterministic empty response
    if row_count == 0:
        return False

    # 2. 1 row result
    if row_count == 1:
        row = rows[0]
        col_count = len(columns)

        # 1 column (e.g. COUNT, SUM, AVG, MAX, MIN, single metric)
        if col_count == 1:
            return False

        # Up to 3 columns if all values are scalars (numbers, strings, dates, booleans)
        if col_count <= 3:
            all_scalar = all(
                v is None or isinstance(v, (int, float, str, bool))
                for v in row.values()
            )
            if all_scalar:
                return False

    # 3. For multi-row results, use LLM for trend analysis, charting & complex insights
    return True


def format_simple_result(
    rows: list[dict],
    columns: list[str],
    query: str,
    user_question: str,
) -> AnalysisResult:
    """
    Lightweight, deterministic result formatter for simple scalar queries.
    Executes in < 1 ms with clean natural phrasing and KPI visualization.
    """
    if not rows:
        return AnalysisResult(
            answer="No matching records were found for this query.",
            insights=["Query executed successfully with 0 matching rows."],
            warnings=[],
            data_quality_notes=[],
            visualization=None,
        )

    row = rows[0]

    # Single column scalar (e.g. COUNT, SUM, AVG, MAX, MIN)
    if len(columns) == 1:
        col_name = columns[0]
        val = row.get(col_name)
        formatted_val = _format_scalar_value(val)
        col_title = _format_column_title(col_name)
        col_lower = col_name.lower()

        # Generate clean natural phrasing
        if "count" in col_lower or "total" in col_lower or "num" in col_lower or "users" in col_lower or "orders" in col_lower:
            if col_title.lower().startswith("number of") or col_title.lower().startswith("total"):
                answer = f"There are **{formatted_val}** {col_title.lower()}."
            else:
                answer = f"The total {col_title.lower()} is **{formatted_val}**."
        elif "avg" in col_lower or "average" in col_lower:
            answer = f"The {col_title.lower()} is **{formatted_val}**."
        elif "max" in col_lower or "highest" in col_lower or "maximum" in col_lower:
            answer = f"The {col_title.lower()} is **{formatted_val}**."
        elif "min" in col_lower or "lowest" in col_lower or "minimum" in col_lower:
            answer = f"The {col_title.lower()} is **{formatted_val}**."
        elif any(kw in col_lower for kw in ("sum", "revenue", "sales", "amount", "cost", "price", "spent", "balance")):
            answer = f"The {col_title.lower()} is **{formatted_val}**."
        else:
            answer = f"The {col_title.lower()} is **{formatted_val}**."

        # KPI visualization for single scalar numbers
        viz = None
        if isinstance(val, (int, float)):
            viz = {
                "required": True,
                "type": "kpi",
                "title": col_title.title(),
                "value_key": col_name,
                "format": "currency" if any(kw in col_lower for kw in ("revenue", "price", "sale", "amount", "cost", "dollar")) else "number",
            }

        return AnalysisResult(
            answer=answer,
            insights=[f"{col_title.title()}: {formatted_val}"],
            warnings=[],
            data_quality_notes=[],
            visualization=viz,
        )

    # 2-3 scalar columns (e.g. {"total_orders": 1200, "total_revenue": 450000.5})
    parts = []
    insights = []
    for col in columns:
        val = row.get(col)
        f_val = _format_scalar_value(val)
        c_title = _format_column_title(col)
        parts.append(f"{c_title.lower()} is **{f_val}**")
        insights.append(f"{c_title.title()}: {f_val}")

    answer = "The " + ", and the ".join(parts) + "."

    return AnalysisResult(
        answer=answer,
        insights=insights,
        warnings=[],
        data_quality_notes=[],
        visualization=None,
    )


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

    # Stage 6: Result Analysis
    t_stage_start = time.perf_counter()
    result_for_analysis = {
        "columns": query_result.columns,
        "rows": query_result.rows,
        "row_count": query_result.row_count,
        "truncated": query_result.truncated,
    }

    use_llm_analysis = should_use_llm_for_result_analysis(
        rows=query_result.rows,
        columns=query_result.columns,
        query=str(validated_query),
        user_question=user_question,
    )

    if use_llm_analysis:
        logger.info(f"🧠 [STAGE 6: Result Analysis] Complex result ({query_result.row_count} rows, {len(query_result.columns)} cols) -> Invoking AI Service...")
        analysis = await ai_service.analyze_results(
            user_question=user_question,
            query=str(validated_query),
            query_result=result_for_analysis,
            db_type=db_type,
            conversation_history=conversation_history,
        )
        t_analysis_ms = (time.perf_counter() - t_stage_start) * 1000.0
        analysis_method_label = "AI Analysis (LLM)"
        logger.info(f"⏱️ [STAGE 6: AI Result Analysis] Took {t_analysis_ms:.1f}ms ({t_analysis_ms/1000:.2f}s) via LLM")
    else:
        logger.info(f"⚡ [STAGE 6: Result Analysis] Simple scalar result ({query_result.row_count} rows, {len(query_result.columns)} cols) -> Fast deterministic formatter (Skipping LLM call)")
        analysis = format_simple_result(
            rows=query_result.rows,
            columns=query_result.columns,
            query=str(validated_query),
            user_question=user_question,
        )
        t_analysis_ms = (time.perf_counter() - t_stage_start) * 1000.0
        analysis_method_label = "Fast Formatter (No LLM)"
        logger.info(f"⏱️ [STAGE 6: Fast Result Formatter] Took {t_analysis_ms:.2f}ms (Saved ~14s LLM call!)")

    # End-of-Pipeline Summary
    t_total_pipeline_ms = (time.perf_counter() - pipeline_t0) * 1000.0

    logger.info(
        f"\n"
        f"================================================================================\n"
        f"📊 [DATADUCK PIPELINE TIMING SUMMARY]\n"
        f"├─ Question           : \"{user_question[:60]}\"\n"
        f"├─ Intent Classify    : {t_intent_ms:8.1f} ms  ({t_intent_ms/t_total_pipeline_ms*100:4.1f}%) [Deterministic: 0 LLM]\n"
        f"├─ Schema Retrieval   : {t_schema_retrieval_ms:8.1f} ms  ({t_schema_retrieval_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ AI Query Gen (LLM) : {t_query_gen_ms:8.1f} ms  ({t_query_gen_ms/t_total_pipeline_ms*100:4.1f}%) [Ollama 1 LLM Call]\n"
        f"├─ AST Validation     : {t_validation_ms:8.1f} ms  ({t_validation_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ DB Execution       : {t_db_exec_ms:8.1f} ms  ({t_db_exec_ms/t_total_pipeline_ms*100:4.1f}%)\n"
        f"├─ Result Analysis    : {t_analysis_ms:8.1f} ms  ({t_analysis_ms/t_total_pipeline_ms*100:4.1f}%) [{analysis_method_label}]\n"
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

