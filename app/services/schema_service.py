"""
Schema service — retrieves and caches database schema metadata.
Implements schema relevance selection for large databases.
"""
import json
import logging
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database_adapters.base import DatabaseAdapter, SchemaInfo, TableInfo
from app.models.schema_metadata import SchemaMetadata
from app.core.config import settings

logger = logging.getLogger(__name__)


def schema_to_dict(schema: SchemaInfo) -> dict:
    """Convert SchemaInfo to JSON-serializable dict."""
    return {
        "db_type": schema.db_type,
        "database_name": schema.database_name,
        "total_tables": schema.total_tables,
        "relationships": schema.relationships,
        "tables": [
            {
                "name": t.name,
                "schema": t.schema,
                "row_count": t.row_count,
                "primary_keys": t.primary_keys,
                "foreign_keys": t.foreign_keys,
                "columns": [
                    {
                        "name": c.name,
                        "data_type": c.data_type,
                        "is_nullable": c.is_nullable,
                        "is_primary_key": c.is_primary_key,
                        "is_foreign_key": c.is_foreign_key,
                        "references_table": c.references_table,
                        "references_column": c.references_column,
                    }
                    for c in t.columns
                ]
            }
            for t in schema.tables
        ]
    }


async def analyze_and_cache_schema(
    db: AsyncSession,
    database_connection_id: str,
    adapter: DatabaseAdapter,
) -> SchemaMetadata:
    """
    Analyze the connected database schema and cache it in the app database.
    """
    schema = await adapter.get_schema()
    schema_dict = schema_to_dict(schema)

    table_summaries = []
    for t in schema.tables:
        table_summaries.append({
            "name": t.name,
            "row_count": t.row_count,
            "column_count": len(t.columns),
            "columns": [c.name for c in t.columns],
        })

    # Check if metadata exists
    result = await db.execute(
        select(SchemaMetadata).where(
            SchemaMetadata.database_connection_id == database_connection_id
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.full_schema = schema_dict
        existing.table_names = [t.name for t in schema.tables]
        existing.table_summaries = table_summaries
        existing.relationships = schema.relationships
        existing.total_tables = schema.total_tables
        existing.total_relationships = len(schema.relationships)
        existing.analyzed_at = datetime.now(timezone.utc)
        await db.flush()
        return existing
    else:
        metadata = SchemaMetadata(
            database_connection_id=database_connection_id,
            db_type=schema.db_type,
            full_schema=schema_dict,
            table_names=[t.name for t in schema.tables],
            table_summaries=table_summaries,
            relationships=schema.relationships,
            total_tables=schema.total_tables,
            total_relationships=len(schema.relationships),
        )
        db.add(metadata)
        await db.flush()
        return metadata


async def get_relevant_schema(
    db: AsyncSession,
    database_connection_id: str,
    user_question: str,
    max_tables: int = None,
) -> dict:
    """
    Get the most relevant schema subset for a user question.
    
    For small databases: return full schema.
    For large databases: select relevant tables based on keyword matching.
    This is a simple implementation — can be upgraded to embedding-based retrieval.
    """
    if max_tables is None:
        max_tables = settings.MAX_SCHEMA_TABLES_TO_GEMINI

    result = await db.execute(
        select(SchemaMetadata).where(
            SchemaMetadata.database_connection_id == database_connection_id
        )
    )
    metadata = result.scalar_one_or_none()

    if not metadata or not metadata.full_schema:
        return {}

    full_schema = metadata.full_schema
    all_tables = full_schema.get("tables", [])

    if len(all_tables) <= max_tables:
        # Small enough — return full schema
        return full_schema

    # Large schema — select relevant tables
    relevant_tables = _select_relevant_tables(user_question, all_tables, max_tables)

    # Also include tables referenced by FK from selected tables
    relevant_names = {t["name"] for t in relevant_tables}
    for table in relevant_tables:
        for fk in table.get("foreign_keys", []):
            ref_table = fk.get("references_table")
            if ref_table and ref_table not in relevant_names:
                # Find and add the referenced table
                ref_data = next((t for t in all_tables if t["name"] == ref_table), None)
                if ref_data:
                    relevant_tables.append(ref_data)
                    relevant_names.add(ref_table)

    # Relevant relationships
    relevant_rels = [
        r for r in full_schema.get("relationships", [])
        if r.get("from_table") in relevant_names or r.get("to_table") in relevant_names
    ]

    return {
        **full_schema,
        "tables": relevant_tables[:max_tables],
        "relationships": relevant_rels,
        "note": f"Showing {len(relevant_tables)} most relevant tables (database has {len(all_tables)} total)."
    }


def _select_relevant_tables(question: str, tables: list[dict], max_tables: int) -> list[dict]:
    """
    Simple keyword-based table relevance scoring.
    
    Can be upgraded to embedding similarity in the future.
    """
    question_lower = question.lower()
    scored = []

    for table in tables:
        score = 0
        table_name_lower = table["name"].lower()

        # Table name match
        if table_name_lower in question_lower:
            score += 10
        # Partial name match
        elif any(word in question_lower for word in table_name_lower.split("_")):
            score += 5

        # Column name matches
        for col in table.get("columns", []):
            col_lower = col["name"].lower() if isinstance(col, dict) else col.lower()
            if col_lower in question_lower:
                score += 3

        # Common important tables get boost
        important_table_keywords = ["user", "order", "product", "customer", "sale",
                                    "payment", "transaction", "revenue", "invoice"]
        for keyword in important_table_keywords:
            if keyword in table_name_lower:
                score += 2

        scored.append((score, table))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:max_tables]]
