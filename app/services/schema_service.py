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
                        "default_value": c.default_value,
                        "max_length": c.max_length,
                        "references_table": c.references_table,
                        "references_column": c.references_column,
                    }
                    for c in t.columns
                ]
            }
            for t in schema.tables
        ]
    }


def _sanitize_mermaid_type(data_type: str) -> str:
    """Clean data type for Mermaid ER diagram syntax."""
    if not data_type:
        return "text"
    # Remove parameters like (255), (10, 2), array brackets, etc.
    cleaned = data_type.split("(")[0].strip()
    cleaned = cleaned.replace("[]", "_array").replace(" ", "_")
    # Clean non-alphanumeric characters
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    return cleaned.lower() or "text"


def _sanitize_mermaid_identifier(name: str) -> str:
    """Ensure identifier has valid Mermaid naming."""
    if not name:
        return "table"
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name.strip())
    return cleaned


def generate_mermaid_er_diagram(schema_dict: dict) -> str:
    """
    Dynamically generate clean, valid Mermaid ER diagram syntax from a database schema.
    """
    if not schema_dict or not schema_dict.get("tables"):
        return "erDiagram\n    DATABASE ||--|| EMPTY_SCHEMA : contains"

    lines = ["erDiagram"]
    tables = schema_dict.get("tables", [])
    relationships = schema_dict.get("relationships", [])

    # 1. Render tables and columns
    for table in tables:
        t_name = _sanitize_mermaid_identifier(table.get("name", "table"))
        lines.append(f"    {t_name} {{")
        columns = table.get("columns", [])
        if not columns:
            lines.append("        string id PK")
        else:
            for col in columns:
                c_name = _sanitize_mermaid_identifier(col.get("name", "col"))
                c_type = _sanitize_mermaid_type(col.get("data_type", "string"))
                is_pk = col.get("is_primary_key", False)
                is_fk = col.get("is_foreign_key", False)

                key_label = ""
                if is_pk and is_fk:
                    key_label = " PK,FK"
                elif is_pk:
                    key_label = " PK"
                elif is_fk:
                    key_label = " FK"

                # Comments can include nullability or defaults
                comment = ""
                if col.get("is_nullable") is False and not is_pk:
                    comment = ' "NOT NULL"'

                lines.append(f"        {c_type} {c_name}{key_label}{comment}")
        lines.append("    }")

    # 2. Render relationships (foreign keys)
    seen_rel = set()
    for rel in relationships:
        from_t = _sanitize_mermaid_identifier(rel.get("from_table", ""))
        to_t = _sanitize_mermaid_identifier(rel.get("to_table", ""))
        from_c = rel.get("from_column", "")
        to_c = rel.get("to_column", "")

        if from_t and to_t:
            rel_key = f"{from_t}_{from_c}_{to_t}_{to_c}"
            if rel_key not in seen_rel:
                seen_rel.add(rel_key)
                label = f"{from_c}_to_{to_c}" if from_c and to_c else "references"
                # to_table (1) has many from_table (*) records
                lines.append(f'    {to_t} ||--o{{ {from_t} : "{label}"')

    # Also inspect foreign keys within table columns if relationships list was empty
    if not relationships:
        for table in tables:
            from_t = _sanitize_mermaid_identifier(table.get("name", ""))
            for col in table.get("columns", []):
                if col.get("is_foreign_key") and col.get("references_table"):
                    to_t = _sanitize_mermaid_identifier(col.get("references_table", ""))
                    from_c = col.get("name", "")
                    to_c = col.get("references_column", "id")
                    rel_key = f"{from_t}_{from_c}_{to_t}_{to_c}"
                    if rel_key not in seen_rel:
                        seen_rel.add(rel_key)
                        label = f"{from_c}_to_{to_c}" if from_c else "references"
                        lines.append(f'    {to_t} ||--o{{ {from_t} : "{label}"')

    return "\n".join(lines)


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


async def get_full_schema_metadata(
    db: AsyncSession,
    database_connection_id: str,
) -> Optional[dict]:
    """Retrieve the full cached schema metadata for a database connection."""
    result = await db.execute(
        select(SchemaMetadata).where(
            SchemaMetadata.database_connection_id == database_connection_id
        )
    )
    metadata = result.scalar_one_or_none()
    if not metadata or not metadata.full_schema:
        return None
    return metadata.full_schema

