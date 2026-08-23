"""
SQLite database adapter.

Uses aiosqlite for async file-based SQLite connections.
Connection string format: sqlite:///path/to/file.db
"""
import asyncio
import time
import logging
from typing import Any, Optional
from urllib.parse import urlparse

import aiosqlite

from app.database_adapters.base import (
    DatabaseAdapter, SchemaInfo, TableInfo, ColumnInfo, QueryResult
)
from app.core.config import settings
from app.core.exceptions import (
    DatabaseConnectionError, QueryExecutionError, QueryTimeoutError
)

logger = logging.getLogger(__name__)


def _parse_sqlite_path(connection_string: str) -> str:
    """Extract file path from sqlite:// connection string."""
    if connection_string.startswith("sqlite:///"):
        return connection_string[10:]  # absolute path
    elif connection_string.startswith("sqlite://"):
        return connection_string[9:]
    return connection_string


class SQLiteAdapter(DatabaseAdapter):
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._db_path = _parse_sqlite_path(connection_string)
        self._db: Optional[aiosqlite.Connection] = None
        self._db_type = "sqlite"

    async def connect(self) -> None:
        try:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            logger.info(f"SQLite adapter connected to: {self._db_path}")
        except Exception as e:
            raise DatabaseConnectionError(f"SQLite connection failed: {e}")

    async def test_connection(self) -> bool:
        try:
            if not self._db:
                await self.connect()
            async with self._db.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
                return row[0] == 1
        except Exception as e:
            logger.warning(f"SQLite test_connection failed: {e}")
            return False

    async def get_database_type(self) -> str:
        return "sqlite"

    async def get_tables(self) -> list[str]:
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_table_info(self, table_name: str) -> TableInfo:
        # Get columns via PRAGMA
        async with self._db.execute(f"PRAGMA table_info('{table_name}')") as cursor:
            col_rows = await cursor.fetchall()

        # Get foreign keys
        async with self._db.execute(f"PRAGMA foreign_key_list('{table_name}')") as cursor:
            fk_rows = await cursor.fetchall()
        fk_map = {row["from"]: row for row in fk_rows}

        # Row count
        try:
            async with self._db.execute(f"SELECT COUNT(*) FROM '{table_name}'") as cursor:
                cnt = await cursor.fetchone()
                row_count = cnt[0] if cnt else None
        except Exception:
            row_count = None

        columns = []
        for row in col_rows:
            col_name = row["name"]
            fk_info = fk_map.get(col_name, {})
            columns.append(ColumnInfo(
                name=col_name,
                data_type=row["type"] or "TEXT",
                is_nullable=not row["notnull"],
                is_primary_key=bool(row["pk"]),
                is_foreign_key=bool(fk_info),
                default_value=str(row["dflt_value"]) if row["dflt_value"] is not None else None,
                max_length=None,
                references_table=fk_info.get("table") if fk_info else None,
                references_column=fk_info.get("to") if fk_info else None,
            ))

        pks = [c.name for c in columns if c.is_primary_key]
        fks = [
            {"column": c.name, "references_table": c.references_table, "references_column": c.references_column}
            for c in columns if c.is_foreign_key
        ]

        return TableInfo(
            name=table_name,
            schema=None,
            row_count=row_count,
            columns=columns,
            primary_keys=pks,
            foreign_keys=fks,
            indexes=[],
        )

    async def get_relationships(self) -> list[dict]:
        tables = await self.get_tables()
        relationships = []
        for table in tables:
            async with self._db.execute(f"PRAGMA foreign_key_list('{table}')") as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    relationships.append({
                        "from_table": table,
                        "from_column": row["from"],
                        "to_table": row["table"],
                        "to_column": row["to"],
                    })
        return relationships

    async def get_schema(self) -> SchemaInfo:
        tables_names = await self.get_tables()
        tables = []
        for name in tables_names:
            try:
                tables.append(await self.get_table_info(name))
            except Exception as e:
                logger.warning(f"Could not get info for table {name}: {e}")

        relationships = await self.get_relationships()
        db_name = self._db_path.split("/")[-1].split("\\")[-1]

        return SchemaInfo(
            db_type="sqlite",
            database_name=db_name,
            tables=tables,
            total_tables=len(tables),
            relationships=relationships,
            raw_metadata={},
        )

    async def execute_read_query(self, query: Any, timeout: int = 30) -> QueryResult:
        if not self._db:
            raise QueryExecutionError("Not connected.")

        start_time = time.monotonic()
        try:
            async def _fetch():
                async with self._db.execute(query) as cursor:
                    rows = await cursor.fetchmany(settings.MAX_QUERY_ROWS + 1)
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    return rows, columns
            rows, columns = await asyncio.wait_for(_fetch(), timeout=timeout)

            elapsed = (time.monotonic() - start_time) * 1000
            truncated = len(rows) > settings.MAX_QUERY_ROWS
            rows = rows[:settings.MAX_QUERY_ROWS]

            if not rows:
                return QueryResult(columns=columns or [], rows=[], row_count=0, execution_time_ms=elapsed, truncated=False)

            return QueryResult(
                columns=columns,
                rows=[dict(zip(columns, row)) for row in rows],
                row_count=len(rows),
                execution_time_ms=elapsed,
                truncated=truncated,
            )
        except asyncio.TimeoutError:
            raise QueryTimeoutError()
        except Exception as e:
            logger.error(f"SQLite query error: {e}")
            raise QueryExecutionError(f"SQLite query failed: {type(e).__name__}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
