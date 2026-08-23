"""
MySQL database adapter.

Uses aiomysql for async connections.
All queries are executed in read-only mode.
"""
import asyncio
import time
import logging
from typing import Any, Optional

import aiomysql

from app.database_adapters.base import (
    DatabaseAdapter, SchemaInfo, TableInfo, ColumnInfo, QueryResult
)
from app.core.config import settings
from app.core.exceptions import (
    DatabaseConnectionError, QueryExecutionError, QueryTimeoutError
)
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _parse_mysql_url(url: str) -> dict:
    """Parse mysql:// or mysql+aiomysql:// URL."""
    url = url.replace("mysql+aiomysql://", "mysql://").replace("mysql+pymysql://", "mysql://")
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": parsed.username or "root",
        "password": parsed.password or "",
        "db": parsed.path.lstrip("/") if parsed.path else "",
    }


class MySQLAdapter(DatabaseAdapter):
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._pool: Optional[aiomysql.Pool] = None
        self._conn_params = _parse_mysql_url(connection_string)
        self._db_type = "mysql"

    async def connect(self) -> None:
        try:
            self._pool = await aiomysql.create_pool(
                host=self._conn_params["host"],
                port=self._conn_params["port"],
                user=self._conn_params["user"],
                password=self._conn_params["password"],
                db=self._conn_params["db"],
                minsize=1,
                maxsize=5,
                connect_timeout=10,
                charset="utf8mb4",
            )
            logger.info("MySQL adapter connected.")
        except Exception as e:
            raise DatabaseConnectionError(f"MySQL connection failed: {str(e)}")

    async def test_connection(self) -> bool:
        try:
            if not self._pool:
                await self.connect()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.warning(f"MySQL test_connection failed: {e}")
            return False

    async def get_database_type(self) -> str:
        return "mysql"

    async def get_tables(self) -> list[str]:
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT TABLE_NAME FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME",
                    (self._conn_params["db"],)
                )
                rows = await cursor.fetchall()
                return [row["TABLE_NAME"] for row in rows]

    async def get_table_info(self, table_name: str) -> TableInfo:
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT,
                           CHARACTER_MAXIMUM_LENGTH, COLUMN_KEY
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """, (self._conn_params["db"], table_name))
                col_rows = await cursor.fetchall()

                # Get FK info
                await cursor.execute("""
                    SELECT kcu.COLUMN_NAME, kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME
                    FROM information_schema.KEY_COLUMN_USAGE kcu
                    JOIN information_schema.TABLE_CONSTRAINTS tc
                        ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                    WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                      AND kcu.TABLE_SCHEMA = %s AND kcu.TABLE_NAME = %s
                """, (self._conn_params["db"], table_name))
                fk_rows = await cursor.fetchall()
                fk_map = {row["COLUMN_NAME"]: row for row in fk_rows}

                # Row count
                try:
                    await cursor.execute(f"SELECT COUNT(*) as cnt FROM `{table_name}`")
                    cnt_row = await cursor.fetchone()
                    row_count = cnt_row["cnt"] if cnt_row else None
                except Exception:
                    row_count = None

                columns = []
                for row in col_rows:
                    fk_info = fk_map.get(row["COLUMN_NAME"], {})
                    columns.append(ColumnInfo(
                        name=row["COLUMN_NAME"],
                        data_type=row["DATA_TYPE"],
                        is_nullable=row["IS_NULLABLE"] == "YES",
                        is_primary_key=row["COLUMN_KEY"] == "PRI",
                        is_foreign_key=bool(fk_info),
                        default_value=str(row["COLUMN_DEFAULT"]) if row["COLUMN_DEFAULT"] is not None else None,
                        max_length=row["CHARACTER_MAXIMUM_LENGTH"],
                        references_table=fk_info.get("REFERENCED_TABLE_NAME"),
                        references_column=fk_info.get("REFERENCED_COLUMN_NAME"),
                    ))

                pks = [c.name for c in columns if c.is_primary_key]
                fks = [
                    {"column": c.name, "references_table": c.references_table, "references_column": c.references_column}
                    for c in columns if c.is_foreign_key
                ]

                return TableInfo(
                    name=table_name,
                    schema=self._conn_params["db"],
                    row_count=row_count,
                    columns=columns,
                    primary_keys=pks,
                    foreign_keys=fks,
                    indexes=[],
                )

    async def get_relationships(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT kcu.TABLE_NAME as from_table, kcu.COLUMN_NAME as from_column,
                           kcu.REFERENCED_TABLE_NAME as to_table, kcu.REFERENCED_COLUMN_NAME as to_column
                    FROM information_schema.KEY_COLUMN_USAGE kcu
                    JOIN information_schema.TABLE_CONSTRAINTS tc ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                    WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY' AND kcu.TABLE_SCHEMA = %s
                    ORDER BY kcu.TABLE_NAME
                """, (self._conn_params["db"],))
                return await cursor.fetchall()

    async def get_schema(self) -> SchemaInfo:
        tables_names = await self.get_tables()
        tables = []
        for name in tables_names:
            try:
                tables.append(await self.get_table_info(name))
            except Exception as e:
                logger.warning(f"Could not get info for table {name}: {e}")

        relationships = await self.get_relationships()

        return SchemaInfo(
            db_type="mysql",
            database_name=self._conn_params["db"],
            tables=tables,
            total_tables=len(tables),
            relationships=relationships,
            raw_metadata={},
        )

    async def execute_read_query(self, query: Any, timeout: int = 30) -> QueryResult:
        if not self._pool:
            raise QueryExecutionError("Not connected to database.")

        start_time = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                # Set read-only mode
                await conn.execute("SET SESSION TRANSACTION READ ONLY")
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    async def _fetch():
                        await cursor.execute(query)
                        return await cursor.fetchmany(settings.MAX_QUERY_ROWS + 1)
                    rows = await asyncio.wait_for(_fetch(), timeout=timeout)

            elapsed = (time.monotonic() - start_time) * 1000
            truncated = len(rows) > settings.MAX_QUERY_ROWS
            rows = rows[:settings.MAX_QUERY_ROWS]

            if not rows:
                return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=elapsed, truncated=False)

            return QueryResult(
                columns=list(rows[0].keys()),
                rows=[dict(r) for r in rows],
                row_count=len(rows),
                execution_time_ms=elapsed,
                truncated=truncated,
            )
        except asyncio.TimeoutError:
            raise QueryTimeoutError()
        except Exception as e:
            logger.error(f"MySQL query execution error: {type(e).__name__}: {e}")
            raise QueryExecutionError(f"Query execution failed: {type(e).__name__}")

    async def close(self) -> None:
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            logger.info("MySQL adapter closed.")
