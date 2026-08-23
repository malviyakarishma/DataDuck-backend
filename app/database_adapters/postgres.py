"""
PostgreSQL database adapter.

Uses asyncpg for async queries with proper timeout and row limiting.
All operations are read-only — queries are wrapped in READ ONLY transactions.
"""
import asyncio
import time
import logging
from typing import Any, Optional

import asyncpg

from app.database_adapters.base import (
    DatabaseAdapter, SchemaInfo, TableInfo, ColumnInfo, QueryResult
)
from app.core.config import settings
from app.core.exceptions import (
    DatabaseConnectionError, QueryExecutionError, QueryTimeoutError
)

logger = logging.getLogger(__name__)


class PostgreSQLAdapter(DatabaseAdapter):
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._pool: Optional[asyncpg.Pool] = None
        self._db_type = "postgresql"

    async def connect(self) -> None:
        try:
            # Convert SQLAlchemy-style URL to asyncpg format
            dsn = self._connection_string
            if dsn.startswith("postgresql+asyncpg://"):
                dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
            elif dsn.startswith("postgresql+psycopg2://"):
                dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")

            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=5,
                command_timeout=settings.QUERY_TIMEOUT_SECONDS,
            )
            logger.info("PostgreSQL adapter connected (pool created).")
        except asyncpg.InvalidCatalogNameError as e:
            raise DatabaseConnectionError(f"Database not found: {e}")
        except asyncpg.InvalidPasswordError:
            raise DatabaseConnectionError("Invalid database credentials.")
        except asyncpg.CannotConnectNowError as e:
            raise DatabaseConnectionError(f"Cannot connect to PostgreSQL: {e}")
        except Exception as e:
            raise DatabaseConnectionError(f"PostgreSQL connection failed: {str(e)}")

    async def test_connection(self) -> bool:
        try:
            if not self._pool:
                await self.connect()
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.warning(f"PostgreSQL test_connection failed: {e}")
            return False

    async def get_database_type(self) -> str:
        return "postgresql"

    async def get_tables(self) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            return [row["table_name"] for row in rows]

    async def get_table_info(self, table_name: str) -> TableInfo:
        async with self._pool.acquire() as conn:
            # Columns
            col_rows = await conn.fetch("""
                SELECT 
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    c.character_maximum_length,
                    CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                    CASE WHEN fk.column_name IS NOT NULL THEN true ELSE false END as is_foreign_key,
                    fk.foreign_table_name,
                    fk.foreign_column_name
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT ku.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage ku
                        ON tc.constraint_name = ku.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND ku.table_name = $1
                      AND ku.table_schema = 'public'
                ) pk ON c.column_name = pk.column_name
                LEFT JOIN (
                    SELECT 
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND kcu.table_name = $1
                ) fk ON c.column_name = fk.column_name
                WHERE c.table_name = $1 AND c.table_schema = 'public'
                ORDER BY c.ordinal_position
            """, table_name)

            # Approximate row count
            try:
                row_count_row = await conn.fetchrow(
                    "SELECT reltuples::bigint as cnt FROM pg_class WHERE relname = $1",
                    table_name
                )
                row_count = int(row_count_row["cnt"]) if row_count_row else None
            except Exception:
                row_count = None

            columns = [
                ColumnInfo(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    is_nullable=row["is_nullable"] == "YES",
                    is_primary_key=row["is_primary_key"],
                    is_foreign_key=row["is_foreign_key"],
                    default_value=str(row["column_default"]) if row["column_default"] else None,
                    max_length=row["character_maximum_length"],
                    references_table=row["foreign_table_name"],
                    references_column=row["foreign_column_name"],
                )
                for row in col_rows
            ]

            pks = [c.name for c in columns if c.is_primary_key]
            fks = [
                {"column": c.name, "references_table": c.references_table, "references_column": c.references_column}
                for c in columns if c.is_foreign_key
            ]

            return TableInfo(
                name=table_name,
                schema="public",
                row_count=row_count,
                columns=columns,
                primary_keys=pks,
                foreign_keys=fks,
                indexes=[],
            )

    async def get_relationships(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    kcu.table_name as from_table,
                    kcu.column_name as from_column,
                    ccu.table_name as to_table,
                    ccu.column_name as to_column,
                    tc.constraint_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.table_schema = 'public'
                ORDER BY kcu.table_name
            """)
            return [dict(row) for row in rows]

    async def get_schema(self) -> SchemaInfo:
        tables_names = await self.get_tables()
        tables = []
        for name in tables_names:
            try:
                table_info = await self.get_table_info(name)
                tables.append(table_info)
            except Exception as e:
                logger.warning(f"Could not get info for table {name}: {e}")

        relationships = await self.get_relationships()

        async with self._pool.acquire() as conn:
            db_name_row = await conn.fetchrow("SELECT current_database()")
            db_name = db_name_row[0] if db_name_row else "unknown"

        return SchemaInfo(
            db_type="postgresql",
            database_name=db_name,
            tables=tables,
            total_tables=len(tables),
            relationships=relationships,
            raw_metadata={},
        )

    async def execute_read_query(self, query: Any, timeout: int = 30) -> QueryResult:
        """Execute a validated SELECT query in a read-only transaction."""
        if not self._pool:
            raise QueryExecutionError("Not connected to database.")

        start_time = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                # Set session to read only for extra safety
                await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
                records = await asyncio.wait_for(conn.fetch(query), timeout=timeout)

            elapsed = (time.monotonic() - start_time) * 1000
            truncated = len(records) >= settings.MAX_QUERY_ROWS

            if not records:
                return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=elapsed, truncated=False)

            columns = list(records[0].keys())
            rows = [dict(r) for r in records[:settings.MAX_QUERY_ROWS]]

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=elapsed,
                truncated=truncated,
            )

        except asyncio.TimeoutError:
            raise QueryTimeoutError()
        except asyncpg.exceptions.InsufficientPrivilegeError:
            raise QueryExecutionError("Insufficient privileges to execute this query.")
        except asyncpg.exceptions.UndefinedTableError as e:
            raise QueryExecutionError(f"Table not found: {e}")
        except asyncpg.exceptions.UndefinedColumnError as e:
            raise QueryExecutionError(f"Column not found: {e}")
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error(f"PostgreSQL query execution error: {type(e).__name__}: {e}")
            raise QueryExecutionError(f"Query execution failed: {type(e).__name__}")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL adapter pool closed.")
