"""
MongoDB adapter using Motor (async MongoDB driver).

Only read operations are allowed. Write operations are blocked at both
the adapter level and the query validator level.
"""
import asyncio
import time
import logging
from typing import Any, Optional
from urllib.parse import urlparse

import motor.motor_asyncio
from pymongo.errors import ConnectionFailure, OperationFailure

from app.database_adapters.base import (
    DatabaseAdapter, SchemaInfo, TableInfo, ColumnInfo, QueryResult
)
from app.core.config import settings
from app.core.exceptions import (
    DatabaseConnectionError, QueryExecutionError, QueryTimeoutError
)

logger = logging.getLogger(__name__)

ALLOWED_OPS = {"find", "findOne", "aggregate", "countDocuments", "distinct", "estimatedDocumentCount", "count"}


class MongoDBAdapter(DatabaseAdapter):
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self._db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        self._db_name: Optional[str] = None
        self._db_type = "mongodb"

    async def connect(self) -> None:
        try:
            parsed = urlparse(self._connection_string)
            db_name = parsed.path.lstrip("/").split("?")[0] if parsed.path else None
            if not db_name:
                raise DatabaseConnectionError("MongoDB connection string must include database name in path.")

            self._client = motor.motor_asyncio.AsyncIOMotorClient(
                self._connection_string,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
            )
            self._db_name = db_name
            self._db = self._client[db_name]
            # Ping to verify
            await self._client.admin.command("ping")
            logger.info(f"MongoDB adapter connected to: {db_name}")
        except DatabaseConnectionError:
            raise
        except Exception as e:
            raise DatabaseConnectionError(f"MongoDB connection failed: {str(e)}")

    async def test_connection(self) -> bool:
        try:
            if not self._client:
                await self.connect()
            await self._client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"MongoDB test_connection failed: {e}")
            return False

    async def get_database_type(self) -> str:
        return "mongodb"

    async def get_tables(self) -> list[str]:
        """Returns collection names."""
        collections = await self._db.list_collection_names()
        return sorted(collections)

    async def _infer_schema_from_sample(self, collection_name: str, sample_size: int = 100) -> list[ColumnInfo]:
        """Infer field types from a sample of documents."""
        collection = self._db[collection_name]
        cursor = collection.find({}, limit=sample_size)
        docs = await cursor.to_list(length=sample_size)

        if not docs:
            return []

        field_types: dict[str, set] = {}
        for doc in docs:
            self._extract_fields(doc, "", field_types)

        columns = []
        for field, types in field_types.items():
            columns.append(ColumnInfo(
                name=field,
                data_type=" | ".join(sorted(types)),
                is_nullable=None in types or "NoneType" in types,
                is_primary_key=field == "_id",
                is_foreign_key=False,
            ))

        return columns

    def _extract_fields(self, doc: dict, prefix: str, field_types: dict, depth: int = 0) -> None:
        """Recursively extract fields from a document."""
        if depth > 3:
            return
        for key, value in doc.items():
            full_key = f"{prefix}.{key}" if prefix else key
            type_name = type(value).__name__
            if full_key not in field_types:
                field_types[full_key] = set()
            field_types[full_key].add(type_name)
            if isinstance(value, dict) and depth < 3:
                self._extract_fields(value, full_key, field_types, depth + 1)

    async def get_table_info(self, collection_name: str) -> TableInfo:
        columns = await self._infer_schema_from_sample(collection_name)
        try:
            row_count = await self._db[collection_name].estimated_document_count()
        except Exception:
            row_count = None

        return TableInfo(
            name=collection_name,
            schema=self._db_name,
            row_count=row_count,
            columns=columns,
            primary_keys=["_id"],
            foreign_keys=[],
            indexes=[],
        )

    async def get_relationships(self) -> list[dict]:
        """MongoDB doesn't have explicit FK relationships."""
        return []

    async def get_schema(self) -> SchemaInfo:
        collection_names = await self.get_tables()
        tables = []
        for name in collection_names[:50]:  # Limit to 50 collections for schema analysis
            try:
                tables.append(await self.get_table_info(name))
            except Exception as e:
                logger.warning(f"Could not get info for collection {name}: {e}")

        return SchemaInfo(
            db_type="mongodb",
            database_name=self._db_name or "unknown",
            tables=tables,
            total_tables=len(tables),
            relationships=[],
            raw_metadata={},
        )

    async def execute_read_query(self, query: Any, timeout: int = 30) -> QueryResult:
        """
        Execute a MongoDB read operation.

        query must be a dict with:
        {
            "collection": str,
            "operation": "find" | "findOne" | "aggregate" | "countDocuments" | "distinct",
            "filter": dict (optional),
            "pipeline": list (optional, for aggregate),
            "projection": dict (optional),
            "sort": dict (optional),
            "limit": int (optional),
            "field": str (optional, for distinct)
        }
        """
        if not self._db:
            raise QueryExecutionError("Not connected to MongoDB.")

        if not isinstance(query, dict):
            raise QueryExecutionError("MongoDB query must be a dictionary.")

        operation = query.get("operation", "").strip()
        if operation not in ALLOWED_OPS:
            raise QueryExecutionError(f"Operation '{operation}' is not allowed.")

        collection_name = query.get("collection", "")
        if not collection_name:
            raise QueryExecutionError("Collection name is required.")

        start_time = time.monotonic()
        collection = self._db[collection_name]
        limit = min(query.get("limit", settings.MAX_QUERY_ROWS), settings.MAX_QUERY_ROWS)

        try:
            async def _execute():
                if operation == "find":
                    cursor = collection.find(
                        query.get("filter", {}),
                        query.get("projection"),
                    ).sort(list(query.get("sort", {}).items()) or [("_id", 1)]).limit(limit + 1)
                    docs = await cursor.to_list(length=limit + 1)
                    truncated = len(docs) > limit
                    docs = docs[:limit]
                    rows = [self._serialize_doc(d) for d in docs]
                    return rows, truncated

                elif operation == "findOne":
                    doc = await collection.find_one(query.get("filter", {}), query.get("projection"))
                    rows = [self._serialize_doc(doc)] if doc else []
                    return rows, False

                elif operation == "aggregate":
                    pipeline = query.get("pipeline", [])
                    has_limit = any("$limit" in stage for stage in pipeline)
                    if not has_limit:
                        pipeline = pipeline + [{"$limit": limit}]
                    cursor = collection.aggregate(pipeline)
                    docs = await cursor.to_list(length=limit + 1)
                    truncated = len(docs) > limit
                    docs = docs[:limit]
                    rows = [self._serialize_doc(d) for d in docs]
                    return rows, truncated

                elif operation == "countDocuments":
                    count = await collection.count_documents(query.get("filter", {}))
                    return [{"count": count}], False

                elif operation == "distinct":
                    field = query.get("field", "_id")
                    values = await collection.distinct(field, query.get("filter", {}))
                    rows = [{field: v} for v in values[:limit]]
                    return rows, len(values) > limit

                else:
                    raise QueryExecutionError(f"Unsupported operation: {operation}")

            rows, truncated = await asyncio.wait_for(_execute(), timeout=timeout)

            elapsed = (time.monotonic() - start_time) * 1000
            columns = list(rows[0].keys()) if rows else []

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=elapsed,
                truncated=truncated,
            )

        except asyncio.TimeoutError:
            raise QueryTimeoutError()
        except OperationFailure as e:
            raise QueryExecutionError(f"MongoDB operation failed: {e.details.get('errmsg', str(e)) if e.details else str(e)}")
        except Exception as e:
            logger.error(f"MongoDB execution error: {type(e).__name__}: {e}")
            raise QueryExecutionError(f"MongoDB query failed: {type(e).__name__}")

    def _serialize_doc(self, doc: dict) -> dict:
        """Convert MongoDB document to JSON-serializable dict."""
        from bson import ObjectId
        from datetime import datetime
        result = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                result[k] = str(v)
            elif isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, dict):
                result[k] = self._serialize_doc(v)
            elif isinstance(v, list):
                result[k] = [self._serialize_doc(i) if isinstance(i, dict) else str(i) if isinstance(i, ObjectId) else i for i in v]
            else:
                result[k] = v
        return result

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            logger.info("MongoDB adapter closed.")
