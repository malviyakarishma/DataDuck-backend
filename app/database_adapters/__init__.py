# backend/app/database_adapters/__init__.py
from app.database_adapters.factory import get_adapter, detect_db_type
from app.database_adapters.base import DatabaseAdapter, SchemaInfo, TableInfo, ColumnInfo, QueryResult

__all__ = ["get_adapter", "detect_db_type", "DatabaseAdapter", "SchemaInfo", "TableInfo", "ColumnInfo", "QueryResult"]
