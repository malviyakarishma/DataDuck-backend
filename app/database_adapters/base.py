"""
Abstract base class for all database adapters.

All adapters must implement this interface. Adding a new database type
requires only implementing this class and registering it in factory.py.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    default_value: Optional[str] = None
    max_length: Optional[int] = None
    references_table: Optional[str] = None
    references_column: Optional[str] = None


@dataclass
class TableInfo:
    name: str
    schema: Optional[str]
    row_count: Optional[int]
    columns: list[ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[dict]
    indexes: list[dict]


@dataclass
class SchemaInfo:
    db_type: str
    database_name: str
    tables: list[TableInfo]
    total_tables: int
    relationships: list[dict]
    raw_metadata: dict


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int
    execution_time_ms: float
    truncated: bool  # True if row limit was hit


class DatabaseAdapter(ABC):
    """
    Abstract interface for database adapters.
    
    All implementations must be READ-ONLY.
    Connection credentials must never be logged or exposed.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        ...

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if connection is alive. Returns True if successful."""
        ...

    @abstractmethod
    async def get_schema(self) -> SchemaInfo:
        """Retrieve full schema information."""
        ...

    @abstractmethod
    async def get_tables(self) -> list[str]:
        """Get list of all table/collection names."""
        ...

    @abstractmethod
    async def get_table_info(self, table_name: str) -> TableInfo:
        """Get detailed info for a specific table/collection."""
        ...

    @abstractmethod
    async def get_relationships(self) -> list[dict]:
        """Get foreign key relationships between tables."""
        ...

    @abstractmethod
    async def execute_read_query(self, query: Any, timeout: int = 30) -> QueryResult:
        """
        Execute a validated read-only query.
        
        For SQL databases: query is a SQL string.
        For MongoDB: query is a dict describing the operation.
        
        Must enforce:
        - Read-only (no writes allowed at adapter level too)
        - Timeout
        - Row limit
        """
        ...

    @abstractmethod
    async def get_database_type(self) -> str:
        """Return database type string: postgresql, mysql, sqlite, mongodb."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close database connection and clean up resources."""
        ...

    def get_sql_dialect(self) -> str:
        """Return SQLGlot dialect string for this database type."""
        dialect_map = {
            "postgresql": "postgres",
            "mysql": "mysql",
            "sqlite": "sqlite",
            "mssql": "tsql",
        }
        return dialect_map.get(self.get_database_type.__func__(self) if hasattr(self, '_db_type') else "postgresql", "postgres")
