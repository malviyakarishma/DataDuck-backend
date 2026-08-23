"""
Database adapter factory.

Returns the correct adapter based on the connection string scheme.
Adding a new database type: implement DatabaseAdapter and add entry here.
"""
from urllib.parse import urlparse
from app.database_adapters.base import DatabaseAdapter
from app.core.exceptions import UnsupportedDatabaseError
import logging

logger = logging.getLogger(__name__)


def detect_db_type(connection_string: str) -> str:
    """Detect database type from connection string scheme."""
    cs = connection_string.strip().lower()

    if cs.startswith("postgresql") or cs.startswith("postgres"):
        return "postgresql"
    elif cs.startswith("mysql"):
        return "mysql"
    elif cs.startswith("sqlite"):
        return "sqlite"
    elif cs.startswith("mongodb"):
        return "mongodb"
    elif cs.startswith("mssql") or cs.startswith("sqlserver"):
        return "mssql"
    elif cs.startswith("oracle"):
        return "oracle"
    else:
        # Try parsing as URL
        try:
            parsed = urlparse(cs)
            scheme = parsed.scheme.split("+")[0]
            if scheme in ("postgresql", "postgres", "pg"):
                return "postgresql"
            elif scheme == "mysql":
                return "mysql"
            elif scheme == "sqlite":
                return "sqlite"
            elif scheme in ("mongodb", "mongodb+srv"):
                return "mongodb"
            elif scheme in ("mssql", "sqlserver"):
                return "mssql"
        except Exception:
            pass
        return "unknown"


def get_adapter(connection_string: str) -> DatabaseAdapter:
    """
    Return the appropriate DatabaseAdapter for the given connection string.
    
    Raises UnsupportedDatabaseError for unimplemented database types.
    """
    db_type = detect_db_type(connection_string)

    if db_type == "postgresql":
        from app.database_adapters.postgres import PostgreSQLAdapter
        return PostgreSQLAdapter(connection_string)
    elif db_type == "mysql":
        from app.database_adapters.mysql import MySQLAdapter
        return MySQLAdapter(connection_string)
    elif db_type == "sqlite":
        from app.database_adapters.sqlite import SQLiteAdapter
        return SQLiteAdapter(connection_string)
    elif db_type == "mongodb":
        from app.database_adapters.mongodb import MongoDBAdapter
        return MongoDBAdapter(connection_string)
    elif db_type in ("mssql", "oracle", "mariadb", "dynamodb", "elasticsearch"):
        raise UnsupportedDatabaseError(db_type)
    else:
        raise UnsupportedDatabaseError(db_type or "unknown")
