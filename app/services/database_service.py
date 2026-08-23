"""
Database connection service — manages user database connections with ownership enforcement.
"""
import logging
from typing import Optional
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.models.database_connection import DatabaseConnection
from app.schemas.database import CreateDatabaseRequest
from app.security.encryption import encrypt_string, mask_connection_string
from app.database_adapters.factory import get_adapter, detect_db_type
from app.services.schema_service import analyze_and_cache_schema
from app.core.exceptions import (
    DatabaseConnectionError, DatabaseNotFoundError, AuthorizationError,
    UnsupportedDatabaseError
)

logger = logging.getLogger(__name__)


def _parse_connection_metadata(connection_string: str) -> dict:
    """Extract metadata from connection string (no passwords)."""
    try:
        parsed = urlparse(connection_string)
        return {
            "host": parsed.hostname,
            "port": parsed.port,
            "database_name": parsed.path.lstrip("/").split("?")[0] if parsed.path else None,
            "username": parsed.username,
        }
    except Exception:
        return {}


async def create_database_connection(
    db: AsyncSession,
    user_id: str,
    request: CreateDatabaseRequest,
) -> DatabaseConnection:
    """Create and test a new database connection for a user."""
    # Validate and test connection
    db_type = detect_db_type(request.connection_string)
    if db_type == "unknown":
        raise DatabaseConnectionError("Unrecognized connection string format.")

    if db_type in ("mssql", "oracle"):
        raise UnsupportedDatabaseError(db_type)

    # Test the connection
    adapter = get_adapter(request.connection_string)
    try:
        await adapter.connect()
        success = await adapter.test_connection()
        if not success:
            raise DatabaseConnectionError("Connection test failed.")
    except (DatabaseConnectionError, UnsupportedDatabaseError):
        raise
    except Exception as e:
        raise DatabaseConnectionError(f"Could not connect to database: {str(e)}")
    finally:
        await adapter.close()

    # Parse metadata
    meta = _parse_connection_metadata(request.connection_string)

    # Encrypt connection string
    encrypted = encrypt_string(request.connection_string)
    masked = mask_connection_string(request.connection_string)

    conn = DatabaseConnection(
        owner_id=user_id,
        name=request.name.strip(),
        db_type=db_type,
        encrypted_connection_string=encrypted,
        masked_connection_string=masked,
        host=meta.get("host"),
        port=meta.get("port"),
        database_name=meta.get("database_name"),
        username=meta.get("username"),
        is_connected=True,
        last_tested_at=datetime.now(timezone.utc),
    )
    db.add(conn)
    await db.flush()
    await db.refresh(conn)

    # Analyze schema in background (quick inline for MVP)
    try:
        from app.security.encryption import decrypt_string
        plain = decrypt_string(encrypted)
        adapter2 = get_adapter(plain)
        await adapter2.connect()
        await analyze_and_cache_schema(db, str(conn.id), adapter2)
        await adapter2.close()
        conn.schema_analyzed_at = datetime.now(timezone.utc)
        await db.flush()
    except Exception as e:
        logger.warning(f"Schema analysis failed for new connection {conn.id}: {e}")

    logger.info(f"Database connection created: {conn.id} ({db_type}) for user {user_id}")
    return conn


async def test_connection_string(connection_string: str) -> dict:
    """Test a connection string without saving it."""
    db_type = detect_db_type(connection_string)
    if db_type == "unknown":
        raise DatabaseConnectionError("Unrecognized connection string format.")

    adapter = get_adapter(connection_string)
    try:
        await adapter.connect()
        success = await adapter.test_connection()
        schema = await adapter.get_schema()
        return {
            "success": success,
            "db_type": db_type,
            "database_name": schema.database_name,
            "table_count": schema.total_tables,
            "message": f"Successfully connected to {db_type} database '{schema.database_name}'.",
        }
    except DatabaseConnectionError:
        raise
    except UnsupportedDatabaseError:
        raise
    except Exception as e:
        raise DatabaseConnectionError(f"Connection failed: {str(e)}")
    finally:
        try:
            await adapter.close()
        except Exception:
            pass


async def get_user_databases(db: AsyncSession, user_id: str) -> list[DatabaseConnection]:
    """Get all database connections owned by a user."""
    result = await db.execute(
        select(DatabaseConnection)
        .where(DatabaseConnection.owner_id == user_id)
        .order_by(DatabaseConnection.created_at.desc())
    )
    return result.scalars().all()


async def get_database_by_id(db: AsyncSession, database_id: str, user_id: str) -> DatabaseConnection:
    """Get a database connection, enforcing ownership."""
    result = await db.execute(
        select(DatabaseConnection).where(DatabaseConnection.id == database_id)
    )
    conn = result.scalar_one_or_none()

    if not conn:
        raise DatabaseNotFoundError()

    if str(conn.owner_id) != str(user_id):
        raise AuthorizationError("You do not have access to this database connection.")

    return conn


async def delete_database(db: AsyncSession, database_id: str, user_id: str) -> None:
    """Delete a database connection, enforcing ownership."""
    conn = await get_database_by_id(db, database_id, user_id)
    await db.delete(conn)
    await db.flush()
    logger.info(f"Database connection {database_id} deleted by user {user_id}")


async def analyze_schema(
    db: AsyncSession,
    database_id: str,
    user_id: str,
) -> dict:
    """Re-analyze schema for an existing connection."""
    from app.security.encryption import decrypt_string

    conn = await get_database_by_id(db, database_id, user_id)
    plain = decrypt_string(conn.encrypted_connection_string)
    adapter = get_adapter(plain)

    try:
        await adapter.connect()
        metadata = await analyze_and_cache_schema(db, database_id, adapter)
        conn.schema_analyzed_at = datetime.now(timezone.utc)
        await db.flush()

        return {
            "total_tables": metadata.total_tables,
            "table_names": metadata.table_names,
            "total_relationships": metadata.total_relationships,
            "analyzed_at": metadata.analyzed_at.isoformat() if metadata.analyzed_at else None,
        }
    finally:
        await adapter.close()
