from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.api.deps import get_current_user_dep
from app.models.user import User
from app.models.schema_metadata import SchemaMetadata
from app.schemas.database import (
    CreateDatabaseRequest, DatabaseResponse, DatabaseListResponse,
    TestConnectionRequest, TestConnectionResponse, SchemaOverviewResponse
)
from app.services.database_service import (
    create_database_connection, test_connection_string,
    get_user_databases, get_database_by_id, delete_database, analyze_schema
)
from app.core.exceptions import (
    DatabaseConnectionError, DatabaseNotFoundError, AuthorizationError,
    UnsupportedDatabaseError
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/databases", tags=["Databases"])


def _db_to_response(conn) -> DatabaseResponse:
    return DatabaseResponse(
        id=str(conn.id),
        name=conn.name,
        db_type=conn.db_type,
        masked_connection_string=conn.masked_connection_string,
        host=conn.host,
        port=conn.port,
        database_name=conn.database_name,
        username=conn.username,
        is_connected=conn.is_connected,
        last_tested_at=conn.last_tested_at,
        schema_analyzed_at=conn.schema_analyzed_at,
        created_at=conn.created_at,
    )


@router.post("/test", response_model=TestConnectionResponse)
async def test_connection(
    request: TestConnectionRequest,
    current_user: User = Depends(get_current_user_dep),
):
    """Test a database connection string without saving it."""
    try:
        result = await test_connection_string(request.connection_string)
        return TestConnectionResponse(
            success=result["success"],
            message=result["message"],
            db_type=result["db_type"],
            database_name=result["database_name"],
        )
    except (DatabaseConnectionError, UnsupportedDatabaseError) as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Connection test failed.")


@router.post("", response_model=DatabaseResponse, status_code=status.HTTP_201_CREATED)
async def add_database(
    request: CreateDatabaseRequest,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Add a new database connection."""
    try:
        conn = await create_database_connection(db, str(current_user.id), request)
        return _db_to_response(conn)
    except (DatabaseConnectionError, UnsupportedDatabaseError) as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.get("", response_model=DatabaseListResponse)
async def list_databases(
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """List all database connections for the current user."""
    conns = await get_user_databases(db, str(current_user.id))
    return DatabaseListResponse(
        databases=[_db_to_response(c) for c in conns],
        total=len(conns),
    )


@router.get("/{database_id}", response_model=DatabaseResponse)
async def get_database(
    database_id: str,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific database connection."""
    try:
        conn = await get_database_by_id(db, database_id, str(current_user.id))
        return _db_to_response(conn)
    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.delete("/{database_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_database(
    database_id: str,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Delete a database connection."""
    try:
        await delete_database(db, database_id, str(current_user.id))
    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.post("/{database_id}/analyze-schema")
async def trigger_schema_analysis(
    database_id: str,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a schema re-analysis for a database connection."""
    try:
        result = await analyze_schema(db, database_id, str(current_user.id))
        return {"success": True, **result}
    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.get("/{database_id}/overview", response_model=SchemaOverviewResponse)
async def get_schema_overview(
    database_id: str,
    current_user: User = Depends(get_current_user_dep),
    db: AsyncSession = Depends(get_db),
):
    """Get schema overview for a database connection."""
    try:
        conn = await get_database_by_id(db, database_id, str(current_user.id))
        result = await db.execute(
            select(SchemaMetadata).where(
                SchemaMetadata.database_connection_id == database_id
            )
        )
        meta = result.scalar_one_or_none()
        if not meta:
            raise HTTPException(status_code=404, detail="Schema not yet analyzed.")

        return SchemaOverviewResponse(
            database_id=database_id,
            db_type=conn.db_type,
            database_name=conn.database_name or "unknown",
            total_tables=meta.total_tables or 0,
            total_relationships=meta.total_relationships or 0,
            tables=meta.table_summaries or [],
            relationships=meta.relationships or [],
            analyzed_at=meta.analyzed_at,
        )
    except (DatabaseNotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=404, detail=e.message)
