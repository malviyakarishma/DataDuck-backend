"""
FastAPI dependencies — authentication, database session injection.
"""
from fastapi import Depends, Cookie, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.models.user import User
from app.security.auth import decode_token
from app.services.auth_service import get_current_user
from app.core.exceptions import AuthenticationError

security = HTTPBearer(auto_error=False)


async def get_current_user_dep(
    db: AsyncSession = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(None, alias="access_token"),
) -> User:
    """
    Get current authenticated user from JWT.
    Supports both Authorization header and HttpOnly cookie.
    """
    token = None

    # Try Authorization header first
    if credentials and credentials.credentials:
        token = credentials.credentials
    # Fall back to cookie
    elif access_token:
        token = access_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(token, expected_type="access")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token.")
        user = await get_current_user(db, user_id)
        return user
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed.")
