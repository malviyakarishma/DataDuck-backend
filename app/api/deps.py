"""
FastAPI dependencies — authentication, database session injection.
"""
from fastapi import Depends, Cookie, HTTPException, status, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.models.user import User
from app.security.auth import decode_token, create_access_token
from app.services.auth_service import get_current_user
from app.core.exceptions import AuthenticationError
from app.core.config import settings

security = HTTPBearer(auto_error=False)
IS_DEV = settings.ENVIRONMENT == "development"


def _set_access_cookie(response: Response, access_token: str):
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax" if IS_DEV else "none",
        secure=not IS_DEV,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


async def get_current_user_dep(
    response: Response,
    db: AsyncSession = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(None, alias="access_token"),
    refresh_token: Optional[str] = Cookie(None, alias="refresh_token"),
) -> User:
    """
    Get current authenticated user from JWT.
    Supports Authorization header and HttpOnly access_token cookie.
    If access_token is missing or expired, automatically validates refresh_token cookie
    and generates/sets a new access_token cookie so the user is never logged out.
    """
    token = None

    # Try Authorization header first
    if credentials and credentials.credentials:
        token = credentials.credentials
    # Fall back to cookie
    elif access_token:
        token = access_token

    # 1. If access token is provided, attempt to decode it
    if token:
        try:
            payload = decode_token(token, expected_type="access")
            user_id = payload.get("sub")
            if user_id:
                user = await get_current_user(db, user_id)
                return user
        except Exception:
            # Token invalid or expired — fall through to refresh token check
            pass

    # 2. If access token missing or expired, check refresh_token in cookies
    if refresh_token:
        try:
            payload = decode_token(refresh_token, expected_type="refresh")
            user_id = payload.get("sub")
            if user_id:
                user = await get_current_user(db, user_id)
                # Generate new access token and set cookie seamlessly
                new_access_token = create_access_token(str(user.id), user.email)
                _set_access_cookie(response, new_access_token)
                return user
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
        headers={"WWW-Authenticate": "Bearer"},
    )

