from fastapi import APIRouter, Depends, Response, Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.core.database import get_db
from app.schemas.auth import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserResponse,
    VerifyOTPRequest, ResendOTPRequest, OTPResponse, LoginResponse
)
from app.services.auth_service import (
    register_user, authenticate_user, refresh_access_token, get_current_user,
    verify_otp, resend_otp
)
from app.security.auth import create_refresh_token, decode_token
from app.api.deps import get_current_user_dep
from app.models.user import User
from app.core.config import settings
from app.core.exceptions import AuthenticationError
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

IS_DEV = settings.ENVIRONMENT == "development"


def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax" if IS_DEV else "none",
        secure=not IS_DEV,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        samesite="lax" if IS_DEV else "none",
        secure=not IS_DEV,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    )


def clear_auth_cookies(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax" if IS_DEV else "none",
        secure=not IS_DEV,
    )
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        samesite="lax" if IS_DEV else "none",
        secure=not IS_DEV,
    )


@router.post("/register", response_model=OTPResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account and trigger 2MFA verification code."""
    try:
        user = await register_user(db, request)
        return OTPResponse(
            message="Verification code sent to your email.",
            email=user.email,
            requires_otp=True,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post("/verify-otp", response_model=TokenResponse)
async def verify_user_otp(
    request: VerifyOTPRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Verify 2MFA OTP code and log user in."""
    try:
        user, access_token, refresh_token = await verify_otp(db, request.email, request.otp_code)
        set_auth_cookies(response, access_token, refresh_token)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post("/resend-otp")
async def resend_user_otp(
    request: ResendOTPRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resend 2MFA verification OTP code."""
    try:
        await resend_otp(db, request.email)
        return {"message": "Verification code resent successfully.", "email": request.email}
    except AuthenticationError as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Login with email and password."""
    try:
        user, access_token, refresh_token, requires_otp = await authenticate_user(db, request)

        if requires_otp:
            return LoginResponse(
                requires_otp=True,
                email=user.email,
                message="Account not verified. A new verification code has been sent to your email.",
            )

        set_auth_cookies(response, access_token, refresh_token)
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            requires_otp=False,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    body: Optional[RefreshRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """Refresh access token using refresh token."""
    try:
        token_to_use = None
        if body and body.refresh_token:
            token_to_use = body.refresh_token
        else:
            token_to_use = request.cookies.get("refresh_token")

        if not token_to_use:
            raise HTTPException(status_code=401, detail="Refresh token missing.")

        new_access = await refresh_access_token(db, token_to_use)
        set_auth_cookies(response, new_access, token_to_use)
        payload = decode_token(new_access, "access")
        user = await get_current_user(db, payload["sub"])

        return TokenResponse(
            access_token=new_access,
            refresh_token=token_to_use,
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)


@router.post("/logout")
async def logout(response: Response):
    """Logout — clear cookies."""
    clear_auth_cookies(response)
    return {"message": "Logged out successfully."}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user_dep)):
    """Get current user info."""
    return UserResponse(
        id=str(current_user.id),
        full_name=current_user.full_name,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    )
