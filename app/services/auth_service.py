"""
Authentication service — handles user registration, login, and token management.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, VerifyOTPRequest
from app.security.auth import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.services.email_service import generate_otp_code, send_otp_email

logger = logging.getLogger(__name__)


async def register_user(db: AsyncSession, request: RegisterRequest) -> User:
    """Register a new user in unverified state and send 2MFA OTP code."""
    result = await db.execute(select(User).where(User.email == request.email.lower().strip()))
    existing = result.scalar_one_or_none()

    if existing:
        raise AuthenticationError("An account with this email already exists.")

    otp_code = generate_otp_code()
    otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    user = User(
        full_name=request.full_name.strip(),
        email=request.email.lower().strip(),
        hashed_password=hash_password(request.password),
        is_active=True,
        is_verified=False,
        otp_code=otp_code,
        otp_expires_at=otp_expires_at,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    await send_otp_email(user.email, otp_code)
    logger.info(f"New user registered (unverified, OTP sent): {user.email}")
    return user


async def verify_otp(db: AsyncSession, email: str, otp_code: str) -> tuple[User, str, str]:
    """
    Verify 2MFA OTP code.
    If valid, marks user as verified and returns (user, access_token, refresh_token).
    """
    email_clean = email.lower().strip()
    result = await db.execute(select(User).where(User.email == email_clean))
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationError("User not found.")

    if user.is_verified:
        # If user is already verified, proceed to authenticate
        access_token = create_access_token(str(user.id), user.email)
        refresh_token = create_refresh_token(str(user.id))
        return user, access_token, refresh_token

    if not user.otp_code or not user.otp_expires_at:
        raise AuthenticationError("No verification code found. Please request a new code.")

    now = datetime.now(timezone.utc)
    # Ensure timezone aware comparison
    expires_at = user.otp_expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        raise AuthenticationError("Verification code has expired. Please click 'Resend Code'.")

    if user.otp_code.strip() != otp_code.strip():
        raise AuthenticationError("Invalid verification code. Please check your code and try again.")

    # Mark user as verified and clear OTP
    user.is_verified = True
    user.otp_code = None
    user.otp_expires_at = None
    user.last_login_at = now
    await db.flush()

    access_token = create_access_token(str(user.id), user.email)
    refresh_token = create_refresh_token(str(user.id))
    logger.info(f"User email verified successfully via OTP: {user.email}")

    return user, access_token, refresh_token


async def resend_otp(db: AsyncSession, email: str) -> bool:
    """Resend a new 2MFA OTP code to user's email."""
    email_clean = email.lower().strip()
    result = await db.execute(select(User).where(User.email == email_clean))
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationError("User not found.")

    if user.is_verified:
        raise AuthenticationError("This account is already verified.")

    otp_code = generate_otp_code()
    user.otp_code = otp_code
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    await db.flush()

    await send_otp_email(user.email, otp_code)
    logger.info(f"Resent OTP code to: {user.email}")
    return True


async def authenticate_user(db: AsyncSession, request: LoginRequest) -> tuple[User, Optional[str], Optional[str], bool]:
    """
    Authenticate user with email/password.
    If user email is not verified, sends a fresh OTP code to mail and returns (user, None, None, True).
    Otherwise returns (user, access_token, refresh_token, False).
    """
    result = await db.execute(select(User).where(User.email == request.email.lower().strip()))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise AuthenticationError("Invalid email or password.")

    if not user.is_active:
        raise AuthenticationError("Your account has been deactivated.")

    if not user.is_verified:
        # Mandatory 2MFA: Generate new OTP code, update DB and send to mail
        otp_code = generate_otp_code()
        user.otp_code = otp_code
        user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        await db.flush()
        await send_otp_email(user.email, otp_code)
        logger.info(f"Unverified login attempt: Resent OTP code to {user.email}")
        return user, None, None, True

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    await db.flush()

    access_token = create_access_token(str(user.id), user.email)
    refresh_token = create_refresh_token(str(user.id))

    return user, access_token, refresh_token, False


async def get_current_user(db: AsyncSession, user_id: str) -> User:
    """Get user by ID. Used for token validation."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationError("User not found.")
    if not user.is_active:
        raise AuthorizationError("Account deactivated.")

    return user


async def refresh_access_token(db: AsyncSession, refresh_token: str) -> str:
    """Validate refresh token and issue new access token."""
    payload = decode_token(refresh_token, expected_type="refresh")
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid refresh token.")

    user = await get_current_user(db, user_id)
    return create_access_token(str(user.id), user.email)
