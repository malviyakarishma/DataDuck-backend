from pydantic import BaseModel, EmailStr, field_validator
import re


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match.")
        return v

    @field_validator("full_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Full name must be at least 2 characters.")
        return v


from typing import Optional


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str


class LoginResponse(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user_id: Optional[str] = None
    email: str
    full_name: Optional[str] = None
    requires_otp: bool = False
    message: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class OTPResponse(BaseModel):
    message: str
    email: str
    requires_otp: bool = True


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp_code: str

    @field_validator("otp_code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        v = v.strip()
        if not v.isdigit() or len(v) != 6:
            raise ValueError("OTP code must be a 6-digit number.")
        return v


class ResendOTPRequest(BaseModel):
    email: EmailStr

