from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Optional, Any, Union
import json


class Settings(BaseSettings):
    # App
    APP_NAME: str = "DataDuck"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Database (App PostgreSQL)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/querymind"

    # AI Provider
    AI_PROVIDER: str = "ollama"  # "ollama", "gemini", or "openai"

    # Ollama Local AI
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"

    # Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # OpenAI AI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = ""



    # Security
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ENCRYPTION_KEY: str = ""  # Fernet key - generate with: Fernet.generate_key()
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    FRONTEND_URL: str = "https://data-duck-frontend.vercel.app"
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://data-duck-frontend.vercel.app",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v_clean = v.strip()
            if v_clean.startswith("[") and v_clean.endswith("]"):
                try:
                    return json.loads(v_clean)
                except Exception:
                    pass
            origins = [o.strip() for o in v_clean.split(",") if o.strip()]
            if origins:
                return origins
        elif isinstance(v, list):
            return v
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://data-duck-frontend.vercel.app",
        ]

    # Query Limits
    MAX_QUERY_ROWS: int = 10000
    QUERY_TIMEOUT_SECONDS: int = 30
    MAX_SCHEMA_TABLES_TO_GEMINI: int = 20

    # Rate Limiting
    RATE_LIMIT_LOGIN: str = "10/minute"
    RATE_LIMIT_CHAT: str = "30/minute"
    RATE_LIMIT_DB_TEST: str = "5/minute"

    # SMTP / Email
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "[EMAIL_ADDRESS]"
    SMTP_PASSWORD: str = "sggj oexz fqeb dvrw"
    EMAILS_FROM: str = "[EMAIL_ADDRESS]"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
