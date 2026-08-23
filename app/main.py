"""
DataDuck FastAPI Application Entry Point — Ask. Dig. Discover.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.exceptions import (
    QueryMindException, querymind_exception_handler,
    http_exception_handler, generic_exception_handler
)
from app.api.routes import auth, databases, chat
import logging

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize and teardown."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    yield
    await close_db()
    logger.info("Application shutdown complete.")


# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered Database Analyst Chatbot API",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow frontend origin with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    expose_headers=["Set-Cookie"],
)

# Exception handlers
app.add_exception_handler(QueryMindException, querymind_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Routers
API_PREFIX = "/api"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(databases.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/api/health/ollama")
async def ollama_health_check():
    """Health check for Ollama local service."""
    import httpx
    base_url = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    model = settings.OLLAMA_MODEL or "qwen2.5-coder:7b"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                installed_models = [m.get("name") for m in data.get("models", [])]
                model_installed = any(
                    m == model or m == f"{model}:latest" or m.startswith(model)
                    for m in installed_models
                )
                return {
                    "status": "healthy",
                    "provider": "ollama",
                    "base_url": base_url,
                    "model": model,
                    "model_installed": model_installed,
                    "available_models": installed_models,
                }
            else:
                return {
                    "status": "unhealthy",
                    "provider": "ollama",
                    "base_url": base_url,
                    "model": model,
                    "error": f"Ollama returned HTTP {response.status_code}",
                }
    except Exception:
        return {
            "status": "unhealthy",
            "provider": "ollama",
            "base_url": base_url,
            "model": model,
            "error": "Local AI service is unavailable. Make sure Ollama is running.",
        }



@app.get("/")
async def root():
    return {"message": f"{settings.APP_NAME} API is running."}
