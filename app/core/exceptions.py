from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class QueryMindException(Exception):
    """Base exception for QueryMind."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class AuthenticationError(QueryMindException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTH_ERROR")


class AuthorizationError(QueryMindException):
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, "AUTHZ_ERROR")


class DatabaseConnectionError(QueryMindException):
    def __init__(self, message: str):
        super().__init__(message, "DB_CONNECTION_ERROR")


class DatabaseNotFoundError(QueryMindException):
    def __init__(self, message: str = "Database connection not found"):
        super().__init__(message, "DB_NOT_FOUND")


class UnsupportedDatabaseError(QueryMindException):
    def __init__(self, db_type: str):
        super().__init__(
            f"Database type '{db_type}' is not currently supported. "
            f"Supported types: PostgreSQL, MySQL, SQLite, MongoDB.",
            "UNSUPPORTED_DB"
        )


class QueryValidationError(QueryMindException):
    def __init__(self, message: str):
        super().__init__(message, "QUERY_VALIDATION_ERROR")


class QueryExecutionError(QueryMindException):
    def __init__(self, message: str):
        super().__init__(message, "QUERY_EXECUTION_ERROR")


class QueryTimeoutError(QueryMindException):
    def __init__(self):
        super().__init__("Query timed out. Try a more specific question.", "QUERY_TIMEOUT")


class AIServiceError(QueryMindException):
    def __init__(self, message: str = "AI service error"):
        super().__init__(message, "AI_ERROR")


class WriteOperationError(QueryMindException):
    def __init__(self):
        super().__init__(
            "This database connection is read-only. I can analyze the data, but I cannot modify it.",
            "WRITE_BLOCKED"
        )


class EncryptionError(QueryMindException):
    def __init__(self, message: str = "Encryption/decryption error"):
        super().__init__(message, "ENCRYPTION_ERROR")


# FastAPI exception handlers
async def querymind_exception_handler(request: Request, exc: QueryMindException):
    status_map = {
        "AUTH_ERROR": 401,
        "AUTHZ_ERROR": 403,
        "DB_NOT_FOUND": 404,
        "DB_CONNECTION_ERROR": 400,
        "UNSUPPORTED_DB": 400,
        "QUERY_VALIDATION_ERROR": 400,
        "QUERY_EXECUTION_ERROR": 500,
        "QUERY_TIMEOUT": 408,
        "AI_ERROR": 503,
        "WRITE_BLOCKED": 403,
        "ENCRYPTION_ERROR": 500,
    }
    status_code = status_map.get(exc.code, 500)
    logger.error(f"[{exc.code}] {exc.message}")
    return JSONResponse(
        status_code=status_code,
        content={"error": exc.message, "code": exc.code}
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "code": "HTTP_ERROR"}
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred.", "code": "INTERNAL_ERROR"}
    )
