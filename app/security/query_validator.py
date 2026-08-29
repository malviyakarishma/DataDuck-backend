"""
READ-ONLY query security validator.

Uses SQLGlot AST parsing for SQL (not just string matching).
Uses whitelist-based validation for MongoDB operations.

This is a critical security component — never skip validation.
"""
import re
from typing import Literal
import logging

try:
    import sqlglot
    import sqlglot.expressions as exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    sqlglot = None

from app.core.exceptions import QueryValidationError, WriteOperationError

logger = logging.getLogger(__name__)

# Blocked SQL statement types (case-insensitive keywords that indicate write ops)
BLOCKED_SQL_STATEMENTS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "MERGE", "REPLACE", "UPSERT",
    "CALL", "EXECUTE", "EXEC", "SP_", "XP_",
}

# Dangerous SQL patterns (regex for additional defense-in-depth)
DANGEROUS_SQL_PATTERNS = [
    r";\s*\w",              # Multiple statements
    r"--\s*$",             # SQL comments at end (injection attempt)
    r"/\*.*?\*/",           # Block comments
    r"\bINTO\s+OUTFILE\b",  # MySQL file write
    r"\bINTO\s+DUMPFILE\b", # MySQL file write
    r"\bLOAD\s+DATA\b",    # MySQL file read/write
    r"\bCOPY\s+\w+\s+TO\b", # PostgreSQL COPY TO (write)
    r"\bPG_READ_FILE\b",   # PostgreSQL file functions
    r"\bPG_WRITE_FILE\b",
    r"\bDBMS_\w+\b",       # Oracle DBMS packages
    r"\bSYS\.\w+\b",       # Oracle SYS objects
    r"\bINFORMATION_SCHEMA\s*\.\s*COLUMNS\b",  # OK but watch for injection
    r"0x[0-9a-fA-F]+",     # Hex encoding (obfuscation)
    r"CHAR\s*\(\s*\d+",    # CHAR() obfuscation
    r"SLEEP\s*\(",         # Time-based blind injection
    r"BENCHMARK\s*\(",     # MySQL timing attack
    r"WAITFOR\s+DELAY",    # SQL Server timing attack
    r"PG_SLEEP\s*\(",      # PostgreSQL timing attack
]

# Allowed MongoDB read operations
ALLOWED_MONGODB_OPS = {
    "find", "findOne", "aggregate", "countDocuments",
    "distinct", "estimatedDocumentCount", "count",
}

# Blocked MongoDB write operations
BLOCKED_MONGODB_OPS = {
    "insertOne", "insertMany", "updateOne", "updateMany",
    "deleteOne", "deleteMany", "replaceOne", "drop",
    "renameCollection", "bulkWrite", "createIndex",
    "dropIndex", "dropIndexes", "createCollection",
    "dropCollection", "findOneAndDelete", "findOneAndUpdate",
    "findOneAndReplace",
}

# SQLGlot dialect mapping
DIALECT_MAP = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "mssql": "tsql",
}


def validate_sql_query(query: str, db_type: str = "postgresql") -> str:
    """
    Validate that a SQL query is read-only.

    1. Check for multiple statements.
    2. Check for dangerous patterns.
    3. Parse with SQLGlot and verify only SELECT / WITH...SELECT.
    4. Walk AST to find any write operations.

    Returns cleaned query string or raises QueryValidationError / WriteOperationError.
    """
    if not query or not query.strip():
        raise QueryValidationError("Empty query.")

    cleaned = query.strip().rstrip(";")

    # 1. Multiple statements check
    if _has_multiple_statements(cleaned):
        raise QueryValidationError("Multiple SQL statements are not allowed.")

    # 2. Dangerous pattern check (defense in depth)
    _check_dangerous_patterns(cleaned)

    # 3. Keyword-level pre-check (fast path before AST)
    _check_blocked_keywords(cleaned)

    # 4. AST-level validation
    if SQLGLOT_AVAILABLE:
        _validate_ast(cleaned, db_type)
    else:
        logger.warning("SQLGlot not available — falling back to keyword validation only.")
        _keyword_fallback_validation(cleaned)

    return cleaned


def _has_multiple_statements(query: str) -> bool:
    """Check for multiple semicolon-separated statements."""
    # Remove string literals first to avoid false positives
    stripped = re.sub(r"'[^']*'", "''", query)
    stripped = re.sub(r'"[^"]*"', '""', stripped)
    return ";" in stripped


def _check_dangerous_patterns(query: str) -> None:
    """Check for dangerous SQL patterns using regex."""
    query_upper = query.upper()
    for pattern in DANGEROUS_SQL_PATTERNS:
        if re.search(pattern, query_upper, re.IGNORECASE | re.DOTALL):
            raise QueryValidationError(
                f"Query contains a dangerous pattern and was blocked for security."
            )


def _check_blocked_keywords(query: str) -> None:
    """Fast keyword check before expensive AST parsing using word boundaries."""
    query_upper = query.upper()
    first_word = query_upper.strip().split()[0] if query_upper.strip() else ""

    # Immediate block if the primary command keyword is a write command
    if first_word in BLOCKED_SQL_STATEMENTS:
        raise WriteOperationError()

    for keyword in BLOCKED_SQL_STATEMENTS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", query_upper):
            # Block any standalone write keyword in statement
            raise WriteOperationError()


def _validate_ast(query: str, db_type: str) -> None:
    """Use SQLGlot to parse and validate the query AST."""
    dialect = DIALECT_MAP.get(db_type, "")

    try:
        parsed = sqlglot.parse(query, dialect=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    except Exception as e:
        raise QueryValidationError(f"Query could not be parsed: {e}")

    if not parsed:
        raise QueryValidationError("Empty or unparseable query.")

    if len(parsed) > 1:
        raise QueryValidationError("Multiple SQL statements are not allowed.")

    statement = parsed[0]

    # Must be a SELECT statement (or WITH...SELECT CTE)
    allowed_types = (exp.Select,)
    is_cte_select = (
        isinstance(statement, exp.With) and
        isinstance(statement.this, exp.Select)
    )

    if not isinstance(statement, allowed_types) and not is_cte_select:
        statement_type = type(statement).__name__
        raise WriteOperationError()

    # Walk the AST for any write expressions
    for node in statement.walk():
        node_type = type(node).__name__
        if node_type in {
            "Insert", "Update", "Delete", "Drop", "Create", "Alter",
            "Truncate", "Grant", "Revoke", "Merge", "Command",
        }:
            raise WriteOperationError()


def _keyword_fallback_validation(query: str) -> None:
    """Fallback when SQLGlot is not available."""
    query_upper = query.upper().strip()
    first_word = query_upper.split()[0] if query_upper.split() else ""

    if first_word not in ("SELECT", "WITH"):
        raise WriteOperationError()


def validate_mongodb_operation(operation: dict) -> dict:
    """
    Validate a MongoDB operation spec.

    Expected format:
    {
        "collection": "users",
        "operation": "find",
        "filter": {...},
        "pipeline": [...],
        "limit": 1000,
        "projection": {...}
    }
    """
    if not operation:
        raise QueryValidationError("Empty MongoDB operation.")

    op_name = operation.get("operation", "").strip()

    if not op_name:
        raise QueryValidationError("MongoDB operation name is required.")

    if op_name in BLOCKED_MONGODB_OPS:
        raise WriteOperationError()

    if op_name not in ALLOWED_MONGODB_OPS:
        raise QueryValidationError(
            f"MongoDB operation '{op_name}' is not in the allowed list: {', '.join(sorted(ALLOWED_MONGODB_OPS))}"
        )

    # Ensure collection is specified
    if not operation.get("collection"):
        raise QueryValidationError("MongoDB operation must specify a collection.")

    # Enforce limit
    from app.core.config import settings
    if "limit" not in operation or operation["limit"] > settings.MAX_QUERY_ROWS:
        operation["limit"] = settings.MAX_QUERY_ROWS

    return operation


def is_write_intent(user_message: str) -> bool:
    """
    Quick check if the user's natural language message has write intent.
    Used to give early refusal before even calling Gemini.
    """
    write_patterns = [
        r"\b(delete\s+from|update\s+\w+\s+set|insert\s+into|drop\s+table|alter\s+table|truncate\s+table|create\s+table)\b",
        r"\b(delete|remove|drop|truncate|wipe|clear|erase)\b",
        r"\b(update|modify|change|edit|alter|set)\b.*\b(record|row|data|value|field|table|database|price|name|status|column)\b",
        r"\b(insert|add|create|append|put)\b.*\b(record|row|data|entry|user|order|customer|into|values)\b",
        r"\b(update|set)\b.*\bwhere\b",
    ]
    message_lower = user_message.lower()
    for pattern in write_patterns:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return True
    return False
