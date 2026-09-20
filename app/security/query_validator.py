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
    Validate that a SQL query is read-only across all supported database types.

    1. Strip trailing semicolons and check for multiple statements.
    2. Check for dangerous patterns (file writes, time-based attacks).
    3. Keyword-level check for primary write statements.
    4. AST-level validation (supports SELECT, UNION, INTERSECT, EXCEPT, CTEs, subqueries).
    5. Fallback validation if AST parser encounters dialect quirks.
    """
    if not query or not query.strip():
        raise QueryValidationError("Empty query.")

    cleaned = query.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

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
        logger.warning("SQLGlot not available — falling back to keyword validation.")
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
                "Query contains a dangerous pattern and was blocked for security."
            )


def _check_blocked_keywords(query: str) -> None:
    """Fast keyword check before expensive AST parsing using word boundaries."""
    query_upper = query.upper().strip()
    first_word = query_upper.split()[0] if query_upper else ""

    # Immediate block if the primary command keyword is a write command
    if first_word in BLOCKED_SQL_STATEMENTS:
        raise WriteOperationError()


def _validate_ast(query: str, db_type: str) -> None:
    """Use SQLGlot to parse and validate the query AST."""
    dialect = DIALECT_MAP.get(db_type, "")

    parsed = None
    if dialect:
        try:
            parsed = sqlglot.parse(query, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception:
            pass

    if not parsed:
        try:
            parsed = sqlglot.parse(query, error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception:
            pass

    if not parsed:
        # If SQLGlot cannot parse dialect-specific syntax, use keyword fallback
        _keyword_fallback_validation(query)
        return

    non_empty = [s for s in parsed if s is not None]
    if not non_empty:
        _keyword_fallback_validation(query)
        return

    if len(non_empty) > 1:
        raise QueryValidationError("Multiple SQL statements are not allowed.")

    statement = non_empty[0]
    unwrapped = statement.unwrap() if hasattr(statement, "unwrap") else statement

    # Allowed read query AST types in SQLGlot:
    # exp.Select, exp.Union, exp.Intersect, exp.Except, or exp.Query base class
    # or exp.With where query expression is contained
    allowed_query_types = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Query, exp.Subquery)
    is_valid_query = (
        isinstance(unwrapped, allowed_query_types)
        or (isinstance(unwrapped, exp.With) and isinstance(getattr(unwrapped, "this", None), allowed_query_types))
    )

    read_command_classes = tuple(c for c in (getattr(exp, "Describe", None), getattr(exp, "Pragma", None)) if c is not None)
    is_read_command = (read_command_classes and isinstance(unwrapped, read_command_classes)) or (
        type(unwrapped).__name__ in ("Describe", "Pragma", "Explain", "Show")
    )

    if not is_valid_query and not is_read_command:
        first_word = query.upper().strip().split()[0] if query.upper().strip() else ""
        if first_word in ("SELECT", "WITH", "(SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA"):
            is_valid_query = True
        else:
            raise WriteOperationError()

    # Walk AST for write expressions (Insert, Update, Delete, Drop, Alter, Truncate, Create, etc.)
    blocked_node_types = {
        "Insert", "Update", "Delete", "Drop", "Create", "Alter",
        "TruncateTable", "Truncate", "Grant", "Revoke", "Merge",
    }
    for node in statement.walk():
        node_type = type(node).__name__
        if node_type in blocked_node_types:
            raise WriteOperationError()


def _keyword_fallback_validation(query: str) -> None:
    """Fallback when SQLGlot cannot parse or is unavailable."""
    q_no_comments = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
    q_no_comments = re.sub(r"/\*.*?\*/", "", q_no_comments, flags=re.DOTALL).strip()

    first_word = q_no_comments.upper().split()[0] if q_no_comments else ""
    if first_word not in ("SELECT", "WITH", "(SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA"):
        raise WriteOperationError()

    q_no_strings = re.sub(r"'[^']*'", "''", q_no_comments)
    q_no_strings = re.sub(r'"[^"]*"', '""', q_no_strings)

    write_patterns = [
        r"\bDELETE\s+FROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\s+[\w.\"]+\s+SET\b",
        r"\bDROP\s+(?:TABLE|VIEW|DATABASE|INDEX|SCHEMA)\b",
        r"\bALTER\s+(?:TABLE|VIEW|DATABASE|INDEX|SCHEMA)\b",
        r"\bTRUNCATE\s+(?:TABLE\s+)?[\w.\"]+\b",
        r"\bCREATE\s+(?:TABLE|VIEW|DATABASE|INDEX|SCHEMA)\b",
    ]
    for pattern in write_patterns:
        if re.search(pattern, q_no_strings, re.IGNORECASE):
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
    Quick check if the user's natural language message has EXPLICIT write intent.
    Only matches clear imperative SQL-style commands, NOT analytical words.

    Examples that SHOULD match:
    - "delete all inactive users"  -> True (imperative "delete" + target)
    - "drop the users table"       -> True (imperative "drop" + "table")
    - "insert a new row into orders" -> True

    Examples that should NOT match:
    - "what caused the drop in sales?"      -> False (analytical use of "drop")
    - "show me the change in price"         -> False (analytical use of "change")
    - "which column has null values"        -> False
    - "clear picture of customer spending"  -> False
    """
    write_patterns = [
        # Explicit SQL command patterns (very specific, low false-positive)
        r"\b(delete\s+from|delete\s+all|delete\s+(the\s+)?record|delete\s+(the\s+)?row|delete\s+(the\s+)?user|delete\s+(the\s+)?data)\b",
        r"\b(insert\s+into|insert\s+(a\s+)?(new\s+)?record|insert\s+(a\s+)?(new\s+)?row)\b",
        r"\b(update\s+\w+\s+set|update\s+(the\s+)?(record|row|data|value|status|price|name|field))\b",
        r"\b(drop\s+(the\s+)?(table|database|collection|index))\b",
        r"\b(alter\s+(the\s+)?(table|column|database))\b",
        r"\b(truncate\s+(the\s+)?(table|collection))\b",
        r"\b(create\s+(a\s+)?(new\s+)?(table|database|collection|index))\b",
        # Imperative write verbs at the START of the sentence (command-style)
        r"^(delete|remove|drop|truncate|wipe|erase|destroy)\s+",
        r"^(insert|add)\s+(a\s+|new\s+)?(record|row|entry|data|user|order|customer)\b",
        r"^(update|modify|edit|change)\s+(the\s+)?(record|row|data|value|field|status|price|name)\b",
    ]
    message_lower = user_message.strip().lower()
    for pattern in write_patterns:
        if re.search(pattern, message_lower):
            return True
    return False

