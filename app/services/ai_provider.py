"""
AI Provider abstraction layer.

Designed so the underlying AI model (Gemini, OpenAI, Claude, etc.)
can be swapped without changing the rest of the application.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel


INTENT_DATA_QUERY = "DATA_QUERY"
INTENT_SCHEMA_EXPLORATION = "SCHEMA_EXPLORATION"
INTENT_CASUAL_CHAT = "CASUAL_CHAT"
INTENT_WRITE_REQUEST = "WRITE_REQUEST"

ALL_INTENTS = {
    INTENT_DATA_QUERY,
    INTENT_SCHEMA_EXPLORATION,
    INTENT_CASUAL_CHAT,
    INTENT_WRITE_REQUEST,
}


def deterministic_classify_intent(user_question: str) -> str:
    """
    Fast, deterministic code-based intent classification (< 1 ms, 0 LLM calls).
    Routes clearly into:
    - WRITE_REQUEST (security check)
    - CASUAL_CHAT (pleasantries, greetings)
    - SCHEMA_EXPLORATION (tables, columns, constraints, ER diagrams)
    - DATA_QUERY (default for all database data queries)
    """
    import re
    q = user_question.strip().lower()

    # 1. WRITE_REQUEST check
    from app.security.query_validator import is_write_intent
    if is_write_intent(user_question):
        return INTENT_WRITE_REQUEST

    write_direct_patterns = [
        r"\b(insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|alter\s+table|truncate\s+table|create\s+table)\b",
        r"\b(insert|update|delete|drop|truncate|alter|create\s+table|modify\s+data)\b",
    ]
    for pattern in write_direct_patterns:
        if re.search(pattern, q):
            return INTENT_WRITE_REQUEST

    # 2. CASUAL_CHAT patterns
    casual_patterns = [
        r"^(hi|hello|hey|howdy|hola|yo|sup|greetings)[!.,? ]*$",
        r"\b(how\s+are\s+you|how's\s+it\s+going|how\s+are\s+things)\b",
        r"\b(good\s+(morning|afternoon|evening|night|day))\b",
        r"\b(thanks|thank\s+you|thx|appreciate\s+it)\b",
        r"^(who\s+are\s+you|what\s+are\s+you|what\s+is\s+dataduck|what\s+can\s+you\s+do|help\s+me)[!.,? ]*$",
    ]
    for pattern in casual_patterns:
        if re.search(pattern, q):
            return INTENT_CASUAL_CHAT

    # 3. SCHEMA_EXPLORATION patterns
    schema_patterns = [
        r"\b(er[ -]?diagram|erd)\b",
        r"\b(database\s+diagram|schema\s+diagram|table\s+diagram)\b",
        r"\b(show|describe|explain|view|get|list|display)\b.*\b(schema|database\s+structure|architecture)\b",
        r"\b(database\s+schema|schema|data\s+dictionary)\b",
        r"\b(show\s+tables|list\s+tables|what\s+tables|tables\s+in\s+(the\s+)?database|all\s+tables|all\s+collections)\b",
        r"\b(how\s+many\s+(tables?|collections?|entities?|models?))\b",
        r"\b(total\s+(number\s+of\s+)?(tables?|collections?))\b",
        r"\b(count\s+of\s+(tables?|collections?)|(table|collection)\s+count)\b",
        r"^(tables|columns|relationships|collections)$",
        r"\b(explain|describe|show|list|get)\b.*\b(columns|fields|attributes|keys|collections)\b",
        r"\b(primary\s+key|foreign\s+key|pk|fk)\b",
        r"\b(relationships?|table\s+relations?|foreign\s+keys?)\b",
        r"\b(how\s+are\s+\w+\s+and\s+\w+\s+related)\b",
        r"\b(explain\s+the\s+\w+\s+(table|collection)|describe\s+the\s+\w+\s+(table|collection))\b",
        r"\b(what\s+(tables?|collections?)\s+(are\s+in|exist\s+in|does\s+this\s+(app|db|database)\s+have))\b",
        r"\b(what\s+tables\s+are\s+in\s+my\s+database|what\s+tables\s+exist)\b",
    ]
    for pattern in schema_patterns:
        if re.search(pattern, q):
            return INTENT_SCHEMA_EXPLORATION

    # 4. Default: DATA_QUERY (normal database questions)
    return INTENT_DATA_QUERY


def quick_classify_intent(user_question: str) -> str:
    """Alias for deterministic_classify_intent."""
    return deterministic_classify_intent(user_question)


import json
from pydantic import BaseModel, model_validator


class QueryGenerationResult(BaseModel):
    """Structured result from AI query generation."""
    operation: str = "query"  # "query", "aggregate", "count", etc.
    database_type: str = "unknown"
    query: str = ""  # SQL string or MongoDB operation JSON string
    query_dict: Optional[dict] = None  # For MongoDB operations
    explanation: str = "Generated database query."
    is_read_only: bool = True
    refused: bool = False
    refusal_reason: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return {
                "query": str(data) if data else "",
                "operation": "query",
                "database_type": "unknown",
                "explanation": "Generated database query.",
            }

        # If query is a dictionary (common in MongoDB LLM outputs)
        q = data.get("query")
        if isinstance(q, dict):
            data["query_dict"] = q
            data["query"] = json.dumps(q)
        elif not q and any(k in data for k in ("aggregate", "find", "collection", "pipeline")):
            # The root object itself is the MongoDB query
            data["query_dict"] = data.copy()
            data["query"] = json.dumps(data)
        elif q is None:
            data["query"] = ""

        # Default missing operation
        if not data.get("operation"):
            if isinstance(q, dict) and "aggregate" in q:
                data["operation"] = "aggregate"
            elif isinstance(q, dict) and "find" in q:
                data["operation"] = "find"
            else:
                data["operation"] = "query"

        # Default missing database_type
        if not data.get("database_type"):
            data["database_type"] = "mongodb" if isinstance(q, dict) or "collection" in data else "sql"

        # Default missing explanation
        if not data.get("explanation"):
            data["explanation"] = "Generated database query based on your question."

        return data


class AnalysisResult(BaseModel):
    """Structured result from AI result analysis or schema exploration."""
    answer: str = ""
    insights: list[str] = []
    warnings: list[str] = []
    visualization: Optional[dict] = None  # Visualization spec or ER diagram spec
    data_quality_notes: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def normalize_analysis(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"answer": data, "insights": [], "warnings": [], "visualization": None, "data_quality_notes": []}
        if isinstance(data, dict):
            if not data.get("answer"):
                data["answer"] = data.get("explanation") or data.get("response") or data.get("summary") or "Analysis completed."
            if not isinstance(data.get("insights"), list):
                data["insights"] = []
            if not isinstance(data.get("warnings"), list):
                data["warnings"] = []
            if not isinstance(data.get("data_quality_notes"), list):
                data["data_quality_notes"] = []
            return data
        return {"answer": str(data), "insights": [], "warnings": [], "visualization": None, "data_quality_notes": []}


class AIProvider(ABC):
    """Abstract AI provider interface."""

    @abstractmethod
    async def classify_intent(
        self,
        user_question: str,
        conversation_history: list[dict],
    ) -> str:
        """Classify user intent into DATA_QUERY, SCHEMA_EXPLORATION, CASUAL_CHAT, WRITE_REQUEST."""
        ...

    @abstractmethod
    async def generate_query(
        self,
        user_question: str,
        relevant_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> QueryGenerationResult:
        """
        Generate a database query from a natural language question.
        
        Must NEVER receive database credentials.
        Only receives schema metadata and conversation history.
        """
        ...

    @abstractmethod
    async def analyze_results(
        self,
        user_question: str,
        query: str,
        query_result: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """
        Analyze query results and produce a natural language answer.
        """
        ...

    @abstractmethod
    async def answer_schema_question(
        self,
        user_question: str,
        full_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """
        Answer schema exploration questions regarding tables, columns, types, constraints, and relationships.
        """
        ...

    @abstractmethod
    async def handle_casual_chat(
        self,
        user_question: str,
        conversation_history: list[dict],
    ) -> str:
        """Generate a natural conversational response for casual messages without querying the database."""
        ...

    @abstractmethod
    async def handle_write_intent(self, user_question: str) -> str:
        """Generate a refusal message for write intent."""
        ...


def get_ai_service() -> AIProvider:
    """Factory function to get the configured AI provider based on app settings."""
    from app.core.config import settings

    provider = (settings.AI_PROVIDER or "").lower()
    if provider == "ollama":
        from app.services.ollama_service import OllamaService
        return OllamaService()
    if provider in ("openai", "groq") or (settings.GROQ_API_KEY and provider == "groq"):
        from app.services.openai_service import OpenAIService
        return OpenAIService()
    from app.services.gemini_service import GeminiService
    return GeminiService()


