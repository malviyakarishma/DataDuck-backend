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


def quick_classify_intent(user_question: str) -> Optional[str]:
    """
    Fast rule-based intent classification for clear-cut queries.
    Returns intent string or None if LLM classification is needed.
    """
    import re
    q = user_question.strip().lower()

    # 1. Write request check
    from app.security.query_validator import is_write_intent
    if is_write_intent(user_question):
        return INTENT_WRITE_REQUEST

    # 2. ER diagram & schema keywords
    er_patterns = [
        r"\b(er[ -]?diagram|erd)\b",
        r"\b(schema|database|table|entity)\s+(diagram|chart|graph|map|visual|architecture)\b",
        r"\b(show|draw|generate|display|view|give me|render)\b.*\b(diagram|schema|structure|erd)\b",
        r"\bwhat tables\b",
        r"\blist\s+(all\s+)?tables\b",
        r"\bshow\s+(all\s+)?tables\b",
        r"\b(what|which)\s+tables\s+(are\s+in|exist|do i have)\b",
        r"\b(show|list|get|explain|describe)\s+(all\s+)?columns\b",
        r"\b(columns|fields|attributes)\s+(in|of|for)\s+\w+\b",
        r"\b(explain|describe|structure of|overview of)\s+(the\s+|my\s+|our\s+|this\s+)?(database\s+schema|schema|database|tables?|\w+\s+table)\b",
        r"\bhow are\s+\w+\s+and\s+\w+\s+related\b",
        r"\b(database|table|foreign key|primary key|pk|fk)\s+relationships?\b",
        r"\bshow\s+(me\s+)?(the\s+)?relationships?\b",
        r"\bshow\s+(me\s+)?(the\s+)?constraints?\b",
        r"\b(primary|foreign)\s+keys?\b",
    ]
    for pattern in er_patterns:
        if re.search(pattern, q):
            return INTENT_SCHEMA_EXPLORATION

    # 3. Casual chat patterns
    casual_patterns = [
        r"^(hi|hello|hey|greetings|howdy|hola|yo|sup)[!.]*$",
        r"^(good\s+(morning|afternoon|evening|day))[!.]*$",
        r"^(who are you|what are you|what is dataduck|what can you do|help me|what are your capabilities)[?.]*$",
        r"^(thanks|thank you|awesome|great job|cool|nice)[!.]*$",
    ]
    for pattern in casual_patterns:
        if re.search(pattern, q):
            return INTENT_CASUAL_CHAT

    return None


class QueryGenerationResult(BaseModel):
    """Structured result from AI query generation."""
    operation: str  # "query", "aggregate", "count", etc.
    database_type: str
    query: str  # SQL string or MongoDB operation JSON string
    query_dict: Optional[dict] = None  # For MongoDB operations
    explanation: str
    is_read_only: bool = True
    refused: bool = False
    refusal_reason: Optional[str] = None


class AnalysisResult(BaseModel):
    """Structured result from AI result analysis or schema exploration."""
    answer: str
    insights: list[str] = []
    warnings: list[str] = []
    visualization: Optional[dict] = None  # Visualization spec or ER diagram spec
    data_quality_notes: list[str] = []


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


