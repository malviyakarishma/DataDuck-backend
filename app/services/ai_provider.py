"""
AI Provider abstraction layer.

Designed so the underlying AI model (Gemini, OpenAI, Claude, etc.)
can be swapped without changing the rest of the application.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel


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
    """Structured result from AI result analysis."""
    answer: str
    insights: list[str] = []
    warnings: list[str] = []
    visualization: Optional[dict] = None  # Visualization spec
    data_quality_notes: list[str] = []


class AIProvider(ABC):
    """Abstract AI provider interface."""

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
    async def handle_write_intent(self, user_question: str) -> str:
        """Generate a refusal message for write intent."""
        ...
