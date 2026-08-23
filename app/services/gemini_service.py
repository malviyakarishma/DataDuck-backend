"""
Gemini AI provider implementation.

SECURITY RULES:
1. Database credentials are NEVER sent to Gemini.
2. Only schema metadata and query results are sent.
3. All responses are validated with Pydantic before use.
4. Gemini never directly accesses any database.
"""
import json
import logging
import re
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.services.ai_provider import AIProvider, QueryGenerationResult, AnalysisResult
from app.core.config import settings
from app.core.exceptions import AIServiceError

logger = logging.getLogger(__name__)

QUERY_GENERATION_SYSTEM_PROMPT = """You are DataDuck's AI database analyst.
Your job is to generate safe, read-only database queries based on natural language questions.

CRITICAL SECURITY RULES:
1. You ONLY generate SELECT statements for SQL or read-only operations for MongoDB.
2. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
3. If the user asks to modify data, respond with a refusal.
4. Generate queries appropriate for the database dialect specified.
5. Always include LIMIT clauses to prevent huge result sets.
6. Prefer aggregations over returning raw data when possible.

You MUST respond with valid JSON matching this exact structure:
{
  "operation": "query",
  "database_type": "postgresql|mysql|sqlite|mongodb",
  "query": "SELECT ...",
  "explanation": "Brief explanation of what this query does.",
  "is_read_only": true,
  "refused": false,
  "refusal_reason": null
}

If you must refuse (write operation requested):
{
  "operation": "refused",
  "database_type": "postgresql",
  "query": "",
  "explanation": "",
  "is_read_only": false,
  "refused": true,
  "refusal_reason": "This database connection is read-only. I can analyze the data, but I cannot modify it."
}

For MongoDB, the "query" field should be a JSON string of the operation dict:
{
  "collection": "users",
  "operation": "find",
  "filter": {},
  "limit": 100
}

IMPORTANT: Respond ONLY with the JSON object. No markdown, no explanation outside JSON."""

ANALYSIS_SYSTEM_PROMPT = """You are DataDuck's AI data analyst. You receive query results and provide clear, professional analysis.

You MUST respond with valid JSON matching this structure:
{
  "answer": "Clear explanation of the findings.",
  "insights": ["Key insight 1", "Key insight 2"],
  "warnings": ["Warning if data quality issue found"],
  "data_quality_notes": ["Any null values, missing data observations"],
  "visualization": {
    "required": true,
    "type": "bar|line|area|pie|donut|scatter|table|kpi",
    "title": "Chart title",
    "description": "Brief description",
    "x_key": "column_name",
    "y_keys": ["revenue", "count"],
    "value_key": null,
    "label_key": null,
    "format": "currency|percentage|number|null"
  }
}

If no visualization is needed, set visualization.required to false.

Visualization selection rules:
- Category + numeric → bar chart
- Time series → line or area chart
- Part-to-whole (≤7 categories) → pie or donut
- Two numeric variables → scatter
- Single important number → kpi
- Large datasets → table
- Multiple series over time → line with multiple y_keys

CRITICAL: Base your answer ONLY on the data provided. Never invent numbers or facts.
If results are empty, say "No matching records were found for this query."
If data is insufficient to answer fully, say so clearly.

Respond ONLY with the JSON object."""


class GeminiService(AIProvider):
    """Google Gemini implementation of the AI provider."""

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise AIServiceError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    async def generate_query(
        self,
        user_question: str,
        relevant_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> QueryGenerationResult:
        """Generate a database query from natural language."""

        # Build context from conversation history
        history_text = self._format_history(conversation_history)

        schema_text = json.dumps(relevant_schema, indent=2)

        prompt = f"""{QUERY_GENERATION_SYSTEM_PROMPT}

DATABASE TYPE: {db_type}

DATABASE SCHEMA (relevant tables only — no credentials):
{schema_text}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_question}

Generate the appropriate read-only {db_type} query:"""

        try:
            response = await self._model.generate_content_async(prompt)
            raw = response.text.strip()
            raw = self._clean_json_response(raw)
            data = json.loads(raw)
            result = QueryGenerationResult(**data)
            logger.info(f"Gemini generated query for '{user_question[:50]}...'")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Gemini returned invalid JSON for query generation: {e}")
            raise AIServiceError(f"AI returned an invalid response. Please try again.")
        except Exception as e:
            logger.error(f"Gemini query generation error: {type(e).__name__}: {e}")
            raise AIServiceError(f"AI service error: {str(e)}")

    async def analyze_results(
        self,
        user_question: str,
        query: str,
        query_result: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """Analyze query results and generate natural language answer."""

        history_text = self._format_history(conversation_history)

        # Limit result data sent to Gemini (first 100 rows for analysis)
        result_preview = dict(query_result)
        if result_preview.get("rows") and len(result_preview["rows"]) > 100:
            result_preview["rows"] = result_preview["rows"][:100]
            result_preview["note"] = f"Showing first 100 of {query_result['row_count']} rows"

        prompt = f"""{ANALYSIS_SYSTEM_PROMPT}

DATABASE TYPE: {db_type}
USER QUESTION: {user_question}
EXECUTED QUERY: {query}

QUERY RESULTS:
{json.dumps(result_preview, indent=2, default=str)}

CONVERSATION HISTORY (for context):
{history_text}

Analyze these results and provide a clear, professional answer:"""

        try:
            response = await self._model.generate_content_async(prompt)
            raw = response.text.strip()
            raw = self._clean_json_response(raw)
            data = json.loads(raw)
            result = AnalysisResult(**data)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Gemini returned invalid JSON for analysis: {e}")
            raise AIServiceError("AI returned an invalid analysis response.")
        except Exception as e:
            logger.error(f"Gemini analysis error: {type(e).__name__}: {e}")
            raise AIServiceError(f"AI analysis error: {str(e)}")

    async def handle_write_intent(self, user_question: str) -> str:
        return (
            "This database connection is read-only. I can analyze the data, "
            "run queries, generate visualizations, and identify patterns — but I "
            "cannot modify, delete, insert, or alter any data. "
            "If you need to make changes, please connect directly to your database."
        )

    def _format_history(self, history: list[dict]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous messages."
        lines = []
        for msg in history[-6:]:  # Last 6 messages for context
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:500]  # Truncate long messages
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _clean_json_response(self, text: str) -> str:
        """Clean up Gemini response to extract JSON."""
        # Remove markdown code blocks if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text


from app.services.ai_provider import get_ai_service
