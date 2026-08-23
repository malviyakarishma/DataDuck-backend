"""
OpenAI provider implementation.

SECURITY RULES:
1. Database credentials are NEVER sent to OpenAI.
2. Only schema metadata and query results are sent.
3. All responses are validated with Pydantic before use.
"""
import json
import logging
import re
from typing import Optional

import httpx

from app.services.ai_provider import AIProvider, QueryGenerationResult, AnalysisResult
from app.services.gemini_service import QUERY_GENERATION_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT
from app.core.config import settings
from app.core.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class OpenAIService(AIProvider):
    """OpenAI implementation of the AI provider."""

    def __init__(self):
        provider = (settings.AI_PROVIDER or "").lower()
        if provider == "groq" or settings.GROQ_API_KEY:
            self.api_key = settings.GROQ_API_KEY or settings.OPENAI_API_KEY
            if not self.api_key:
                raise AIServiceError("GROQ_API_KEY is not configured.")
            self.model = settings.GROQ_MODEL or "llama-3.1-70b-versatile"
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        else:
            self.api_key = settings.OPENAI_API_KEY
            if not self.api_key:
                raise AIServiceError("OPENAI_API_KEY is not configured.")
            self.model = settings.OPENAI_MODEL or "gpt-4o-mini"
            base_url = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
            self.api_url = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                if response.status_code != 200:
                    err_data = response.json() if response.content else {}
                    err_msg = err_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                    logger.error(f"OpenAI API Error ({response.status_code}): {err_msg}")
                    raise AIServiceError(f"OpenAI API Error: {err_msg}")

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(self._clean_json_response(content))
            except httpx.HTTPError as e:
                logger.error(f"OpenAI HTTP connection error: {e}")
                raise AIServiceError(f"OpenAI connection error: {str(e)}")
            except json.JSONDecodeError as e:
                logger.error(f"OpenAI returned invalid JSON: {e}")
                raise AIServiceError("AI returned an invalid JSON response.")

    async def generate_query(
        self,
        user_question: str,
        relevant_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> QueryGenerationResult:
        """Generate a database query from natural language using OpenAI."""
        history_text = self._format_history(conversation_history)
        schema_text = json.dumps(relevant_schema, indent=2)

        prompt = f"""DATABASE TYPE: {db_type}

DATABASE SCHEMA (relevant tables only — no credentials):
{schema_text}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_question}

Generate the appropriate read-only {db_type} query:"""

        try:
            parsed_data = await self._call_openai(QUERY_GENERATION_SYSTEM_PROMPT, prompt)
            result = QueryGenerationResult(**parsed_data)
            logger.info(f"OpenAI generated query for '{user_question[:50]}...'")
            return result
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            logger.error(f"OpenAI query generation error: {e}")
            raise AIServiceError(f"AI service error: {str(e)}")

    async def analyze_results(
        self,
        user_question: str,
        query: str,
        query_result: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """Analyze query results and generate natural language answer using OpenAI."""
        history_text = self._format_history(conversation_history)

        result_preview = dict(query_result)
        if result_preview.get("rows") and len(result_preview["rows"]) > 100:
            result_preview["rows"] = result_preview["rows"][:100]
            result_preview["note"] = f"Showing first 100 of {query_result['row_count']} rows"

        prompt = f"""DATABASE TYPE: {db_type}
USER QUESTION: {user_question}
EXECUTED QUERY: {query}

QUERY RESULTS:
{json.dumps(result_preview, indent=2, default=str)}

CONVERSATION HISTORY (for context):
{history_text}

Analyze these results and provide a clear, professional answer:"""

        try:
            parsed_data = await self._call_openai(ANALYSIS_SYSTEM_PROMPT, prompt)
            result = AnalysisResult(**parsed_data)
            return result
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            logger.error(f"OpenAI analysis error: {e}")
            raise AIServiceError(f"AI analysis error: {str(e)}")

    async def handle_write_intent(self, user_question: str) -> str:
        return (
            "This database connection is read-only. I can analyze the data, "
            "run queries, generate visualizations, and identify patterns — but I "
            "cannot modify, delete, insert, or alter any data. "
            "If you need to make changes, please connect directly to your database."
        )

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous messages."
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:500]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _clean_json_response(self, text: str) -> str:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
