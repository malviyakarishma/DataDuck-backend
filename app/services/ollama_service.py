"""
Ollama local LLM provider implementation.

SECURITY & ARCHITECTURE RULES:
1. Database credentials are NEVER sent to Ollama.
2. Only schema metadata and query results are sent.
3. All requests go strictly through the backend (never directly from frontend).
4. All responses are validated with Pydantic before use.
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


class OllamaService(AIProvider):
    """Ollama local model implementation of the AI provider."""

    def __init__(self):
        self.base_url = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        self.model = settings.OLLAMA_MODEL or "qwen2.5-coder:7b"
        self.api_url = f"{self.base_url}/api/chat"

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict:
        """Call Ollama chat endpoint and parse JSON response."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(self.api_url, json=payload)
                
                if response.status_code == 404:
                    logger.error(f"Ollama model '{self.model}' not found.")
                    raise AIServiceError(
                        f"The configured Ollama model '{self.model}' was not found. "
                        f"Please run 'ollama pull {self.model}' in your terminal."
                    )
                
                if response.status_code != 200:
                    err_text = response.text[:200]
                    logger.error(f"Ollama API Error ({response.status_code}): {err_text}")
                    raise AIServiceError(f"Ollama API returned status {response.status_code}: {err_text}")

                data = response.json()
                content = data.get("message", {}).get("content", "")
                if not content:
                    raise AIServiceError("Ollama returned an empty response.")
                
                cleaned = self._clean_json_response(content)
                return json.loads(cleaned)

            except (httpx.ConnectError, httpx.ConnectTimeout):
                logger.error(f"Failed to connect to Ollama at {self.base_url}")
                raise AIServiceError(
                    f"Local AI service is unavailable. Make sure Ollama is running at {self.base_url}."
                )
            except httpx.ReadTimeout:
                logger.error("Ollama request timed out.")
                raise AIServiceError("Ollama request timed out. Please check if the model is loaded and responding.")
            except httpx.HTTPError as e:
                logger.error(f"Ollama HTTP error: {e}")
                raise AIServiceError(f"Ollama connection error: {str(e)}")
            except json.JSONDecodeError as e:
                logger.error(f"Ollama returned invalid JSON: {e}")
                raise AIServiceError("Ollama returned an invalid JSON response.")

    async def generate_query(
        self,
        user_question: str,
        relevant_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> QueryGenerationResult:
        """Generate a database query from natural language using Ollama."""
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
            parsed_data = await self._call_ollama(QUERY_GENERATION_SYSTEM_PROMPT, prompt)
            
            # Clean any leftover markdown code blocks in raw SQL field if present
            if isinstance(parsed_data.get("query"), str):
                parsed_data["query"] = self._strip_markdown_code_fences(parsed_data["query"])

            result = QueryGenerationResult(**parsed_data)
            logger.info(f"Ollama generated query using model '{self.model}' for '{user_question[:50]}...'")
            return result
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            logger.error(f"Ollama query generation error: {e}")
            raise AIServiceError(f"AI service error: {str(e)}")

    async def analyze_results(
        self,
        user_question: str,
        query: str,
        query_result: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """Analyze query results and generate natural language answer using Ollama."""
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
            parsed_data = await self._call_ollama(ANALYSIS_SYSTEM_PROMPT, prompt)
            result = AnalysisResult(**parsed_data)
            return result
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            logger.error(f"Ollama analysis error: {e}")
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

    def _strip_markdown_code_fences(self, text: str) -> str:
        text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)
        return text.strip()

    def _clean_json_response(self, text: str) -> str:
        text = self._strip_markdown_code_fences(text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
