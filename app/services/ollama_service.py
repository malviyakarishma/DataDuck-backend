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
from app.services.gemini_service import (
    QUERY_GENERATION_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT, SCHEMA_EXPLORATION_SYSTEM_PROMPT,
    CASUAL_CHAT_SYSTEM_PROMPT
)
from app.core.config import settings
from app.core.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class OllamaService(AIProvider):
    """Ollama local model implementation of the AI provider."""

    def __init__(self):
        self.base_url = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        self.model = settings.OLLAMA_MODEL or "qwen2.5-coder:7b"
        self.api_url = f"{self.base_url}/api/chat"

    async def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        action_name: str = "llm_call",
        temperature: float = 0.1,
        num_predict: Optional[int] = None,
    ) -> dict:
        """Call Ollama chat endpoint and parse JSON response with detailed performance timing."""
        import time

        options = {
            "temperature": temperature,
        }
        if num_predict:
            options["num_predict"] = num_predict

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": options,
        }

        prompt_char_len = len(system_prompt) + len(user_prompt)
        logger.info(f"⏳ [OLLAMA START] Action: '{action_name}' | Model: '{self.model}' | Prompt Length: ~{prompt_char_len} chars")

        t_start = time.perf_counter()

        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                response = await client.post(self.api_url, json=payload)
                t_end = time.perf_counter()
                http_duration_ms = (t_end - t_start) * 1000.0

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

                # Extract Ollama's internal timing and token metrics
                total_duration_ms = data.get("total_duration", 0) / 1e6
                load_duration_ms = data.get("load_duration", 0) / 1e6
                prompt_eval_duration_ms = data.get("prompt_eval_duration", 0) / 1e6
                eval_duration_ms = data.get("eval_duration", 0) / 1e6
                prompt_eval_count = data.get("prompt_eval_count", 0)
                eval_count = data.get("eval_count", 0)
                
                tok_per_sec = (eval_count / (eval_duration_ms / 1000.0)) if eval_duration_ms > 0 else 0.0

                logger.info(
                    f"⏱️ [OLLAMA TIMING] Action: '{action_name}' | Total HTTP: {http_duration_ms:.1f}ms ({http_duration_ms/1000:.2f}s) | "
                    f"Ollama Internal: {total_duration_ms:.1f}ms | Load: {load_duration_ms:.1f}ms | "
                    f"Prompt Eval: {prompt_eval_duration_ms:.1f}ms ({prompt_eval_count} tokens) | "
                    f"Generation: {eval_duration_ms:.1f}ms ({eval_count} tokens @ {tok_per_sec:.1f} tok/s)"
                )

                if http_duration_ms > 15000:
                    logger.warning(
                        f"⚠️ [OLLAMA SLOW] Action '{action_name}' took {http_duration_ms/1000:.1f}s. "
                        f"Speed: {tok_per_sec:.1f} tokens/s. (If < 15 tok/s, model is likely running on CPU instead of GPU)."
                    )

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
            parsed_data = await self._call_ollama(
                system_prompt=QUERY_GENERATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                action_name="query_generation",
                temperature=0.1,
                num_predict=512,
            )
            
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
            parsed_data = await self._call_ollama(
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                user_prompt=prompt,
                action_name="result_analysis",
                temperature=0.1,
                num_predict=1024,
            )
            result = AnalysisResult(**parsed_data)
            return result
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            logger.error(f"Ollama analysis error: {e}")
            raise AIServiceError(f"AI analysis error: {str(e)}")

    async def classify_intent(
        self,
        user_question: str,
        conversation_history: list[dict],
    ) -> str:
        """Classify user intent using Ollama with fast heuristic check."""
        from app.services.ai_provider import (
            quick_classify_intent, INTENT_DATA_QUERY, INTENT_SCHEMA_EXPLORATION,
            INTENT_CASUAL_CHAT, INTENT_WRITE_REQUEST, ALL_INTENTS
        )
        quick = quick_classify_intent(user_question)
        if quick:
            return quick

        history_text = self._format_history(conversation_history)
        prompt = f"""{INTENT_CLASSIFICATION_SYSTEM_PROMPT}

CONVERSATION HISTORY:
{history_text}

USER MESSAGE: {user_question}

Classify the user intent:"""

        try:
            parsed = await self._call_ollama(
                system_prompt=INTENT_CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                action_name="intent_classification",
                temperature=0.1,
                num_predict=64,
            )
            intent = parsed.get("intent", "").upper().strip()
            if intent in ALL_INTENTS:
                return intent
            return INTENT_DATA_QUERY
        except Exception as e:
            logger.warning(f"Ollama intent classification failed: {e}. Defaulting to DATA_QUERY.")
            return INTENT_DATA_QUERY

    async def answer_schema_question(
        self,
        user_question: str,
        full_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """Answer schema exploration questions using Ollama."""
        history_text = self._format_history(conversation_history)
        schema_text = json.dumps(full_schema, indent=2, default=str)

        prompt = f"""{SCHEMA_EXPLORATION_SYSTEM_PROMPT}

DATABASE TYPE: {db_type}

DATABASE SCHEMA METADATA:
{schema_text}

CONVERSATION HISTORY:
{history_text}

USER QUESTION ABOUT SCHEMA: {user_question}

Explain the requested schema details accurately and clearly:"""

        try:
            parsed_data = await self._call_ollama(
                system_prompt=SCHEMA_EXPLORATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                action_name="schema_explanation",
                temperature=0.1,
                num_predict=1024,
            )
            return AnalysisResult(**parsed_data)
        except Exception as e:
            logger.error(f"Ollama schema exploration error: {e}")
            return AnalysisResult(
                answer=f"Here is the schema overview for this {db_type} database:\n\n" +
                       f"- Total tables: {full_schema.get('total_tables', len(full_schema.get('tables', [])))}\n" +
                       f"- Tables: {', '.join(t.get('name', '') for t in full_schema.get('tables', []))}",
                insights=["Use the Schema Explorer button to inspect full columns and relationships."],
            )

    async def handle_casual_chat(
        self,
        user_question: str,
        conversation_history: list[dict],
    ) -> str:
        """Generate a natural conversational response using Ollama."""
        import time
        history_text = self._format_history(conversation_history)
        prompt = f"""{CASUAL_CHAT_SYSTEM_PROMPT}

CONVERSATION HISTORY:
{history_text}

USER MESSAGE: {user_question}

Response:"""

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": CASUAL_CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 256},
            }
            t_start = time.perf_counter()
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(self.api_url, json=payload)
                t_end = time.perf_counter()
                if resp.status_code == 200:
                    data = resp.json()
                    dur_ms = (t_end - t_start) * 1000.0
                    eval_count = data.get("eval_count", 0)
                    eval_duration_ms = data.get("eval_duration", 0) / 1e6
                    tok_s = (eval_count / (eval_duration_ms / 1000.0)) if eval_duration_ms > 0 else 0
                    logger.info(f"⏱️ [OLLAMA TIMING] Action: 'casual_chat' | HTTP: {dur_ms:.1f}ms | Generated {eval_count} tokens @ {tok_s:.1f} tok/s")
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        return content
        except Exception as e:
            logger.warning(f"Ollama casual chat call error: {e}")

        return "Hello! I am DataDuck, your read-only database analyst. Ask me questions about your data, inspect table schemas, or request an ER diagram!"

    async def handle_write_intent(self, user_question: str) -> str:
        return (
            "This database connection is read-only. I can analyze the data, "
            "run queries, generate visualizations, explain schemas, and create ER diagrams — but I "
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
