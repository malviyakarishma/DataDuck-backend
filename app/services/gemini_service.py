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
If data is insufficient to answer fully, say so clearly."""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """You are DataDuck's AI query intent classifier.
Analyze the user's message in the context of database analysis.

Classify the message into EXACTLY ONE of these categories:
1. DATA_QUERY: The user wants to retrieve, calculate, filter, count, aggregate, or analyze actual data records (e.g., "top 5 products", "revenue this year", "users who signed up yesterday", "find duplicate emails").
2. SCHEMA_EXPLORATION: The user asks about database structure, schema, tables list, column definitions, data types, primary keys, foreign keys, relationships, constraints, or requests an ER diagram (e.g., "explain database schema", "what tables exist", "explain users table", "show columns in orders", "how are users and orders related", "show ER diagram").
3. CASUAL_CHAT: Greetings, introductory questions, capabilities, assistant identity, pleasantries, or general help (e.g., "hi", "who are you", "what can you do", "thanks").
4. WRITE_REQUEST: Requests to write, insert, update, modify, delete, drop, alter, truncate, or create tables/data (e.g., "delete user 10", "update status to active", "drop table test").

You MUST respond with valid JSON:
{
  "intent": "DATA_QUERY" | "SCHEMA_EXPLORATION" | "CASUAL_CHAT" | "WRITE_REQUEST"
}
IMPORTANT: Respond ONLY with the JSON object."""

SCHEMA_EXPLORATION_SYSTEM_PROMPT = """You are DataDuck's database schema expert.
Your job is to clearly, thoroughly, and professionally explain database architecture, tables, columns, constraints, foreign keys, and relationships based on the provided schema metadata.

GUIDELINES:
1. Use clean GitHub markdown with headings, bullet points, bold keywords, and concise tables where helpful.
2. When explaining tables: mention table name, row count if known, primary keys, foreign keys, and purpose.
3. When explaining columns: list column names, data types, whether they are nullable or non-nullable, and default values if present.
4. When explaining relationships: explain how tables connect (e.g., `orders.user_id` -> `users.id`), noting the direction (e.g., each user can have multiple orders).
5. When the user asks for an ER diagram or schema diagram: provide a clear summary of all entities and relationships in the schema and state that the ER diagram has been rendered.
6. Base your response strictly on the schema provided. Never invent non-existent tables or columns.

You MUST respond with valid JSON:
{
  "answer": "Detailed markdown explanation of the schema/tables/columns/relationships.",
  "insights": ["Key schema insight 1", "Key schema insight 2"],
  "warnings": [],
  "data_quality_notes": []
}
IMPORTANT: Respond ONLY with the JSON object."""

CASUAL_CHAT_SYSTEM_PROMPT = """You are DataDuck, an AI-powered read-only database analyst and assistant ("Ask. Dig. Discover.").
You help users explore database schemas, query data using plain English, generate charts and visualizations, and render dynamic ER diagrams.
Always maintain a friendly, knowledgeable, and helpful tone.
Emphasize that DataDuck is strictly read-only for database safety.
Respond with natural, concise conversational text."""



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
        self._text_model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
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

    async def classify_intent(
        self,
        user_question: str,
        conversation_history: list[dict],
    ) -> str:
        """Classify user intent using Gemini with rule-based fallback."""
        from app.services.ai_provider import (
            quick_classify_intent, INTENT_DATA_QUERY, INTENT_SCHEMA_EXPLORATION,
            INTENT_CASUAL_CHAT, INTENT_WRITE_REQUEST, ALL_INTENTS
        )
        # Fast path check
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
            response = await self._model.generate_content_async(prompt)
            raw = self._clean_json_response(response.text.strip())
            data = json.loads(raw)
            intent = data.get("intent", "").upper().strip()
            if intent in ALL_INTENTS:
                return intent
            return INTENT_DATA_QUERY
        except Exception as e:
            logger.warning(f"Gemini intent classification failed: {e}. Defaulting to DATA_QUERY.")
            return INTENT_DATA_QUERY

    async def answer_schema_question(
        self,
        user_question: str,
        full_schema: dict,
        db_type: str,
        conversation_history: list[dict],
    ) -> AnalysisResult:
        """Answer database schema structure and relationship questions using Gemini."""
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
            response = await self._model.generate_content_async(prompt)
            raw = self._clean_json_response(response.text.strip())
            data = json.loads(raw)
            return AnalysisResult(**data)
        except Exception as e:
            logger.error(f"Gemini schema exploration error: {e}")
            # Fallback direct response
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
        """Generate a natural conversational response."""
        history_text = self._format_history(conversation_history)
        prompt = f"""{CASUAL_CHAT_SYSTEM_PROMPT}

CONVERSATION HISTORY:
{history_text}

USER MESSAGE: {user_question}

Response:"""

        try:
            response = await self._text_model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini casual chat error: {e}")
            return "Hello! I am DataDuck, your read-only database assistant. Ask me questions about your data, database schema, or request an ER diagram!"

    async def handle_write_intent(self, user_question: str) -> str:
        return (
            "This database connection is read-only. I can analyze the data, "
            "run queries, generate visualizations, explain schemas, and create ER diagrams — but I "
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
