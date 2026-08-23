from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ChatRequest(BaseModel):
    database_id: str
    conversation_id: Optional[str] = None
    message: str


class VisualizationSpec(BaseModel):
    required: bool
    type: Optional[str] = None  # bar, line, pie, donut, scatter, area, table, kpi
    title: Optional[str] = None
    description: Optional[str] = None
    x_key: Optional[str] = None
    y_keys: Optional[list[str]] = None
    value_key: Optional[str] = None  # for pie/KPI
    label_key: Optional[str] = None  # for pie
    format: Optional[str] = None  # currency, percentage, number


class QueryInfo(BaseModel):
    display: bool
    language: str  # sql, mongodb
    content: str


class QueryResultData(BaseModel):
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    execution_time_ms: Optional[float] = None


class MessageResponse(BaseModel):
    id: str
    role: str
    answer: str
    insights: list[str] = []
    warnings: list[str] = []
    query: Optional[QueryInfo] = None
    result: Optional[QueryResultData] = None
    visualization: Optional[VisualizationSpec] = None
    created_at: datetime


class ChatResponse(BaseModel):
    conversation_id: str
    conversation_title: str
    message: MessageResponse


class ConversationResponse(BaseModel):
    id: str
    title: str
    database_id: str
    database_name: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int


class MessageListResponse(BaseModel):
    conversation_id: str
    messages: list[MessageResponse]
    total: int
