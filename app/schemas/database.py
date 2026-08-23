from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CreateDatabaseRequest(BaseModel):
    name: str
    connection_string: str

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Production DB",
                "connection_string": "postgresql://user:password@host:5432/database"
            }
        }


class TestConnectionRequest(BaseModel):
    connection_string: str


class DatabaseResponse(BaseModel):
    id: str
    name: str
    db_type: str
    masked_connection_string: str
    host: Optional[str]
    port: Optional[int]
    database_name: Optional[str]
    username: Optional[str]
    is_connected: bool
    last_tested_at: Optional[datetime]
    schema_analyzed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class DatabaseListResponse(BaseModel):
    databases: list[DatabaseResponse]
    total: int


class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    db_type: Optional[str] = None
    database_name: Optional[str] = None


class SchemaOverviewResponse(BaseModel):
    database_id: str
    db_type: str
    database_name: str
    total_tables: int
    total_relationships: int
    tables: list[dict]
    relationships: list[dict]
    analyzed_at: Optional[datetime]


class ColumnSummary(BaseModel):
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    references_table: Optional[str] = None


class TableSummary(BaseModel):
    name: str
    row_count: Optional[int]
    column_count: int
    columns: list[ColumnSummary]
    primary_keys: list[str]
    foreign_keys: list[dict]
