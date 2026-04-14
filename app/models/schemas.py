"""
Pydantic models for request/response validation.
All API inputs and outputs are typed and validated here.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    chunks_created: Optional[int] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = Field(
        default=None,
        description="Optionally restrict retrieval to a specific document"
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank.")
        return v.strip()


class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float
    model_used: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
