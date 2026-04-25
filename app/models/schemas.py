"""Pydantic API models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    session_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("session_id")
    @classmethod
    def session_id_stripped_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("session_id cannot be empty or whitespace-only")
        return s

    @field_validator("message")
    @classmethod
    def message_stripped_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("message cannot be empty or whitespace-only")
        return s


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    response: str
    sources: list[str] = Field(default_factory=list)
    query_type: str = ""
    low_confidence: bool = False


class HealthResponse(BaseModel):
    """GET /health payload: process up, FAISS loaded, and non-secret model routing."""

    status: str
    llm_base_url: str
    llm_model: str
    embedding_model_id: str
    index_ready: bool
    index_error: str | None = None


__all__ = ["ChatRequest", "ChatResponse", "HealthResponse"]
