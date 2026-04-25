"""Pydantic API models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    session_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=8000)


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


__all__ = ["ChatRequest", "ChatResponse", "HealthResponse"]
