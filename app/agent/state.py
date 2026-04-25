"""LangGraph state schema for the life insurance assistant."""

from __future__ import annotations

from typing import List, TypedDict


class GraphState(TypedDict, total=False):
    """State carried through the agent graph (in-memory per request)."""

    session_id: str
    user_message: str
    # Last N messages for prompt injection (user/assistant turns)
    history: List[dict]

    query_type: str
    # True when the question is very short/underspecified (stricter retrieval threshold).
    vague_query: bool
    # Serialized retrieval results
    retrieved: List[dict]
    best_score: float
    context_text: str
    low_confidence: bool

    system_prompt: str
    user_payload: str
    raw_llm_response: str
    # Set when chat.completions raised (timeout, 5xx, etc.) so the API can flag low_confidence.
    llm_call_failed: bool
    final_response: str
    sources: List[str]
    grounded_ok: bool


FALLBACK_MESSAGE = "I don't know."

__all__ = ["GraphState", "FALLBACK_MESSAGE"]
