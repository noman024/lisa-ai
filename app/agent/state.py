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
    # Serialized retrieval results
    retrieved: List[dict]
    best_score: float
    context_text: str
    low_confidence: bool

    system_prompt: str
    user_payload: str
    raw_llm_response: str
    final_response: str
    sources: List[str]
    grounded_ok: bool


FALLBACK_MESSAGE = "I don't know."

__all__ = ["GraphState", "FALLBACK_MESSAGE"]
