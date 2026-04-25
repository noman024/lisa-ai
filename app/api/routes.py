"""FastAPI routes: chat and health."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent.state import GraphState

_log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    """Liveness: API up; reports FAISS index, chat endpoint target, and embedding model (no secrets)."""
    s = request.app.state.settings
    r = request.app.state.retriever
    return HealthResponse(
        status="ok",
        llm_base_url=s.llm_base_url,
        llm_model=s.llm_model,
        embedding_model_id=s.embedding_model_id,
        index_ready=bool(r.is_ready),
        index_error=r.load_error if not r.is_ready else None,
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Run one user turn: load prior turns for this ``session_id``, run the LangGraph
    (history is formatted into the prompt for follow-ups; facts still come from RAG),
    then append this turn to the session store.
    """
    store = request.app.state.session_store
    graph = request.app.state.graph
    history = store.get_history(body.session_id)
    # Current message is the active query; previous turns are in ``history`` only
    state_in: GraphState = {
        "session_id": body.session_id,
        "user_message": body.message,
        "history": list(history),
    }
    try:
        out = await graph.ainvoke(state_in)
    except Exception as exc:
        _log.exception("LangGraph /chat failed")
        raise HTTPException(
            status_code=503,
            detail="Chat pipeline failed. Ensure the FAISS index is built, the LLM server is reachable, and check server logs.",
        ) from exc
    final = out.get("final_response", "I don't know.")
    src = out.get("sources", [])
    if not isinstance(src, list):
        src = []
    store.append(body.session_id, "user", body.message)
    store.append(body.session_id, "assistant", str(final))
    return ChatResponse(
        response=str(final),
        sources=[str(s) for s in src],
        query_type=str(out.get("query_type", "")),
        low_confidence=bool(out.get("low_confidence", False)),
    )


__all__ = ["router"]
