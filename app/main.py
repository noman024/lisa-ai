"""
Life Insurance Support Assistant — FastAPI entry (async) with LangGraph + RAG
and an OpenAI-compatible chat LLM (OpenAI API, vLLM, or similar).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.agent.context import AgentContext
from app.agent.graph import build_graph
from app.config import get_settings
from app.llm.client import get_llm_client
from app.memory.store import SessionStore
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import FAISSRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retriever, embeddings, LLM client, and compile the graph once per process."""
    settings = get_settings()
    emb = get_embedding_model(settings)
    ret = FAISSRetriever(settings.data_dir, emb)
    ret.load()
    ctx = AgentContext(
        settings=settings,
        retriever=ret,
        llm=get_llm_client(settings),
    )
    g = build_graph(ctx)
    app.state.settings = settings
    app.state.session_store = SessionStore(max_messages=settings.memory_max_messages)
    app.state.retriever = ret
    app.state.graph = g
    yield


def create_app() -> FastAPI:
    """Application factory (useful for tests and ASGI hosting)."""
    app = FastAPI(
        title="Life Insurance Support Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
