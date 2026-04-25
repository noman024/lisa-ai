"""Runtime dependencies for the LangGraph node bundle."""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import Settings
from app.rag.retriever import FAISSRetriever


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Wired once at application startup and shared by the compiled graph."""

    settings: Settings
    retriever: FAISSRetriever
    llm: AsyncOpenAI


__all__ = ["AgentContext"]
