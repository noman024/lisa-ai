"""OpenAI-compatible async client (OpenAI API or local vLLM / other compatible servers)."""

from __future__ import annotations

import threading
from openai import AsyncOpenAI

from app.config import Settings, get_settings


def build_async_client(settings: Settings) -> AsyncOpenAI:
    """Create an AsyncOpenAI client for the configured base URL and API key."""
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )


_client: AsyncOpenAI | None = None
_lock = threading.Lock()


def get_llm_client(settings: Settings | None = None) -> AsyncOpenAI:
    """Process-wide LLM client (singleton)."""
    global _client
    s = settings or get_settings()
    with _lock:
        if _client is None:
            _client = build_async_client(s)
    return _client


async def chat_text(
    client: AsyncOpenAI,
    settings: Settings,
    system: str,
    user: str,
) -> str:
    """Run a single-turn chat and return the assistant message content."""
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0]
    if not choice.message or not choice.message.content:
        return ""
    return choice.message.content.strip()


__all__ = ["build_async_client", "get_llm_client", "chat_text"]
