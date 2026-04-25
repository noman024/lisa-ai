"""LLM client: optional ``seed`` and message assembly."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.llm.client import chat_text

from tests.conftest import settings_for_tests


def _make_completion(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    ch = MagicMock()
    ch.message = msg
    resp = MagicMock()
    resp.choices = [ch]
    return resp


@pytest.mark.asyncio
async def test_chat_text_passes_seed_when_configured() -> None:
    captured: dict = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return _make_completion("OK")

    client = MagicMock()
    client.chat.completions.create = fake_create
    s: Settings = settings_for_tests(
        llm_seed=42,
        llm_model="test-model",
        llm_base_url="http://127.0.0.1:1/v1",
    )
    out = await chat_text(client, s, "system text", "user text")
    assert out == "OK"
    assert captured.get("seed") == 42
    assert captured.get("model") == "test-model"
    assert len(captured.get("messages", [])) == 2  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_chat_text_omits_seed_when_unset() -> None:
    captured: dict = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return _make_completion("x")

    client = MagicMock()
    client.chat.completions.create = fake_create
    s: Settings = settings_for_tests(
        llm_model="m",
        llm_base_url="http://127.0.0.1:1/v1",
    )
    assert s.llm_seed is None
    await chat_text(client, s, "s", "u")
    assert "seed" not in captured
