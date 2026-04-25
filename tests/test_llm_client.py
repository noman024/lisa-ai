"""LLM client: optional ``seed`` and message assembly."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.llm.client import build_async_client, chat_text

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


@pytest.mark.asyncio
async def test_chat_text_empty_or_missing_content() -> None:
    """Completions with no assistant text become empty string."""

    async def fake_create(**_kwargs: object) -> MagicMock:
        ch = MagicMock()
        ch.message = MagicMock()
        ch.message.content = None
        resp = MagicMock()
        resp.choices = [ch]
        return resp

    client = MagicMock()
    client.chat.completions.create = fake_create
    s: Settings = settings_for_tests(
        llm_model="m",
        llm_base_url="http://127.0.0.1:1/v1",
    )
    out = await chat_text(client, s, "sys", "user")
    assert out == ""


def test_get_llm_client_is_process_singleton() -> None:
    """The global LLM client is created once; later calls ignore new settings (documented)."""
    from app.llm import client as client_mod

    client_mod._client = None  # type: ignore[attr-defined]
    try:
        c1 = client_mod.get_llm_client(
            settings_for_tests(llm_base_url="http://a/v1", llm_model="m")
        )
        c2 = client_mod.get_llm_client(
            settings_for_tests(llm_base_url="http://b-different/v1", llm_model="m2")
        )
        assert c1 is c2
    finally:
        client_mod._client = None  # type: ignore[attr-defined]


def test_build_async_client_wires_base_url_and_key() -> None:
    s: Settings = settings_for_tests(
        llm_base_url="http://127.0.0.1:9999/v1",
        llm_api_key="secret-test-key",
        llm_timeout_seconds=77.0,
    )
    c = build_async_client(s)
    assert str(c.base_url).rstrip("/").endswith("/v1")
    # API key is stored on client (not echoed in repr for security)
    assert c.api_key == "secret-test-key"
