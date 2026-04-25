"""HTTP API tests with mocked startup (no vLLM, no FAISS files, no embedding download)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@contextmanager
def mocked_lisa_app(
    ainvoke_result: dict | None = None,
    ainvoke_side_effect: Exception | None = None,
):
    """
    Keep ``app.main`` patches active for the whole ``TestClient`` lifetime.

    Patches must not be closed before the ASGI lifespan runs, or the real
    embedding model and LLM client would load.
    """
    ainvoke = AsyncMock(
        return_value=ainvoke_result
        or {
            "final_response": "Answer from test graph.",
            "sources": ["Term Life Insurance"],
            "query_type": "informational",
            "low_confidence": False,
        }
    )
    if ainvoke_side_effect is not None:
        ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    g = MagicMock()
    g.ainvoke = ainvoke
    ret = MagicMock()
    ret.is_ready = True
    ret.load = MagicMock()
    with (
        patch("app.main.get_embedding_model", return_value=MagicMock()),
        patch("app.main.get_llm_client", return_value=MagicMock()),
        patch("app.main.FAISSRetriever", return_value=ret),
        patch("app.main.build_graph", return_value=g),
    ):
        from app.main import create_app

        yield create_app(), ainvoke


@patch("app.main.get_embedding_model", return_value=MagicMock())
@patch("app.main.get_llm_client", return_value=MagicMock())
@patch("app.main.FAISSRetriever")
@patch("app.main.build_graph")
def test_get_health(
    mock_graph: MagicMock,
    mock_faiss: MagicMock,
    _mock_llm: MagicMock,
    _mock_emb: MagicMock,
) -> None:
    ret = MagicMock()
    ret.is_ready = True
    ret.load = MagicMock()
    mock_faiss.return_value = ret
    mock_graph.return_value = MagicMock(ainvoke=AsyncMock())
    from app.main import create_app

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "llm_model" in data
    assert data["index_ready"] is True
    assert data.get("index_error") in (None, "")


@patch("app.main.get_embedding_model", return_value=MagicMock())
@patch("app.main.get_llm_client", return_value=MagicMock())
@patch("app.main.FAISSRetriever")
@patch("app.main.build_graph")
def test_get_health_shows_index_error_when_not_ready(
    mock_graph: MagicMock,
    mock_faiss: MagicMock,
    _mock_llm: MagicMock,
    _mock_emb: MagicMock,
) -> None:
    ret = MagicMock()
    ret.is_ready = False
    ret.load = MagicMock()
    ret.load_error = "Missing data/faiss.index or data/metadata.json. Run scripts/ingest_kb.py"
    mock_faiss.return_value = ret
    mock_graph.return_value = MagicMock(ainvoke=AsyncMock())
    from app.main import create_app

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["index_ready"] is False
    assert "ingest_kb" in (data.get("index_error") or "")


def test_post_chat_200() -> None:
    with mocked_lisa_app() as (app, ainvoke):
        with TestClient(app) as client:
            r = client.post(
                "/chat",
                json={"session_id": "sess-a", "message": "What is term life?"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "Answer from test graph."
    assert body["query_type"] == "informational"
    ainvoke.assert_awaited()


def test_post_chat_passes_history_on_second_turn() -> None:
    with mocked_lisa_app() as (app, ainvoke):
        with TestClient(app) as client:
            client.post(
                "/chat",
                json={"session_id": "sess-b", "message": "First question?"},
            )
            client.post(
                "/chat",
                json={"session_id": "sess-b", "message": "And a follow-up?"},
            )
    assert ainvoke.await_count == 2
    second_state = ainvoke.call_args_list[1].args[0]
    assert len(second_state.get("history", [])) >= 1


def test_post_chat_422_whitespace_message() -> None:
    with mocked_lisa_app() as (app, _a):
        with TestClient(app) as client:
            r = client.post(
                "/chat",
                json={"session_id": "s", "message": "  \t  "},
            )
    assert r.status_code == 422


def test_post_chat_503_on_graph_failure() -> None:
    with mocked_lisa_app(ainvoke_side_effect=RuntimeError("graph boom")) as (app, ainvoke):
        with TestClient(app) as client:
            r = client.post(
                "/chat",
                json={"session_id": "s", "message": "hello"},
            )
    assert r.status_code == 503
    ainvoke.assert_awaited()
