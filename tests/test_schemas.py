"""Request / response validation and edge cases at the API boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import ChatRequest


def test_chat_request_strips_and_accepts() -> None:
    r = ChatRequest(session_id="  sid-1  ", message="  What is term life?  ")
    assert r.session_id == "sid-1"
    assert r.message == "What is term life?"


def test_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s", message="")


def test_whitespace_only_message_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s", message="   \t  ")


def test_whitespace_session_id_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id="  \n  ", message="hi")


def test_max_length_message() -> None:
    msg = "a" * 8000
    r = ChatRequest(session_id="s", message=msg)
    assert len(r.message) == 8000


def test_message_over_max_length_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s", message="a" * 8001)


def test_session_id_max_length_boundary() -> None:
    s = "a" * 256
    r = ChatRequest(session_id=s, message="hi")
    assert len(r.session_id) == 256
    with pytest.raises(ValidationError):
        ChatRequest(session_id="a" * 257, message="hi")


def test_chat_response_model_defaults() -> None:
    from app.models.schemas import ChatResponse, HealthResponse

    cr = ChatResponse(response="x")
    assert cr.sources == []
    assert cr.query_type == ""
    assert cr.low_confidence is False
    h = HealthResponse(
        status="ok",
        llm_base_url="u",
        llm_model="m",
        embedding_model_id="e",
        index_ready=True,
    )
    assert h.index_error is None
