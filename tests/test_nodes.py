"""Unit tests for graph nodes, conversation formatting, and retrieval classification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.context import AgentContext
from openai import OpenAIError

from app.agent.nodes import (
    NodeBundle,
    _classify_query_type,
    _is_off_topic,
    _is_vague_query,
    format_conversation_for_prompt,
)
from app.agent.state import FALLBACK_MESSAGE, GraphState
from app.config import Settings
from app.rag.retriever import RetrievedChunk
from tests.conftest import settings_for_tests
def make_settings(tmp_path: Path) -> Settings:
    return settings_for_tests(
        data_dir=tmp_path,
        llm_base_url="http://127.0.0.1:9/v1",
        llm_model="test-model",
        memory_max_messages=8,
        memory_prompt_max_chars=2000,
        retrieval_min_score=0.1,
    )


def test_format_conversation_trims_and_truncates() -> None:
    h = [
        {"role": "user", "content": "  hi  "},
        {"role": "assistant", "content": "hello"},
    ]
    out = format_conversation_for_prompt(h, max_messages=8, max_chars=1000)
    assert "User: hi" in out
    assert "Assistant: hello" in out
    long = "x" * 5000
    out2 = format_conversation_for_prompt(
        [{"role": "user", "content": long}], max_messages=1, max_chars=100
    )
    assert out2.startswith("…")
    assert len(out2) <= 102  # "…\n" + 100 char tail
    assert format_conversation_for_prompt(None, 5, 100) == ""
    assert format_conversation_for_prompt([], 5, 100) == ""


def test_classify_query_type() -> None:
    assert _classify_query_type("How do I file a claim?") == "claims"
    assert _classify_query_type("Am I eligible at age 60?") == "eligibility"
    assert _classify_query_type("term vs whole life") == "comparison"
    assert _classify_query_type("What is a beneficiary?") == "informational"


def test_off_topic_and_vague_heuristics() -> None:
    assert _is_off_topic("What's the weather tomorrow?")
    assert not _is_off_topic("What is term life insurance?")
    assert not _is_off_topic("life insurance and the weather")
    assert _is_vague_query("ok")
    assert _is_vague_query("हाँ")
    assert not _is_vague_query("What is term life insurance?")


def test_router_sets_off_topic_and_vague(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    assert b.router_node({"user_message": "What's the bitcoin price?"}) == {
        "query_type": "off_topic",
        "vague_query": False,
    }
    out = b.router_node({"user_message": "What is term life?"})
    assert out["query_type"] == "informational"
    assert out["vague_query"] is False


@pytest.mark.asyncio
async def test_retriever_node_off_topic_skips_search(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    r.is_ready = True
    r.search = MagicMock()
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = await b.retriever_node(
        {"user_message": "What's the forecast?", "query_type": "off_topic"}
    )
    assert out["low_confidence"] is True
    assert out["retrieved"] == []
    assert out["best_score"] == 0.0
    r.search.assert_not_called()


@pytest.mark.asyncio
async def test_retriever_node_vague_stricter_threshold(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    s.retrieval_min_score = 0.32
    r = MagicMock()
    r.is_ready = True
    ch = RetrievedChunk(id=0, text="x", source_section="S", score=0.33)
    r.search = MagicMock(return_value=([ch], 0.33))
    r.format_context = MagicMock(return_value="ctx")
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = await b.retriever_node(
        {
            "user_message": "Tell me about that",
            "query_type": "informational",
            "vague_query": True,
            "history": [{"role": "user", "content": "What is term life?"}],
        }
    )
    assert out["low_confidence"] is True
    r.search.assert_called_once()


@pytest.mark.asyncio
async def test_retriever_node_vague_without_history_skips_search(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    r.is_ready = True
    r.search = MagicMock()
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = await b.retriever_node(
        {
            "user_message": "ok",
            "query_type": "informational",
            "vague_query": True,
            "history": [],
        }
    )
    assert out["low_confidence"] is True
    assert out["retrieved"] == []
    r.search.assert_not_called()


@pytest.mark.asyncio
async def test_retriever_node_low_confidence_when_no_chunks(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    r.is_ready = True
    r.search = MagicMock(return_value=([], 0.0))
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = await b.retriever_node({"user_message": "something"})  # type: ignore[arg-type]
    assert out["low_confidence"] is True
    r.search.assert_called_once()


def test_prompt_builder_includes_conversation_block(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    state: GraphState = {
        "user_message": "Does it build cash value?",
        "history": [
            {"role": "user", "content": "What is term life?"},
            {"role": "assistant", "content": "It covers a fixed period."},
        ],
        "low_confidence": False,
        "retrieved": [
            {
                "id": 0,
                "text": "Term life is temporary coverage.",
                "source_section": "Term Life Insurance",
                "score": 0.9,
            },
        ],
    }
    out = b.prompt_builder_node(state)
    payload = out["user_payload"]
    assert "Recent conversation:" in payload
    assert "User: What is term life?" in payload
    assert "Question: Does it build cash value?" in payload
    assert "Context:" in payload


@pytest.mark.asyncio
async def test_llm_node_falls_back_on_openai_error(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    r.is_ready = True
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))

    async def openai_boom(*_a, **_k):
        raise OpenAIError("simulated API failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.agent.nodes.chat_text", openai_boom)
        out = await b.llm_node(
            {
                "low_confidence": False,
                "system_prompt": "sys",
                "user_payload": "u",
            }
        )
    assert out["raw_llm_response"] == FALLBACK_MESSAGE


@pytest.mark.asyncio
async def test_llm_node_falls_back_on_non_openai_error(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    r.is_ready = True
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))

    async def boom(*_a, **_k):
        raise RuntimeError("simulated transport failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.agent.nodes.chat_text", boom)
        out = await b.llm_node(
            {
                "low_confidence": False,
                "system_prompt": "sys",
                "user_payload": "u",
            }
        )
    assert out["raw_llm_response"] == FALLBACK_MESSAGE


def test_validator_llm_call_failed_flags_low_confidence(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    r.is_ready = True
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = b.validator_node(
        {
            "low_confidence": False,
            "raw_llm_response": "I don't know.",
            "context_text": "Term life provides temporary coverage.",
            "retrieved": [{"id": 0, "text": "t", "source_section": "Term", "score": 0.9}],
            "llm_call_failed": True,
        }
    )
    assert out["final_response"] == FALLBACK_MESSAGE
    assert out["low_confidence"] is True
    assert out["sources"] == []


def test_validator_ungrounded_answer_flags_low_confidence(tmp_path: Path) -> None:
    s = settings_for_tests(
        data_dir=tmp_path,
        llm_base_url="http://127.0.0.1:9/v1",
        llm_model="test-model",
        memory_max_messages=8,
        memory_prompt_max_chars=2000,
        retrieval_min_score=0.1,
        grounding_min_overlap=0.99,
    )
    r = MagicMock()
    r.is_ready = True
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = b.validator_node(
        {
            "low_confidence": False,
            "raw_llm_response": "The zebra runs fast across the savanna.",
            "context_text": "Term life insurance provides temporary coverage.",
            "retrieved": [{"id": 0, "text": "t", "source_section": "Term", "score": 0.9}],
        }
    )
    assert out["final_response"] == FALLBACK_MESSAGE
    assert out["low_confidence"] is True


def test_prompt_builder_omits_block_when_no_history(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    r = MagicMock()
    b = NodeBundle(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    state: GraphState = {
        "user_message": "What is life insurance?",
        "history": [],
        "low_confidence": False,
        "retrieved": [
            {"id": 0, "text": "x", "source_section": "S", "score": 0.5},
        ],
    }
    out = b.prompt_builder_node(state)
    assert "Recent conversation:" not in out["user_payload"]
