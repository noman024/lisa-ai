"""SQA: compiled LangGraph end-to-end with mocked I/O (no vLLM, no FAISS files)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.context import AgentContext
from app.agent.graph import build_graph
from app.agent.state import FALLBACK_MESSAGE
from app.rag.retriever import RetrievedChunk
from tests.conftest import settings_for_tests


@pytest.mark.asyncio
async def test_graph_returns_grounded_answer_with_sources(tmp_path: Path) -> None:
    s = settings_for_tests(
        data_dir=tmp_path,
        llm_base_url="http://127.0.0.1:9/v1",
        llm_model="test-model",
        retrieval_min_score=0.1,
        grounding_min_overlap=0.2,
    )
    r = MagicMock()
    r.is_ready = True
    ch = RetrievedChunk(
        id=0,
        text="Term life insurance provides temporary coverage for a set period of years.",
        source_section="Term Life Insurance",
        score=0.9,
    )
    r.search = MagicMock(return_value=([ch], 0.9))
    r.format_context = MagicMock(
        return_value="Term life insurance provides temporary coverage for a set period of years."
    )
    llm = MagicMock()
    with patch("app.agent.nodes.chat_text", new_callable=AsyncMock) as m_chat:
        m_chat.return_value = (
            "Term life insurance offers temporary coverage for a specified number of years."
        )
        g = build_graph(AgentContext(settings=s, retriever=r, llm=llm))
        out = await g.ainvoke(
            {
                "session_id": "e2e-1",
                "user_message": "What is term life insurance?",
                "history": [],
            }
        )
    assert out.get("query_type") == "informational"
    assert "temporary coverage" in (out.get("final_response") or "").lower()
    assert out.get("low_confidence") is False
    assert "Term Life" in " ".join(out.get("sources") or [])


@pytest.mark.asyncio
async def test_graph_off_topic_skips_retrieval_path(tmp_path: Path) -> None:
    s = settings_for_tests(
        data_dir=tmp_path,
        llm_base_url="http://127.0.0.1:9/v1",
        llm_model="test-model",
    )
    r = MagicMock()
    r.is_ready = True
    r.search = MagicMock()
    g = build_graph(AgentContext(settings=s, retriever=r, llm=MagicMock()))
    out = await g.ainvoke(
        {
            "session_id": "e2e-2",
            "user_message": "What is the weather in Chicago?",
            "history": [],
        }
    )
    assert out.get("query_type") == "off_topic"
    r.search.assert_not_called()
    assert out.get("final_response") == FALLBACK_MESSAGE
