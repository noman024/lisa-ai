"""FAISS retriever edge cases (no real index required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from app.rag.retriever import FAISSRetriever


def test_faiss_retriever_empty_or_whitespace_query_no_search(tmp_path: Path) -> None:
    """Inner-product search is skipped when the query is blank after strip."""
    emb = MagicMock()
    ret = FAISSRetriever(tmp_path, emb)  # type: ignore[arg-type]
    # Simulate a loaded index so only the query guard should fire:
    ret._loaded = True  # noqa: SLF001
    ret._index = MagicMock()  # noqa: SLF001
    ret._chunks = [{"text": "t", "source_section": "S"}]  # noqa: SLF001
    out, best = ret.search("  \n\t  ", k=3)
    assert out == [] and best == 0.0
    emb.encode.assert_not_called()


def test_faiss_retriever_truncates_long_query_for_embedding(tmp_path: Path) -> None:
    """Only the head of a very long string is embedded; retrieval still runs."""
    emb = MagicMock()
    emb.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    ret = FAISSRetriever(tmp_path, emb)  # type: ignore[arg-type]

    d = 2
    index = faiss.IndexFlatIP(d)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    ret._loaded = True  # noqa: SLF001
    ret._index = index  # noqa: SLF001
    ret._chunks = [{"text": "chunk0", "source_section": "S0"}]  # noqa: SLF001

    long_q = "x" * 5000
    ret.search(long_q, k=3)
    emb.encode.assert_called_once()
    passed = emb.encode.call_args[0][0]
    assert len(passed) == 1
    assert len(passed[0]) == 4000
    assert passed[0] == "x" * 4000


def test_format_context_joins_section_and_relevance() -> None:
    from app.rag.retriever import RetrievedChunk

    a = RetrievedChunk(id=0, text="t1", source_section="A", score=0.5)
    b = RetrievedChunk(id=1, text="t2", source_section="B", score=0.9)
    out = FAISSRetriever.format_context([a, b])
    assert "### A" in out and "0.500" in out
    assert "### B" in out and "0.900" in out
    assert "t1" in out and "t2" in out


def test_faiss_load_sets_error_when_index_missing(tmp_path: Path) -> None:
    emb = MagicMock()
    ret = FAISSRetriever(tmp_path, emb)  # type: ignore[arg-type]
    ret.load()
    assert ret.is_ready is False
    assert "ingest" in (ret.load_error or "").lower()
    ret.search("term life", k=3)
    emb.encode.assert_not_called()


def test_faiss_index_dimension_mismatch_clears_index(
    tmp_path: Path,
) -> None:
    """Mismatched embedding dim vs index marks error and not ready."""
    ddir = tmp_path
    d = 2
    index = faiss.IndexFlatIP(d)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(ddir / "faiss.index"))
    meta = {"chunks": [{"text": "x", "source_section": "S", "id": 0}, {"id": 1, "text": "y", "source_section": "S2"}]}
    (ddir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    emb = MagicMock()
    emb.dimension = 999  # not 2
    emb.encode.return_value = np.zeros((1, 999), dtype=np.float32)
    ret = FAISSRetriever(ddir, emb)  # type: ignore[arg-type]
    ret.load()
    assert ret.is_ready is False
    assert "dim" in (ret.load_error or "").lower()
    out, best = ret.search("q", k=1)
    assert out == [] and best == 0.0
