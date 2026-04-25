"""FAISS retriever edge cases (no real index required)."""

from __future__ import annotations

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
