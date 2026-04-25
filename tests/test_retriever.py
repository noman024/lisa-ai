"""FAISS retriever edge cases (no real index required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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
