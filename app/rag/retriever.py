"""FAISS-based dense retrieval with metadata sidecar (JSON)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import faiss
import numpy as np

from app.rag.embeddings import EmbeddingModel

# Embedding models have finite context; long questions still use the full text in the LLM prompt.
_RETRIEVAL_QUERY_MAX_CHARS = 4000


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk with similarity score and provenance."""

    id: int
    text: str
    source_section: str
    score: float


def _l2_renorm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


class FAISSRetriever:
    """
    Loads a flat inner-product FAISS index (normalized vectors = cosine).
    """

    def __init__(
        self,
        data_dir: Path,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._emb = embedding_model
        self._lock = threading.Lock()
        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []
        self._loaded = False
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._index is not None and len(self._chunks) > 0

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Load ``faiss.index`` and ``metadata.json`` from data directory."""
        index_path = self._data_dir / "faiss.index"
        meta_path = self._data_dir / "metadata.json"
        with self._lock:
            if not index_path.is_file() or not meta_path.is_file():
                self._load_error = f"Missing {index_path} or {meta_path}. Run scripts/ingest_kb.py"
                self._loaded = True
                return
            self._index = faiss.read_index(str(index_path))
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            self._chunks = list(raw.get("chunks", []))
            self._loaded = True
            self._load_error = None
            d = int(self._index.d)  # type: ignore[attr-defined]
            if d != self._emb.dimension and self._chunks:
                self._load_error = (
                    f"Index dim {d} != embedding model dim {self._emb.dimension}"
                )
                self._index = None
                self._chunks = []

    def search(
        self,
        query: str,
        k: int,
    ) -> tuple[List[RetrievedChunk], float]:
        """
        Return top-``k`` chunks and the best score (0 if no index / empty).
        """
        q_raw = (query or "").strip()
        if not self.is_ready or not q_raw or not self._index:
            return [], 0.0
        q_embed = (
            q_raw
            if len(q_raw) <= _RETRIEVAL_QUERY_MAX_CHARS
            else q_raw[:_RETRIEVAL_QUERY_MAX_CHARS]
        )
        q = self._emb.encode([q_embed])
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = _l2_renorm(q)
        n = int(self._index.ntotal)  # type: ignore[attr-defined]
        if n == 0:
            return [], 0.0
        kk = min(k, n)
        sims, idxs = self._index.search(q, kk)  # type: ignore[union-attr]
        best = float(sims[0, 0]) if sims.size else 0.0
        out: list[RetrievedChunk] = []
        for j in range(sims.shape[1]):
            i = int(idxs[0, j])
            if i < 0:
                continue
            row = self._chunks[i]
            out.append(
                RetrievedChunk(
                    id=i,
                    text=row.get("text", ""),
                    source_section=row.get("source_section", ""),
                    score=float(sims[0, j]),
                )
            )
        return out, best

    @staticmethod
    def format_context(chunks: Sequence[RetrievedChunk]) -> str:
        """Join chunks into a single block for the LLM."""
        parts: list[str] = []
        for c in chunks:
            parts.append(f"### {c.source_section} (relevance: {c.score:.3f})\n{c.text}")
        return "\n\n".join(parts) if parts else ""


__all__ = ["FAISSRetriever", "RetrievedChunk"]
