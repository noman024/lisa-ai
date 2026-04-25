"""Local embedding model (BGE via sentence-transformers), normalized for cosine / FAISS IP."""

from __future__ import annotations

import asyncio
import threading
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings


class EmbeddingModel:
    """
    Wraps a sentence-transformers model with process-safe lazy init.

    Embeddings are L2-normalized for use with FAISS inner product (cosine).
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._st: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._st is not None:
            return self._st
        with self._lock:
            if self._st is None:
                self._st = SentenceTransformer(self._model_id)
        assert self._st is not None
        return self._st

    @property
    def dimension(self) -> int:
        m = self._load()
        if hasattr(m, "get_embedding_dimension"):
            return int(m.get_embedding_dimension())  # sentence-transformers >= 3.0
        return int(m.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        m = self._load()
        arr = m.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(arr, list):
            return np.array(arr, dtype=np.float32)
        return arr.astype(np.float32, copy=False)

    async def aencode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return await asyncio.to_thread(self.encode, list(texts), batch_size)


_model_singleton: EmbeddingModel | None = None
_model_lock = threading.Lock()


def get_embedding_model(settings: Settings | None = None) -> EmbeddingModel:
    """Return a process-wide embedding model (singleton)."""
    global _model_singleton
    s = settings or get_settings()
    with _model_lock:
        if _model_singleton is None or _model_singleton._model_id != s.embedding_model_id:
            _model_singleton = EmbeddingModel(s.embedding_model_id)
    return _model_singleton


__all__ = ["EmbeddingModel", "get_embedding_model"]
