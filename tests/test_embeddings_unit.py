"""SQA: embedding wrapper without downloading models (all mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app.rag.embeddings as _emb_mod
from app.rag.embeddings import EmbeddingModel


def _patch_st_constructs(instance: object) -> object:
    """Replaces the class used in :meth:`EmbeddingModel._load` (bound name in this module)."""
    return patch.object(_emb_mod, "SentenceTransformer", return_value=instance)


class _STNew:
    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, *_a: object, **_k: object) -> np.ndarray:
        return np.array([[0.0] * 4], dtype=np.float32)


class _STLegacy:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, *_a: object, **_k: object) -> np.ndarray:
        return np.array([[0.0] * 3], dtype=np.float32)


def test_embedding_model_dimension_uses_get_embedding_dimension() -> None:
    # _load is lazy: touch ``dimension`` while the patch is active.
    with _patch_st_constructs(_STNew()):
        e = EmbeddingModel("e2e-dim-new")
        assert e.dimension == 4


def test_embedding_model_dimension_falls_back_to_get_sentence_embedding_dimension() -> None:
    with _patch_st_constructs(_STLegacy()):
        e = EmbeddingModel("e2e-dim-legacy")
        assert e.dimension == 3


def test_encode_returns_float32_2d() -> None:
    m = MagicMock()
    m.get_embedding_dimension = MagicMock(return_value=2)
    m.encode = MagicMock(
        return_value=np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    )
    with _patch_st_constructs(m):
        e = EmbeddingModel("enc-1")
        out = e.encode(["a", "b"])
    assert out.shape == (2, 2)
    assert out.dtype == np.float32


@pytest.mark.asyncio
async def test_aencode_runs() -> None:
    m = MagicMock()
    m.get_embedding_dimension = MagicMock(return_value=1)
    m.encode = MagicMock(
        return_value=np.array([[1.0]], dtype=np.float32)
    )
    with _patch_st_constructs(m):
        e = EmbeddingModel("aenc-1")
        a = await e.aencode(["x"], batch_size=8)
    assert a.shape[0] == 1
    m.encode.assert_called_once()
