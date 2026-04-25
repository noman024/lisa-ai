#!/usr/bin/env python3
"""
Build FAISS index and metadata from ``knowledge/insurance_kb.md``.

Chunks: ~300--500 token windows with overlap (see ``app.utils.chunking``).
Run from the repository root::

    export PYTHONPATH=.
    python scripts/ingest_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np

# Repository root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.embeddings import get_embedding_model
from app.utils.chunking import chunk_markdown


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / n).astype(np.float32)


def main() -> None:
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    raw = s.knowledge_path.read_text(encoding="utf-8")
    text_chunks = chunk_markdown(raw)
    texts = [c.text for c in text_chunks]
    sections = [c.source_section for c in text_chunks]
    if not texts:
        raise SystemExit("No chunks produced; check the knowledge file.")

    model = get_embedding_model(s)
    embs = model.encode(texts, batch_size=32)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    d = int(embs.shape[1])
    embs = _l2_normalize(embs)

    index = faiss.IndexFlatIP(d)
    index.add(embs)

    faiss_path = s.data_dir / "faiss.index"
    meta_path = s.data_dir / "metadata.json"
    faiss.write_index(index, str(faiss_path))

    rows = [
        {
            "id": i,
            "text": texts[i],
            "source_section": sections[i],
        }
        for i in range(len(texts))
    ]
    meta = {"model": s.embedding_model_id, "n_chunks": len(texts), "chunks": rows}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(texts)} chunks to {faiss_path} and {meta_path}")


if __name__ == "__main__":
    main()
