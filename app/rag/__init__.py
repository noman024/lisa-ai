"""RAG: local embeddings and FAISS retriever."""

from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.retriever import FAISSRetriever, RetrievedChunk

__all__ = [
    "EmbeddingModel",
    "FAISSRetriever",
    "RetrievedChunk",
    "get_embedding_model",
]
