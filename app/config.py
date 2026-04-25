"""Application configuration (environment and defaults)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment; vLLM + local model is the default dev path."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chat LLM: vLLM (default) or OpenAI / any OpenAI-compatible HTTP API.
    # Alias order: vLLM / generic LLM first, then OpenAI — explicit .env wins per field.
    llm_base_url: str = Field(
        default="http://127.0.0.1:8001/v1",
        description="OpenAI-compatible API base (include /v1).",
        validation_alias=AliasChoices(
            "VLLM_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
        ),
    )
    llm_model: str = Field(
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        description="Model id (served name in vLLM, or OpenAI model name).",
        validation_alias=AliasChoices("VLLM_MODEL", "LLM_MODEL", "OPENAI_MODEL"),
    )
    llm_api_key: str = Field(
        default="not-needed",
        description="API key (optional for local vLLM; required for OpenAI).",
        validation_alias=AliasChoices("VLLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"),
    )

    embedding_model_id: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence-transformers model id (BGE small English).",
    )

    # Project root: parent of the app package. Override via env to point at alternate KB / index.
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data",
        description="FAISS index and metadata directory.",
        validation_alias=AliasChoices("DATA_DIR", "LISA_DATA_DIR"),
    )
    knowledge_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "knowledge"
        / "insurance_kb.md",
        description="Life insurance RAG source (markdown, ## section headers).",
        validation_alias=AliasChoices(
            "KNOWLEDGE_PATH",
            "INSURANCE_KB_PATH",
            "LISA_KNOWLEDGE_PATH",
        ),
    )

    retriever_top_k: int = 5
    retrieval_min_score: float = Field(
        default=0.32,
        description="Minimum best similarity (inner product on normalized BGE) to trust retrieval.",
    )
    grounding_min_overlap: float = Field(
        default=0.25,
        description="Min fraction of answer content words that must appear in context.",
    )

    memory_max_messages: int = Field(
        default=20,
        description="Max user+assistant lines per session; also caps how many are formatted into the LLM prompt.",
    )
    memory_prompt_max_chars: int = Field(
        default=4000,
        description="Max characters of prior conversation to include in the RAG user prompt (tail-truncated).",
    )
    llm_seed: int | None = Field(
        default=None,
        description="If set, passed to chat.completions (OpenAI/vLLM) for more deterministic sampling when supported.",
        validation_alias=AliasChoices("LLM_SEED", "VLLM_SEED"),
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "VLLM_TIMEOUT_SECONDS"),
    )
    llm_max_tokens: int = Field(
        default=600,
        validation_alias=AliasChoices("LLM_MAX_TOKENS", "VLLM_MAX_TOKENS"),
    )
    llm_temperature: float = Field(
        default=0.1,
        validation_alias=AliasChoices("LLM_TEMPERATURE", "VLLM_TEMPERATURE"),
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings singleton."""
    return Settings()


__all__ = ["Settings", "get_settings"]
