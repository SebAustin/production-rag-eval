"""Runtime configuration via pydantic-settings.

All values are read from environment variables (or a local ``.env`` file). API
keys are required and have no defaults so that a misconfigured environment fails
loudly at construction time rather than at the first API call.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings. See ``.env.example`` for the full surface."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys (required) ---
    anthropic_api_key: str
    cohere_api_key: str
    voyage_api_key: str

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "financebench_v1"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval ---
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rerank_top_n: int = 10

    # --- Embedding (voyage-3-large, Matryoshka) ---
    voyage_model: str = "voyage-3-large"
    embedding_dim: int = 256

    # --- Conformal abstention ---
    conformal_alpha: float = 0.10

    # --- Models ---
    haiku_model: str = "claude-haiku-4-5-20250929"
    sonnet_model: str = "claude-sonnet-4-5-20251022"

    # --- Cost guardrails ---
    max_api_spend_usd: float = 10.0
