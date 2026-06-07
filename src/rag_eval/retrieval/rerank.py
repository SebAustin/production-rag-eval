"""Cohere Rerank 3.5 over fused candidates. (Implementation: Prompt 3.)"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.generation.citations import Chunk


class CohereReranker:
    """Reranks fused candidates with Cohere ``rerank-v3.5``."""

    def __init__(self, api_key: str, model: str = "rerank-v3.5") -> None:
        self._api_key = api_key
        self._model = model

    async def rerank(
        self,
        query: str,
        chunk_ids: list[str],
        chunks_map: dict[str, Chunk],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Return top-n ``(chunk_id, relevance_score)`` ordered by score desc."""
        raise NotImplementedError  # Prompt 3
