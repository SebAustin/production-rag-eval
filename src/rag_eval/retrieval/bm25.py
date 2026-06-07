"""BM25 lexical retrieval over the pickled index. (Implementation: Prompt 3.)"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.generation.citations import Chunk


class BM25Retriever:
    """Loads a pickled BM25Okapi index and retrieves top-k chunk ids."""

    def __init__(self, bm25: object, chunks: list[Chunk]) -> None:
        self._bm25 = bm25
        self._chunks = chunks

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        """Load a BM25 index + chunk list from a pickle file."""
        raise NotImplementedError  # Prompt 3

    def retrieve(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top-k ``(chunk_id, bm25_score)`` ordered by score descending."""
        raise NotImplementedError  # Prompt 3
