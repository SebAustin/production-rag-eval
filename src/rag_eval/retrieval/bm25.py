"""BM25 lexical retrieval over the pickled index built by :class:`HybridIndexer`.

The pickle holds a ``BM25Okapi`` fitted over the *contextualized* chunk texts and
the ordered chunk list it was built from, so document index ``i`` in the BM25
model maps to ``chunks[i]``. Queries are tokenized with the same tokenizer used
at index time so the term statistics line up.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

from rag_eval.generation.citations import Chunk
from rag_eval.text import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Loads a pickled BM25Okapi index and retrieves top-k chunk ids by score."""

    def __init__(self, bm25: BM25Okapi, chunks: Sequence[Chunk]) -> None:
        self._bm25 = bm25
        self._chunks = list(chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """The ordered chunk list this index was built from."""
        return self._chunks

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        """Load a BM25 index + chunk list from the pickle written at index time."""
        with path.open("rb") as fh:
            payload: dict[str, Any] = pickle.load(fh)  # noqa: S301 — local, trusted artifact
        bm25: BM25Okapi = payload["bm25"]
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return cls(bm25, chunks)

    def retrieve(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, bm25_score)`` ordered by score desc.

        Ties break deterministically by chunk id.
        """
        tokens = tokenize(query)
        if not tokens or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip((c.chunk_id for c in self._chunks), scores, strict=True),
            key=lambda kv: (-float(kv[1]), kv[0]),
        )
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k]]
