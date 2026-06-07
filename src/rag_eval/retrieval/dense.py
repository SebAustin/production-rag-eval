"""Dense retrieval via Qdrant + voyage-3-large. (Implementation: Prompt 3.)"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class DenseRetriever:
    """Embeds the query with voyage-3-large and searches Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        voyage_api_key: str,
        voyage_model: str = "voyage-3-large",
        embedding_dim: int = 256,
    ) -> None:
        self._client = client
        self._collection = collection
        self._voyage_api_key = voyage_api_key
        self._voyage_model = voyage_model
        self._embedding_dim = embedding_dim

    async def retrieve(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top-k ``(chunk_id, similarity)`` from the dense index."""
        raise NotImplementedError  # Prompt 3
