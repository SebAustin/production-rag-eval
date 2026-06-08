"""Dense retrieval via Qdrant + voyage-3-large.

Returns ``(chunk_id, score)`` — chunk ids, not Qdrant point ids — so the result
fuses cleanly with BM25 in RRF. The chunk id is read from the point payload that
:class:`HybridIndexer` stores. The voyage client (sync) and Qdrant client (sync)
are both called off the event loop via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import voyageai

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class DenseRetriever:
    """Embeds the query with voyage-3-large and searches the Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        voyage_api_key: str,
        voyage_model: str = "voyage-3-large",
        embedding_dim: int = 256,
        voyage_client: voyageai.Client | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._voyage_model = voyage_model
        self._embedding_dim = embedding_dim
        self._voyage = voyage_client or voyageai.Client(api_key=voyage_api_key)

    async def retrieve(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, similarity)`` from the dense index."""
        vector = await asyncio.to_thread(self._embed_query, query)
        points = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        results: list[tuple[str, float]] = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id is not None:
                results.append((str(chunk_id), float(point.score)))
        return results

    def _embed_query(self, query: str) -> list[float]:
        result = self._voyage.embed(
            [query],
            model=self._voyage_model,
            input_type="query",
            output_dimension=self._embedding_dim,
        )
        return [float(x) for x in result.embeddings[0]]
