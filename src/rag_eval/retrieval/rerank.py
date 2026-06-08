"""Cohere Rerank 3.5 over the fused candidate set.

Reranks on the *contextualized* chunk text (the same text that was indexed) and
returns ``(chunk_id, relevance_score)``. Retries with exponential backoff on HTTP
429 (rate limit), up to ``max_retries`` times.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import cohere

from rag_eval.logging import get_logger

if TYPE_CHECKING:
    from rag_eval.generation.citations import Chunk

log = get_logger(__name__)

_RATE_LIMIT_STATUS = 429


def _is_rate_limit(exc: Exception) -> bool:
    """True if ``exc`` looks like an HTTP 429 from the Cohere SDK."""
    return getattr(exc, "status_code", None) == _RATE_LIMIT_STATUS


class CohereReranker:
    """Reranks fused candidates with Cohere ``rerank-v3.5``."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        client: cohere.AsyncClientV2 | None = None,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
    ) -> None:
        self._model = model
        self._client = client or cohere.AsyncClientV2(api_key=api_key)
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    async def rerank(
        self,
        query: str,
        chunk_ids: list[str],
        chunks_map: dict[str, Chunk],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Return up to ``top_n`` ``(chunk_id, relevance_score)`` ordered by score."""
        if not chunk_ids:
            return []
        documents = [chunks_map[cid].contextualized_text for cid in chunk_ids]
        response = await self._rerank_with_backoff(query, documents, top_n)
        results: list[tuple[str, float]] = []
        for item in response.results:
            results.append((chunk_ids[item.index], float(item.relevance_score)))
        results.sort(key=lambda kv: (-kv[1], kv[0]))
        return results

    async def _rerank_with_backoff(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> Any:  # noqa: ANN401 — SDK response type is dynamic
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.rerank(
                    model=self._model,
                    query=query,
                    documents=documents,
                    top_n=min(top_n, len(documents)),
                    return_documents=False,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised below unless retriable
                if not _is_rate_limit(exc) or attempt == self._max_retries:
                    raise
                last_exc = exc
                delay = self._backoff_base_s * (2**attempt)
                log.warning("cohere_rate_limited", attempt=attempt, delay_s=delay)
                await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover — loop always returns or raises
