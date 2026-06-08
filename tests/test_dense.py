"""Tests for DenseRetriever (Qdrant + voyage faked, runs offline)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval.retrieval.dense import DenseRetriever


class _FakeVoyage:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def embed(self, texts: list[str], **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"texts": texts, **kwargs})
        return SimpleNamespace(embeddings=[[0.1] * self.dim])


class _FakeQdrant:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return self.points


def _point(chunk_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(id="uuid-x", score=score, payload={"chunk_id": chunk_id})


@pytest.mark.asyncio
async def test_retrieve_returns_chunk_ids_and_scores() -> None:
    qdrant = _FakeQdrant([_point("chunk_a", 0.91), _point("chunk_b", 0.72)])
    retriever = DenseRetriever(
        client=qdrant,  # type: ignore[arg-type]
        collection="financebench_v1",
        voyage_api_key="test",
        voyage_client=_FakeVoyage(),  # type: ignore[arg-type]
    )

    results = await retriever.retrieve("Apple revenue", top_k=10)

    assert results == [("chunk_a", 0.91), ("chunk_b", 0.72)]
    assert all(isinstance(cid, str) and isinstance(s, float) for cid, s in results)


@pytest.mark.asyncio
async def test_query_embedding_uses_query_input_type() -> None:
    voyage = _FakeVoyage()
    retriever = DenseRetriever(
        client=_FakeQdrant([]),  # type: ignore[arg-type]
        collection="c",
        voyage_api_key="test",
        voyage_client=voyage,  # type: ignore[arg-type]
    )

    await retriever.retrieve("a question", top_k=5)

    assert voyage.calls[0]["input_type"] == "query"
    assert voyage.calls[0]["output_dimension"] == 256


@pytest.mark.asyncio
async def test_points_without_chunk_id_are_skipped() -> None:
    bad = SimpleNamespace(id="u", score=0.5, payload={})
    retriever = DenseRetriever(
        client=_FakeQdrant([bad, _point("good", 0.4)]),  # type: ignore[arg-type]
        collection="c",
        voyage_api_key="test",
        voyage_client=_FakeVoyage(),  # type: ignore[arg-type]
    )

    results = await retriever.retrieve("q", top_k=5)
    assert results == [("good", 0.4)]
