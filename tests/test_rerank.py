"""Tests for CohereReranker (Cohere client faked, runs offline)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval.generation.citations import Chunk
from rag_eval.retrieval.rerank import CohereReranker


def _chunks_map(n: int) -> dict[str, Chunk]:
    out: dict[str, Chunk] = {}
    for i in range(n):
        text = f"passage number {i} about Apple revenue"
        out[f"chunk{i}"] = Chunk(
            chunk_id=f"chunk{i}",
            source_filename="doc.pdf",
            page_number=i,
            text=text,
            contextualized_text=text,
            char_start=0,
            char_end=len(text),
        )
    return out


class _FakeCohere:
    """Returns a fixed rerank result, optionally failing first with 429s."""

    def __init__(self, results: list[SimpleNamespace], fail_times: int = 0) -> None:
        self._results = results
        self._fail_times = fail_times
        self.call_count = 0

    async def rerank(self, **_kwargs: Any) -> SimpleNamespace:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            exc = RuntimeError("rate limited")
            exc.status_code = 429  # type: ignore[attr-defined]
            raise exc
        return SimpleNamespace(results=self._results)


@pytest.mark.asyncio
async def test_rerank_maps_indices_to_chunk_ids() -> None:
    results = [
        SimpleNamespace(index=2, relevance_score=0.93),
        SimpleNamespace(index=0, relevance_score=0.61),
        SimpleNamespace(index=1, relevance_score=0.18),
    ]
    reranker = CohereReranker("test", client=_FakeCohere(results))  # type: ignore[arg-type]
    out = await reranker.rerank("q", ["chunk0", "chunk1", "chunk2"], _chunks_map(3), top_n=3)

    assert out[0] == ("chunk2", 0.93)
    assert [cid for cid, _ in out] == ["chunk2", "chunk0", "chunk1"]
    assert all(0.0 <= s <= 1.0 for _, s in out)


@pytest.mark.asyncio
async def test_rerank_empty_returns_empty() -> None:
    reranker = CohereReranker("test", client=_FakeCohere([]))  # type: ignore[arg-type]
    assert await reranker.rerank("q", [], {}, top_n=10) == []


@pytest.mark.asyncio
async def test_rerank_retries_on_429_then_succeeds() -> None:
    results = [SimpleNamespace(index=0, relevance_score=0.8)]
    fake = _FakeCohere(results, fail_times=2)
    reranker = CohereReranker("test", client=fake, backoff_base_s=0.0)  # type: ignore[arg-type]

    out = await reranker.rerank("q", ["chunk0"], _chunks_map(1), top_n=1)

    assert out == [("chunk0", 0.8)]
    assert fake.call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_rerank_gives_up_after_max_retries() -> None:
    fake = _FakeCohere([], fail_times=99)
    reranker = CohereReranker(
        "test",
        client=fake,  # type: ignore[arg-type]
        max_retries=2,
        backoff_base_s=0.0,
    )
    with pytest.raises(RuntimeError):
        await reranker.rerank("q", ["chunk0"], _chunks_map(1), top_n=1)
    assert fake.call_count == 3  # initial + 2 retries
