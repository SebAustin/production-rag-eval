"""Tests for HybridIndexer with faked Qdrant + voyage clients (offline)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval.config import Settings
from rag_eval.generation.citations import Chunk
from rag_eval.ingestion import indexer as indexer_module
from rag_eval.ingestion.indexer import HybridIndexer, point_id_for


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        anthropic_api_key="x",
        cohere_api_key="x",
        voyage_api_key="x",
    )


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"chunk{i}",
            source_filename="doc.pdf",
            page_number=i,
            text=f"original text {i}",
            contextualized_text=f"context {i}. original text {i}",
            char_start=0,
            char_end=10,
        )
        for i in range(n)
    ]


class _FakeVoyage:
    def embed(self, texts: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(embeddings=[[0.1] * 256 for _ in texts])


class _FakeQdrant:
    def __init__(self) -> None:
        self.recreated: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []

    def recreate_collection(self, **kwargs: Any) -> None:
        self.recreated.append(kwargs)

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)


def test_point_id_is_deterministic_uuid() -> None:
    assert point_id_for("chunk0") == point_id_for("chunk0")
    assert point_id_for("chunk0") != point_id_for("chunk1")


@pytest.mark.asyncio
async def test_build_indexes_dense_and_bm25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(indexer_module, "_BM25_PATH", tmp_path / "bm25.pkl")
    monkeypatch.setattr(indexer_module, "_CHUNKS_PATH", tmp_path / "chunks.jsonl")

    qdrant = _FakeQdrant()
    indexer = HybridIndexer(_settings(), qdrant_client=qdrant, voyage_client=_FakeVoyage())  # type: ignore[arg-type]
    chunks = _chunks(3)

    await indexer.build(chunks)

    assert qdrant.recreated, "collection should be (re)created"
    assert len(qdrant.upserts[0]["points"]) == 3
    assert (tmp_path / "bm25.pkl").exists()
    assert (tmp_path / "chunks.jsonl").read_text().count("\n") == 2  # 3 lines

    # load_bm25 round-trips the just-written pickle.
    _bm25, loaded = indexer.load_bm25(tmp_path / "bm25.pkl")
    assert [c.chunk_id for c in loaded] == ["chunk0", "chunk1", "chunk2"]


@pytest.mark.asyncio
async def test_build_empty_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant = _FakeQdrant()
    indexer = HybridIndexer(_settings(), qdrant_client=qdrant, voyage_client=_FakeVoyage())  # type: ignore[arg-type]
    await indexer.build([])
    assert qdrant.recreated == []
