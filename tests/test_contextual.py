"""Tests for ContextualRetriever (Anthropic Contextual Retrieval).

The Anthropic client is faked (see ``conftest.FakeAnthropic``) so these run
offline and deterministically.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_eval.generation.citations import Chunk
from rag_eval.ingestion.contextual import ContextualRetriever
from tests.conftest import FakeAnthropic


def _chunk(text: str, chunk_id: str = "id0000000000abcd") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_filename="AAPL_FY2022_10K.pdf",
        page_number=1,
        text=text,
        contextualized_text="",
        char_start=0,
        char_end=len(text),
    )


@pytest.mark.asyncio
async def test_add_context_prepends_prefix(
    fake_anthropic: FakeAnthropic,
    tmp_path: Path,
) -> None:
    retriever = ContextualRetriever(
        fake_anthropic,  # type: ignore[arg-type]  # duck-typed test double
        cache_path=tmp_path / "cache.sqlite",
    )
    chunk = _chunk("Net sales were $394,328 million for fiscal 2022." * 4)
    out = await retriever.add_context(chunk, "full document text " * 100)
    retriever.close()

    assert out.contextualized_text.startswith("This chunk discusses Apple")
    assert chunk.text in out.contextualized_text
    assert out.text == chunk.text  # original preserved for citations


@pytest.mark.asyncio
async def test_context_prefix_is_short(
    fake_anthropic: FakeAnthropic,
    tmp_path: Path,
) -> None:
    retriever = ContextualRetriever(fake_anthropic, cache_path=tmp_path / "c.sqlite")  # type: ignore[arg-type]
    out = await retriever.add_context(_chunk("A short chunk."), "doc")
    retriever.close()

    prefix = out.contextualized_text.split("\n\n")[0]
    sentences = [s for s in re.split(r"[.!?]+", prefix) if s.strip()]
    assert len(sentences) <= 3


@pytest.mark.asyncio
async def test_cache_avoids_second_api_call(
    fake_anthropic: FakeAnthropic,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache.sqlite"
    chunk = _chunk("Cacheable chunk text.")

    r1 = ContextualRetriever(fake_anthropic, cache_path=cache)  # type: ignore[arg-type]
    await r1.add_context(chunk, "doc")
    r1.close()
    assert len(fake_anthropic.calls) == 1

    # New retriever, same cache file -> served from cache, no new call.
    r2 = ContextualRetriever(fake_anthropic, cache_path=cache)  # type: ignore[arg-type]
    await r2.add_context(chunk, "doc")
    r2.close()
    assert len(fake_anthropic.calls) == 1


@pytest.mark.asyncio
async def test_process_batch_with_semaphore(
    fake_anthropic: FakeAnthropic,
    tmp_path: Path,
) -> None:
    retriever = ContextualRetriever(fake_anthropic, cache_path=tmp_path / "c.sqlite")  # type: ignore[arg-type]
    chunks = [_chunk(f"chunk number {i}", chunk_id=f"id{i:014d}") for i in range(5)]
    docs = {"AAPL_FY2022_10K.pdf": "the full document"}

    out = await retriever.process_batch(chunks, docs, semaphore_n=2)
    retriever.close()

    assert len(out) == 5
    assert all(c.contextualized_text for c in out)
