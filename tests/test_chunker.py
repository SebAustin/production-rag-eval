"""Tests for FinancialChunker."""

from __future__ import annotations

from rag_eval.ingestion.chunker import FinancialChunker


def test_chunk_offsets_reproduce_text(sample_10k_text: str) -> None:
    chunker = FinancialChunker(chunk_size=256, overlap=32)
    chunks = chunker.chunk_document(sample_10k_text, "AAPL_FY2022_10K.pdf", page_number=28)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        # The recorded offsets must point back at the chunk body.
        assert sample_10k_text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_metadata_propagates(sample_10k_text: str) -> None:
    chunker = FinancialChunker(chunk_size=256, overlap=32)
    chunks = chunker.chunk_document(sample_10k_text, "AAPL_FY2022_10K.pdf", page_number=28)

    assert all(c.source_filename == "AAPL_FY2022_10K.pdf" for c in chunks)
    assert all(c.page_number == 28 for c in chunks)
    assert all(c.contextualized_text == "" for c in chunks)  # filled later


def test_chunk_ids_are_stable() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunker = FinancialChunker(chunk_size=64, overlap=0)
    first = chunker.chunk_document(text, "doc.pdf")
    second = chunker.chunk_document(text, "doc.pdf")

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(len(c.chunk_id) == 16 for c in first)
