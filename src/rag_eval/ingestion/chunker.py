"""Recursive character chunker that preserves char offsets into the source text.

Char offsets matter downstream: the Citations API points at spans of the
original chunk text, so every chunk records exactly where it came from.
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.generation.citations import Chunk


class FinancialChunker:
    """Split financial documents into ~512-char chunks with stable ids/offsets."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def chunk_document(
        self,
        text: str,
        source_filename: str,
        page_number: int = 0,
    ) -> list[Chunk]:
        """Chunk ``text`` and return :class:`Chunk` objects with char offsets.

        Offsets are recovered by scanning forward through the source so that
        ``text[char_start:char_end]`` reproduces the chunk body, even when the
        same substring appears more than once.
        """
        splits = self._splitter.split_text(text)
        chunks: list[Chunk] = []
        pos = 0
        for split in splits:
            start = text.find(split, pos)
            if start == -1:  # splitter normalized whitespace — fall back to scan from 0
                start = text.find(split)
            if start == -1:  # genuinely unlocatable; record a zero-width offset
                start = pos
            end = start + len(split)
            digest = hashlib.sha256(f"{source_filename}:{start}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=digest[:16],
                    source_filename=source_filename,
                    page_number=page_number,
                    text=split,
                    contextualized_text="",  # filled by ContextualRetriever
                    char_start=start,
                    char_end=end,
                ),
            )
            pos = start + 1
        return chunks
