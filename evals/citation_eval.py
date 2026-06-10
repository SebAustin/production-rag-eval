"""Citation coverage and char-offset verification (pure; no live deps)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.generation.citations import Chunk, CitedAnswer

# Marker phrase the generator emits when the passages don't support an answer.
_INSUFFICIENT_INFO_MARKER = "do not contain"


def citation_coverage(answer: CitedAnswer) -> float:
    """Is this answer grounded? 1.0 if so, 0.0 if a claim is left uncited.

    Aligned with the >=1-citation generation contract: a substantive answer must
    carry at least one citation. Abstentions and honest "the documents do not
    contain enough information" responses have no claim to ground, so they score
    1.0 (an honest "I don't know" need not cite).
    """
    if answer.abstained or _INSUFFICIENT_INFO_MARKER in answer.answer_text.lower():
        return 1.0
    return 1.0 if answer.citations else 0.0


def verify_offsets(answer: CitedAnswer, chunks_map: dict[str, Chunk]) -> bool:
    """Verify each citation's ``cited_text`` matches its source span.

    For every :class:`CitedSpan`, look up its source chunk (by ``document_index``
    into the chunk list, or by filename) and confirm that
    ``chunk.text[start:end] == cited_text``. Returns True only if all verify.
    """
    chunks = list(chunks_map.values())
    for span in answer.citations:
        chunk = _resolve_chunk(span.document_index, span.source_filename, chunks)
        if chunk is None:
            return False
        actual = chunk.text[span.start_char_index : span.end_char_index]
        if actual != span.cited_text:
            return False
    return True


def _resolve_chunk(
    document_index: int,
    source_filename: str,
    chunks: list[Chunk],
) -> Chunk | None:
    """Resolve a citation's source chunk by index, falling back to filename."""
    if 0 <= document_index < len(chunks):
        return chunks[document_index]
    for chunk in chunks:
        if chunk.source_filename == source_filename:
            return chunk
    return None
