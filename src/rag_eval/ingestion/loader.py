"""Load FinanceBench rows and turn evidence passages into indexable chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_eval.generation.citations import Chunk
from rag_eval.ingestion.chunker import FinancialChunker

_EVIDENCE_RECHUNK_THRESHOLD = 512


def load_financebench(split_path: str | Path) -> list[dict[str, Any]]:
    """Load FinanceBench rows from a JSONL file.

    Each row is expected to carry at least: ``question_id``, ``question``,
    ``answer``, ``evidence`` (a source passage), and ``source_filename``.
    """
    path = Path(split_path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _evidence_text(row: dict[str, Any]) -> str:
    """Extract the evidence passage from a row, tolerating schema variation.

    FinanceBench has shipped ``evidence`` as a plain string and as a list of
    ``{"evidence_text": ...}`` objects across releases; handle both.
    """
    raw = row.get("evidence", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("evidence_text") or item.get("text") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(p for p in parts if p)
    return ""


def load_passages(
    rows: list[dict[str, Any]],
    chunker: FinancialChunker | None = None,
) -> list[Chunk]:
    """Convert FinanceBench rows into chunks.

    Each row's evidence passage becomes one chunk, unless it exceeds
    ``_EVIDENCE_RECHUNK_THRESHOLD`` characters, in which case it is re-chunked
    with :class:`FinancialChunker`.
    """
    chunker = chunker or FinancialChunker()
    chunks: list[Chunk] = []
    for row in rows:
        evidence = _evidence_text(row)
        if not evidence.strip():
            continue
        source = str(row.get("source_filename") or row.get("doc_name") or "unknown.pdf")
        page = int(row.get("page_number", 0) or 0)
        if len(evidence) > _EVIDENCE_RECHUNK_THRESHOLD:
            chunks.extend(chunker.chunk_document(evidence, source, page))
        else:
            chunks.append(_single_chunk(evidence, source, page))
    return chunks


def _single_chunk(text: str, source_filename: str, page_number: int) -> Chunk:
    """Wrap a short evidence passage as a single chunk spanning the whole text."""
    digest = hashlib.sha256(f"{source_filename}:0:{text[:64]}".encode()).hexdigest()
    return Chunk(
        chunk_id=digest[:16],
        source_filename=source_filename,
        page_number=page_number,
        text=text,
        contextualized_text="",
        char_start=0,
        char_end=len(text),
    )
