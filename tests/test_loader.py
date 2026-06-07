"""Tests for the FinanceBench loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_eval.ingestion.loader import load_financebench, load_passages


def test_load_financebench_reads_jsonl(tmp_path: Path) -> None:
    rows = [
        {"question_id": "fb_001", "question": "Q1", "evidence": "E1", "source_filename": "a.pdf"},
        {"question_id": "fb_002", "question": "Q2", "evidence": "E2", "source_filename": "b.pdf"},
    ]
    path = tmp_path / "split.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    loaded = load_financebench(path)
    assert [r["question_id"] for r in loaded] == ["fb_001", "fb_002"]


def test_load_passages_short_evidence_is_single_chunk(
    sample_financebench_row: dict[str, Any],
) -> None:
    chunks = load_passages([sample_financebench_row])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_filename == "AAPL_FY2022_10K.pdf"
    assert chunk.text == sample_financebench_row["evidence"]


def test_load_passages_long_evidence_is_rechunked() -> None:
    long_evidence = "Sentence about revenue. " * 80  # > 512 chars
    row = {"question_id": "x", "evidence": long_evidence, "source_filename": "big.pdf"}
    chunks = load_passages([row])
    assert len(chunks) > 1


def test_load_passages_handles_list_evidence() -> None:
    row = {
        "question_id": "x",
        "evidence": [{"evidence_text": "Part one."}, {"evidence_text": "Part two."}],
        "source_filename": "c.pdf",
    }
    chunks = load_passages([row])
    assert chunks
    assert "Part one." in chunks[0].text


def test_load_passages_skips_empty_evidence() -> None:
    assert load_passages([{"question_id": "x", "evidence": "", "source_filename": "d.pdf"}]) == []
