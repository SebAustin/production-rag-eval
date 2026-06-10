"""Tests for citation coverage and offset verification."""

from __future__ import annotations

from rag_eval.generation.citations import Chunk, CitedAnswer, CitedSpan
from evals.citation_eval import citation_coverage, verify_offsets


def _answer(n_citations: int, *, abstained: bool = False) -> CitedAnswer:
    spans = [
        CitedSpan(
            document_index=0,
            start_char_index=0,
            end_char_index=3,
            cited_text="Net",
            source_filename="a.pdf",
        )
        for _ in range(n_citations)
    ]
    return CitedAnswer(
        question="q",
        answer_text="a",
        citations=spans,
        abstained=abstained,
    )


def test_coverage_full_with_two_citations() -> None:
    assert citation_coverage(_answer(2)) == 1.0


def test_coverage_full_with_one_citation() -> None:
    # >=1 grounding citation satisfies the contract.
    assert citation_coverage(_answer(1)) == 1.0


def test_coverage_zero_with_no_citations() -> None:
    assert citation_coverage(_answer(0)) == 0.0


def test_abstention_scores_full_coverage() -> None:
    assert citation_coverage(_answer(0, abstained=True)) == 1.0


def test_insufficient_info_answer_scores_full_coverage() -> None:
    answer = CitedAnswer(
        question="q",
        answer_text="The provided documents do not contain enough information.",
        citations=[],
        abstained=False,
    )
    assert citation_coverage(answer) == 1.0


def test_verify_offsets_matches_source() -> None:
    chunk = Chunk(
        chunk_id="c0",
        source_filename="a.pdf",
        page_number=0,
        text="Net sales were high.",
        contextualized_text="",
        char_start=0,
        char_end=20,
    )
    answer = CitedAnswer(
        question="q",
        answer_text="a",
        citations=[
            CitedSpan(
                document_index=0,
                start_char_index=0,
                end_char_index=3,
                cited_text="Net",
                source_filename="a.pdf",
            ),
        ],
    )
    assert verify_offsets(answer, {"c0": chunk}) is True


def test_verify_offsets_detects_mismatch() -> None:
    chunk = Chunk(
        chunk_id="c0",
        source_filename="a.pdf",
        page_number=0,
        text="Net sales were high.",
        contextualized_text="",
        char_start=0,
        char_end=20,
    )
    answer = CitedAnswer(
        question="q",
        answer_text="a",
        citations=[
            CitedSpan(
                document_index=0,
                start_char_index=0,
                end_char_index=3,
                cited_text="WRONG",
                source_filename="a.pdf",
            ),
        ],
    )
    assert verify_offsets(answer, {"c0": chunk}) is False
