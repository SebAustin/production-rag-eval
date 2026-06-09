"""Tests for RAGAnswerGenerator (Citations API) and RAGPipeline orchestration.

The Anthropic client and all pipeline components are faked, so these run offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval.config import Settings
from rag_eval.errors import CitationContractError
from rag_eval.generation.answer import RAGAnswerGenerator
from rag_eval.generation.citations import Chunk, CitedAnswer
from rag_eval.pipeline import RAGPipeline


def _chunks_map(n: int) -> dict[str, Chunk]:
    return {
        f"chunk{i}": Chunk(
            chunk_id=f"chunk{i}",
            source_filename=f"DOC{i}.pdf",
            page_number=i,
            text=f"Net sales were ${i}00 million.",
            contextualized_text=f"context {i}. Net sales were ${i}00 million.",
            char_start=0,
            char_end=10,
        )
        for i in range(n)
    }


def _citation(doc_index: int, start: int, end: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="char_location",
        document_index=doc_index,
        start_char_index=start,
        end_char_index=end,
        cited_text=text,
    )


def _text_block(text: str, citations: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text, citations=citations)


class _FakeMessages:
    def __init__(self, content: list[SimpleNamespace], usage: SimpleNamespace) -> None:
        self._content = content
        self._usage = usage
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return SimpleNamespace(content=self._content, usage=self._usage)


class _FakeGenClient:
    def __init__(self, content: list[SimpleNamespace]) -> None:
        self.messages = _FakeMessages(content, SimpleNamespace(input_tokens=1200, output_tokens=80))


# --- generator ---


@pytest.mark.asyncio
async def test_generate_returns_cited_answer() -> None:
    content = [
        _text_block(
            "Apple net sales were $394 billion.",
            [_citation(0, 0, 14, "Net sales were"), _citation(1, 0, 3, "Net")],
        ),
    ]
    gen = RAGAnswerGenerator(_FakeGenClient(content), "sonnet")  # type: ignore[arg-type]
    reranked = [("chunk0", 0.9), ("chunk1", 0.7)]

    answer = await gen.generate("q", reranked, _chunks_map(2), nonconformity_score=0.1)

    assert isinstance(answer, CitedAnswer)
    assert len(answer.citations) == 2
    assert answer.citations[0].source_filename == "DOC0.pdf"
    assert answer.abstained is False
    assert answer.cost_usd > 0
    assert answer.retrieval_scores == [0.9, 0.7]


@pytest.mark.asyncio
async def test_zero_citations_raises_contract_error() -> None:
    content = [_text_block("Apple net sales were $394 billion.", [])]
    gen = RAGAnswerGenerator(_FakeGenClient(content), "sonnet")  # type: ignore[arg-type]
    with pytest.raises(CitationContractError):
        await gen.generate("q", [("chunk0", 0.9)], _chunks_map(1), nonconformity_score=0.1)


@pytest.mark.asyncio
async def test_insufficient_info_answer_does_not_raise() -> None:
    content = [
        _text_block("The provided documents do not contain enough information.", []),
    ]
    gen = RAGAnswerGenerator(_FakeGenClient(content), "sonnet")  # type: ignore[arg-type]
    answer = await gen.generate("q", [("chunk0", 0.9)], _chunks_map(1), nonconformity_score=0.1)
    assert answer.citations == []
    assert "do not contain" in answer.answer_text.lower()


@pytest.mark.asyncio
async def test_documents_use_original_text() -> None:
    content = [_text_block("x", [_citation(0, 0, 3, "Net"), _citation(0, 0, 3, "Net")])]
    client = _FakeGenClient(content)
    gen = RAGAnswerGenerator(client, "sonnet")  # type: ignore[arg-type]
    await gen.generate("q", [("chunk0", 0.9)], _chunks_map(1), nonconformity_score=0.1)

    docs = [
        b
        for b in client.messages.last_kwargs["messages"][0]["content"]
        if b.get("type") == "document"
    ]
    assert docs[0]["source"]["data"] == "Net sales were $000 million."  # original text


# --- pipeline ---


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        anthropic_api_key="a",
        cohere_api_key="c",
        voyage_api_key="v",
    )


class _FakeBM25:
    def retrieve(self, _q: str, top_k: int = 50) -> list[tuple[str, float]]:
        return [("chunk0", 1.0), ("chunk1", 0.5)]


class _FakeDense:
    async def retrieve(self, _q: str, top_k: int = 50) -> list[tuple[str, float]]:
        return [("chunk1", 0.9), ("chunk2", 0.8)]


class _FakeReranker:
    def __init__(self, scores: list[tuple[str, float]]) -> None:
        self._scores = scores

    async def rerank(self, *_args: Any, **_kwargs: Any) -> list[tuple[str, float]]:
        return self._scores


class _FakePredictor:
    def __init__(self, *, abstain: bool, score: float) -> None:
        self._abstain = abstain
        self._score = score

    def predict(self, _scores: list[float]) -> tuple[bool, float]:
        return self._abstain, self._score


class _FakeGenerator:
    def __init__(self) -> None:
        self.called = False

    async def generate(self, question: str, *_args: Any, **_kwargs: Any) -> CitedAnswer:
        self.called = True
        return CitedAnswer(question=question, answer_text="generated", abstained=False)


def _pipeline(*, abstain: bool, generator: _FakeGenerator) -> RAGPipeline:
    return RAGPipeline(
        _settings(),
        _FakeBM25(),  # type: ignore[arg-type]
        _FakeDense(),  # type: ignore[arg-type]
        _FakeReranker([("chunk1", 0.8), ("chunk2", 0.6)]),  # type: ignore[arg-type]
        _FakePredictor(abstain=abstain, score=0.42),  # type: ignore[arg-type]
        generator,  # type: ignore[arg-type]
        _chunks_map(3),
    )


@pytest.mark.asyncio
async def test_pipeline_abstains_when_predictor_abstains() -> None:
    gen = _FakeGenerator()
    pipeline = _pipeline(abstain=True, generator=gen)

    answer = await pipeline.ask("hard question")

    assert answer.abstained is True
    assert answer.abstention_reason == "insufficient_retrieval_confidence"
    assert answer.nonconformity_score == pytest.approx(0.42)
    assert gen.called is False  # generator must NOT be called on abstention


@pytest.mark.asyncio
async def test_pipeline_answers_when_confident() -> None:
    gen = _FakeGenerator()
    pipeline = _pipeline(abstain=False, generator=gen)

    answer = await pipeline.ask("easy question")

    assert answer.abstained is False
    assert answer.answer_text == "generated"
    assert gen.called is True
    assert answer.latency_ms >= 0.0
