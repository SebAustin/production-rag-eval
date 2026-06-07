"""Pydantic data models for the RAG pipeline.

These are the typed contracts that flow between pipeline stages: a :class:`Chunk`
is what the ingestion side produces, a :class:`CitedAnswer` is what the
generation side produces, and :class:`EvalResult` is what the eval harness emits
per question.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC now (``datetime.utcnow`` is deprecated in 3.12)."""
    return datetime.now(tz=UTC)


class CitedSpan(BaseModel):
    """A single citation pointing at a span of a source document.

    Char offsets are relative to the *original* chunk text (``Chunk.text``),
    which is what the Anthropic Citations API points at — not the
    contextualized text used for retrieval.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_index: int
    start_char_index: int
    end_char_index: int
    cited_text: str
    source_filename: str


class CitedAnswer(BaseModel):
    """The output of the generation stage (or an abstention decision)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    answer_text: str
    citations: list[CitedSpan] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    nonconformity_score: float = 0.0
    retrieval_scores: list[float] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


class Chunk(BaseModel):
    """A unit of indexed text plus its provenance and Contextual Retrieval prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    source_filename: str
    page_number: int
    text: str  # original chunk text — citations point here
    contextualized_text: str  # Contextual Retrieval prefix + text — indexed/embedded
    char_start: int
    char_end: int


class EvalResult(BaseModel):
    """Per-question eval record written to ``evals/runs/<sha>/per_question.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    ground_truth: str
    answer: CitedAnswer | None
    ragas_faithfulness: float | None = None
    ragas_relevancy: float | None = None
    ragas_context_precision: float | None = None
    hhem_score: float | None = None
    deepeval_score: float | None = None
    citation_coverage: float = 0.0
    abstained: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
