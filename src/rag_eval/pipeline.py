"""End-to-end RAG orchestration. (Impl: Prompt 5.)

Query flow: bm25 -> dense -> rrf_fuse -> cohere_rerank -> conformal_predictor ->
(abstain | generate_with_citations). The retrieval path contains no LLM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_eval.generation.citations import CitedAnswer

if TYPE_CHECKING:
    from rag_eval.config import Settings


class RAGPipeline:
    """Wires together retrieval, abstention, and generation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def ask(self, question: str) -> CitedAnswer:
        """Answer a question end-to-end, or abstain.

        Returns a :class:`CitedAnswer` with ``abstained=True`` when the
        conformal predictor's nonconformity score exceeds the calibrated tau.
        """
        raise NotImplementedError  # Prompt 5
