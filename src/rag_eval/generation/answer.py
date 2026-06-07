"""Answer generation with the Anthropic Citations API. (Impl: Prompt 5.)

Each reranked chunk is passed as a Citations API document using its *original*
``text`` (not the contextualized text) so citation offsets land on real source
spans. Answers with fewer than two citations violate the citation contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from rag_eval.generation.citations import Chunk, CitedAnswer


class RAGAnswerGenerator:
    """Generate a :class:`CitedAnswer` from reranked chunks via Claude Sonnet."""

    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def generate(
        self,
        question: str,
        reranked_chunks: list[tuple[str, float]],
        chunks_map: dict[str, Chunk],
        nonconformity_score: float,
    ) -> CitedAnswer:
        """Generate a cited answer. Raises CitationContractError if < 2 citations."""
        raise NotImplementedError  # Prompt 5
