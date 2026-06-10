"""Answer generation with the Anthropic Citations API.

Each reranked chunk is passed as a Citations API document using its *original*
``text`` (not the contextualized text) so citation char offsets land on real
source spans. The model is instructed to answer only from the passages and cite
every claim; a non-abstention answer with fewer than two citations violates the
citation contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from rag_eval.errors import CitationContractError
from rag_eval.generation.citations import CitedAnswer, CitedSpan
from rag_eval.logging import get_logger

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from rag_eval.generation.citations import Chunk

log = get_logger(__name__)

_MAX_TOKENS = 1024
# Require at least one grounding citation on a non-abstention answer. (The spec
# floated >=2, but many valid FinanceBench answers cite a single source — a
# calculation, a one-line fact — and a hard >=2 rule discards them as errors.
# Citation *quality* is graded separately by evals.citation_eval.)
_MIN_CITATIONS = 1
_ABSTENTION_MARKER = "do not contain"
# Claude Sonnet 4.5 list pricing (USD per million tokens).
_INPUT_USD_PER_MTOK = 3.0
_OUTPUT_USD_PER_MTOK = 15.0

_SYSTEM = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "provided document passages. Cite the specific passage for every claim. If "
    "the passages do not contain sufficient information to answer, say 'The "
    "provided documents do not contain enough information to answer this "
    "question.' Do not hallucinate figures."
)


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
        documents = [
            self._document_block(chunks_map[chunk_id]) for chunk_id, _score in reranked_chunks
        ]
        content = [*documents, {"type": "text", "text": f"Question: {question}"}]
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            system=_SYSTEM,
            # Runtime-correct Citations API blocks; cast past the SDK's TypedDict union.
            messages=cast("Any", [{"role": "user", "content": content}]),
        )

        answer_text, citations = self._parse(response, reranked_chunks, chunks_map)
        if len(citations) < _MIN_CITATIONS and _ABSTENTION_MARKER not in answer_text.lower():
            msg = f"answer has {len(citations)} citations (< {_MIN_CITATIONS})"
            raise CitationContractError(msg)

        return CitedAnswer(
            question=question,
            answer_text=answer_text,
            citations=citations,
            abstained=False,
            nonconformity_score=nonconformity_score,
            retrieval_scores=[score for _, score in reranked_chunks],
            cost_usd=self._cost(response),
        )

    @staticmethod
    def _document_block(chunk: Chunk) -> dict[str, Any]:
        return {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": f"{chunk.source_filename} p{chunk.page_number}",
            "citations": {"enabled": True},
        }

    @staticmethod
    def _parse(
        response: Any,  # noqa: ANN401 — SDK Message type
        reranked_chunks: list[tuple[str, float]],
        chunks_map: dict[str, Chunk],
    ) -> tuple[str, list[CitedSpan]]:
        texts: list[str] = []
        spans: list[CitedSpan] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
            for citation in getattr(block, "citations", None) or []:
                span = RAGAnswerGenerator._to_span(citation, reranked_chunks, chunks_map)
                if span is not None:
                    spans.append(span)
        return "".join(texts), spans

    @staticmethod
    def _to_span(
        citation: Any,  # noqa: ANN401 — SDK citation type
        reranked_chunks: list[tuple[str, float]],
        chunks_map: dict[str, Chunk],
    ) -> CitedSpan | None:
        doc_idx = getattr(citation, "document_index", None)
        start = getattr(citation, "start_char_index", None)
        end = getattr(citation, "end_char_index", None)
        if doc_idx is None or start is None or end is None:
            return None
        source = "unknown"
        if 0 <= doc_idx < len(reranked_chunks):
            source = chunks_map[reranked_chunks[doc_idx][0]].source_filename
        return CitedSpan(
            document_index=doc_idx,
            start_char_index=start,
            end_char_index=end,
            cited_text=getattr(citation, "cited_text", ""),
            source_filename=source,
        )

    @staticmethod
    def _cost(response: Any) -> float:  # noqa: ANN401 — SDK Message type
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        return in_tok / 1e6 * _INPUT_USD_PER_MTOK + out_tok / 1e6 * _OUTPUT_USD_PER_MTOK
