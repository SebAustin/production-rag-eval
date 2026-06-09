"""End-to-end RAG orchestration.

Query flow: bm25 -> dense -> rrf_fuse -> cohere_rerank -> conformal_predictor ->
(abstain | generate_with_citations). The retrieval path contains no LLM calls.

Components are dependency-injected; ``RAGPipeline.from_settings`` builds the
default wiring from disk artifacts (BM25 pickle, Qdrant collection, calibrator).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from rag_eval.generation.citations import CitedAnswer
from rag_eval.retrieval.rrf import rrf_fuse

if TYPE_CHECKING:
    from rag_eval.abstention.predictor import ConformalPredictor
    from rag_eval.config import Settings
    from rag_eval.generation.answer import RAGAnswerGenerator
    from rag_eval.generation.citations import Chunk
    from rag_eval.retrieval.bm25 import BM25Retriever
    from rag_eval.retrieval.dense import DenseRetriever
    from rag_eval.retrieval.rerank import CohereReranker

_BM25_PATH = Path("data/index/bm25.pkl")
_CALIBRATOR_PATH = Path("data/calibration/calibrator.json")
_RRF_K = 60
_ABSTENTION_REASON = "insufficient_retrieval_confidence"


class RAGPipeline:
    """Wires together retrieval, conformal abstention, and generation."""

    def __init__(  # noqa: PLR0913 — explicit component injection
        self,
        settings: Settings,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: CohereReranker,
        predictor: ConformalPredictor,
        generator: RAGAnswerGenerator,
        chunks_map: dict[str, Chunk],
    ) -> None:
        self._settings = settings
        self._bm25 = bm25
        self._dense = dense
        self._reranker = reranker
        self._predictor = predictor
        self._generator = generator
        self._chunks_map = chunks_map

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        """Build the default pipeline from on-disk index + calibrator artifacts."""
        from anthropic import AsyncAnthropic
        from qdrant_client import QdrantClient

        from rag_eval.abstention.calibration import ConformalCalibrator
        from rag_eval.abstention.predictor import ConformalPredictor
        from rag_eval.generation.answer import RAGAnswerGenerator
        from rag_eval.retrieval.bm25 import BM25Retriever
        from rag_eval.retrieval.dense import DenseRetriever
        from rag_eval.retrieval.rerank import CohereReranker

        bm25 = BM25Retriever.load(_BM25_PATH)
        chunks_map = {c.chunk_id: c for c in bm25.chunks}
        dense = DenseRetriever(
            QdrantClient(url=settings.qdrant_url),
            settings.qdrant_collection,
            settings.voyage_api_key,
            settings.voyage_model,
            settings.embedding_dim,
        )
        reranker = CohereReranker(settings.cohere_api_key)
        calibrator = ConformalCalibrator()
        calibrator.load(_CALIBRATOR_PATH)
        predictor = ConformalPredictor(calibrator)
        generator = RAGAnswerGenerator(
            AsyncAnthropic(api_key=settings.anthropic_api_key),
            settings.sonnet_model,
        )
        return cls(settings, bm25, dense, reranker, predictor, generator, chunks_map)

    async def ask(self, question: str) -> CitedAnswer:
        """Answer a question end-to-end, or abstain when retrieval is too weak."""
        t0 = time.perf_counter()
        bm25_results = self._bm25.retrieve(question, top_k=self._settings.bm25_top_k)
        dense_results = await self._dense.retrieve(question, top_k=self._settings.dense_top_k)
        fused = rrf_fuse([bm25_results, dense_results], k=_RRF_K)
        candidate_ids = [
            cid for cid, _ in fused[: self._settings.dense_top_k] if cid in self._chunks_map
        ]
        reranked = await self._reranker.rerank(
            question, candidate_ids, self._chunks_map, top_n=self._settings.rerank_top_n
        )

        abstain, nonconformity = self._predictor.predict([score for _, score in reranked])
        if abstain:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return CitedAnswer(
                question=question,
                answer_text="",
                abstained=True,
                abstention_reason=_ABSTENTION_REASON,
                nonconformity_score=nonconformity,
                retrieval_scores=[score for _, score in reranked],
                latency_ms=latency_ms,
            )

        answer = await self._generator.generate(question, reranked, self._chunks_map, nonconformity)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return answer.model_copy(update={"latency_ms": latency_ms})
