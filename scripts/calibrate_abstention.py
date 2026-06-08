"""Fit the conformal abstention threshold on the FinanceBench calibration split.

For each calibration question, run the full retrieval cascade
(bm25 -> dense -> rrf -> cohere rerank), compute the nonconformity score
(1 - max rerank score), fit the conformal threshold tau, and persist it to
data/calibration/calibrator.json.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

from rag_eval.abstention import ConformalCalibrator, compute_nonconformity
from rag_eval.config import Settings
from rag_eval.ingestion.loader import load_financebench
from rag_eval.logging import configure_logging, get_logger
from rag_eval.retrieval import BM25Retriever, CohereReranker, DenseRetriever, rrf_fuse

if TYPE_CHECKING:
    from rag_eval.generation.citations import Chunk

log = get_logger(__name__)

_CALIB_PATH = Path("data/calibration/calib_split.jsonl")
_CALIBRATOR_PATH = Path("data/calibration/calibrator.json")
_BM25_PATH = Path("data/index/bm25.pkl")
_RRF_K = 60


async def _nonconformity(  # noqa: PLR0913 — wires the full retrieval cascade
    question: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    reranker: CohereReranker,
    chunks_map: dict[str, Chunk],
    settings: Settings,
) -> float:
    bm25_results = bm25.retrieve(question, top_k=settings.bm25_top_k)
    dense_results = await dense.retrieve(question, top_k=settings.dense_top_k)
    fused = rrf_fuse([bm25_results, dense_results], k=_RRF_K)
    candidate_ids = [cid for cid, _ in fused[: settings.dense_top_k] if cid in chunks_map]
    reranked = await reranker.rerank(
        question, candidate_ids, chunks_map, top_n=settings.rerank_top_n
    )
    return compute_nonconformity([score for _, score in reranked])


async def _run() -> None:
    configure_logging()
    settings = Settings()  # values from env/.env

    rows = load_financebench(_CALIB_PATH)
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

    scores: list[float] = []
    for i, row in enumerate(rows):
        score = await _nonconformity(row["question"], bm25, dense, reranker, chunks_map, settings)
        scores.append(score)
        log.info("calibrated_question", i=i + 1, of=len(rows), nonconformity=round(score, 4))

    calibrator = ConformalCalibrator(alpha=settings.conformal_alpha)
    tau = calibrator.fit(scores)
    calibrator.save(_CALIBRATOR_PATH)

    in_sample_abstain = sum(1 for s in scores if s > tau) / len(scores)
    print(
        f"Threshold τ = {tau:.4f} at alpha={settings.conformal_alpha} "
        f"on {len(scores)} calibration questions.\n"
        f"In-sample abstention rate at τ: {in_sample_abstain:.1%}. "
        f"Expected abstention rate on future data: ~{settings.conformal_alpha:.0%}.\n"
        f"Saved to {_CALIBRATOR_PATH}.",
    )


def main() -> None:
    """Entrypoint for ``make calibrate``."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
