"""Build the hybrid index from the FinanceBench calibration split.

Pipeline: load rows -> evidence passages to chunks -> Contextual Retrieval
prefixes (Claude Haiku, cached) -> embed + index (Qdrant dense + BM25 pickle).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from anthropic import AsyncAnthropic

from rag_eval.config import Settings
from rag_eval.ingestion.contextual import ContextualRetriever
from rag_eval.ingestion.indexer import HybridIndexer
from rag_eval.ingestion.loader import load_financebench, load_passages
from rag_eval.logging import configure_logging, get_logger

log = get_logger(__name__)

_CALIB_PATH = Path("data/calibration/calib_split.jsonl")


def _documents_by_source(rows: list[dict[str, object]]) -> dict[str, str]:
    """Aggregate evidence text per source filename for Contextual Retrieval."""
    docs: dict[str, str] = {}
    for row in rows:
        source = str(row.get("source_filename") or "unknown.pdf")
        evidence = str(row.get("evidence") or "")
        docs[source] = f"{docs.get(source, '')}\n\n{evidence}".strip()
    return docs


async def _run() -> None:
    configure_logging()
    settings = Settings()  # type: ignore[call-arg]  # values from env/.env

    rows = load_financebench(_CALIB_PATH)
    chunks = load_passages(rows)
    docs = _documents_by_source(rows)
    log.info("loaded", rows=len(rows), chunks=len(chunks), docs=len(docs))

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    contextualizer = ContextualRetriever(anthropic_client, model=settings.haiku_model)
    try:
        contextualized = await contextualizer.process_batch(chunks, docs)
    finally:
        contextualizer.close()

    indexer = HybridIndexer(settings)
    await indexer.build(contextualized)
    log.info("index_complete", n_chunks=len(contextualized))


def main() -> None:
    """Entrypoint for ``make build-index``."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
