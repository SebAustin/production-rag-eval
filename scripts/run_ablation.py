"""Retriever ablation across 5 configurations. (Impl: Prompt 7.)

Configurations (same 30-question test split):
  1. BM25 only
  2. Dense only
  3. BM25 + Dense + RRF
  4. + Cohere Rerank
  5. + Contextual Retrieval (full pipeline)

For each, compute RAGAS faithfulness and context recall, then write
docs/ablation_results.md. This produces the ablation table in the README.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval.logging import configure_logging, get_logger

log = get_logger(__name__)

_OUTPUT = Path("docs/ablation_results.md")
CONFIGURATIONS: tuple[str, ...] = (
    "bm25_only",
    "dense_only",
    "rrf",
    "rrf_rerank",
    "full",
)


def main() -> None:
    """Entrypoint for ``make ablation`` (implemented in Prompt 7)."""
    configure_logging()
    log.warning(
        "ablation_not_implemented",
        detail="Depends on the retrieval cascade (Prompt 3) and eval harness (Prompt 6).",
        configurations=list(CONFIGURATIONS),
        output=str(_OUTPUT),
    )
    raise NotImplementedError


if __name__ == "__main__":
    main()
