"""Vectara HHEM-2.1-Open hallucination score. (Impl: Prompt 6.)

Will load ``vectara/hallucination_evaluation_model`` via a transformers
CrossEncoder (~1.3GB download). Returns ``None`` when the model is unavailable
so the metric is reported as pending rather than failing the run.
"""

from __future__ import annotations

from rag_eval.logging import get_logger

log = get_logger(__name__)


def score_hhem(answer_text: str, contexts: list[str]) -> float | None:
    """Return the mean HHEM score (higher = less hallucination), or None."""
    log.warning("hhem_eval_pending", n_contexts=len(contexts))
    return None
