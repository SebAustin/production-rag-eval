"""Vectara HHEM-2.1-Open hallucination score.

Loads ``vectara/hallucination_evaluation_model`` (a ~1.3GB local model) and
scores each (context, answer) pair: the model returns the probability the answer
is *consistent* with the context (higher = less hallucination). We report the
mean over contexts.

If the model can't be loaded (not downloaded / offline / no transformers), the
scorer returns ``None`` and the harness reports HHEM as pending.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from rag_eval.logging import get_logger

log = get_logger(__name__)

_MODEL_ID = "vectara/hallucination_evaluation_model"


@cache
def _load_model() -> Any | None:  # noqa: ANN401 — transformers model type
    try:
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001 — optional dependency / large download
        log.warning("hhem_model_unavailable", model=_MODEL_ID)
        return None


def score_hhem(answer_text: str, contexts: list[str]) -> float | None:
    """Return the mean HHEM consistency score in [0, 1], or None if unavailable."""
    if not answer_text.strip() or not contexts:
        return None
    model = _load_model()
    if model is None:
        return None
    try:
        # HHEM expects (premise, hypothesis) = (source_context, generated_answer).
        pairs = [(context, answer_text) for context in contexts]
        scores = model.predict(pairs)
        values = [float(s) for s in scores]
        return sum(values) / len(values) if values else None
    except Exception:
        log.exception("hhem_failed")
        return None
