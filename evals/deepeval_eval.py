"""DeepEval G-Eval with a financial-accuracy rubric. (Impl: Prompt 6.)

Rubric: "Does the answer correctly state the financial metric, company name,
fiscal year, and unit (dollars, percentage, etc.) as evidenced by the source
passages?" Returns ``None`` until implemented (Prompt 6).
"""

from __future__ import annotations

from rag_eval.logging import get_logger

log = get_logger(__name__)

FINANCIAL_CRITERIA = (
    "Does the answer correctly state the financial metric, company name, fiscal "
    "year, and unit (dollars, percentage, etc.) as evidenced by the source passages?"
)


def score_deepeval(question: str, answer: str, contexts: list[str]) -> float | None:
    """Return the DeepEval G-Eval score in [0, 1], or None (pending)."""
    log.warning("deepeval_eval_pending", question=question[:60])
    return None
