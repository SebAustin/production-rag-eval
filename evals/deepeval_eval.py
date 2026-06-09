"""DeepEval G-Eval with a financial-accuracy rubric, judged by Claude.

Returns a score in [0, 1], or ``None`` if G-Eval can't run (reported as pending).
"""

from __future__ import annotations

from rag_eval.logging import get_logger

log = get_logger(__name__)

FINANCIAL_CRITERIA = (
    "Does the answer correctly state the financial metric, company name, fiscal "
    "year, and unit (dollars, percentage, etc.) as evidenced by the source passages?"
)


def score_deepeval(question: str, answer: str, contexts: list[str]) -> float | None:
    """Return the DeepEval G-Eval score in [0, 1], or None if unavailable/fails."""
    if not answer.strip() or not contexts:
        return None
    try:
        return _run_deepeval(question, answer, contexts)
    except Exception:
        log.exception("deepeval_failed", question=question[:60])
        return None


def _run_deepeval(question: str, answer: str, contexts: list[str]) -> float | None:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    from evals._claude_deepeval import ClaudeDeepEval
    from rag_eval.config import Settings

    settings = Settings()  # type: ignore[call-arg]
    metric = GEval(
        name="FinancialAccuracy",
        criteria=FINANCIAL_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=ClaudeDeepEval(settings.sonnet_model, settings.anthropic_api_key),
    )
    metric.measure(
        LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts),
    )
    return None if metric.score is None else float(metric.score)
