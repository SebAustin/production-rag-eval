"""RAGAS faithfulness / relevancy / context-precision. (Impl: Prompt 6.)

Will wrap Claude Sonnet 4.5 as the judge LLM via ``LangchainLLMWrapper``. Until
then this returns an empty dict so the harness records the metrics as pending
rather than fabricating numbers (see the eval-honesty contract in .cursorrules).
"""

from __future__ import annotations

from rag_eval.logging import get_logger

log = get_logger(__name__)


async def score_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """Return RAGAS metric scores. Currently pending (Prompt 6)."""
    log.warning("ragas_eval_pending", question=question[:60])
    return {}
