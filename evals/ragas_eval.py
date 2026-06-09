"""RAGAS faithfulness / answer-relevancy / context-precision.

Judge LLM: Claude Sonnet 4.5 (temperature 0) via ``LangchainLLMWrapper``.
Answer-relevancy embeddings: voyage-3-large via :class:`VoyageEmbeddings`.

On any failure the scorer returns ``{}`` so the harness records the metrics as
pending rather than crashing or fabricating numbers (eval-honesty contract).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from rag_eval.logging import get_logger

log = get_logger(__name__)

# Map RAGAS output columns -> our metric keys (tolerant of naming variation).
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    "context_precision": (
        "llm_context_precision_without_reference",
        "context_precision",
    ),
}


async def score_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """Return RAGAS metric scores, or ``{}`` if scoring is unavailable/fails."""
    if not answer.strip() or not contexts:
        return {}
    try:
        return await asyncio.to_thread(_run_ragas, question, answer, contexts, ground_truth)
    except Exception:
        log.exception("ragas_failed", question=question[:60])
        return {}


def _run_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    from evals._embeddings import VoyageEmbeddings
    from rag_eval.config import Settings

    settings = Settings()  # type: ignore[call-arg]
    judge = LangchainLLMWrapper(
        ChatAnthropic(
            model_name=settings.sonnet_model,
            temperature=0,
            timeout=60,
            stop=None,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        ),
    )
    embeddings = LangchainEmbeddingsWrapper(VoyageEmbeddings(settings))

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )
    result = evaluate(
        dataset=EvaluationDataset([sample]),
        metrics=[
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=embeddings),
            LLMContextPrecisionWithoutReference(llm=judge),
        ],
        show_progress=False,
    )
    return _extract_scores(result.to_pandas().iloc[0].to_dict())


def _extract_scores(row: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for col in aliases:
            value = row.get(col)
            if isinstance(value, (int | float)) and not math.isnan(float(value)):
                scores[key] = float(value)
                break
    return scores
