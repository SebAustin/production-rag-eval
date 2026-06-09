"""Live smoke test for the eval judges (marked ``eval`` — deselected in CI).

Requires real ANTHROPIC_API_KEY / VOYAGE_API_KEY. Run with:
    uv run pytest -m eval -q
"""

from __future__ import annotations

import os

import pytest

from evals.deepeval_eval import score_deepeval
from evals.hhem_eval import score_hhem
from evals.ragas_eval import score_ragas

pytestmark = pytest.mark.eval

_QUESTION = "What was Apple's net sales for fiscal year 2022?"
_ANSWER = "Apple's net sales were $394,328 million for fiscal 2022."
_CONTEXTS = ["Net sales were $394,328 million for fiscal 2022."]
_GROUND_TRUTH = "$394.3 billion"


def _have_keys() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("VOYAGE_API_KEY"))


@pytest.mark.asyncio
async def test_ragas_live_returns_dict() -> None:
    if not _have_keys():
        pytest.skip("requires ANTHROPIC_API_KEY + VOYAGE_API_KEY")
    scores = await score_ragas(_QUESTION, _ANSWER, _CONTEXTS, _GROUND_TRUTH)
    assert isinstance(scores, dict)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_hhem_live_returns_score_or_none() -> None:
    score = score_hhem(_ANSWER, _CONTEXTS)
    assert score is None or 0.0 <= score <= 1.0


def test_deepeval_live_returns_score_or_none() -> None:
    if not _have_keys():
        pytest.skip("requires ANTHROPIC_API_KEY")
    score = score_deepeval(_QUESTION, _ANSWER, _CONTEXTS)
    assert score is None or 0.0 <= score <= 1.0
