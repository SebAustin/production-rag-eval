"""Inference-time abstention decision wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_eval.abstention.scorer import compute_nonconformity

if TYPE_CHECKING:
    from rag_eval.abstention.calibration import ConformalCalibrator


class ConformalPredictor:
    """Compute the nonconformity score and the abstain/answer decision."""

    def __init__(self, calibrator: ConformalCalibrator) -> None:
        self._calibrator = calibrator

    def predict(self, reranker_scores: list[float]) -> tuple[bool, float]:
        """Return ``(should_abstain, nonconformity_score)`` for a query."""
        score = compute_nonconformity(reranker_scores)
        return self._calibrator.should_abstain(score), score
