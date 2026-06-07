"""Inference-time abstention decision wrapper. (Impl: Prompt 4.)"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.abstention.calibration import ConformalCalibrator


class ConformalPredictor:
    """Compute the nonconformity score and the abstain/answer decision."""

    def __init__(self, calibrator: ConformalCalibrator) -> None:
        self._calibrator = calibrator

    def predict(self, reranker_scores: list[float]) -> tuple[bool, float]:
        """Return ``(should_abstain, nonconformity_score)``."""
        raise NotImplementedError  # Prompt 4
