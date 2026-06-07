"""Conformal calibrator: fit the abstention threshold tau. (Impl: Prompt 4.)

Signatures only for now (kickoff scaffold). The threshold is the conformal
quantile at level ``ceil((n + 1)(1 - alpha)) / n`` of the calibration
nonconformity scores; abstain when a query's score exceeds tau.
"""

from __future__ import annotations

from pathlib import Path


class ConformalCalibrator:
    """Fit, persist, and apply the conformal abstention threshold."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.threshold: float | None = None

    def fit(self, nonconformity_scores: list[float]) -> float:
        """Compute tau at the conformal quantile of calibration scores."""
        raise NotImplementedError  # Prompt 4

    def should_abstain(self, nonconformity_score: float) -> bool:
        """Return True if ``score > tau``. Raises if not yet fitted."""
        raise NotImplementedError  # Prompt 4

    def save(self, path: Path) -> None:
        """Persist ``{threshold, alpha}`` as JSON."""
        raise NotImplementedError  # Prompt 4

    def load(self, path: Path) -> None:
        """Load ``{threshold, alpha}`` from JSON."""
        raise NotImplementedError  # Prompt 4
