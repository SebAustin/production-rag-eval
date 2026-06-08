"""Conformal calibrator: fit and apply the abstention threshold tau.

Implements split-conformal calibration (Yadkori et al., arXiv 2405.01563). Given
nonconformity scores on a calibration set, ``fit`` computes the conformal
quantile tau; at inference, abstain when a query's score exceeds tau.
"""

from __future__ import annotations

import json
from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from rag_eval.errors import CalibratorNotFittedError

if TYPE_CHECKING:
    from pathlib import Path


class ConformalCalibrator:
    """Fit, persist, and apply the conformal abstention threshold."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.threshold: float | None = None

    def fit(self, nonconformity_scores: list[float]) -> float:
        """Compute tau at the conformal quantile of calibration scores.

        Uses the finite-sample-valid level ``ceil((n + 1)(1 - alpha)) / n``
        (clipped to 1.0), which gives marginal coverage >= 1 - alpha.
        """
        if not nonconformity_scores:
            msg = "cannot fit on an empty calibration set"
            raise ValueError(msg)
        n = len(nonconformity_scores)
        level = min(ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.threshold = float(np.quantile(nonconformity_scores, level))
        return self.threshold

    def should_abstain(self, nonconformity_score: float) -> bool:
        """Return True if ``score > tau``. Raises if not yet fitted/loaded."""
        if self.threshold is None:
            raise CalibratorNotFittedError
        return nonconformity_score > self.threshold

    def save(self, path: Path) -> None:
        """Persist ``{threshold, alpha}`` as JSON."""
        if self.threshold is None:
            raise CalibratorNotFittedError
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"threshold": self.threshold, "alpha": self.alpha}, indent=2),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        """Load ``{threshold, alpha}`` from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.threshold = float(data["threshold"])
        self.alpha = float(data["alpha"])
