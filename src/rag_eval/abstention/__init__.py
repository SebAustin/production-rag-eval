"""Conformal abstention (Yadkori et al., arXiv 2405.01563).

Nonconformity scorer + conformal calibrator + inference-time predictor.
"""

from __future__ import annotations

from rag_eval.abstention.calibration import ConformalCalibrator
from rag_eval.abstention.predictor import ConformalPredictor
from rag_eval.abstention.scorer import compute_nonconformity

__all__ = ["ConformalCalibrator", "ConformalPredictor", "compute_nonconformity"]
