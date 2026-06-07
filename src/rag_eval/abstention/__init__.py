"""Conformal abstention (Yadkori et al., arXiv 2405.01563).

Scorer + calibrator + predictor. Full implementation lands in Prompt 4; the
scorer is pure and implemented now.
"""

from __future__ import annotations

from rag_eval.abstention.scorer import compute_nonconformity

__all__ = ["compute_nonconformity"]
