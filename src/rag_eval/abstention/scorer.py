"""Nonconformity scoring for conformal abstention.

The nonconformity score is ``1 - max(reranker_scores)``: high reranker
confidence -> low nonconformity -> answer; low confidence -> high
nonconformity -> abstain. An empty score list is maximally nonconforming.
"""

from __future__ import annotations


def compute_nonconformity(reranker_scores: list[float]) -> float:
    """Return ``1 - max(reranker_scores)``, or ``1.0`` if the list is empty."""
    return 1.0 - max(reranker_scores) if reranker_scores else 1.0
