"""Retrieval-quality metrics (recall@k, hit@k, reciprocal rank).

Pure functions over ranked id lists and a set of relevant ids. Used by the
ablation to compare retriever configurations without invoking generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def first_relevant_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> int | None:
    """1-based rank of the first relevant id in ``ranked_ids``, or None."""
    for index, item_id in enumerate(ranked_ids):
        if item_id in relevant_ids:
            return index + 1
    return None


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> bool:
    """True if any relevant id appears in the top-k."""
    return bool(set(ranked_ids[:k]) & relevant_ids)


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids retrieved within the top-k (0.0 if none relevant)."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant id, or 0.0 if none retrieved."""
    rank = first_relevant_rank(ranked_ids, relevant_ids)
    return 0.0 if rank is None else 1.0 / rank


def mean(values: list[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
