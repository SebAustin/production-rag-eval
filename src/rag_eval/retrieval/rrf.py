"""Reciprocal Rank Fusion. Pure function — no external dependencies."""

from __future__ import annotations


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Each input ranking is a list of ``(item_id, score)`` ordered best-first. The
    original scores are ignored — only ranks matter. For an item at zero-based
    ``rank`` in a list, its contribution is ``1 / (k + rank + 1)``; contributions
    sum across all lists.

    Args:
        rankings: Two or more ranked lists to fuse.
        k: RRF damping constant (60 is the canonical default).

    Returns:
        ``(item_id, fused_score)`` sorted by fused score descending. Ties break
        deterministically by item id.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item_id, _score) in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
