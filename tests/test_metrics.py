"""Tests for retrieval-quality metrics."""

from __future__ import annotations

import pytest

from rag_eval.retrieval.metrics import (
    first_relevant_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
)

_RANKED = ["a", "b", "c", "d", "e"]


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(_RANKED, {"c"}) == 3
    assert first_relevant_rank(_RANKED, {"a", "e"}) == 1
    assert first_relevant_rank(_RANKED, {"z"}) is None


def test_hit_at_k() -> None:
    assert hit_at_k(_RANKED, {"c"}, 3) is True
    assert hit_at_k(_RANKED, {"c"}, 2) is False
    assert hit_at_k(_RANKED, {"z"}, 5) is False


def test_recall_at_k() -> None:
    assert recall_at_k(_RANKED, {"a", "c"}, 3) == pytest.approx(1.0)
    assert recall_at_k(_RANKED, {"a", "z"}, 3) == pytest.approx(0.5)
    assert recall_at_k(_RANKED, set(), 3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(_RANKED, {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(_RANKED, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(_RANKED, {"z"}) == 0.0


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert mean([]) == 0.0
