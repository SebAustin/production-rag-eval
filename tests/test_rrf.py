"""Tests for Reciprocal Rank Fusion (pure function — full coverage)."""

from __future__ import annotations

from rag_eval.retrieval.rrf import rrf_fuse


def test_fused_length_is_union() -> None:
    list_a = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    list_b = [("b", 0.9), ("d", 0.8), ("e", 0.7)]
    fused = rrf_fuse([list_a, list_b])
    assert {item_id for item_id, _ in fused} == {"a", "b", "c", "d", "e"}


def test_top_result_appears_in_both_lists() -> None:
    list_a = [("shared", 9.0), ("a2", 8.0), ("a3", 7.0)]
    list_b = [("shared", 0.9), ("b2", 0.8), ("b3", 0.7)]
    fused = rrf_fuse([list_a, list_b])
    assert fused[0][0] == "shared"


def test_scores_sorted_descending() -> None:
    fused = rrf_fuse([[("a", 1.0), ("b", 0.5)], [("b", 1.0), ("a", 0.5)]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_k_changes_weighting() -> None:
    rankings = [[("a", 1.0), ("b", 0.5)]]
    high_k = dict(rrf_fuse(rankings, k=1000))
    low_k = dict(rrf_fuse(rankings, k=1))
    # Lower k amplifies rank differences.
    assert (low_k["a"] - low_k["b"]) > (high_k["a"] - high_k["b"])


def test_empty_input_returns_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_ties_break_by_id() -> None:
    fused = rrf_fuse([[("b", 1.0)], [("a", 1.0)]])
    # Both at rank 0 in separate lists -> equal score -> id order.
    assert [item_id for item_id, _ in fused] == ["a", "b"]
