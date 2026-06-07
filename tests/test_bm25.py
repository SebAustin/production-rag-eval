"""Tests for BM25Retriever. Implemented in Prompt 3."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="BM25Retriever implemented in Prompt 3")


def test_bm25_retrieve_returns_sorted_scores() -> None:
    # Load a 5-doc fixture index, query "Apple revenue", assert 5 non-zero
    # scores sorted descending.
    raise NotImplementedError
