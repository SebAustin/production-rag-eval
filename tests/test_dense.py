"""Tests for DenseRetriever. Implemented in Prompt 3."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="DenseRetriever implemented in Prompt 3")


def test_dense_retrieve_returns_id_score_pairs() -> None:
    # Mock Qdrant + voyageai, assert returns list[(str, float)].
    raise NotImplementedError
