"""Tests for CohereReranker. Implemented in Prompt 3."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="CohereReranker implemented in Prompt 3")


def test_rerank_returns_top_n_scored() -> None:
    # Mock Cohere via respx (see fixtures/sample_cohere_rerank.json); assert
    # top_n results with scores in [0.0, 1.0].
    raise NotImplementedError
