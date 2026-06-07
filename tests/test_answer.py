"""Tests for RAGAnswerGenerator + RAGPipeline. Implemented in Prompt 5."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Generation + pipeline implemented in Prompt 5")


def test_generate_returns_cited_answer() -> None:
    # Mock Anthropic with a fixture carrying >=2 char_location citations.
    raise NotImplementedError


def test_zero_citations_raises_contract_error() -> None:
    raise NotImplementedError


def test_pipeline_abstains_when_predictor_abstains() -> None:
    raise NotImplementedError
