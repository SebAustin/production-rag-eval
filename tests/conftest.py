"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval.generation.citations import Chunk

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_10k_text() -> str:
    """A ~300-word excerpt from a public Apple 10-K."""
    return (FIXTURES / "sample_10k_chunk.txt").read_text(encoding="utf-8")


@pytest.fixture
def sample_financebench_row() -> dict[str, Any]:
    """One normalized FinanceBench row."""
    return json.loads((FIXTURES / "sample_financebench_q.json").read_text())


@pytest.fixture
def sample_cohere_rerank() -> dict[str, Any]:
    """A mock Cohere rerank response with 3 results."""
    return json.loads((FIXTURES / "sample_cohere_rerank.json").read_text())


@pytest.fixture
def sample_chunk() -> Chunk:
    """A minimal chunk for unit tests."""
    text = "Net sales were $394,328 million for fiscal 2022."
    return Chunk(
        chunk_id="abc123def4567890",
        source_filename="AAPL_FY2022_10K.pdf",
        page_number=28,
        text=text,
        contextualized_text="",
        char_start=0,
        char_end=len(text),
    )


class _FakeMessages:
    """Stand-in for ``AsyncAnthropic().messages`` returning a fixed text block."""

    def __init__(self, reply: str, calls: list[dict[str, Any]]) -> None:
        self._reply = reply
        self._calls = calls

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self._calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._reply)])


class FakeAnthropic:
    """Duck-typed AsyncAnthropic for offline tests.

    Records every ``messages.create`` call in ``.calls`` and returns ``reply``.
    """

    def __init__(self, reply: str = "This chunk discusses Apple's net sales for FY2022.") -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = _FakeMessages(reply, self.calls)


@pytest.fixture
def fake_anthropic() -> FakeAnthropic:
    """A fresh fake Anthropic client per test."""
    return FakeAnthropic()
