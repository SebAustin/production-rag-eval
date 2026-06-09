"""Unit tests for the eval judge adapters.

The heavy RAGAS / HHEM / DeepEval calls are monkeypatched, so these run offline
and validate the wrapper logic: input guards, passthrough, and graceful
degradation to pending (``{}`` / ``None``) on failure.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from evals import deepeval_eval, hhem_eval, ragas_eval


# --- RAGAS ---


@pytest.mark.asyncio
async def test_ragas_empty_inputs_return_pending() -> None:
    assert await ragas_eval.score_ragas("q", "", ["ctx"], "gt") == {}
    assert await ragas_eval.score_ragas("q", "answer", [], "gt") == {}


@pytest.mark.asyncio
async def test_ragas_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ragas_eval,
        "_run_ragas",
        lambda *_a, **_k: {"faithfulness": 0.9, "answer_relevancy": 0.8},
    )
    scores = await ragas_eval.score_ragas("q", "a", ["ctx"], "gt")
    assert scores["faithfulness"] == 0.9


@pytest.mark.asyncio
async def test_ragas_degrades_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> dict[str, float]:
        raise RuntimeError("ragas down")

    monkeypatch.setattr(ragas_eval, "_run_ragas", _boom)
    assert await ragas_eval.score_ragas("q", "a", ["ctx"], "gt") == {}


def test_ragas_extract_scores_handles_aliases_and_nan() -> None:
    row = {
        "faithfulness": 0.88,
        "response_relevancy": 0.81,
        "llm_context_precision_without_reference": float("nan"),
    }
    scores = ragas_eval._extract_scores(row)
    assert scores["faithfulness"] == 0.88
    assert scores["answer_relevancy"] == 0.81
    assert "context_precision" not in scores  # NaN dropped


# --- HHEM ---


def test_hhem_empty_returns_none() -> None:
    assert hhem_eval.score_hhem("", ["ctx"]) is None
    assert hhem_eval.score_hhem("answer", []) is None


def test_hhem_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hhem_eval, "_load_model", lambda: None)
    assert hhem_eval.score_hhem("answer", ["ctx"]) is None


def test_hhem_mean_over_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(predict=lambda _pairs: [0.8, 0.6])
    monkeypatch.setattr(hhem_eval, "_load_model", lambda: fake)
    assert hhem_eval.score_hhem("answer", ["c1", "c2"]) == pytest.approx(0.7)


def test_hhem_degrades_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_pairs: Any) -> list[float]:
        raise RuntimeError("model crash")

    monkeypatch.setattr(hhem_eval, "_load_model", lambda: SimpleNamespace(predict=_boom))
    assert hhem_eval.score_hhem("answer", ["c1"]) is None


# --- DeepEval ---


def test_deepeval_empty_returns_none() -> None:
    assert deepeval_eval.score_deepeval("q", "", ["ctx"]) is None


def test_deepeval_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deepeval_eval, "_run_deepeval", lambda *_a, **_k: 0.78)
    assert deepeval_eval.score_deepeval("q", "a", ["ctx"]) == 0.78


def test_deepeval_degrades_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> float:
        raise RuntimeError("geval down")

    monkeypatch.setattr(deepeval_eval, "_run_deepeval", _boom)
    assert deepeval_eval.score_deepeval("q", "a", ["ctx"]) is None
