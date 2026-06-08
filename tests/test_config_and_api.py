"""Tests for Settings, the FastAPI surface, logging, and errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import server
from rag_eval.config import Settings
from rag_eval.errors import CitationContractError, RagEvalError
from rag_eval.logging import configure_logging, get_logger


def test_settings_defaults_and_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.setenv("VOYAGE_API_KEY", "v")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.anthropic_api_key == "a"
    assert settings.chunk_size == 512
    assert settings.embedding_dim == 256
    assert settings.conformal_alpha == pytest.approx(0.10)


def test_health_endpoint() -> None:
    client = TestClient(server.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_endpoint_not_implemented() -> None:
    client = TestClient(server.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 501


def test_eval_summary_404_when_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_RUNS_DIR", tmp_path)
    client = TestClient(server.app)
    assert client.get("/eval-summary").status_code == 404


def test_eval_summary_returns_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "abc123"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"ragas_faithfulness": 0.9}))
    monkeypatch.setattr(server, "_RUNS_DIR", tmp_path)

    client = TestClient(server.app)
    resp = client.get("/eval-summary")
    assert resp.status_code == 200
    assert resp.json()["ragas_faithfulness"] == 0.9


def test_logging_configures_and_binds() -> None:
    configure_logging(json_logs=True)
    log = get_logger("test")
    log.info("hello", k="v")  # must not raise


def test_error_hierarchy() -> None:
    assert issubclass(CitationContractError, RagEvalError)
