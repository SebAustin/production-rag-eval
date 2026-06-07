"""FastAPI app exposing /health, /ask, and /eval-summary.

The pipeline itself is wired in Prompt 5; ``/ask`` returns 501 until then.
``/health`` and ``/eval-summary`` are functional now.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_eval import __version__

app = FastAPI(title="production-rag-eval", version=__version__)

_RUNS_DIR = Path("evals/runs")


class AskRequest(BaseModel):
    """Body for POST /ask."""

    question: str


class HealthResponse(BaseModel):
    """Body for GET /health."""

    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/ask")
async def ask(_request: AskRequest) -> dict[str, Any]:
    """Answer a question end-to-end. Wired in Prompt 5."""
    raise HTTPException(status_code=501, detail="pipeline not yet implemented")


@app.get("/eval-summary")
def eval_summary() -> dict[str, Any]:
    """Return the most recent eval run's summary.json, if any."""
    summaries = sorted(_RUNS_DIR.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise HTTPException(status_code=404, detail="no eval runs found")
    latest = summaries[-1]
    return json.loads(latest.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
