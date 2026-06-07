"""Eval harness for production-rag-eval.

Evaluates the RAG pipeline on the FinanceBench test split across RAGAS
faithfulness/relevancy/precision, Vectara HHEM, DeepEval G-Eval, citation
coverage, and conformal abstention coverage.

Usage:
    uv run python -m evals.run_eval --limit 30 --seed 42
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import typer

from evals.abstention_eval import evaluate_calibration_coverage
from evals.cache import SQLiteEvalCache
from evals.citation_eval import citation_coverage
from evals.deepeval_eval import score_deepeval
from evals.hhem_eval import score_hhem
from evals.ragas_eval import score_ragas
from rag_eval.config import Settings
from rag_eval.generation.citations import CitedAnswer, EvalResult
from rag_eval.logging import configure_logging, get_logger
from rag_eval.pipeline import RAGPipeline

log = get_logger(__name__)
app = typer.Typer(add_completion=False)

COST_CAP = float(os.environ.get("EVAL_COST_CAP_USD", "5.00"))
CI_THRESHOLDS: dict[str, float] = {"ragas_faithfulness": 0.85, "citation_coverage": 1.00}


def _git_sha() -> str:
    try:
        return subprocess.check_output(  # noqa: S603, S607
            ["git", "rev-parse", "--short", "HEAD"],
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _load_questions(limit: int, seed: int) -> list[dict[str, Any]]:
    path = Path("data/calibration/test_split.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


async def _run_one(
    pipeline: RAGPipeline,
    q: dict[str, Any],
    cache: SQLiteEvalCache,
) -> EvalResult:
    cache_key = f"{q['question_id']}_{q['question'][:40]}"
    if hit := cache.get(cache_key):
        return EvalResult(**hit)

    t0 = time.perf_counter()
    answer: CitedAnswer | None = None
    try:
        answer = await pipeline.ask(q["question"])
    except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
        log.error("pipeline_error", id=q["question_id"], err=str(exc))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    contexts: list[str] = [c.cited_text for c in (answer.citations if answer else [])]
    scoreable = bool(answer) and not (answer.abstained if answer else True)

    ragas_scores = (
        await score_ragas(
            question=q["question"],
            answer=answer.answer_text if answer else "",
            contexts=contexts,
            ground_truth=q.get("answer", ""),
        )
        if scoreable
        else {}
    )
    hhem = (
        score_hhem(answer_text=answer.answer_text if answer else "", contexts=contexts)
        if scoreable
        else None
    )
    deepeval = (
        score_deepeval(
            question=q["question"],
            answer=answer.answer_text if answer else "",
            contexts=contexts,
        )
        if scoreable
        else None
    )
    cite_cov = citation_coverage(answer) if answer else 0.0

    result = EvalResult(
        question_id=q["question_id"],
        question=q["question"],
        ground_truth=q.get("answer", ""),
        answer=answer,
        ragas_faithfulness=ragas_scores.get("faithfulness"),
        ragas_relevancy=ragas_scores.get("answer_relevancy"),
        ragas_context_precision=ragas_scores.get("context_precision"),
        hhem_score=hhem,
        deepeval_score=deepeval,
        citation_coverage=cite_cov,
        abstained=answer.abstained if answer else True,
        latency_ms=latency_ms,
        cost_usd=answer.cost_usd if answer else 0.0,
    )
    cache.put(cache_key, result.model_dump(mode="json"))
    return result


def _safe_mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(int(pct * len(ordered)), len(ordered) - 1)
    return ordered[idx]


def _check_ci_gate(summary: dict[str, Any]) -> bool:
    passed = True
    for metric, threshold in CI_THRESHOLDS.items():
        val = summary.get(metric)
        if val is None:
            log.warning("ci_gate_pending", metric=metric)
            continue
        if val < threshold:
            log.error("ci_gate_failed", metric=metric, value=val, threshold=threshold)
            passed = False
    return passed


def _build_summary(results: list[EvalResult], sha: str, seed: int) -> dict[str, Any]:
    answered = [r for r in results if not r.abstained]
    abstained_n = sum(1 for r in results if r.abstained)
    abstention_meta = evaluate_calibration_coverage(results)
    return {
        "git_sha": sha,
        "seed": seed,
        "n_total": len(results),
        "n_answered": len(answered),
        "n_abstained": abstained_n,
        "abstention_rate": abstained_n / max(len(results), 1),
        "ragas_faithfulness": _safe_mean([r.ragas_faithfulness for r in answered]),
        "ragas_relevancy": _safe_mean([r.ragas_relevancy for r in answered]),
        "ragas_context_precision": _safe_mean(
            [r.ragas_context_precision for r in answered],
        ),
        "hhem_score": _safe_mean([r.hhem_score for r in answered]),
        "deepeval_score": _safe_mean([r.deepeval_score for r in answered]),
        "citation_coverage": _safe_mean([r.citation_coverage for r in answered]),
        "conformal_coverage_met": abstention_meta.get("coverage_met"),
        "p50_latency_ms": _percentile([r.latency_ms for r in results], 0.50),
        "p95_latency_ms": _percentile([r.latency_ms for r in results], 0.95),
        "total_cost_usd": sum(r.cost_usd for r in results),
    }


@app.command()
def main(
    limit: int = typer.Option(0, "--limit"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Run the eval, write summary.json + per_question.jsonl, gate on thresholds."""
    configure_logging()
    sha = _git_sha()
    out = output_dir or (Path("evals/runs") / sha)
    out.mkdir(parents=True, exist_ok=True)

    settings = Settings()  # type: ignore[call-arg]  # values from env/.env
    pipeline = RAGPipeline(settings)
    cache = SQLiteEvalCache(Path(".eval_cache.sqlite"))

    questions = _load_questions(limit, seed)
    results: list[EvalResult] = []
    total_cost = 0.0
    for q in questions:
        if total_cost >= COST_CAP:
            log.warning("cost_cap_hit", cap=COST_CAP, spent=total_cost)
            break
        result = asyncio.run(_run_one(pipeline, q, cache))
        results.append(result)
        total_cost += result.cost_usd
        log.info(
            "done",
            id=result.question_id,
            abstained=result.abstained,
            faithfulness=result.ragas_faithfulness,
            cite_cov=result.citation_coverage,
        )

    summary = _build_summary(results, sha, seed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_question.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in results),
    )
    typer.echo(json.dumps(summary, indent=2))
    raise typer.Exit(0 if _check_ci_gate(summary) else 1)


if __name__ == "__main__":
    app()
