"""CLI entrypoint: ``rag-eval ask "<question>"`` / ``python -m rag_eval ask ...``."""

from __future__ import annotations

import asyncio

import typer

from rag_eval.config import Settings
from rag_eval.logging import configure_logging
from rag_eval.pipeline import RAGPipeline

app = typer.Typer(add_completion=False, help="production-rag-eval CLI")


@app.command()
def ask(question: str = typer.Argument(..., help="Natural-language question")) -> None:
    """Run a single question end-to-end and print the CitedAnswer as JSON."""
    configure_logging()
    settings = Settings()  # type: ignore[call-arg]  # values come from env/.env
    pipeline = RAGPipeline(settings)
    answer = asyncio.run(pipeline.ask(question))
    typer.echo(answer.model_dump_json(indent=2))


def main() -> None:
    """Console-script entrypoint declared in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()
