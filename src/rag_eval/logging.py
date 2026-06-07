"""structlog configuration. ``src/`` uses structlog only — never ``print``."""

from __future__ import annotations

import logging

import structlog


def configure_logging(*, json_logs: bool = False, level: int = logging.INFO) -> None:
    """Configure structlog once at process start.

    Args:
        json_logs: Emit JSON lines (for prod/CI) instead of the console renderer.
        level: Standard-library log level threshold.
    """
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
