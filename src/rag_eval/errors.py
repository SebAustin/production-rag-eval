"""Shared exception types for the rag_eval package."""

from __future__ import annotations


class RagEvalError(Exception):
    """Base class for all rag_eval errors."""


class CitationContractError(RagEvalError):
    """Raised when a generated answer violates the citation contract.

    The contract requires every non-abstention answer to carry at least two
    :class:`~rag_eval.generation.citations.CitedSpan` objects.
    """


class CalibratorNotFittedError(RagEvalError):
    """Raised when the conformal calibrator is used before ``fit``/``load``."""


class CostCapExceededError(RagEvalError):
    """Raised when an operation would exceed the configured API spend cap."""
