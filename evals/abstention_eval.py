"""Abstention / conformal-coverage metrics over a set of EvalResults.

Coverage requires a per-question correctness label. We don't have a hard
correctness oracle in this harness, so we use a faithfulness-threshold proxy:
an *answered* question counts as correct when ``ragas_faithfulness`` meets
``correct_threshold``. This proxy is documented in docs/eval_methodology.md and
should be revisited when a stronger correctness judge lands (Prompt 6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_eval.generation.citations import EvalResult

_COVERAGE_TOLERANCE = 0.02


def _is_correct(result: EvalResult, correct_threshold: float) -> bool | None:
    """Best-effort correctness label for an answered question (None if unknown)."""
    if result.abstained:
        return None
    if result.ragas_faithfulness is None:
        return None
    return result.ragas_faithfulness >= correct_threshold


def evaluate_calibration_coverage(
    results: list[EvalResult],
    alpha: float = 0.10,
    correct_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute abstention rate, conditional accuracy, and empirical coverage.

    Returns a dict with ``abstention_rate``, ``conditional_accuracy``,
    ``empirical_coverage``, ``target_coverage`` (= 1 - alpha), and
    ``coverage_met`` (empirical within tolerance of target). Fields that cannot
    be computed from the available labels are ``None``.
    """
    n_total = len(results)
    n_abstained = sum(1 for r in results if r.abstained)
    answered = [r for r in results if not r.abstained]

    labels = [_is_correct(r, correct_threshold) for r in answered]
    known = [label for label in labels if label is not None]

    conditional_accuracy: float | None = None
    if known:
        n_correct = sum(1 for label in known if label)
        conditional_accuracy = n_correct / len(known)

    target_coverage = 1.0 - alpha
    empirical_coverage = conditional_accuracy
    coverage_met = (
        None
        if empirical_coverage is None
        else empirical_coverage >= target_coverage - _COVERAGE_TOLERANCE
    )

    return {
        "n_total": n_total,
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / n_total if n_total else 0.0,
        "conditional_accuracy": conditional_accuracy,
        "empirical_coverage": empirical_coverage,
        "target_coverage": target_coverage,
        "coverage_met": coverage_met,
    }
