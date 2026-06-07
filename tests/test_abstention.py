"""Tests for conformal abstention.

The nonconformity scorer is implemented now; calibrator/predictor tests are
marked xfail until Prompt 4.
"""

from __future__ import annotations

import pytest

from rag_eval.abstention.scorer import compute_nonconformity

_PROMPT_4 = "ConformalCalibrator implemented in Prompt 4"


def test_nonconformity_is_one_minus_max() -> None:
    assert compute_nonconformity([0.2, 0.9, 0.5]) == pytest.approx(0.1)


def test_nonconformity_empty_is_maximal() -> None:
    assert compute_nonconformity([]) == 1.0


def test_high_confidence_low_nonconformity() -> None:
    confident = compute_nonconformity([0.95])
    unsure = compute_nonconformity([0.30])
    assert confident < unsure


@pytest.mark.xfail(reason=_PROMPT_4, raises=NotImplementedError, strict=True)
def test_fit_returns_threshold() -> None:
    from rag_eval.abstention.calibration import ConformalCalibrator

    calibrator = ConformalCalibrator(alpha=0.10)
    tau = calibrator.fit([0.2, 0.5, 0.8, 0.3, 0.6])
    assert 0.0 <= tau <= 1.0


@pytest.mark.xfail(reason=_PROMPT_4, raises=NotImplementedError, strict=True)
def test_should_abstain_above_threshold() -> None:
    from rag_eval.abstention.calibration import ConformalCalibrator

    calibrator = ConformalCalibrator()
    calibrator.fit([0.1, 0.2, 0.3, 0.4, 0.5])
    assert calibrator.should_abstain(0.95) is True
