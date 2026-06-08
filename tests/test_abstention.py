"""Tests for conformal abstention: scorer, calibrator, predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_eval.abstention.calibration import ConformalCalibrator
from rag_eval.abstention.predictor import ConformalPredictor
from rag_eval.abstention.scorer import compute_nonconformity
from rag_eval.errors import CalibratorNotFittedError

# --- scorer ---


def test_nonconformity_is_one_minus_max() -> None:
    assert compute_nonconformity([0.2, 0.9, 0.5]) == pytest.approx(0.1)


def test_nonconformity_empty_is_maximal() -> None:
    assert compute_nonconformity([]) == 1.0


def test_high_confidence_low_nonconformity() -> None:
    assert compute_nonconformity([0.95]) < compute_nonconformity([0.30])


# --- calibrator ---


def test_fit_returns_threshold_in_range() -> None:
    calibrator = ConformalCalibrator(alpha=0.10)
    tau = calibrator.fit([0.2, 0.5, 0.8, 0.3, 0.6])
    assert 0.0 <= tau <= 1.0
    assert calibrator.threshold == tau


def test_fit_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty calibration set"):
        ConformalCalibrator().fit([])


def test_should_abstain_above_threshold() -> None:
    calibrator = ConformalCalibrator()
    calibrator.threshold = 0.65
    assert calibrator.should_abstain(0.95) is True


def test_answer_below_threshold() -> None:
    calibrator = ConformalCalibrator()
    calibrator.threshold = 0.65
    assert calibrator.should_abstain(0.30) is False


def test_should_abstain_unfitted_raises() -> None:
    with pytest.raises(CalibratorNotFittedError):
        ConformalCalibrator().should_abstain(0.5)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    calibrator = ConformalCalibrator(alpha=0.10)
    calibrator.fit([0.1, 0.2, 0.3, 0.4, 0.5])
    path = tmp_path / "calibrator.json"
    calibrator.save(path)

    reloaded = ConformalCalibrator()
    reloaded.load(path)
    assert reloaded.threshold == calibrator.threshold
    assert reloaded.alpha == pytest.approx(0.10)


def test_save_unfitted_raises(tmp_path: Path) -> None:
    with pytest.raises(CalibratorNotFittedError):
        ConformalCalibrator().save(tmp_path / "x.json")


def test_coverage_guarantee() -> None:
    """On exchangeable data, P(score <= tau) >= 1 - alpha (within tolerance)."""
    rng = np.random.default_rng(42)
    calib = rng.uniform(0.0, 1.0, size=200).tolist()
    calibrator = ConformalCalibrator(alpha=0.10)
    calibrator.fit(calib)

    test = rng.uniform(0.0, 1.0, size=1000)
    assert calibrator.threshold is not None
    covered = float(np.mean(test <= calibrator.threshold))
    assert covered >= 0.88  # target 0.90, allow 2pp slack


# --- predictor ---


def test_predictor_returns_decision_and_score() -> None:
    calibrator = ConformalCalibrator()
    calibrator.threshold = 0.5
    predictor = ConformalPredictor(calibrator)

    # max rerank score 0.9 -> nonconformity 0.1 <= 0.5 -> answer
    abstain, score = predictor.predict([0.9, 0.4])
    assert abstain is False
    assert score == pytest.approx(0.1)

    # max rerank score 0.2 -> nonconformity 0.8 > 0.5 -> abstain
    abstain, score = predictor.predict([0.2, 0.1])
    assert abstain is True
    assert score == pytest.approx(0.8)
