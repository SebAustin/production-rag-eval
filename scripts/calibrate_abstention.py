"""Fit the conformal abstention threshold on the calibration split. (Impl: Prompt 4.)

Procedure (to be completed in Prompt 4):
  1. Load the BM25 + Qdrant index.
  2. For each calibration question, run bm25 -> dense -> rrf -> cohere_rerank
     and compute the nonconformity score (1 - max rerank score).
  3. Fit ``ConformalCalibrator`` and save tau to data/calibration/calibrator.json.
  4. Print the threshold and expected abstention rate.

The retrieval cascade it depends on is implemented in Prompt 3 and the
calibrator in Prompt 4; this script wires them together.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval.logging import configure_logging, get_logger

log = get_logger(__name__)

_CALIB_PATH = Path("data/calibration/calib_split.jsonl")
_CALIBRATOR_PATH = Path("data/calibration/calibrator.json")


def main() -> None:
    """Entrypoint for ``make calibrate`` (implemented in Prompt 4)."""
    configure_logging()
    log.warning(
        "calibrate_not_implemented",
        detail="Depends on the retrieval cascade (Prompt 3) and calibrator (Prompt 4).",
        calib_path=str(_CALIB_PATH),
        output=str(_CALIBRATOR_PATH),
    )
    raise NotImplementedError


if __name__ == "__main__":
    main()
