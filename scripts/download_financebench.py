"""Download FinanceBench from HuggingFace and write deterministic JSONL splits.

Splits are deterministic by index (no shuffle): the first 120 rows become the
calibration split, the remaining rows the test split. Rows are normalized to the
schema the rest of the pipeline expects:
``question_id, question, answer, evidence, source_filename, page_number``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

_CALIB_N = 120
_CALIB_PATH = Path("data/calibration/calib_split.jsonl")
_TEST_PATH = Path("data/calibration/test_split.jsonl")


def _evidence(row: dict[str, Any]) -> str:
    raw = row.get("evidence")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("evidence_text") or item.get("text") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)
    return ""


def _normalize(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "question_id": str(
            row.get("financebench_id") or row.get("question_id") or f"fb_{index:04d}",
        ),
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "evidence": _evidence(row),
        "source_filename": str(row.get("doc_name") or row.get("source_filename") or ""),
        "page_number": int(row.get("page_number", 0) or 0),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def main() -> None:
    """Download, normalize, split, and persist FinanceBench."""
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = [_normalize(dict(r), i) for i, r in enumerate(ds)]
    calib, test = rows[:_CALIB_N], rows[_CALIB_N:]
    _write_jsonl(_CALIB_PATH, calib)
    _write_jsonl(_TEST_PATH, test)
    print(  # noqa: T201
        f"Downloaded {len(rows)} examples. "
        f"Calib: {len(calib)} -> {_CALIB_PATH}. Test: {len(test)} -> {_TEST_PATH}.",
    )


if __name__ == "__main__":
    main()
