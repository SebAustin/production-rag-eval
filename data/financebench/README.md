# FinanceBench data

This directory holds FinanceBench artifacts downloaded at runtime — nothing here
is committed except this README and `.gitkeep`.

## Source

FinanceBench (`PatronusAI/financebench` on HuggingFace) — 150 Q&A pairs over real
10-K filings, with ground-truth answers and evidence passages. Public and
citable (Islam et al., arXiv 2311.11944, 2023).

## How it lands

`make download-data` runs `scripts/download_financebench.py`, which:

1. Loads the `train` split from HuggingFace.
2. Normalizes each row to `{question_id, question, answer, evidence,
   source_filename, page_number}`.
3. Splits deterministically by index: the first 120 rows ->
   `data/calibration/calib_split.jsonl`, the rest ->
   `data/calibration/test_split.jsonl`.

The calibration split fits the conformal threshold; the test split is the
held-out eval set. The split is deterministic so eval numbers are reproducible.
