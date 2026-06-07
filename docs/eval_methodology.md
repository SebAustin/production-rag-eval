# Eval Methodology

> Status: harness scaffolded (Prompt 1). Citation + abstention metrics are
> implemented; RAGAS / HHEM / DeepEval judges land in Prompt 6. Until then the
> harness records those metrics as `pending` rather than fabricating numbers
> (eval-honesty contract, `.cursorrules`).

## Dataset split

FinanceBench (150 Q&A) is split **deterministically by index**: first 120 rows ->
calibration, remaining 30 -> held-out test. No shuffle at download time; the eval
runner shuffles the test split with a fixed `--seed 42` for reporting order only.

## Metrics

| Metric | Source | Notes |
|---|---|---|
| RAGAS faithfulness | RAGAS 0.2.x, Claude Sonnet 4.5 judge (temp 0) | CI gate ≥ 0.85 |
| RAGAS answer relevancy | RAGAS | |
| RAGAS context precision | RAGAS | |
| Vectara HHEM | `vectara/hallucination_evaluation_model` CrossEncoder | local ~1.3GB; `pending` if absent |
| DeepEval G-Eval | DeepEval, financial rubric | see criteria below |
| Citation coverage | local (`evals/citation_eval.py`) | 1.0 for ≥2 citations; CI gate = 1.00 |
| Abstention rate | local | n_abstained / n_total |
| Conformal coverage | local (`evals/abstention_eval.py`) | empirical vs 1 − α |

### Why Claude Sonnet 4.5 as the RAGAS judge
Consistency with the generation model family and strong instruction-following on
financial text; temperature 0 for reproducibility. The judge is a configurable
wrapper, so swapping it is a one-line change.

### DeepEval G-Eval criteria
> "Does the answer correctly state the financial metric, company name, fiscal
> year, and unit (dollars, percentage, etc.) as evidenced by the source passages?"

### Correctness proxy (interim)
Until a hard correctness oracle lands, `abstention_eval` labels an answered
question "correct" when its RAGAS faithfulness clears a threshold (0.5). This
proxy is documented in `evals/abstention_eval.py` and should be replaced by a
dedicated judge.

## Cost controls
`EVAL_COST_CAP_USD` (default 5.00) hard-stops the run when accumulated cost
reaches the cap. The smoke eval (`make eval-smoke`, n=5) runs with a tighter cap
in CI.

## Reproduce
```bash
make download-data       # deterministic split
make build-index         # contextualize + embed + index
make calibrate           # fit conformal τ on the calibration split
make eval                # full eval, n=30, seed=42
```
Outputs land in `evals/runs/<git_sha>/summary.json` and `per_question.jsonl`.
The README eval table is generated from the latest `summary.json` — never edited
by hand.
