# Ablation results

> Not yet generated on this checkout. Run `make ablation` (after `make
> build-index`) to populate this file. The table below shows the configurations
> measured; values fill in from the run.

Retriever ablation on the FinanceBench test split. Metric is gold-evidence
retrieval quality (hit@10 / recall@10 / MRR) — no generation or LLM judge.

| Config | hit@10 | recall@10 | MRR |
|---|---|---|---|
| BM25 only | _pending_ | _pending_ | _pending_ |
| Dense only (voyage-3-large) | _pending_ | _pending_ | _pending_ |
| BM25 + Dense + RRF | _pending_ | _pending_ | _pending_ |
| + Cohere Rerank 3.5 (full cascade) | _pending_ | _pending_ | _pending_ |

> All configs run on the contextualized index, so Contextual Retrieval is already
> baked in. A true +Contextual-Retrieval ablation needs a parallel index over raw
> chunk text (see `scripts/run_ablation.py`).
