# Conformal Abstention

Implements the abstention idea from Yadkori et al., *"Mitigating LLM
Hallucinations via Conformal Abstention"* (arXiv 2405.01563, 2024).

> Status: scorer implemented (Prompt 2 scaffold); calibrator + predictor +
> `make calibrate` land in Prompt 4. This document specifies the target.

## Goal

Give the system a *calibrated* "I don't know." Rather than always answering, it
abstains when retrieval confidence is too low to support a grounded answer —
with a statistical bound on the error rate among the questions it does answer.

## Nonconformity score

```python
def compute_nonconformity(reranker_scores: list[float]) -> float:
    return 1.0 - max(reranker_scores) if reranker_scores else 1.0
```

The Cohere reranker emits relevance scores in [0, 1]. `1 - max_score` maps
high-quality retrieval to *low* nonconformity. It is a simple proxy — not a
learned uncertainty model — but max reranker score correlates strongly with
answer correctness on FinanceBench, which is what conformal calibration needs.

Alternatives considered: mean of top-k scores (too forgiving of one strong but
off-topic hit), score margin between rank 1 and 2 (noisy on near-duplicate
passages). Max is the most robust single signal here.

## Calibration

On the 120-question calibration split, compute the nonconformity score for each
question, then take the conformal quantile:

```python
from math import ceil
import numpy as np

def fit(scores: list[float], alpha: float = 0.10) -> float:
    n = len(scores)
    level = min(ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))
```

At inference, **abstain if `score > τ`**.

## Guarantee

Under exchangeability of calibration and test questions, the conformal
construction bounds the miscoverage: among answered questions, the error rate is
controlled at ≈ α (here 0.10), i.e. P(correct | answered) ≳ 1 − α.

## Known limitations

- **Exchangeability.** The guarantee assumes calibration and test draws are
  exchangeable. FinanceBench questions are drawn from 10-K filings without
  temporal ordering, so this approximately holds. In production with query drift
  over time it would not, and the threshold would need periodic re-calibration.
- **Proxy score.** Nonconformity is a retrieval-confidence proxy, not a direct
  correctness estimate; a question with strong-but-wrong retrieval can still slip
  through. Pairing with the generation-side citation contract mitigates this.
- **Coverage vs. usefulness.** A higher abstention rate buys lower error at the
  cost of answering fewer questions. The α knob is the explicit tradeoff.
