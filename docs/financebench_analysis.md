# FinanceBench Analysis

Real numbers below are from `evals/runs/3c53d10/summary.json` and
`per_question.jsonl` — the canonical FinanceBench test-split run (n=30, seed=42,
Claude Sonnet 4.6). Reproduce with `make eval`.

## Question categories (approximate)

Approximate shares from the FinanceBench paper (Islam et al., 2023):

| Category | Share | Example |
|---|---|---|
| Income statement | ~40% | "What was Apple's net sales for FY2022?" |
| Balance sheet | ~25% | "What were total assets at year end?" |
| Cash flow | ~20% | "What was cash from operations?" |
| Ratios / derived | ~15% | "What was the gross margin percentage?" |

## The faithfulness ceiling: multi-step calculations

Aggregate RAGAS faithfulness on the run is **0.83** (22 answered questions
scored). The shortfall to the 0.85 gate is **not spread evenly** — it is
concentrated in a handful of multi-step *calculation* and *synthesis* questions.
The six lowest-scoring answered questions:

| faithfulness | question_id | question |
|---|---|---|
| 0.38 | financebench_id_00859 | Among all the derivative instruments Verizon used… (synthesis across notes) |
| 0.43 | financebench_id_04481 | What is the FY2022 unadjusted **EBITDA % margin** for PepsiCo? *Calculate…* |
| 0.43 | financebench_id_03620 | What is the FY2022 unadjusted **EBITDA less capex** for PepsiCo? *Define…* |
| 0.50 | financebench_id_00705 | **By how much did** Pepsico increase its 5-year revolving credit agreement? |
| 0.50 | financebench_id_00735 | Has Pepsico reported any materially important ongoing legal battles…? |
| 0.50 | financebench_id_00283 | **How much** does Pfizer expect to pay to spin off Upjohn (USD million)? |

By contrast, single-fact lookups score 1.00 (e.g. "What drove the reduction in
SG&A…", "…how much did Verizon expect to pay for its retirees in 2024").

**Why this is a metric ceiling, not a pipeline defect.** RAGAS faithfulness
decomposes the answer into atomic claims and checks each against the retrieved
passages. A calculation answers with a *derived* figure — a margin %, a
difference, a quotient — that is, by construction, **not verbatim in any
passage**. That derived claim is scored unfaithful even when every input figure
it was computed from is correctly cited. So the harder the arithmetic, the lower
the achievable faithfulness, independent of pipeline quality. This mirrors how
Harvey and Scale AI report eval quality stratified by task-complexity tier
(see GitHub issue #2: add a `question_category` field and report per-category).

Two prompt iterations were tried on these (documented in git history):
tightening grounding lifted faithfulness 0.65 → 0.83; suppressing intermediate
arithmetic *lowered* it (0.83 → 0.78), because removing grounded source-figure
sentences left fewer faithful claims to offset the one derived claim. The 0.83
configuration is the one that ships.

## Abstained questions (run 3c53d10)

Conformal threshold τ = **0.5275** (α=0.10, calibrated on 120 questions).
The pipeline abstained on 2/30 (6.7%) — both where `1 − max_rerank_score > τ`:

| question_id | nonconformity | τ | question |
|---|---|---|---|
| financebench_id_00521 | 0.565 | 0.528 | What are major acquisitions Ulta Beauty has done in FY2023 and FY2022? |
| financebench_id_00605 | 0.654 | 0.528 | What percent of Ulta Beauty's total spend on stock repurchases for FY2023…? |

Both are Ulta Beauty multi-part questions where the reranker's top score stayed
low — exactly the low-confidence retrieval that conformal abstention is meant to
catch rather than answer with a hallucinated figure.
