# FinanceBench Analysis

> Status: scaffold. The category distribution below is approximate (from the
> FinanceBench paper); the abstained-question list populates after the first eval
> run (`make eval`) and must not be hand-filled with fabricated ids.

## Question categories (approximate)

| Category | Share | Example |
|---|---|---|
| Income statement | ~40% | "What was Apple's net sales for FY2022?" |
| Balance sheet | ~25% | "What were total assets at year end?" |
| Cash flow | ~20% | "What was cash from operations?" |
| Ratios / derived | ~15% | "What was the gross margin percentage?" |

## Known hard cases

**Multi-step calculations.** Questions like *"What was the gross margin
percentage?"* (= gross profit ÷ revenue) require two figures that may live in
different passages. Retrieval can surface the components without surfacing the
ratio, and the reranker confidence on the *question as asked* stays low — exactly
the case conformal abstention is designed to catch. See GitHub issue #2 for the
proposed multi-step hard-case category.

## Abstained questions (latest run)

_Populated from `evals/runs/<git_sha>/per_question.jsonl` after `make eval`._
For each abstained question, record: `question_id`, category, nonconformity
score, and τ at the time of the run.

| question_id | category | nonconformity | τ |
|---|---|---|---|
| _pending_ | | | |
