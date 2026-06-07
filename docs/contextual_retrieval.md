# Contextual Retrieval

Implements Anthropic's Contextual Retrieval methodology
([anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval),
Nov 2024).

## The problem it solves

Chunking destroys context. A passage like *"The amount rose 8% to $394,328
million as of that date"* is unembeddable on its own — "the amount", "that date",
and the company are all defined elsewhere in the filing. Standard chunking
indexes that passage with no idea what it refers to, so neither BM25 nor dense
retrieval can match it to *"What was Apple's FY2022 net sales?"*.

## The method

Before indexing, each chunk is passed to Claude Haiku **together with its full
document**, with the instruction to write a 1–2 sentence description situating
the chunk. That prefix is **prepended to the chunk body** to form
`contextualized_text`:

```
This chunk discusses Apple's net sales for fiscal year 2022, reporting $394,328
million in total revenue.

Net sales were $394,328 million for fiscal 2022, an increase of 8%...
```

Both the dense embedding **and** the BM25 index are built over
`contextualized_text`. The original `text` is preserved untouched — it is what
the Citations API points at, so citation offsets remain valid against the source.

### Why the full document, not neighboring chunks
Adjacent chunks often lack the defining context (the metric name, the fiscal
year, the company) that may appear pages earlier. Passing the whole document
(truncated to 8,000 chars here, given FinanceBench evidence passages are short)
lets Haiku resolve those references. For very large filings this truncation is a
known simplification; a production system would summarize or window the document.

## Caching

Re-contextualizing on every index rebuild would be wasteful and expensive. Each
prefix is cached in SQLite keyed by `sha256(chunk_id + text[:100])`, so a rebuild
only pays for new or changed chunks. The cache lives at
`.contextual_cache.sqlite` (gitignored).

## Concurrency

`process_batch` runs chunks through an `asyncio.Semaphore` (default 5 concurrent
Haiku calls) and gathers results, keeping throughput up without tripping rate
limits.

## Cost

FinanceBench's calibration split is ~120 short evidence passages. At Claude Haiku
pricing with ≤8K-char documents and ≤128 output tokens per call, contextualizing
the full set is on the order of **~$0.80** — a one-time index-build cost,
amortized away by the cache on subsequent rebuilds.

## Measured effect

The ablation (`make ablation`, Prompt 7) isolates the Contextual Retrieval
contribution by running the full pipeline with and without the prefix. Numbers
populate `docs/ablation_results.md` and the README ablation table once run.
