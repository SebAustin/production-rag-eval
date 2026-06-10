# Architecture

`production-rag-eval` is a hybrid RAG pipeline with a conformal abstention gate
and a triple-eval harness, benchmarked on FinanceBench.

## Pipeline stages

```
ingest -> chunk -> contextualize -> index
                                       |
query --> bm25_retrieve ┐              |
      --> dense_retrieve ┴-> rrf_fuse -> cohere_rerank -> conformal_abstention
                                                              |          |
                                                          abstain   generate_with_citations
                                                                          |
                                                                       evaluate
```

### Ingestion (`src/rag_eval/ingestion/`)
- **loader** — reads FinanceBench JSONL; each evidence passage becomes a chunk
  (re-chunked if > 512 chars).
- **chunker** — `RecursiveCharacterTextSplitter`, 512-token target / 64 overlap,
  recording char offsets so citations can recover source spans.
- **contextual** — Anthropic Contextual Retrieval: a Claude Haiku prefix per
  chunk, prepended before embedding/BM25 indexing. SQLite-cached. See
  [contextual_retrieval.md](contextual_retrieval.md).
- **indexer** — voyage-3-large (dim=256, Matryoshka) into Qdrant (cosine,
  on-disk) + a pickled BM25Okapi index over the same contextualized text.

### Retrieval (`src/rag_eval/retrieval/`) — *Prompt 3*
- **bm25 / dense** — top-50 each.
- **rrf** — Reciprocal Rank Fusion, k=60 (pure function, implemented).
- **rerank** — Cohere `rerank-v3.5`, top-10.

### Abstention (`src/rag_eval/abstention/`) — *Prompt 4*
- Nonconformity score `1 - max(rerank_score)` (implemented).
- Conformal threshold τ calibrated at α=0.10. See
  [conformal_abstention.md](conformal_abstention.md).

### Generation (`src/rag_eval/generation/`) — *Prompt 5*
- Claude Sonnet 4.6 + Anthropic Citations API; original chunk text is the
  citation source. ≥1 grounding citation enforced (`CitationContractError`).

### Eval (`evals/`) — *Prompt 6*
- RAGAS, Vectara HHEM, DeepEval G-Eval, citation coverage, conformal coverage.

## Invariants
- No LLM calls in the retrieval path.
- Embeddings/BM25 always run over the *contextualized* text; citations always
  point at the *original* text.
- All external clients are dependency-injected — no mutable module globals.
- `src/` logs via structlog only; no `print`.
