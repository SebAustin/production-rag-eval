# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffold: package layout, `pyproject.toml`, tooling config, CI workflows.
- Core pydantic models: `Chunk`, `CitedSpan`, `CitedAnswer`, `EvalResult`.
- `Settings` (pydantic-settings) for all runtime configuration.
- Ingestion pipeline (P2):
  - `FinancialChunker` — recursive 512-token splitter with char offsets.
  - `loader` — FinanceBench JSONL loader; evidence passages -> `Chunk`s.
  - `ContextualRetriever` — Anthropic Contextual Retrieval prefix via Claude
    Haiku, with SQLite caching and bounded-concurrency batch processing.
  - `HybridIndexer` — voyage-3-large dense vectors into Qdrant + BM25 pickle.
- Retrieval cascade (P3):
  - `BM25Retriever` — loads the pickled BM25Okapi index; top-k by score.
  - `DenseRetriever` — voyage-3-large query embedding + Qdrant search, returning
    chunk ids (off-loop via `asyncio.to_thread`).
  - `rrf_fuse` — Reciprocal Rank Fusion (k=60), pure function.
  - `CohereReranker` — `rerank-v3.5` with exponential backoff on HTTP 429.
- Conformal abstention (P4): `ConformalCalibrator` (fit/save/load),
  `ConformalPredictor`, and `scripts/calibrate_abstention.py`.
- Generation (P5): `RAGAnswerGenerator` (Anthropic Citations API, ≥2-citation
  contract, Sonnet cost accounting) and `RAGPipeline.ask` orchestration
  (bm25 → dense → rrf → rerank → conformal → answer/abstain), DI-friendly with
  `RAGPipeline.from_settings`. `make ask` is now functional.
- Bumped `anthropic` 0.42.0 → 0.49.0: the Citations API is absent below 0.43.
- Skeletons (impl in later prompts): eval LLM judges (RAGAS/HHEM/DeepEval),
  ablation script, FastAPI `/ask` wiring.

## [0.1.0] — TBD (target Fri Jun 13, 2026)

Initial release target. See README for the eval table (populated by `make eval`).
