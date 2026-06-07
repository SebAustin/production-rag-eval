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
- Skeletons (impl in later prompts): retrieval (BM25/dense/RRF/rerank),
  abstention (scorer/calibration/predictor), generation (answer/pipeline),
  FastAPI server, eval harness.

## [0.1.0] — TBD (target Fri Jun 13, 2026)

Initial release target. See README for the eval table (populated by `make eval`).
