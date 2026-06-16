# Production RAG Eval

> A production-grade retrieval-augmented generation pipeline for financial Q&A —
> hybrid retrieval, reranking, **conformal abstention so it knows when to say
> "I don't know,"** grounded generation with inline citations, and a triple
> evaluation harness. Every number in this README is the harness's own output,
> published even where it misses target.

[![CI](https://github.com/SebAustin/production-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/SebAustin/production-rag-eval/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-2a6db2)](https://mypy-lang.org)
[![lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![benchmark: FinanceBench](https://img.shields.io/badge/benchmark-FinanceBench-orange)](https://huggingface.co/datasets/PatronusAI/financebench)

This project answers questions over real SEC 10-K filings and — just as
importantly — **measures how well it does so and refuses to guess when retrieval
is weak.** It is built the way a production RAG system in a regulated domain
should be: grounded, evaluated, and reproducible end to end.

## Highlights

- **End-to-end and runnable** — ingestion → hybrid retrieval → reranking →
  grounded generation → evaluation, every stage driven from a `Makefile`.
- **Knows its limits** — conformal prediction calibrates a confidence threshold
  so the system abstains instead of hallucinating (**7% abstention at α = 0.10**).
- **Grounded answers** — the Anthropic Citations API gives **100% citation
  coverage**: every answered question cites the source passages it used.
- **Evaluation you can trust** — three independent judges (RAGAS, Vectara HHEM,
  DeepEval G-Eval) plus a hard citation check; results are reproducible and
  reported honestly, including the gates not yet met.
- **Measured, not asserted** — a retrieval ablation isolates BM25 / dense / RRF /
  rerank with hit@10, recall@10, and MRR.
- **Cost-aware** — **$0.011 per question** for generation, ~$0.33 for a full
  evaluation run including all judges.
- **Production hygiene** — typed (mypy), linted (ruff), unit-tested (pytest +
  coverage), CI on GitHub Actions, reproducible with `uv` and pinned model IDs.

## What this project demonstrates

| Capability | Where it shows up |
|---|---|
| Production RAG architecture | Hybrid retrieval cascade, RRF fusion, cross-encoder reranking |
| LLM evaluation & hallucination detection | RAGAS, Vectara HHEM-2.1, DeepEval G-Eval, citation-coverage gate |
| Uncertainty quantification | Split conformal prediction with a calibrated abstention threshold |
| Research → working code | Implements 3 papers: Contextual Retrieval, conformal abstention, RAGAS |
| Depth on the Anthropic API | Citations API, Haiku contextualization, Sonnet generation, cost accounting |
| Software-engineering rigor | Typing, tests, CI, dependency-injection-friendly design, response caching |
| Financial domain | 10-K filings — income statements, balance sheets, cash flow, ratios, segments |

## Architecture

![Production RAG architecture: offline indexing, online query-to-answer with a conformal abstention gate, and a triple evaluation layer.](docs/assets/architecture.png)

Three phases: an **offline indexing** pass builds a hybrid index once; an
**online query path** retrieves, fuses, reranks, and gates every question through
conformal abstention before Claude answers with citations; an **evaluation
layer** scores each answered question with three independent judges plus a
citation check.

> A landscape version for slides and social is at
> [`docs/assets/architecture-wide.png`](docs/assets/architecture-wide.png).

<details>
<summary>Same diagram as Mermaid (renders natively on GitHub)</summary>

```mermaid
flowchart TB
    subgraph INDEX["① Offline · indexing (run once)"]
        direction LR
        FB[FinanceBench<br/>150 Q&A · 10-K chunks] --> CH[Chunker<br/>512-token recursive]
        CH --> CR[Contextual prefix<br/>Claude Haiku per chunk]
        CR --> IX[(Hybrid index<br/>Qdrant voyage-3-large dim=256<br/>+ BM25 rank-bm25)]
    end

    subgraph SERVE["② Online · query → answer"]
        direction TB
        Q([Query]) --> B[BM25 retrieve<br/>top-50 lexical]
        Q --> D[Dense retrieve<br/>voyage-3-large top-50]
        B & D --> RRF[RRF fusion<br/>k=60]
        RRF --> RR[Cohere Rerank 3.5<br/>top-10 passages]
        RR --> CA{Conformal gate<br/>τ calibrated · α=0.10}
        CA -->|answer| GEN[Claude Sonnet 4.6<br/>Citations API · grounded]
        CA -->|abstain| ABS[abstained: true<br/>insufficient confidence]
    end

    subgraph EVAL["③ Evaluation"]
        direction LR
        RG[RAGAS<br/>faithfulness]
        HH[Vectara HHEM<br/>hallucination]
        DE[DeepEval<br/>G-Eval financial]
        CC[Citation coverage<br/>≥1 per answer]
    end

    IX -.serves.-> B
    IX -.serves.-> D
    GEN --> RG & HH & DE & CC

    classDef store fill:#EEEDFE,stroke:#534AB7,color:#26215C;
    classDef proc fill:#E1F5EE,stroke:#0F6E56,color:#04342C;
    classDef gate fill:#FAEEDA,stroke:#854F0B,color:#412402;
    classDef good fill:#EAF3DE,stroke:#3B6D11,color:#173404;
    classDef neutral fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    class FB,IX store;
    class CH,CR,B,D,RRF,RR proc;
    class CA gate;
    class GEN,CC good;
    class Q,ABS neutral;
    class RG,HH,DE neutral;
```

</details>

## Tech stack

| Layer | Tools |
|---|---|
| Language & tooling | Python 3.12, `uv`, `ruff`, `mypy`, `pytest`, GitHub Actions |
| Retrieval | `rank-bm25`, Voyage `voyage-3-large`, Qdrant, Reciprocal Rank Fusion, Cohere `rerank-v3.5` |
| Generation | Claude Sonnet 4.6 (Citations API), Claude Haiku (contextual prefixes) |
| Evaluation | RAGAS, Vectara HHEM-2.1-Open, DeepEval G-Eval |
| Uncertainty | Split conformal prediction |
| Core | `pydantic` / `pydantic-settings`, SQLite caching, FastAPI scaffold (CLI path live) |

## Results

FinanceBench test split, n = 30, seed = 42. Generation and the RAGAS/DeepEval
judge both use Claude Sonnet 4.6; embeddings `voyage-3-large`; reranker Cohere
`rerank-v3.5`. Numbers are copied from `evals/runs/3c53d10/summary.json` per the
eval-honesty contract in `.cursorrules` — they are the harness's own output,
deliberately published even where they miss target. Reproduce with `make eval`.

| Metric | Target | Actual | Notes |
|---|---|---|---|
| RAGAS faithfulness | ≥ 0.85 | **0.83** | Below gate; residual gap is multi-step calculations — see [financebench_analysis.md](docs/financebench_analysis.md) |
| RAGAS answer relevancy | ≥ 0.80 | **0.77** | |
| RAGAS context precision | ≥ 0.75 | **0.70** | Measured over the 10 reranked passages |
| Vectara HHEM score | ≥ 0.80 | _pending_ | Local ~1.3GB model not installed for this run |
| DeepEval G-Eval (financial) | ≥ 0.75 | **0.85** | ✅ |
| Citation coverage | 1.00 | **1.00** | ✅ every answered question grounded (≥1 citation) |
| Abstention rate | report | **7%** | 2/30; conformal α = 0.10 |
| Conditional accuracy | ≥ 0.85 | **0.86** | faithfulness ≥ 0.5 proxy (no hard oracle yet) |
| Conformal coverage | ≥ 0.90 | **0.86** | not met on n=30 (small sample + calc hard cases) |
| p50 latency | < 4s | 8.6s\* | \*inflated by Voyage free-tier rate-limit backoff, not real compute |
| Cost per question | < $0.05 | **$0.011** | generation only (~$0.33/run incl. judges) |

The honest headline: **faithfulness 0.83**, lifted from 0.51 over three
documented iterations (Sonnet 4.6, judge-context fix, prompt tightening). The
remaining 0.02 gap to the 0.85 gate is concentrated in multi-step calculation
questions, where RAGAS penalizes a *computed* figure that isn't verbatim in any
passage. Reproduce: `make eval` (full, n=30) or `make eval-smoke` (n=5).

> **Why publish the misses?** In a regulated domain, an evaluation you can trust
> is worth more than a number you can't. Showing the gaps — and the analysis
> behind them — is the point.

## Retrieval ablation

From [docs/ablation_results.md](docs/ablation_results.md) (`make ablation`). This
measures **retrieval quality** directly — whether each config surfaces a
question's gold evidence chunk(s) in the top-10 — so it needs no generation or
LLM judge.

| Retriever | hit@10 | recall@10 | MRR |
|---|---|---|---|
| BM25 only | 0.80 | 0.59 | 0.565 |
| Dense only (voyage-3-large) | 1.00 | 0.91 | **0.894** |
| BM25 + Dense + RRF | 0.93 | 0.74 | 0.638 |
| + Cohere Rerank 3.5 | 1.00 | 0.92 | 0.823 |

Honest finding: on this split **dense-only has the best MRR** (0.894) — Cohere
rerank ties on hit/recall but slightly lowers ranking quality here. All configs
run on the contextualized index; isolating the Contextual Retrieval contribution
needs a parallel index over raw chunk text (see the script docstring).

## Engineering practices

- **Typed end to end** — `pydantic` models (`Chunk`, `CitedSpan`, `CitedAnswer`,
  `EvalResult`) and `mypy`; runtime config via `pydantic-settings`.
- **Tested** — `pytest` with coverage; unit tests mock the heavy LLM/network
  calls, and the live eval path is a smoke test marked `eval` (deselected in CI).
- **CI** — GitHub Actions on every push (see badge).
- **Reproducible** — seeded splits, pinned model IDs, and SQLite caches for chunk
  contextualization and evaluation so reruns are deterministic and cheap.
- **Composable** — a dependency-injection-friendly pipeline
  (`RAGPipeline.from_settings`) and graceful degradation (each eval scorer falls
  back to *pending* rather than crashing the run).

## Quickstart

```bash
git clone https://github.com/SebAustin/production-rag-eval && cd production-rag-eval
uv sync && cp .env.example .env       # then put REAL keys in .env (Anthropic/Cohere/Voyage)

# Qdrant must be running for build-index / calibrate / ask / eval:
docker run -d --name rag-eval-qdrant -p 6333:6333 \
  -v "$(pwd)/data/index/qdrant_storage:/qdrant/storage" qdrant/qdrant

make download-data    # pulls FinanceBench from HuggingFace (~1 min)
make build-index      # contextualizes (~550 Haiku calls) + embeds + indexes (~$1)
make calibrate        # fits conformal threshold on 120-Q calibration split
make ask Q="What was Apple's revenue in fiscal year 2022?"
```

> **Model IDs** are pinned in `.env` (`HAIKU_MODEL`, `SONNET_MODEL`) — set them to
> IDs your Anthropic account actually exposes (list via the `/v1/models` endpoint).

## Why FinanceBench

FinanceBench (Islam et al., 2023) is 150 Q&A pairs over real 10-K filings from
publicly traded companies. Questions span income statements, balance sheets, cash
flow, ratios, and segment data, with ground-truth answers and evidence passages.
It is the closest public benchmark to real financial-services RAG work — a domain
where a wrong-but-confident answer is worse than an abstention.

## Sources

1. Islam et al. "FINANCEBENCH: A New Benchmark for Financial Question Answering." arXiv 2311.11944, 2023.
2. Anthropic. "Contextual Retrieval." anthropic.com/news/contextual-retrieval, Nov 2024.
3. Yadkori et al. "Mitigating LLM Hallucinations via Conformal Abstention." arXiv 2405.01563, 2024.
4. Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." arXiv 2309.15217, 2023.
5. Saad-Falcon et al. "HHEM-2.1-Open: an open-source hallucination detection model." Vectara, 2024.

## License

MIT — see [LICENSE](LICENSE).
