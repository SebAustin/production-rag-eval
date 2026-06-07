"""Ingestion: load FinanceBench, chunk, contextualize, and index."""

from __future__ import annotations

from rag_eval.ingestion.chunker import FinancialChunker
from rag_eval.ingestion.contextual import ContextualRetriever
from rag_eval.ingestion.indexer import HybridIndexer
from rag_eval.ingestion.loader import load_financebench, load_passages

__all__ = [
    "ContextualRetriever",
    "FinancialChunker",
    "HybridIndexer",
    "load_financebench",
    "load_passages",
]
