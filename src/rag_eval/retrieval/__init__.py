"""Retrieval cascade: BM25 + dense + RRF fusion + Cohere rerank.

Implementations land in Prompt 3; this package currently exposes the public
surface so the pipeline and tests can import against stable names.
"""

from __future__ import annotations

from rag_eval.retrieval.rrf import rrf_fuse

__all__ = ["rrf_fuse"]
