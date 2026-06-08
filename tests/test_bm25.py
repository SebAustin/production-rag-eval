"""Tests for BM25Retriever."""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_eval.generation.citations import Chunk
from rag_eval.retrieval.bm25 import BM25Retriever
from rag_eval.text import tokenize

_DOCS = [
    "Apple revenue grew to 394 billion dollars in fiscal 2022.",
    "Microsoft cloud revenue increased year over year.",
    "Apple iPhone net sales were the largest product line.",
    "The balance sheet listed total assets and liabilities.",
    "Apple services revenue reached 78 billion dollars.",
]


def _build_retriever() -> BM25Retriever:
    chunks = [
        Chunk(
            chunk_id=f"chunk{i}",
            source_filename="doc.pdf",
            page_number=i,
            text=doc,
            contextualized_text=doc,
            char_start=0,
            char_end=len(doc),
        )
        for i, doc in enumerate(_DOCS)
    ]
    bm25 = BM25Okapi([tokenize(c.contextualized_text) for c in chunks])
    return BM25Retriever(bm25, chunks)


def test_retrieve_sorted_descending() -> None:
    retriever = _build_retriever()
    results = retriever.retrieve("Apple revenue", top_k=5)

    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert len(results) <= 5


def test_top_result_is_most_relevant() -> None:
    retriever = _build_retriever()
    results = retriever.retrieve("Apple revenue", top_k=5)
    top_id = results[0][0]
    # Docs 0 and 4 mention both "apple" and "revenue".
    assert top_id in {"chunk0", "chunk4"}


def test_top_k_caps_results() -> None:
    retriever = _build_retriever()
    assert len(retriever.retrieve("Apple", top_k=2)) == 2


def test_empty_query_returns_empty() -> None:
    retriever = _build_retriever()
    assert retriever.retrieve("   ", top_k=5) == []


def test_load_roundtrip(tmp_path: Path) -> None:
    retriever = _build_retriever()
    payload = {
        "bm25": retriever._bm25,
        "chunks": [c.model_dump() for c in retriever._chunks],
    }
    path = tmp_path / "bm25.pkl"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)

    loaded = BM25Retriever.load(path)
    assert loaded.retrieve("Apple revenue", top_k=3)[0][0] in {"chunk0", "chunk4"}
