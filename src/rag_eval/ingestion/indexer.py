"""Build the hybrid index: Qdrant dense vectors + an on-disk BM25 pickle.

Both indexes are built over ``contextualized_text`` (Contextual Retrieval
prefix + body), per the architecture invariant. The Qdrant payload keeps the
original ``text`` and char offsets so generation/citations can recover spans.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import voyageai
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from rag_eval.generation.citations import Chunk
from rag_eval.logging import get_logger
from rag_eval.text import tokenize

if TYPE_CHECKING:
    from rag_eval.config import Settings

log = get_logger(__name__)

_QDRANT_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000f1ce")
_EMBED_BATCH = 128
_BM25_PATH = Path("data/index/bm25.pkl")
_CHUNKS_PATH = Path("data/index/chunks.jsonl")


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id (UUID) derived from a chunk id."""
    return str(uuid.uuid5(_QDRANT_NAMESPACE, chunk_id))


class HybridIndexer:
    """Index chunks into Qdrant (dense) and a pickled BM25Okapi (lexical)."""

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient | None = None,
        voyage_client: voyageai.Client | None = None,
    ) -> None:
        self._settings = settings
        self._qdrant = qdrant_client or QdrantClient(url=settings.qdrant_url)
        self._voyage = voyage_client or voyageai.Client(api_key=settings.voyage_api_key)
        self._bm25: BM25Okapi | None = None

    async def build(self, chunks: list[Chunk]) -> None:
        """Embed + upsert into Qdrant and build/persist the BM25 index."""
        if not chunks:
            log.warning("build_index_empty")
            return
        self._build_dense(chunks)
        self._build_bm25(chunks)
        self._persist_chunks(chunks)
        log.info("index_built", n_chunks=len(chunks))

    def _build_dense(self, chunks: list[Chunk]) -> None:
        collection = self._settings.qdrant_collection
        self._qdrant.recreate_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=self._settings.embedding_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        )
        points: list[models.PointStruct] = []
        for batch in _batched(chunks, _EMBED_BATCH):
            vectors = self._embed_documents([c.contextualized_text for c in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                points.append(
                    models.PointStruct(
                        id=point_id_for(chunk.chunk_id),
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "source_filename": chunk.source_filename,
                            "page_number": chunk.page_number,
                            "text": chunk.text,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                        },
                    ),
                )
        self._qdrant.upsert(collection_name=collection, points=points)
        log.info("dense_indexed", points=len(points), collection=collection)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._voyage.embed(
            texts,
            model=self._settings.voyage_model,
            input_type="document",
            output_dimension=self._settings.embedding_dim,
        )
        return [list(map(float, v)) for v in result.embeddings]

    def _build_bm25(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(c.contextualized_text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self.save_bm25(_BM25_PATH, chunks)
        log.info("bm25_indexed", n=len(chunks), path=str(_BM25_PATH))

    def save_bm25(self, path: Path, chunks: list[Chunk]) -> None:
        """Pickle the BM25 index alongside the chunk list it was built from."""
        if self._bm25 is None:
            msg = "BM25 index has not been built"
            raise RuntimeError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunks": [c.model_dump() for c in chunks],
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    def load_bm25(self, path: Path = _BM25_PATH) -> tuple[BM25Okapi, list[Chunk]]:
        """Load a pickled BM25 index and its chunk list."""
        with path.open("rb") as fh:
            payload: dict[str, Any] = pickle.load(fh)  # noqa: S301 — local, trusted artifact
        self._bm25 = payload["bm25"]
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return self._bm25, chunks

    @staticmethod
    def _persist_chunks(chunks: list[Chunk]) -> None:
        _CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CHUNKS_PATH.write_text(
            "\n".join(c.model_dump_json() for c in chunks),
            encoding="utf-8",
        )


def _batched(items: list[Chunk], size: int) -> list[list[Chunk]]:
    """Split ``items`` into consecutive batches of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]
