"""LangChain-compatible Voyage embeddings adapter for RAGAS answer-relevancy.

RAGAS's ``ResponseRelevancy`` needs an embeddings model; this wraps our
voyage-3-large client (with the shared rate-limit backoff) behind the
``langchain_core.embeddings.Embeddings`` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

from rag_eval.embedding import embed_with_backoff

if TYPE_CHECKING:
    from rag_eval.config import Settings


class VoyageEmbeddings(Embeddings):
    """Minimal langchain Embeddings backed by voyage-3-large."""

    def __init__(self, settings: Settings) -> None:
        import voyageai

        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._model = settings.voyage_model
        self._dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_with_backoff(
            self._client,
            list(texts),
            model=self._model,
            input_type="document",
            output_dimension=self._dim,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
