"""Shared voyage embedding helper with rate-limit backoff.

Used by both the indexer (document embedding) and the dense retriever (query
embedding) so the Voyage free-tier rate limit (3 RPM / 10K TPM until a payment
method is added) is handled in one place.
"""

from __future__ import annotations

import time

import voyageai
import voyageai.error

from rag_eval.logging import get_logger

log = get_logger(__name__)

_MAX_RETRIES = 6
_BACKOFF_BASE_S = 15.0  # free tier is ~3 RPM -> ~20s between requests


def embed_with_backoff(  # noqa: PLR0913 — explicit embed params; all meaningful
    client: voyageai.Client,
    texts: list[str],
    *,
    model: str,
    input_type: str,
    output_dimension: int,
    max_retries: int = _MAX_RETRIES,
    backoff_base_s: float = _BACKOFF_BASE_S,
) -> list[list[float]]:
    """Embed ``texts`` via Voyage, retrying on RateLimitError with linear backoff.

    Returns one float vector per input text. Raises the underlying
    ``RateLimitError`` if the limit persists past ``max_retries``.
    """
    for attempt in range(max_retries + 1):
        try:
            result = client.embed(
                texts,
                model=model,
                input_type=input_type,
                output_dimension=output_dimension,
            )
        except voyageai.error.RateLimitError:
            if attempt == max_retries:
                raise
            delay = backoff_base_s * (attempt + 1)
            log.warning("voyage_rate_limited", attempt=attempt, delay_s=delay)
            time.sleep(delay)
        else:
            return [[float(x) for x in vector] for vector in result.embeddings]
    msg = "unreachable: embed retry loop always returns or raises"
    raise AssertionError(msg)  # pragma: no cover
