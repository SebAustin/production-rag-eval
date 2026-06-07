"""Shared text utilities (used by both indexing and BM25 retrieval)."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization.

    Used identically at index time and query time so BM25 statistics line up.
    """
    return _TOKEN_RE.findall(text.lower())
