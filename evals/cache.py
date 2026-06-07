"""SQLite-backed cache for eval results, keyed by question + question text.

Lets an interrupted eval run resume without re-paying for already-scored
questions. Values are JSON-serializable ``EvalResult.model_dump(mode="json")``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteEvalCache:
    """Tiny key/value store over SQLite holding serialized eval results."""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload TEXT)",
        )
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached payload for ``key`` or ``None``."""
        row = self._conn.execute(
            "SELECT payload FROM results WHERE key = ?", (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])  # type: ignore[no-any-return]

    def put(self, key: str, payload: dict[str, Any]) -> None:
        """Store ``payload`` under ``key``."""
        self._conn.execute(
            "INSERT OR REPLACE INTO results (key, payload) VALUES (?, ?)",
            (key, json.dumps(payload)),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
