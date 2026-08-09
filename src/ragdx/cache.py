"""On-disk content-addressed cache.

Determinism is a hard requirement (PLAN.md §4): two runs on the same input must
produce byte-identical output. Embeddings are already deterministic; LLM calls
are not, so every judge response is cached under a hash of its prompt. The
second run reads the first run's answers instead of asking again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def content_key(*parts: Any) -> str:
    """Stable hash over any JSON-serializable parts."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ContentCache:
    """A namespace of content-addressed entries.

    With ``root=None`` the cache is memory-only, which is what the test suite
    uses: no stray files, same semantics.
    """

    def __init__(self, root: Path | None = None, namespace: str = "default") -> None:
        self.namespace = namespace
        self._root = (root / namespace) if root is not None else None
        self._memory: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path | None:
        return None if self._root is None else self._root / f"{key}.json"

    def get(self, key: str) -> str | None:
        if key in self._memory:
            self.hits += 1
            return self._memory[key]
        path = self._path(key)
        if path is not None and path.is_file():
            value = str(json.loads(path.read_text(encoding="utf-8"))["value"])
            self._memory[key] = value
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, key: str, value: str) -> None:
        self._memory[key] = value
        path = self._path(key)
        if path is not None:
            path.write_text(json.dumps({"value": value}, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._memory)
