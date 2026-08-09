"""Chunkers.

Chunks carry exact character offsets into the source document body. Everything
downstream — gold-span coverage, boundary detection, the re-chunking ablation —
is arithmetic over those offsets, so they have to be exact rather than
approximate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ragdx.corpus import Document
from ragdx.schema import Chunk

_WORD_RE = re.compile(r"\S+")


class Chunker(Protocol):
    """Cuts a document into retrievable chunks."""

    @property
    def name(self) -> str:
        """Stable identifier, used in chunk ids and cache keys."""
        ...

    def chunk(self, doc: Document) -> list[Chunk]:
        """Return chunks covering ``doc``, in document order."""
        ...


@dataclass(frozen=True)
class FixedSizeChunker:
    """Greedy fixed-size chunker that never splits a word.

    ``size`` and ``overlap`` are in characters. Words are packed until adding the
    next one would exceed ``size``; the next chunk backs up over whole words
    until at least ``overlap`` characters are repeated.
    """

    size: int = 220
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("chunk size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be >= 0 and < size")

    @property
    def name(self) -> str:
        return f"fixed{self.size}x{self.overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        words = [(m.start(), m.end()) for m in _WORD_RE.finditer(doc.text)]
        if not words:
            return []

        chunks: list[Chunk] = []
        i = 0
        while i < len(words):
            start = words[i][0]
            j = i
            while j < len(words) and (words[j][1] - start) <= self.size:
                j += 1
            if j == i:  # single word longer than `size`; emit it alone
                j = i + 1
            end = words[j - 1][1]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{self.name}::{start}-{end}",
                    doc_id=doc.doc_id,
                    text=doc.text[start:end],
                    char_start=start,
                    char_end=end,
                    metadata=dict(doc.metadata),
                )
            )
            if j >= len(words):
                break
            i = _next_start(words, i, j, end, self.overlap)
        return chunks

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        return [c for doc in docs for c in self.chunk(doc)]


def _next_start(words: list[tuple[int, int]], i: int, j: int, end: int, overlap: int) -> int:
    """Index of the first word of the next chunk, honouring ``overlap``."""
    if overlap == 0:
        return j
    back = j
    while back > i + 1 and (end - words[back - 1][0]) < overlap:
        back -= 1
    return max(back, i + 1)
