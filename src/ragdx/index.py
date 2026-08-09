"""In-memory reference retrievers.

These back the fixture corpus and the ablations. Real deployments plug their own
retriever in through ``adapters/``; these exist so ragdx can (a) run entirely
offline and (b) construct the counterfactual retrievals an ablation needs when
the user's own retriever cannot be reconfigured.

Ranking is deterministic: scores are sorted descending with ``chunk_id`` as the
tie-break, so equal scores never reorder between runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from ragdx.embedding import Embedder, StubEmbedder, Vectors
from ragdx.schema import Chunk, RetrievedChunk
from ragdx.text import tokenize


def matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    """Metadata equality, with list values meaning "any of"."""
    if not filters:
        return True
    for key, wanted in filters.items():
        actual = chunk.metadata.get(key)
        if isinstance(wanted, list | tuple | set):
            if actual not in wanted:
                return False
        elif actual != wanted:
            return False
    return True


def _top_k(
    chunks: Sequence[Chunk],
    scores: Sequence[float],
    k: int,
    filters: dict[str, Any] | None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    eligible = [
        (float(score), chunk)
        for chunk, score in zip(chunks, scores, strict=True)
        if matches_filters(chunk, filters) and (min_score is None or score > min_score)
    ]
    eligible.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
    return [
        RetrievedChunk(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(eligible[:k])
    ]


class DenseRetriever:
    """Cosine similarity over embedded chunks."""

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.embedder: Embedder = embedder or StubEmbedder()
        self._matrix: Vectors = self.embedder.embed([c.text for c in self.chunks])

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self._matrix @ q
        return _top_k(self.chunks, scores.tolist(), k, filters)

    def similarity(self, query: str, chunk_ids: Sequence[str]) -> dict[str, float]:
        """Cosine similarity for specific chunks, filters and k ignored.

        The classifier uses this for its distribution check when no ablation
        recovers the gold chunk.
        """
        q = self.embedder.embed([query])[0]
        wanted = set(chunk_ids)
        scores = self._matrix @ q
        return {
            chunk.chunk_id: float(score)
            for chunk, score in zip(self.chunks, scores.tolist(), strict=True)
            if chunk.chunk_id in wanted
        }

    def reindex(self, chunks: list[Chunk]) -> DenseRetriever:
        return DenseRetriever(chunks, self.embedder)


class LexicalRetriever:
    """BM25 over the same tokenization the dense side uses."""

    name = "lexical"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: list[Chunk] = list(chunks)
        corpus = [tokenize(c.text) or ["\0empty"] for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or not self.chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        # A chunk sharing no query term scores 0. Returning those as "results"
        # would let the lexical ablation "recover" a gold chunk on a chunk_id
        # tie-break, inventing a vocabulary mismatch that is not there.
        return _top_k(self.chunks, scores.tolist(), k, filters, min_score=0.0)

    def reindex(self, chunks: list[Chunk]) -> LexicalRetriever:
        return LexicalRetriever(chunks)


class HybridRetriever:
    """Reciprocal-rank fusion of the dense and lexical planes."""

    name = "hybrid"

    def __init__(
        self, chunks: Sequence[Chunk], embedder: Embedder | None = None, rrf_k: int = 60
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.dense = DenseRetriever(self.chunks, embedder)
        self.lexical = LexicalRetriever(self.chunks)
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        depth = max(k, 100)
        fused: dict[str, float] = {}
        for plane in (self.dense, self.lexical):
            for item in plane.retrieve(query, depth, filters):
                fused[item.chunk.chunk_id] = fused.get(item.chunk.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + item.rank + 1
                )
        by_id = {c.chunk_id: c for c in self.chunks}
        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [
            RetrievedChunk(chunk=by_id[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked[:k])
        ]

    def reindex(self, chunks: list[Chunk]) -> HybridRetriever:
        return HybridRetriever(chunks, self.dense.embedder, self.rrf_k)
