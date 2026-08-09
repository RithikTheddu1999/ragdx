"""The retriever contract every adapter satisfies.

Anything that can answer ``retrieve(query, k, filters)`` can be diagnosed. That
is deliberately the whole surface area — see the trace-file adapter for the
zero-integration path.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ragdx.schema import Chunk, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """A single-turn retriever."""

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks for ``query``, rank 0 first.

        ``filters`` is a metadata equality map; ``None`` means no filtering.
        Implementations must return ranks that are contiguous from 0.
        """
        ...


@runtime_checkable
class Replayed(Protocol):
    """A retriever backed by recorded traces rather than a live index.

    Counterfactuals that need a *new* retrieval — retrieving deeper, dropping a
    filter — cannot be run against a recording. Ablations check for this and
    report themselves skipped, because "we re-ran it and it still failed" and
    "we could not re-run it" must never collapse into the same answer.
    """

    @property
    def recorded_depth(self) -> int:
        """How many results the recording actually contains."""
        ...

    @property
    def recorded_filters(self) -> dict[str, Any] | None:
        """The filter that was in force when the trace was captured."""
        ...


@runtime_checkable
class Indexable(Protocol):
    """A retriever that can be rebuilt over a different chunking of the corpus.

    Only the ``alternate_chunking`` ablation needs this; retrievers that cannot
    be re-indexed simply skip that ablation.
    """

    def reindex(self, chunks: list[Chunk]) -> Retriever:
        """Return a retriever backed by ``chunks`` instead of the current index."""
        ...
