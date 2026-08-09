"""LangChain adapter. Install with ``pip install 'ragdx[langchain]'``.

Wraps any LangChain retriever into the ragdx protocol. The core package does not
import this module, and this module does not import LangChain at module level,
so ragdx runs with LangChain absent.

**Character offsets are the thing to get right.** Gold evidence is a span in the
source document (PLAN.md §6), so every chunk needs to know where it came from.
Put ``char_start`` and ``char_end`` in each Document's metadata when you index —
that is the reliable path. Failing that, pass ``docs`` and the adapter locates
each returned chunk's text in its source document instead; that costs a search
per result and breaks if the same passage appears twice.
"""

from __future__ import annotations

from typing import Any, Protocol

from ragdx.corpus import Document
from ragdx.schema import Chunk, RetrievedChunk
from ragdx.spans import AmbiguousSpanError, SpanNotFoundError, find_span

DOC_ID_KEYS = ("doc_id", "source", "file_path")


class _LangChainRetriever(Protocol):
    """The slice of LangChain's BaseRetriever this adapter uses."""

    def invoke(self, input: str, config: Any = None, **kwargs: Any) -> list[Any]: ...


class OffsetsUnavailableError(ValueError):
    """A returned chunk could not be located in its source document."""


class LangChainRetrieverAdapter:
    """Adapts ``BaseRetriever`` to ragdx's ``Retriever`` protocol."""

    name = "langchain"

    def __init__(
        self,
        retriever: _LangChainRetriever,
        docs: list[Document] | None = None,
        doc_id_keys: tuple[str, ...] = DOC_ID_KEYS,
    ) -> None:
        self.retriever = retriever
        self.docs = {d.doc_id: d for d in (docs or [])}
        self.doc_id_keys = doc_id_keys

    def _doc_id(self, metadata: dict[str, Any], index: int) -> str:
        for key in self.doc_id_keys:
            if metadata.get(key):
                return str(metadata[key])
        return f"unknown-{index}"

    def _offsets(self, doc_id: str, text: str, metadata: dict[str, Any]) -> tuple[int, int]:
        start, end = metadata.get("char_start"), metadata.get("char_end")
        if isinstance(start, int) and isinstance(end, int):
            return start, end
        source = self.docs.get(doc_id)
        if source is None:
            raise OffsetsUnavailableError(
                f"chunk from {doc_id!r} has no char_start/char_end in metadata and "
                f"no source document was supplied. ragdx matches gold evidence by "
                f"character span, so it cannot score this chunk. Add the offsets at "
                f"index time, or pass docs=load_corpus(...) to the adapter."
            )
        try:
            return find_span(source.text, text)
        except (SpanNotFoundError, AmbiguousSpanError) as exc:
            raise OffsetsUnavailableError(f"cannot locate chunk in {doc_id!r}: {exc}") from exc

    def _to_chunk(self, document: Any, index: int) -> Chunk:
        metadata = dict(getattr(document, "metadata", {}) or {})
        text = str(getattr(document, "page_content", ""))
        doc_id = self._doc_id(metadata, index)
        start, end = self._offsets(doc_id, text, metadata)
        return Chunk(
            chunk_id=str(metadata.get("chunk_id") or f"{doc_id}::{start}-{end}"),
            doc_id=doc_id,
            text=text,
            char_start=start,
            char_end=end,
            metadata=metadata,
        )

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # LangChain retrievers carry k and filters in `search_kwargs`. Setting
        # them per call is what lets the rank-cutoff and filter ablations run;
        # a retriever without search_kwargs is queried as configured and those
        # ablations will simply not find anything new.
        search_kwargs = getattr(self.retriever, "search_kwargs", None)
        if isinstance(search_kwargs, dict):
            previous = dict(search_kwargs)
            search_kwargs["k"] = k
            if filters is None:
                search_kwargs.pop("filter", None)
            else:
                search_kwargs["filter"] = filters
            try:
                documents = self.retriever.invoke(query)
            finally:
                search_kwargs.clear()
                search_kwargs.update(previous)
        else:
            documents = self.retriever.invoke(query)

        return [
            RetrievedChunk(
                chunk=self._to_chunk(document, index),
                # LangChain rarely surfaces scores; a strictly decreasing stand-in
                # keeps rank order meaningful without inventing a similarity.
                score=float(len(documents) - index),
                rank=index,
            )
            for index, document in enumerate(documents[:k])
        ]
