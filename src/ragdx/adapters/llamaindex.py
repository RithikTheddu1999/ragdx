"""LlamaIndex adapter. Install with ``pip install 'ragdx[llamaindex]'``.

Wraps a LlamaIndex retriever into the ragdx protocol. Neither the core package
nor this module imports LlamaIndex at module level, so ragdx runs without it.

LlamaIndex is the easier of the two adapters, because its nodes already carry
``start_char_idx`` / ``end_char_idx`` — exactly the character span ragdx needs
to match gold evidence (PLAN.md §6). Where those are absent, pass ``docs`` and
the adapter locates the node text in its source document instead.
"""

from __future__ import annotations

from typing import Any, Protocol

from ragdx.corpus import Document
from ragdx.schema import Chunk, RetrievedChunk
from ragdx.spans import AmbiguousSpanError, SpanNotFoundError, find_span

DOC_ID_KEYS = ("doc_id", "file_name", "file_path", "source")


class _LlamaRetriever(Protocol):
    """The slice of LlamaIndex's BaseRetriever this adapter uses."""

    def retrieve(self, str_or_query_bundle: Any) -> list[Any]: ...


class OffsetsUnavailableError(ValueError):
    """A returned node could not be located in its source document."""


class LlamaIndexRetrieverAdapter:
    """Adapts a LlamaIndex retriever to ragdx's ``Retriever`` protocol."""

    name = "llamaindex"

    def __init__(
        self,
        retriever: _LlamaRetriever,
        docs: list[Document] | None = None,
        doc_id_keys: tuple[str, ...] = DOC_ID_KEYS,
    ) -> None:
        self.retriever = retriever
        self.docs = {d.doc_id: d for d in (docs or [])}
        self.doc_id_keys = doc_id_keys

    def _doc_id(self, node: Any, metadata: dict[str, Any], index: int) -> str:
        for key in self.doc_id_keys:
            if metadata.get(key):
                return str(metadata[key])
        ref = getattr(node, "ref_doc_id", None) or getattr(node, "source_node", None)
        node_id = getattr(ref, "node_id", ref)
        return str(node_id) if node_id else f"unknown-{index}"

    def _offsets(self, node: Any, doc_id: str, text: str) -> tuple[int, int]:
        start = getattr(node, "start_char_idx", None)
        end = getattr(node, "end_char_idx", None)
        if isinstance(start, int) and isinstance(end, int):
            return start, end
        source = self.docs.get(doc_id)
        if source is None:
            raise OffsetsUnavailableError(
                f"node from {doc_id!r} has no start_char_idx/end_char_idx and no "
                f"source document was supplied. ragdx matches gold evidence by "
                f"character span, so it cannot score this node. Pass "
                f"docs=load_corpus(...) to the adapter."
            )
        try:
            return find_span(source.text, text)
        except (SpanNotFoundError, AmbiguousSpanError) as exc:
            raise OffsetsUnavailableError(f"cannot locate node in {doc_id!r}: {exc}") from exc

    def _to_chunk(self, scored_node: Any, index: int) -> Chunk:
        node = getattr(scored_node, "node", scored_node)
        text = str(
            node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
        )
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_id = self._doc_id(node, metadata, index)
        start, end = self._offsets(node, doc_id, text)
        return Chunk(
            chunk_id=str(getattr(node, "node_id", None) or f"{doc_id}::{start}-{end}"),
            doc_id=doc_id,
            text=text,
            char_start=start,
            char_end=end,
            metadata=metadata,
        )

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # `similarity_top_k` is what makes the rank-cutoff ablation possible;
        # without it the retriever answers at whatever depth it was built with.
        previous = getattr(self.retriever, "similarity_top_k", None)
        if previous is not None:
            self.retriever.similarity_top_k = k  # type: ignore[attr-defined]
        try:
            nodes = self.retriever.retrieve(query)
        finally:
            if previous is not None:
                self.retriever.similarity_top_k = previous  # type: ignore[attr-defined]

        return [
            RetrievedChunk(
                chunk=self._to_chunk(scored, index),
                score=float(getattr(scored, "score", None) or (len(nodes) - index)),
                rank=index,
            )
            for index, scored in enumerate(nodes[:k])
        ]
