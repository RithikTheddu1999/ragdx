"""The ablation contract and the shared state ablations run against.

An ablation is one counterfactual: *change exactly one thing about retrieval and
see whether the gold chunk comes back*. Everything else — the query, the value
of `k`, the coverage threshold — is held constant, because an ablation that
changes two things at once names nothing.

``DiagnosisTarget`` owns the expensive shared objects (the BM25 index, the
re-chunked index) and builds each at most once per run, not once per query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ragdx.adapters.base import Retriever
from ragdx.chunking import FixedSizeChunker
from ragdx.corpus import Document
from ragdx.embedding import Embedder, StubEmbedder
from ragdx.index import DenseRetriever, LexicalRetriever
from ragdx.matching import COVERAGE_THRESHOLD, satisfiable
from ragdx.schema import AblationResult, Chunk, Golden


@dataclass(frozen=True)
class AblationConfig:
    """Knobs shared by the battery."""

    #: Depth the rank-cutoff ablation retrieves to. PLAN.md §2 illustrates 100;
    #: it must stay well below the index size or the "ablation" degenerates into
    #: returning most of the corpus, which is not a fix anyone can ship.
    rank_cutoff_k: int = 100
    alternate_chunk_size: int = 960
    alternate_chunk_overlap: int = 480
    coverage_threshold: float = COVERAGE_THRESHOLD


@dataclass
class DiagnosisTarget:
    """The production retrieval setup, plus whatever ragdx needs to vary it.

    ``chunks``/``docs``/``embedder`` are optional. A user whose retriever is a
    hosted service cannot hand ragdx its corpus, and in that case the lexical
    and re-chunking ablations report themselves *skipped* rather than quietly
    returning "did not recover".
    """

    retriever: Retriever
    k: int
    filters: dict[str, Any] | None = None
    #: Which plane production retrieval runs on: dense, lexical or hybrid.
    plane: str = "dense"
    chunks: list[Chunk] | None = None
    docs: list[Document] | None = None
    embedder: Embedder | None = None
    config: AblationConfig = field(default_factory=AblationConfig)

    _lexical: LexicalRetriever | None = field(default=None, init=False, repr=False)
    _dense: DenseRetriever | None = field(default=None, init=False, repr=False)
    _rechunked: tuple[list[Chunk], DenseRetriever] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def n_chunks(self) -> int:
        return len(self.chunks) if self.chunks is not None else 0

    def lexical_index(self) -> LexicalRetriever | None:
        """BM25 over the production chunking, built once."""
        if self.chunks is None:
            return None
        if self._lexical is None:
            self._lexical = LexicalRetriever(self.chunks)
        return self._lexical

    def dense_index(self) -> DenseRetriever | None:
        """A dense index over the production chunking, built once."""
        if self.chunks is None:
            return None
        if self._dense is None:
            self._dense = DenseRetriever(self.chunks, self.embedder or StubEmbedder())
        return self._dense

    def rechunked_index(self) -> tuple[list[Chunk], DenseRetriever] | None:
        """The corpus re-chunked larger and with overlap, indexed once.

        This is the expensive one, which is why it is cached across the whole
        run rather than rebuilt per query.
        """
        if self.docs is None:
            return None
        if self._rechunked is None:
            chunker = FixedSizeChunker(
                size=self.config.alternate_chunk_size,
                overlap=self.config.alternate_chunk_overlap,
            )
            chunks = chunker.chunk_all(self.docs)
            self._rechunked = (chunks, DenseRetriever(chunks, self.embedder or StubEmbedder()))
        return self._rechunked

    def satisfiable(self, golden: Golden) -> bool:
        """Can *any* chunk of the production chunking answer this golden?

        ``False`` means the chunker is the problem: no reranker, filter change
        or second retrieval plane can surface a chunk that does not exist.
        """
        if self.chunks is None:
            return True
        return satisfiable(self.chunks, golden, self.config.coverage_threshold)


class Ablation(Protocol):
    """One counterfactual retrieval."""

    @property
    def name(self) -> str: ...

    @property
    def cost(self) -> int:
        """Rough relative expense; the battery runs cheap ones first."""
        ...

    def applicable(self, target: DiagnosisTarget, golden: Golden) -> bool:
        """False when this ablation cannot say anything about this golden."""
        ...

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        """Re-run retrieval under this counterfactual."""
        ...


def skipped(name: str, why: str) -> AblationResult:
    return AblationResult(ablation_name=name, recovered=False, skipped=True, detail=why)
