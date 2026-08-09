"""Core data model for ragdx.

Everything in the package depends on these types, so they are defined first and
changed reluctantly.

The one design decision worth calling out: gold evidence is a **character span in
the source document** (``gold_doc_id`` + ``gold_char_start`` / ``gold_char_end``),
never a chunk id. Chunk ids are invalidated the moment the corpus is re-chunked,
which would make the ``alternate_chunking`` ablation impossible to express.
Spans survive re-chunking.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FailureCause(StrEnum):
    """Root causes the classifier can assign.

    ``UNCLASSIFIED`` is a first-class outcome, not an error: the classifier
    abstains rather than guessing.
    """

    RANK_CUTOFF = "rank_cutoff"
    VOCABULARY_MISMATCH = "vocabulary_mismatch"
    PARAPHRASE_GAP = "paraphrase_gap"
    CHUNK_BOUNDARY = "chunk_boundary"
    METADATA_FILTER = "metadata_filter"
    EMBEDDING_BLIND_SPOT = "embedding_blind_spot"
    GENERATION_UNGROUNDED = "generation_ungrounded"
    UNCLASSIFIED = "unclassified"


class Chunk(BaseModel):
    """A unit of retrievable text, with its offsets in the source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_span(self) -> Chunk:
        if self.char_end < self.char_start:
            raise ValueError(
                f"chunk {self.chunk_id}: char_end ({self.char_end}) "
                f"< char_start ({self.char_start})"
            )
        return self

    def overlaps(self, doc_id: str, char_start: int, char_end: int) -> bool:
        """True if this chunk shares any characters with the given span."""
        if self.doc_id != doc_id:
            return False
        return self.char_start < char_end and char_start < self.char_end

    def overlap_fraction(self, doc_id: str, char_start: int, char_end: int) -> float:
        """Fraction of the *span* that this chunk covers, in ``[0.0, 1.0]``.

        Used by the chunk-boundary ablation: a span split across two chunks
        yields a best-chunk coverage well below 1.0.
        """
        if not self.overlaps(doc_id, char_start, char_end):
            return 0.0
        span_len = char_end - char_start
        if span_len <= 0:
            return 0.0
        covered = min(self.char_end, char_end) - max(self.char_start, char_start)
        return max(0.0, covered / span_len)


class RetrievedChunk(BaseModel):
    """A chunk as returned by a retriever, with its score and 0-indexed rank."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    rank: int = Field(ge=0)


class Golden(BaseModel):
    """A query paired with the evidence span that answers it."""

    model_config = ConfigDict(frozen=True)

    golden_id: str
    query: str
    gold_doc_id: str
    gold_char_start: int = Field(ge=0)
    gold_char_end: int = Field(ge=0)
    expected_answer: str | None = None
    origin: Literal["synthetic", "human"]
    synth_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_span(self) -> Golden:
        if self.gold_char_end <= self.gold_char_start:
            raise ValueError(
                f"golden {self.golden_id}: gold_char_end ({self.gold_char_end}) "
                f"must be > gold_char_start ({self.gold_char_start})"
            )
        return self


class Trace(BaseModel):
    """One observed retrieval (and optionally generation) for a query."""

    trace_id: str
    query: str
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    answer: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


class AblationResult(BaseModel):
    """Outcome of re-running retrieval under one counterfactual condition.

    ``skipped`` matters: "we re-ran with the filter off and it still failed" and
    "we could not re-run without the filter" are different findings, and
    collapsing them into ``recovered=False`` would let the classifier rule out a
    cause it never actually tested.
    """

    model_config = ConfigDict(frozen=True)

    ablation_name: str
    recovered: bool
    recovered_at_rank: int | None = Field(default=None, ge=0)
    skipped: bool = False
    detail: str = ""

    @model_validator(mode="after")
    def _check_rank(self) -> AblationResult:
        if self.recovered and self.recovered_at_rank is None:
            raise ValueError(f"{self.ablation_name}: recovered=True requires recovered_at_rank")
        if not self.recovered and self.recovered_at_rank is not None:
            raise ValueError(
                f"{self.ablation_name}: recovered=False must not set recovered_at_rank"
            )
        if self.skipped and self.recovered:
            raise ValueError(f"{self.ablation_name}: a skipped ablation cannot have recovered")
        return self


class Diagnosis(BaseModel):
    """The verdict for a single golden query."""

    golden_id: str
    outcome: Literal["hit", "retrieval_failure", "generation_failure"]
    cause: FailureCause | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    ablation_results: list[AblationResult] = Field(default_factory=list)
    evidence: str = ""
    gold_rank: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_cause(self) -> Diagnosis:
        if self.outcome == "hit" and self.cause is not None:
            raise ValueError(f"{self.golden_id}: outcome 'hit' must not carry a cause")
        if self.outcome != "hit" and self.cause is None:
            raise ValueError(f"{self.golden_id}: outcome '{self.outcome}' requires a cause")
        return self
