"""Shared result types for both ways goldens get in.

Synthesis and import both produce *some* goldens and reject others, and the
rejection rate is the number worth looking at: a synthetic set that kept
everything it generated was not verified.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ragdx.schema import Golden


class RejectReason(StrEnum):
    """Why a candidate golden did not make it into the set."""

    MALFORMED = "malformed_response"
    EVIDENCE_NOT_FOUND = "evidence_not_found"
    EVIDENCE_AMBIGUOUS = "evidence_ambiguous"
    UNKNOWN_DOCUMENT = "unknown_document"
    SPAN_OUT_OF_RANGE = "span_out_of_range"
    #: The question could not be answered from its own gold chunk. Usually the
    #: generator asked about something it only half saw.
    NOT_ANSWERABLE_FROM_GOLD = "not_answerable_from_gold"
    #: The question could also be answered from an unrelated chunk, so it does
    #: not test retrieval at all. This is the rejection that matters most.
    ANSWERABLE_FROM_DISTRACTOR = "answerable_from_distractor"
    JUDGE_ABSTAINED = "judge_abstained"
    #: The corpus is too small to hold out a distractor, so the candidate could
    #: not be verified. Keeping it unverified would be worse than dropping it.
    NO_DISTRACTOR = "no_distractor_available"


class Rejection(BaseModel):
    """A candidate that was thrown away, and why."""

    model_config = ConfigDict(frozen=True)

    reason: RejectReason
    detail: str = ""
    source: str = ""


class GoldenBatch(BaseModel):
    """The outcome of building goldens: what was kept and what was not."""

    goldens: list[Golden] = Field(default_factory=list)
    rejections: list[Rejection] = Field(default_factory=list)

    @property
    def n_considered(self) -> int:
        return len(self.goldens) + len(self.rejections)

    @property
    def rejection_rate(self) -> float:
        return len(self.rejections) / self.n_considered if self.n_considered else 0.0

    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason.value] = counts.get(rejection.reason.value, 0) + 1
        return dict(sorted(counts.items()))
