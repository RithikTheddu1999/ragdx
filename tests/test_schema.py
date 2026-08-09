"""Round-trip and invariant tests for the core data model."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ragdx.schema import (
    AblationResult,
    Chunk,
    Diagnosis,
    FailureCause,
    Golden,
    RetrievedChunk,
    Trace,
)


def _chunk(**kw: Any) -> Chunk:
    base: dict[str, Any] = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "hello world",
        "char_start": 0,
        "char_end": 11,
    }
    base.update(kw)
    return Chunk(**base)


class TestRoundTrip:
    def test_chunk(self) -> None:
        c = _chunk(metadata={"department": "legal"})
        assert Chunk.model_validate_json(c.model_dump_json()) == c

    def test_retrieved_chunk(self) -> None:
        r = RetrievedChunk(chunk=_chunk(), score=0.42, rank=3)
        assert RetrievedChunk.model_validate_json(r.model_dump_json()) == r

    def test_golden(self) -> None:
        g = Golden(
            golden_id="g1",
            query="what is the refund window?",
            gold_doc_id="d1",
            gold_char_start=10,
            gold_char_end=90,
            expected_answer="30 days",
            origin="synthetic",
            synth_confidence=0.8,
        )
        assert Golden.model_validate_json(g.model_dump_json()) == g

    def test_trace(self) -> None:
        t = Trace(
            trace_id="t1",
            query="q",
            retrieved=[RetrievedChunk(chunk=_chunk(), score=1.0, rank=0)],
            answer="a",
            config_snapshot={"k": 5, "retriever": "dense"},
        )
        assert Trace.model_validate_json(t.model_dump_json()) == t

    def test_diagnosis(self) -> None:
        d = Diagnosis(
            golden_id="g1",
            outcome="retrieval_failure",
            cause=FailureCause.RANK_CUTOFF,
            confidence=1.0,
            ablation_results=[
                AblationResult(ablation_name="rank_cutoff", recovered=True, recovered_at_rank=11)
            ],
            evidence="gold chunk recovered at rank 11 with k=100; current k=5",
            gold_rank=None,
        )
        assert Diagnosis.model_validate_json(d.model_dump_json()) == d

    def test_diagnosis_json_is_stable(self) -> None:
        """Two dumps of the same object are byte-identical (determinism floor)."""
        d = Diagnosis(golden_id="g1", outcome="hit", confidence=1.0, evidence="gold at rank 2")
        assert d.model_dump_json() == d.model_dump_json()
        assert json.loads(d.model_dump_json())["cause"] is None


class TestChunkSpans:
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [(0, 5, True), (10, 20, True), (20, 30, False), (30, 40, False), (5, 25, True)],
    )
    def test_overlaps(self, start: int, end: int, expected: bool) -> None:
        c = _chunk(char_start=0, char_end=20)
        assert c.overlaps("d1", start, end) is expected

    def test_overlaps_requires_same_doc(self) -> None:
        assert _chunk(char_start=0, char_end=20).overlaps("other", 0, 5) is False

    def test_overlap_fraction_full_and_partial(self) -> None:
        c = _chunk(char_start=0, char_end=100)
        assert c.overlap_fraction("d1", 10, 20) == pytest.approx(1.0)
        # span 90..110 is only half covered by a chunk ending at 100
        assert c.overlap_fraction("d1", 90, 110) == pytest.approx(0.5)
        assert c.overlap_fraction("d1", 200, 210) == 0.0

    def test_rejects_inverted_span(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(char_start=10, char_end=5)


class TestInvariants:
    def test_golden_span_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            Golden(
                golden_id="g",
                query="q",
                gold_doc_id="d",
                gold_char_start=5,
                gold_char_end=5,
                origin="human",
            )

    def test_recovered_requires_rank(self) -> None:
        with pytest.raises(ValidationError):
            AblationResult(ablation_name="a", recovered=True)

    def test_not_recovered_forbids_rank(self) -> None:
        with pytest.raises(ValidationError):
            AblationResult(ablation_name="a", recovered=False, recovered_at_rank=3)

    def test_hit_must_not_carry_cause(self) -> None:
        with pytest.raises(ValidationError):
            Diagnosis(
                golden_id="g",
                outcome="hit",
                cause=FailureCause.RANK_CUTOFF,
                confidence=1.0,
            )

    def test_failure_must_carry_cause(self) -> None:
        with pytest.raises(ValidationError):
            Diagnosis(golden_id="g", outcome="retrieval_failure", confidence=1.0)

    def test_unclassified_is_a_real_cause(self) -> None:
        d = Diagnosis(
            golden_id="g",
            outcome="retrieval_failure",
            cause=FailureCause.UNCLASSIFIED,
            confidence=0.0,
            evidence="no ablation recovered the gold chunk",
        )
        assert d.cause is FailureCause.UNCLASSIFIED
