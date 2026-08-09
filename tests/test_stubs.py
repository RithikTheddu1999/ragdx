"""The offline stubs. Determinism here is what makes byte-identical runs possible."""

from __future__ import annotations

import numpy as np
import pytest

from ragdx.embedding import StubEmbedder
from ragdx.index import DenseRetriever, HybridRetriever, LexicalRetriever, matches_filters
from ragdx.judge import JudgeVerdict, StubJudge
from ragdx.schema import Chunk

CHUNKS = [
    Chunk(
        chunk_id="c0",
        doc_id="d0",
        text="Reefer units are pre-cooled for ninety minutes before loading.",
        char_start=0,
        char_end=62,
        metadata={"status": "current", "department": "operations"},
    ),
    Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="A refund request older than ninety days is declined automatically.",
        char_start=0,
        char_end=66,
        metadata={"status": "archived", "department": "finance"},
    ),
    Chunk(
        chunk_id="c2",
        doc_id="d2",
        text="Pallets must be shrink wrapped with four full turns at the base.",
        char_start=0,
        char_end=64,
        metadata={"status": "current", "department": "logistics"},
    ),
]


class TestStubEmbedder:
    def test_vectors_are_unit_length(self) -> None:
        vectors = StubEmbedder(dim=64).embed([c.text for c in CHUNKS])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_is_deterministic_across_instances(self) -> None:
        a = StubEmbedder(dim=64).embed(["reefer units"])
        b = StubEmbedder(dim=64).embed(["reefer units"])
        assert a.tobytes() == b.tobytes()

    def test_empty_text_gives_a_zero_vector_rather_than_nan(self) -> None:
        assert not np.isnan(StubEmbedder(dim=64).embed(["", "   "])).any()

    def test_shared_wording_scores_above_unrelated_wording(self) -> None:
        embedder = StubEmbedder(dim=256)
        query, related, unrelated = embedder.embed(
            ["shrink wrapped pallets", "Pallets must be shrink wrapped.", "OAuth token expiry"]
        )
        assert float(query @ related) > float(query @ unrelated)

    def test_synonyms_bridge_where_surface_forms_do_not(self) -> None:
        embedder = StubEmbedder(dim=256)
        query, bridged, unrelated = embedder.embed(["reimbursement", "refund", "pallet"])
        assert float(query @ bridged) == pytest.approx(1.0)
        assert float(query @ unrelated) < 0.5

    def test_repeated_tokens_are_damped(self) -> None:
        """Sublinear tf: saying it four times is not four times the signal."""
        embedder = StubEmbedder(dim=256)
        once, repeated = embedder.embed(["rollover", "rollover rollover rollover rollover"])
        assert float(once @ repeated) == pytest.approx(1.0)


class TestFilters:
    def test_equality_and_membership(self) -> None:
        assert matches_filters(CHUNKS[0], {"status": "current"})
        assert not matches_filters(CHUNKS[1], {"status": "current"})
        assert matches_filters(CHUNKS[1], {"status": ["current", "archived"]})

    def test_absent_key_never_matches(self) -> None:
        assert not matches_filters(CHUNKS[0], {"missing": "value"})

    def test_empty_filters_admit_everything(self) -> None:
        assert all(matches_filters(c, None) and matches_filters(c, {}) for c in CHUNKS)


class TestRetrievers:
    def test_dense_ranks_are_contiguous_from_zero(self) -> None:
        results = DenseRetriever(CHUNKS).retrieve("shrink wrapped pallets", k=3)
        assert [r.rank for r in results] == [0, 1, 2]

    def test_dense_respects_filters(self) -> None:
        results = DenseRetriever(CHUNKS).retrieve(
            "refund request", k=5, filters={"status": "current"}
        )
        assert all(r.chunk.metadata["status"] == "current" for r in results)

    def test_dense_is_deterministic(self) -> None:
        first = DenseRetriever(CHUNKS).retrieve("ninety", k=3)
        second = DenseRetriever(CHUNKS).retrieve("ninety", k=3)
        assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]

    def test_lexical_excludes_chunks_sharing_no_term(self) -> None:
        results = LexicalRetriever(CHUNKS).retrieve("pallets", k=5)
        assert [r.chunk.chunk_id for r in results] == ["c2"]

    def test_lexical_returns_nothing_for_a_stopword_only_query(self) -> None:
        assert LexicalRetriever(CHUNKS).retrieve("the and of", k=5) == []

    def test_reindex_swaps_the_backing_chunks(self) -> None:
        smaller = DenseRetriever(CHUNKS).reindex(CHUNKS[:1])
        assert len(smaller.retrieve("anything", k=5)) == 1

    def test_hybrid_fuses_both_planes(self) -> None:
        results = HybridRetriever(CHUNKS).retrieve("refund request pallets", k=3)
        assert [r.rank for r in results] == [0, 1, 2]

    def test_empty_index_returns_nothing(self) -> None:
        assert DenseRetriever([]).retrieve("q", k=5) == []
        assert LexicalRetriever([]).retrieve("q", k=5) == []


class TestStubJudge:
    def test_abstains_on_an_unknown_prompt(self) -> None:
        verdict = StubJudge().judge("never seen", ("grounded", "ungrounded"))
        assert verdict.abstained
        assert verdict.confidence == 0.0

    def test_returns_the_canned_verdict(self) -> None:
        judge = StubJudge()
        judge.set_verdict("is it grounded?", JudgeVerdict(label="ungrounded", confidence=0.9))
        verdict = judge.judge("is it grounded?", ("grounded", "ungrounded"))
        assert verdict.label == "ungrounded"
        assert not verdict.abstained

    def test_completion_defaults_to_empty_rather_than_invented(self) -> None:
        assert StubJudge().complete("write me a question") == ""

    def test_records_calls(self) -> None:
        judge = StubJudge()
        judge.complete("a")
        judge.judge("b", ("x",))
        assert len(judge.calls) == 2
