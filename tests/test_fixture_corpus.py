"""The fixture is the yardstick, so it gets checked on every run.

These tests assert that each *planted* failure still manifests as a primitive
retrieval fact. They deliberately do not import anything from ``ragdx.diagnose``:
if the ground truth were established by running the classifier, the Milestone 5
gate would be measuring the classifier against itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ragdx.corpus import Document, corpus_hash
from ragdx.index import DenseRetriever, LexicalRetriever
from ragdx.matching import best_coverage, gold_rank, satisfiable
from ragdx.schema import Chunk
from support import (
    ALTERNATE_CHUNKER,
    EXPECTED_PATH,
    PRODUCTION_CHUNKER,
    PRODUCTION_FILTERS,
    PRODUCTION_K,
    RANK_CUTOFF_K,
    FixtureGolden,
    load_fixture_corpus,
    load_fixture_goldens,
)

COVERAGE_THRESHOLD = 0.75


@dataclass(frozen=True)
class Facts:
    """Primitive retrieval facts for one golden. Arithmetic, not judgement."""

    production: int | None
    deep: int | None
    unfiltered: int | None
    lexical: int | None
    alternate: int | None
    coverage: float
    coverage_alt: float


class World:
    """The fixture corpus indexed under the production and ablated conditions."""

    def __init__(self) -> None:
        self.docs: list[Document] = load_fixture_corpus()
        self.fixtures: list[FixtureGolden] = load_fixture_goldens(self.docs)
        self.base: list[Chunk] = PRODUCTION_CHUNKER.chunk_all(self.docs)
        self.alt: list[Chunk] = ALTERNATE_CHUNKER.chunk_all(self.docs)
        self.dense = DenseRetriever(self.base)
        self.dense_alt = DenseRetriever(self.alt)
        self.lexical = LexicalRetriever(self.base)
        self.by_id = {f.golden.golden_id: f for f in self.fixtures}
        self.by_doc = {d.doc_id: d for d in self.docs}

    def ids(self, planted: str) -> list[str]:
        return [f.golden.golden_id for f in self.fixtures if f.planted == planted]

    def facts(self, golden_id: str) -> Facts:
        g = self.by_id[golden_id].golden
        return Facts(
            production=gold_rank(self.dense.retrieve(g.query, PRODUCTION_K, PRODUCTION_FILTERS), g),
            deep=gold_rank(self.dense.retrieve(g.query, RANK_CUTOFF_K, PRODUCTION_FILTERS), g),
            unfiltered=gold_rank(self.dense.retrieve(g.query, PRODUCTION_K, None), g),
            lexical=gold_rank(self.lexical.retrieve(g.query, PRODUCTION_K, PRODUCTION_FILTERS), g),
            alternate=gold_rank(
                self.dense_alt.retrieve(g.query, PRODUCTION_K, PRODUCTION_FILTERS), g
            ),
            coverage=best_coverage(self.base, g),
            coverage_alt=best_coverage(self.alt, g),
        )


@pytest.fixture(scope="module")
def world() -> World:
    return World()


class TestCorpusLoads:
    def test_documents_and_metadata(self, world: World) -> None:
        assert len(world.docs) == 21
        for doc in world.docs:
            assert doc.metadata["status"] in {"current", "archived"}
            assert doc.metadata["department"]
            assert not doc.text.startswith("---")

    def test_chunk_offsets_are_exact(self, world: World) -> None:
        for chunk in world.base:
            source = world.by_doc[chunk.doc_id].text
            assert source[chunk.char_start : chunk.char_end] == chunk.text

    def test_every_golden_resolves_to_a_real_span(self, world: World) -> None:
        for fx in world.fixtures:
            g = fx.golden
            assert world.by_doc[g.gold_doc_id].text[g.gold_char_start : g.gold_char_end].strip()

    def test_expected_file_matches_the_golden_set(self, world: World) -> None:
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        assert {d["golden_id"] for d in expected["diagnoses"]} == set(world.by_id)
        assert expected["_meta"]["corpus_hash"] == corpus_hash(world.docs)

    def test_expected_file_agrees_with_the_planted_labels(self, world: World) -> None:
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        for row in expected["diagnoses"]:
            planted = world.by_id[row["golden_id"]].planted
            if planted == "hit":
                assert row["outcome"] == "hit" and row["cause"] is None
            elif planted == "generation_failure":
                assert row["outcome"] == "generation_failure"
            else:
                assert row["outcome"] == "retrieval_failure"
                assert row["cause"] == planted


class TestPlantsStillManifest:
    """One test per planted failure mode. These are the yardstick's guarantees."""

    def test_true_negatives_are_retrieved(self, world: World) -> None:
        ids = world.ids("hit") + world.ids("generation_failure")
        assert len(ids) == 38
        for gid in ids:
            assert world.facts(gid).production is not None, f"{gid}: gold not in top-k"

    def test_rank_cutoff_is_ranked_but_below_k(self, world: World) -> None:
        assert world.ids("rank_cutoff")
        for gid in world.ids("rank_cutoff"):
            facts = world.facts(gid)
            assert facts.production is None, f"{gid}: should miss at k={PRODUCTION_K}"
            assert facts.deep is not None, f"{gid}: should be found at k={RANK_CUTOFF_K}"

    def test_vocabulary_mismatch_needs_the_lexical_plane(self, world: World) -> None:
        assert world.ids("vocabulary_mismatch")
        for gid in world.ids("vocabulary_mismatch"):
            facts = world.facts(gid)
            assert facts.deep is None, f"{gid}: dense should not reach it even at depth"
            assert facts.lexical is not None, f"{gid}: BM25 should find it within k"
            assert facts.coverage >= COVERAGE_THRESHOLD, f"{gid}: must not be a boundary case"

    def test_chunk_boundary_cannot_be_covered_by_one_chunk(self, world: World) -> None:
        assert world.ids("chunk_boundary")
        for gid in world.ids("chunk_boundary"):
            facts = world.facts(gid)
            assert facts.coverage < COVERAGE_THRESHOLD, f"{gid}: span is not actually split"
            assert facts.coverage_alt >= COVERAGE_THRESHOLD, f"{gid}: re-chunking must fix it"
            assert not satisfiable(world.base, world.by_id[gid].golden)
            assert facts.alternate is not None, f"{gid}: re-chunked gold must rank within k"

    def test_metadata_filter_excludes_the_gold_document(self, world: World) -> None:
        assert world.ids("metadata_filter")
        for gid in world.ids("metadata_filter"):
            facts = world.facts(gid)
            assert facts.production is None, f"{gid}: the filter should exclude it"
            assert facts.unfiltered is not None, f"{gid}: removing the filter must recover it"
            doc_id = world.by_id[gid].golden.gold_doc_id
            assert world.by_doc[doc_id].metadata["status"] == "archived"

    def test_blind_spot_is_recovered_by_nothing(self, world: World) -> None:
        assert world.ids("embedding_blind_spot")
        for gid in world.ids("embedding_blind_spot"):
            facts = world.facts(gid)
            assert facts.production is None
            assert facts.deep is None, f"{gid}: depth must not recover it"
            assert facts.lexical is None, f"{gid}: BM25 must not recover it"
            assert facts.unfiltered is None, f"{gid}: filters are not the cause"
            # Re-chunking only counts as a recovery when production chunking
            # could not cover the span at all. Here it can, so a better rank
            # under re-chunking would be a ranking accident, not a boundary fix.
            assert facts.coverage >= COVERAGE_THRESHOLD

    def test_generation_failures_retrieve_correctly(self, world: World) -> None:
        assert world.ids("generation_failure")
        for gid in world.ids("generation_failure"):
            assert world.by_id[gid].answer, f"{gid}: needs a recorded answer to judge"
            assert world.facts(gid).production is not None
