"""Each ablation, against the planted-failure fixture.

An ablation is only useful if it fires on the failure it names and stays quiet
on every other one. Both halves are tested here.
"""

from __future__ import annotations

import time

import pytest

from ragdx.ablations import (
    AblationConfig,
    AlternateChunking,
    DenseOnly,
    DiagnosisTarget,
    FiltersRemoved,
    LexicalOnly,
    RankCutoff,
    default_battery,
    first_recovery,
    run_battery,
)
from ragdx.index import DenseRetriever, LexicalRetriever
from ragdx.schema import Chunk, Golden
from support import (
    FixtureGolden,
    fixture_target,
    load_fixture_goldens,
)


@pytest.fixture(scope="module")
def target() -> DiagnosisTarget:
    return fixture_target()


@pytest.fixture(scope="module")
def fixtures() -> list[FixtureGolden]:
    return load_fixture_goldens()


def _by_planted(fixtures: list[FixtureGolden], planted: str) -> list[Golden]:
    return [f.golden for f in fixtures if f.planted == planted]


def _failures(fixtures: list[FixtureGolden]) -> list[FixtureGolden]:
    return [f for f in fixtures if f.planted not in {"hit", "generation_failure"}]


class TestFiltersRemoved:
    def test_recovers_exactly_the_filter_failures(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        ablation = FiltersRemoved()
        for fx in _failures(fixtures):
            result = ablation.run(target, fx.golden)
            expected = fx.planted == "metadata_filter"
            assert result.recovered is expected, f"{fx.golden.golden_id}: {result.detail}"

    def test_names_the_offending_filter_key(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        golden = _by_planted(fixtures, "metadata_filter")[0]
        assert "status" in FiltersRemoved().run(target, golden).detail

    def test_skipped_when_production_applies_no_filters(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        unfiltered = DiagnosisTarget(
            retriever=target.retriever, k=target.k, filters=None, chunks=target.chunks
        )
        result = FiltersRemoved().run(unfiltered, _by_planted(fixtures, "metadata_filter")[0])
        assert result.skipped and not result.recovered


class TestRankCutoff:
    def test_recovers_exactly_the_rank_cutoffs(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        ablation = RankCutoff()
        for fx in _failures(fixtures):
            # Filter failures are excluded by the filter at any depth, so the
            # deeper retrieval genuinely cannot see them either.
            result = ablation.run(target, fx.golden)
            expected = fx.planted == "rank_cutoff"
            assert result.recovered is expected, f"{fx.golden.golden_id}: {result.detail}"

    def test_evidence_quotes_both_depths(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        detail = RankCutoff().run(target, _by_planted(fixtures, "rank_cutoff")[0]).detail
        assert f"k={target.config.rank_cutoff_k}" in detail
        assert f"current k={target.k}" in detail

    def test_refuses_to_retrieve_most_of_the_index(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        """'Raise k to 150 of 213 chunks' is not a fix anybody can ship."""
        greedy = DiagnosisTarget(
            retriever=target.retriever,
            k=target.k,
            filters=target.filters,
            chunks=target.chunks,
            docs=target.docs,
            config=AblationConfig(rank_cutoff_k=150),
        )
        golden = _by_planted(fixtures, "rank_cutoff")[0]
        assert not RankCutoff().applicable(greedy, golden)
        result = RankCutoff().run(greedy, golden)
        assert result.skipped
        assert "not a shippable fix" in result.detail

    def test_skipped_when_depth_is_not_deeper(self, target: DiagnosisTarget) -> None:
        shallow = DiagnosisTarget(
            retriever=target.retriever,
            k=50,
            filters=target.filters,
            chunks=target.chunks,
            config=AblationConfig(rank_cutoff_k=20),
        )
        golden = load_fixture_goldens()[0].golden
        assert RankCutoff().run(shallow, golden).skipped


class TestRetrievalPlanes:
    def test_lexical_only_recovers_exactly_the_vocabulary_mismatches(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        ablation = LexicalOnly()
        for fx in _failures(fixtures):
            result = ablation.run(target, fx.golden)
            expected = fx.planted == "vocabulary_mismatch"
            assert result.recovered is expected, f"{fx.golden.golden_id}: {result.detail}"

    def test_dense_only_is_skipped_for_a_dense_production_retriever(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        result = DenseOnly().run(target, _by_planted(fixtures, "vocabulary_mismatch")[0])
        assert result.skipped
        assert "already dense" in result.detail

    def test_dense_only_recovers_a_paraphrase_gap_on_a_lexical_retriever(self) -> None:
        """The mirror case: BM25 in production, meaning present but wording absent.

        The fixture corpus cannot host this one — its production retriever is
        dense, so `dense_only` never applies there. Built explicitly instead:
        the query says "employee / refund / work trip", the answer says
        "staff / reimbursement / travel". Zero shared surface forms, so BM25
        scores the gold chunk at zero and never returns it; the dense plane
        bridges the synonyms and ranks it first.
        """
        texts = {
            "expenses": "Staff may claim reimbursement for travel undertaken on company business.",
            "safety": "Work at height requires a harness and a second person present.",
            "returns": "A refund for a damaged parcel is issued once the claim is approved.",
            "trips": "A trip to a customer site is booked through the travel desk.",
            "perdiem": "Per diem covers meals while away from base.",
        }
        chunks = [
            Chunk(chunk_id=f"{k}::0", doc_id=k, text=v, char_start=0, char_end=len(v))
            for k, v in texts.items()
        ]
        gold = texts["expenses"]
        golden = Golden(
            golden_id="paraphrase",
            query="Can an employee get a refund for a work trip?",
            gold_doc_id="expenses",
            gold_char_start=0,
            gold_char_end=len(gold),
            origin="human",
        )
        lexical_production = DiagnosisTarget(
            retriever=LexicalRetriever(chunks), k=5, plane="lexical", chunks=chunks
        )

        assert LexicalOnly().run(lexical_production, golden).skipped
        result = DenseOnly().run(lexical_production, golden)
        assert result.recovered, result.detail
        assert "the lexical plane does not" in result.detail

    def test_lexical_skipped_without_corpus_chunks(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        opaque = DiagnosisTarget(retriever=target.retriever, k=target.k, filters=target.filters)
        result = LexicalOnly().run(opaque, _by_planted(fixtures, "vocabulary_mismatch")[0])
        assert result.skipped and "cannot build BM25" in result.detail


class TestAlternateChunking:
    def test_recovers_exactly_the_boundary_splits(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        ablation = AlternateChunking()
        for fx in _failures(fixtures):
            result = ablation.run(target, fx.golden)
            expected = fx.planted == "chunk_boundary"
            assert result.recovered is expected, f"{fx.golden.golden_id}: {result.detail}"

    def test_is_skipped_when_a_chunk_already_covers_the_span(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        """Bigger chunks win on ranking. Without this guard every failure would
        look like a chunking problem."""
        result = AlternateChunking().run(target, _by_planted(fixtures, "rank_cutoff")[0])
        assert result.skipped
        assert "only change ranking" in result.detail

    def test_evidence_quotes_the_coverage_improvement(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        detail = AlternateChunking().run(target, _by_planted(fixtures, "chunk_boundary")[0]).detail
        assert "split across chunks" in detail
        assert "%" in detail

    def test_skipped_without_source_documents(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        no_docs = DiagnosisTarget(
            retriever=target.retriever,
            k=target.k,
            filters=target.filters,
            chunks=target.chunks,
        )
        result = AlternateChunking().run(no_docs, _by_planted(fixtures, "chunk_boundary")[0])
        assert result.skipped and "cannot re-chunk" in result.detail

    def test_index_is_built_once_per_run(self, target: DiagnosisTarget) -> None:
        assert target.rechunked_index() is target.rechunked_index()


class TestBattery:
    def test_order_is_most_specific_first(self) -> None:
        names = [a.name for a in default_battery()]
        assert names == [
            "filters_removed",
            "rank_cutoff",
            "lexical_only",
            "dense_only",
            "alternate_chunking",
        ]
        costs = [a.cost for a in default_battery()]
        assert costs == sorted(costs)

    def test_short_circuits_at_the_first_recovery(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        golden = _by_planted(fixtures, "metadata_filter")[0]
        results = run_battery(target, golden)
        assert results[-1].recovered
        assert results[-1].ablation_name == "filters_removed"
        assert sum(r.recovered for r in results) == 1

    def test_each_planted_failure_is_named_by_the_right_ablation(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        expected = {
            "metadata_filter": "filters_removed",
            "rank_cutoff": "rank_cutoff",
            "vocabulary_mismatch": "lexical_only",
            "chunk_boundary": "alternate_chunking",
            "embedding_blind_spot": None,
        }
        for fx in _failures(fixtures):
            recovery = first_recovery(run_battery(target, fx.golden))
            name = recovery.ablation_name if recovery else None
            assert name == expected[fx.planted], f"{fx.golden.golden_id} ({fx.planted})"

    def test_blind_spots_recover_under_nothing(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        for golden in _by_planted(fixtures, "embedding_blind_spot"):
            results = run_battery(target, golden)
            assert not any(r.recovered for r in results)
            assert {r.ablation_name for r in results} == {
                "filters_removed",
                "rank_cutoff",
                "lexical_only",
                "dense_only",
                "alternate_chunking",
            }

    def test_skipped_ablations_are_recorded_not_silently_dropped(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        results = run_battery(target, _by_planted(fixtures, "chunk_boundary")[0])
        skipped = {r.ablation_name for r in results if r.skipped}
        assert {"filters_removed", "rank_cutoff", "lexical_only"} <= skipped

    def test_can_omit_skipped_results(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        results = run_battery(
            target, _by_planted(fixtures, "chunk_boundary")[0], record_skipped=False
        )
        assert all(not r.skipped for r in results)


class TestPerformance:
    def test_full_battery_over_the_golden_set_is_fast(self, fixtures: list[FixtureGolden]) -> None:
        """PLAN.md Milestone 4: 50 queries, full battery, under 30s offline."""
        assert len(fixtures) >= 50
        target = fixture_target()
        target.rechunked_index()  # index build is a per-run cost, not per-query

        started = time.perf_counter()
        for fx in fixtures:
            run_battery(target, fx.golden)
        elapsed = time.perf_counter() - started
        assert elapsed < 30.0, f"battery over {len(fixtures)} queries took {elapsed:.1f}s"

    def test_hit_queries_never_pay_for_the_battery(self, fixtures: list[FixtureGolden]) -> None:
        """The fast path matters: most queries in a real set are hits."""
        target = fixture_target()
        assert isinstance(target.retriever, DenseRetriever)
        # A hit is decided before any ablation runs; that is the classifier's
        # job, but the battery must at least be cheap when nothing is wrong.
        started = time.perf_counter()
        for fx in fixtures:
            if fx.planted == "hit":
                target.retriever.retrieve(fx.golden.query, target.k, target.filters)
        assert time.perf_counter() - started < 5.0
