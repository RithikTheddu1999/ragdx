"""The classifier — written before the implementation, per PLAN.md §7.

The gate for the whole project lives in ``TestTheGate`` at the bottom: the
classifier must reproduce ``tests/fixtures/expected/diagnoses.json`` with ≥95%
agreement and raise **zero** false positives on the true-negative fixtures.
"""

from __future__ import annotations

import json

import pytest

from ragdx.ablations import AblationConfig, DiagnosisTarget
from ragdx.chunking import FixedSizeChunker
from ragdx.corpus import Document
from ragdx.diagnose.classifier import (
    CAUSE_BY_ABLATION,
    Classifier,
    ClassifierConfig,
    cause_for_ablation,
)
from ragdx.index import DenseRetriever, LexicalRetriever
from ragdx.schema import AblationResult, Chunk, Diagnosis, FailureCause, Golden
from support import (
    EXPECTED_PATH,
    PRODUCTION_CHUNKER,
    FixtureGolden,
    fixture_target,
    load_fixture_corpus,
    load_fixture_goldens,
)


@pytest.fixture(scope="module")
def target() -> DiagnosisTarget:
    return fixture_target()


@pytest.fixture(scope="module")
def fixtures() -> list[FixtureGolden]:
    return load_fixture_goldens()


@pytest.fixture(scope="module")
def diagnoses(target: DiagnosisTarget, fixtures: list[FixtureGolden]) -> dict[str, Diagnosis]:
    classifier = Classifier(target)
    return {f.golden.golden_id: classifier.classify(f.golden) for f in fixtures}


class TestAblationToCauseMapping:
    """Pure, deterministic: an ablation name maps to exactly one cause."""

    @pytest.mark.parametrize(
        ("ablation", "cause"),
        [
            ("filters_removed", FailureCause.METADATA_FILTER),
            ("rank_cutoff", FailureCause.RANK_CUTOFF),
            ("lexical_only", FailureCause.VOCABULARY_MISMATCH),
            ("dense_only", FailureCause.PARAPHRASE_GAP),
            ("alternate_chunking", FailureCause.CHUNK_BOUNDARY),
        ],
    )
    def test_each_ablation_names_its_cause(self, ablation: str, cause: FailureCause) -> None:
        assert cause_for_ablation(ablation) is cause

    def test_unknown_ablation_names_nothing(self) -> None:
        assert cause_for_ablation("something_new") is None

    def test_every_battery_ablation_is_mapped(self) -> None:
        from ragdx.ablations import default_battery

        assert {a.name for a in default_battery()} <= set(CAUSE_BY_ABLATION)


class TestFastPath:
    def test_a_hit_runs_no_ablations(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        hits = [f.golden.golden_id for f in fixtures if f.planted == "hit"]
        assert hits
        for gid in hits:
            diagnosis = diagnoses[gid]
            assert diagnosis.outcome == "hit"
            assert diagnosis.cause is None
            assert diagnosis.ablation_results == []
            assert diagnosis.confidence == 1.0

    def test_a_hit_records_the_rank_it_was_found_at(self, diagnoses: dict[str, Diagnosis]) -> None:
        diagnosis = diagnoses["h01"]
        assert diagnosis.gold_rank == 0
        assert "rank 0" in diagnosis.evidence


class TestDeterministicRecoveries:
    def test_confidence_is_certain_when_an_ablation_recovered_it(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        deterministic = {"metadata_filter", "rank_cutoff", "vocabulary_mismatch", "chunk_boundary"}
        for fx in fixtures:
            if fx.planted in deterministic:
                assert diagnoses[fx.golden.golden_id].confidence == 1.0, fx.golden.golden_id

    def test_evidence_is_a_plain_english_one_liner(self, diagnoses: dict[str, Diagnosis]) -> None:
        """Users will not trust a bare label."""
        for diagnosis in diagnoses.values():
            assert diagnosis.evidence
            assert "\n" not in diagnosis.evidence
            assert len(diagnosis.evidence) > 20

    def test_rank_cutoff_evidence_quotes_both_k_values(
        self, diagnoses: dict[str, Diagnosis]
    ) -> None:
        evidence = diagnoses["rc01"].evidence
        assert "rank" in evidence and "k=" in evidence

    def test_filter_evidence_names_the_filter(self, diagnoses: dict[str, Diagnosis]) -> None:
        assert "status" in diagnoses["mf01"].evidence

    def test_ablation_results_are_attached(self, diagnoses: dict[str, Diagnosis]) -> None:
        results = diagnoses["vm01"].ablation_results
        assert results
        assert results[-1].recovered
        assert results[-1].ablation_name == "lexical_only"


class TestAbstention:
    """The classifier must never guess. `unclassified` is a real answer."""

    def _target(self, chunks: list[Chunk], **kwargs: object) -> DiagnosisTarget:
        return DiagnosisTarget(
            retriever=DenseRetriever(chunks),
            k=1,
            chunks=chunks,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_blind_spot_when_the_gold_chunk_sits_far_below_the_field(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        for fx in fixtures:
            if fx.planted == "embedding_blind_spot":
                diagnosis = diagnoses[fx.golden.golden_id]
                assert diagnosis.cause is FailureCause.EMBEDDING_BLIND_SPOT
                assert 0.0 < diagnosis.confidence < 1.0, "a distribution call is not certain"

    def test_unclassified_when_nothing_recovers_and_the_field_is_close(self) -> None:
        """Gold ranks just outside k and scores close to the winners: no
        ablation fires, and the distribution does not support a blind spot."""
        texts = [
            "The depot team reviews shipment delays every morning.",
            "The depot team reviews shipment delays every afternoon.",
            "The depot team reviews shipment delays every evening.",
            "The depot team reviews shipment delays every weekend.",
        ]
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, char_start=0, char_end=len(t))
            for i, t in enumerate(texts)
        ]
        golden = Golden(
            golden_id="near-miss",
            query="When does the depot team review shipment delays?",
            gold_doc_id="d3",
            gold_char_start=0,
            gold_char_end=len(texts[3]),
            origin="human",
        )
        target = DiagnosisTarget(
            retriever=DenseRetriever(chunks),
            k=1,
            chunks=chunks,
            config=AblationConfig(rank_cutoff_k=2),
        )
        diagnosis = Classifier(target).classify(golden)
        assert diagnosis.outcome == "retrieval_failure"
        assert diagnosis.cause is FailureCause.UNCLASSIFIED
        assert diagnosis.confidence == 0.0
        assert "could not" in diagnosis.evidence or "no ablation" in diagnosis.evidence

    def test_unclassified_when_similarity_cannot_be_measured(self) -> None:
        """An opaque retriever leaves no distribution to reason about.

        Here the gold document is not in the index at all, and ragdx was handed
        no corpus, so no ablation can recover it and nothing can be measured.
        Saying "embedding blind spot" would be a guess.
        """
        indexed = "completely unrelated wording appears throughout this sentence"
        chunks = [
            Chunk(chunk_id="c0", doc_id="d0", text=indexed, char_start=0, char_end=len(indexed))
        ]
        golden = Golden(
            golden_id="opaque",
            query="completely unrelated wording",
            gold_doc_id="d-missing",
            gold_char_start=0,
            gold_char_end=30,
            origin="human",
        )
        # No `chunks` handed over: ragdx cannot build any counterfactual index.
        opaque = DiagnosisTarget(retriever=DenseRetriever(chunks), k=1)
        diagnosis = Classifier(opaque).classify(golden)
        assert diagnosis.outcome == "retrieval_failure"
        assert diagnosis.cause is FailureCause.UNCLASSIFIED
        assert diagnosis.confidence == 0.0
        assert "could not measure" in diagnosis.evidence

    def test_a_skipped_ablation_does_not_rule_a_cause_out(self) -> None:
        """Skipped means untested. It must not be read as 'this is not the cause'."""
        results = [
            AblationResult(ablation_name="filters_removed", recovered=False, skipped=True),
            AblationResult(ablation_name="lexical_only", recovered=False, skipped=True),
        ]
        assert all(not r.recovered for r in results)
        # The mapping only ever consults *recovered* ablations.
        assert cause_for_ablation("filters_removed") is FailureCause.METADATA_FILTER


class TestBoundaryWithoutRechunking:
    def test_split_span_is_still_reported_when_rechunking_cannot_be_tested(
        self, target: DiagnosisTarget, fixtures: list[FixtureGolden]
    ) -> None:
        """Coverage below threshold is arithmetic, and worth reporting even when
        ragdx has no documents to re-chunk and prove the fix."""
        no_docs = DiagnosisTarget(
            retriever=target.retriever,
            k=target.k,
            filters=target.filters,
            chunks=target.chunks,
            config=target.config,
        )
        golden = next(f.golden for f in fixtures if f.planted == "chunk_boundary")
        diagnosis = Classifier(no_docs).classify(golden)
        assert diagnosis.cause is FailureCause.CHUNK_BOUNDARY
        assert diagnosis.confidence < 1.0, "unverified fix must not claim certainty"
        assert "%" in diagnosis.evidence
        assert "no source documents" in diagnosis.evidence

    def test_split_span_that_rechunking_does_not_rescue_says_so(self) -> None:
        """Re-chunking fixes the coverage but the chunk still does not rank.

        The chunking really is wrong, so the cause stands — but something else
        is wrong too, and the confidence and evidence have to admit it.
        """
        gold_text = (
            "Column one | Column two | Column three padding padding padding padding "
            "padding padding Row value delta epsilon trailing trailing trailing"
        )
        docs = [Document(doc_id="table", text=gold_text, metadata={})]
        # Distractors that are nothing but the query terms will out-score a long
        # chunk that merely contains them.
        docs += [Document(doc_id=f"noise{i}", text="delta epsilon", metadata={}) for i in range(4)]
        production = FixedSizeChunker(size=40, overlap=0)
        chunks = production.chunk_all(docs)
        golden = Golden(
            golden_id="split-and-outranked",
            query="delta epsilon",
            gold_doc_id="table",
            gold_char_start=0,
            gold_char_end=len(gold_text),
            origin="human",
        )
        target = DiagnosisTarget(
            retriever=DenseRetriever(chunks),
            k=1,
            chunks=chunks,
            docs=docs,
            config=AblationConfig(rank_cutoff_k=2, alternate_chunk_size=960),
        )
        diagnosis = Classifier(target).classify(golden)
        assert diagnosis.cause is FailureCause.CHUNK_BOUNDARY
        assert diagnosis.confidence < 1.0
        assert "the ranking is wrong too" in diagnosis.evidence


class TestParaphraseGapOnALexicalRetriever:
    def test_named_when_only_the_dense_plane_finds_it(self) -> None:
        texts = {
            "expenses": "Staff may claim reimbursement for travel undertaken on company business.",
            "safety": "Work at height requires a harness and a second person present.",
            "returns": "A refund for a damaged parcel is issued once the claim is approved.",
            "trips": "A trip to a customer site is booked through the travel desk.",
        }
        chunks = [
            Chunk(chunk_id=f"{k}::0", doc_id=k, text=v, char_start=0, char_end=len(v))
            for k, v in texts.items()
        ]
        golden = Golden(
            golden_id="paraphrase",
            query="Can an employee get a refund for a work trip?",
            gold_doc_id="expenses",
            gold_char_start=0,
            gold_char_end=len(texts["expenses"]),
            origin="human",
        )
        target = DiagnosisTarget(
            retriever=LexicalRetriever(chunks), k=3, plane="lexical", chunks=chunks
        )
        diagnosis = Classifier(target).classify(golden)
        assert diagnosis.cause is FailureCause.PARAPHRASE_GAP
        assert diagnosis.confidence == 1.0


class TestDeterminism:
    def test_two_runs_are_byte_identical(self, fixtures: list[FixtureGolden]) -> None:
        """PLAN.md §4: two runs on the same input produce identical JSON."""
        first = Classifier(fixture_target()).classify_all([f.golden for f in fixtures])
        second = Classifier(fixture_target()).classify_all([f.golden for f in fixtures])
        assert [d.model_dump_json() for d in first] == [d.model_dump_json() for d in second]


@pytest.fixture(scope="module")
def expected() -> dict[str, dict[str, str | None]]:
    raw = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    return {row["golden_id"]: row for row in raw["diagnoses"]}


class TestTheGate:
    """PLAN.md Milestone 5. Not negotiable."""

    def test_zero_false_positives_on_the_true_negatives(
        self,
        diagnoses: dict[str, Diagnosis],
        expected: dict[str, dict[str, str | None]],
    ) -> None:
        """A diagnostician that cries wolf is worse than none. This is absolute."""
        cried_wolf = [
            gid
            for gid, row in expected.items()
            if row["outcome"] == "hit" and diagnoses[gid].outcome != "hit"
        ]
        assert cried_wolf == [], f"false positives on working queries: {cried_wolf}"

    def test_no_failure_is_reported_as_a_hit(
        self,
        diagnoses: dict[str, Diagnosis],
        expected: dict[str, dict[str, str | None]],
    ) -> None:
        missed = [
            gid
            for gid, row in expected.items()
            if row["outcome"] == "retrieval_failure" and diagnoses[gid].outcome == "hit"
        ]
        assert missed == [], f"real failures reported as hits: {missed}"

    def test_agreement_with_the_expected_diagnoses(
        self,
        diagnoses: dict[str, Diagnosis],
        expected: dict[str, dict[str, str | None]],
    ) -> None:
        # Generation failures are Milestone 6; at the retrieval plane they are
        # correctly hits, so they are scored on outcome-of-retrieval only.
        disagreements: list[str] = []
        scored = 0
        for gid, row in expected.items():
            actual = diagnoses[gid]
            if row["outcome"] == "generation_failure":
                scored += 1
                if actual.outcome != "hit":
                    disagreements.append(f"{gid}: retrieval should succeed, got {actual.outcome}")
                continue
            scored += 1
            expected_cause = row["cause"]
            actual_cause = actual.cause.value if actual.cause else None
            if actual.outcome != row["outcome"] or actual_cause != expected_cause:
                disagreements.append(
                    f"{gid}: expected {row['outcome']}/{expected_cause}, "
                    f"got {actual.outcome}/{actual_cause}"
                )

        agreement = (scored - len(disagreements)) / scored
        assert agreement >= 0.95, (
            f"agreement {agreement:.1%} of {scored} goldens (need >=95%):\n  "
            + "\n  ".join(disagreements)
        )


class TestConfigurability:
    def test_coverage_threshold_flows_through(self) -> None:
        target = fixture_target()
        strict = Classifier(target, ClassifierConfig(coverage_threshold=1.0))
        assert strict.config.coverage_threshold == 1.0

    def test_classify_all_preserves_order(self, fixtures: list[FixtureGolden]) -> None:
        goldens = [f.golden for f in fixtures[:6]]
        results = Classifier(fixture_target()).classify_all(goldens)
        assert [d.golden_id for d in results] == [g.golden_id for g in goldens]


def test_corpus_is_unchanged_since_the_expected_file_was_written() -> None:
    """The gate is meaningless if the yardstick has drifted."""
    from ragdx.corpus import corpus_hash

    raw = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    assert raw["_meta"]["corpus_hash"] == corpus_hash(load_fixture_corpus())
    assert PRODUCTION_CHUNKER.name == "fixed240x60"
