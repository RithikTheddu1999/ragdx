"""Milestone 6: separating the generator's fault from the retriever's.

The judge is consulted only when the gold chunk was retrieved. Everything here
checks that boundary holds, and that an unsure judge never invents a failure.
"""

from __future__ import annotations

import pytest

from ragdx.diagnose.classifier import Classifier, ClassifierConfig
from ragdx.judge.base import JudgeVerdict, StubJudge
from ragdx.judge.faithfulness import (
    FAITHFULNESS_LABELS,
    GROUNDED,
    UNGROUNDED,
    assess_faithfulness,
    faithfulness_prompt,
)
from ragdx.schema import Chunk, Diagnosis, FailureCause, RetrievedChunk
from support import (
    FixtureGolden,
    ScriptedFaithfulnessJudge,
    fixture_answers,
    fixture_target,
    load_fixture_goldens,
    numeric_claims,
)


@pytest.fixture(scope="module")
def fixtures() -> list[FixtureGolden]:
    return load_fixture_goldens()


@pytest.fixture(scope="module")
def diagnoses(fixtures: list[FixtureGolden]) -> dict[str, Diagnosis]:
    classifier = Classifier(fixture_target(), judge=ScriptedFaithfulnessJudge())
    answers = fixture_answers()
    return {
        f.golden.golden_id: classifier.classify(f.golden, answers.get(f.golden.golden_id))
        for f in fixtures
    }


class TestPrompt:
    def test_contains_question_answer_and_context(self) -> None:
        prompt = faithfulness_prompt("q?", "an answer", ["ctx one", "ctx two"])
        assert "QUESTION:\nq?" in prompt
        assert "ANSWER:\nan answer" in prompt
        assert "ctx one" in prompt and "ctx two" in prompt

    def test_tells_the_judge_not_to_penalise_omissions(self) -> None:
        assert "leaving things out" in faithfulness_prompt("q", "a", ["c"])

    def test_labels_are_binary(self) -> None:
        assert FAITHFULNESS_LABELS == (GROUNDED, UNGROUNDED)


class TestAssessment:
    def _retrieved(self, text: str) -> list[RetrievedChunk]:
        chunk = Chunk(chunk_id="c", doc_id="d", text=text, char_start=0, char_end=len(text))
        return [RetrievedChunk(chunk=chunk, score=1.0, rank=0)]

    def test_flags_a_number_the_context_never_states(self) -> None:
        verdict = assess_faithfulness(
            ScriptedFaithfulnessJudge(),
            "How fast are claims paid?",
            "Claims are paid within 48 hours.",
            self._retrieved("Approved claims are paid with the next payroll run."),
        )
        assert verdict.label == UNGROUNDED
        assert "48" in verdict.rationale

    def test_accepts_an_answer_the_context_supports(self) -> None:
        verdict = assess_faithfulness(
            ScriptedFaithfulnessJudge(),
            "How long are records kept?",
            "Records are kept for seven years.",
            self._retrieved("Shipment records are retained for seven years."),
        )
        assert verdict.label == GROUNDED

    def test_abstains_with_no_retrieved_context(self) -> None:
        verdict = assess_faithfulness(ScriptedFaithfulnessJudge(), "q", "an answer", [])
        assert verdict.abstained

    def test_abstains_with_no_answer(self) -> None:
        verdict = assess_faithfulness(
            ScriptedFaithfulnessJudge(), "q", "   ", self._retrieved("context")
        )
        assert verdict.abstained

    def test_numeric_claim_extraction(self) -> None:
        assert numeric_claims("paid within 48 hours or twice a week") == {"48", "twice"}


class TestSeparatingTheTwoPlanes:
    def test_planted_generation_failures_are_labelled(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        planted = [f.golden.golden_id for f in fixtures if f.planted == "generation_failure"]
        assert planted
        for gid in planted:
            diagnosis = diagnoses[gid]
            assert diagnosis.outcome == "generation_failure", diagnosis.evidence
            assert diagnosis.cause is FailureCause.GENERATION_UNGROUNDED
            assert diagnosis.gold_rank is not None, "retrieval succeeded, so record the rank"

    def test_faithful_answers_on_hits_stay_hits(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        answered_hits = [f.golden.golden_id for f in fixtures if f.planted == "hit" and f.answer]
        assert answered_hits
        for gid in answered_hits:
            assert diagnoses[gid].outcome == "hit", diagnoses[gid].evidence

    def test_retrieval_failures_are_never_relabelled(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        """A wrong answer on a failed retrieval says nothing about the generator."""
        for fx in fixtures:
            if fx.planted not in {"hit", "generation_failure"}:
                assert diagnoses[fx.golden.golden_id].outcome == "retrieval_failure"

    def test_evidence_names_which_component_is_at_fault(
        self, diagnoses: dict[str, Diagnosis], fixtures: list[FixtureGolden]
    ) -> None:
        gid = next(f.golden.golden_id for f in fixtures if f.planted == "generation_failure")
        evidence = diagnoses[gid].evidence
        assert "not grounded" in evidence
        assert "generator's fault" in evidence


class TestTheJudgeNeverInventsAFailure:
    def _classify(self, judge: object, config: ClassifierConfig | None = None) -> Diagnosis:
        fixtures = load_fixture_goldens()
        fx = next(f for f in fixtures if f.planted == "generation_failure")
        classifier = Classifier(fixture_target(), config=config, judge=judge)  # type: ignore[arg-type]
        return classifier.classify(fx.golden, fx.answer)

    def test_an_abstaining_judge_leaves_the_hit_standing(self) -> None:
        assert self._classify(ScriptedFaithfulnessJudge(abstain=True)).outcome == "hit"

    def test_the_canned_stub_judge_leaves_the_hit_standing(self) -> None:
        assert self._classify(StubJudge()).outcome == "hit"

    def test_a_low_confidence_verdict_leaves_the_hit_standing(self) -> None:
        class Unsure(ScriptedFaithfulnessJudge):
            def judge(self, prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
                return JudgeVerdict(label=UNGROUNDED, confidence=0.2, rationale="not sure")

        assert self._classify(Unsure()).outcome == "hit"
        # ...and it is the threshold doing the work, not the verdict being ignored.
        lenient = ClassifierConfig(faithfulness_min_confidence=0.1)
        assert self._classify(Unsure(), lenient).outcome == "generation_failure"

    def test_no_judge_configured_means_no_generation_verdict(self) -> None:
        fixtures = load_fixture_goldens()
        fx = next(f for f in fixtures if f.planted == "generation_failure")
        diagnosis = Classifier(fixture_target()).classify(fx.golden, fx.answer)
        assert diagnosis.outcome == "hit"

    def test_no_answer_recorded_means_no_judge_call(self) -> None:
        judge = ScriptedFaithfulnessJudge()
        calls: list[str] = []

        def spy(prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
            calls.append(prompt)
            return JudgeVerdict(label=UNGROUNDED, confidence=0.9)

        judge.judge = spy  # type: ignore[method-assign]
        fixtures = load_fixture_goldens()
        fx = next(f for f in fixtures if f.planted == "hit")
        Classifier(fixture_target(), judge=judge).classify(fx.golden, None)
        assert calls == []

    def test_failed_retrieval_never_reaches_the_judge(self) -> None:
        judge = ScriptedFaithfulnessJudge()
        calls: list[str] = []

        def spy(prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
            calls.append(prompt)
            return JudgeVerdict(label=UNGROUNDED, confidence=0.9)

        judge.judge = spy  # type: ignore[method-assign]
        fixtures = load_fixture_goldens()
        fx = next(f for f in fixtures if f.planted == "rank_cutoff")
        Classifier(fixture_target(), judge=judge).classify(fx.golden, "a wrong answer about 99")
        assert calls == [], "the judge was consulted for a query retrieval already failed"


class TestClassifyAll:
    def test_answers_are_matched_by_golden_id(self, fixtures: list[FixtureGolden]) -> None:
        classifier = Classifier(fixture_target(), judge=ScriptedFaithfulnessJudge())
        results = classifier.classify_all([f.golden for f in fixtures], fixture_answers())
        by_id = {d.golden_id: d for d in results}
        generation = [d for d in results if d.outcome == "generation_failure"]
        assert {d.golden_id for d in generation} == {
            f.golden.golden_id for f in fixtures if f.planted == "generation_failure"
        }
        assert by_id["h01"].outcome == "hit"


def test_the_full_pipeline_reproduces_the_expected_diagnoses_exactly() -> None:
    """With the generation plane wired in, every row of the yardstick matches —
    including the two that are the generator's fault rather than the retriever's."""
    import json

    from support import EXPECTED_PATH

    fixtures = load_fixture_goldens()
    classifier = Classifier(fixture_target(), judge=ScriptedFaithfulnessJudge())
    actual = {
        d.golden_id: d
        for d in classifier.classify_all([f.golden for f in fixtures], fixture_answers())
    }
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))["diagnoses"]

    disagreements: list[str] = []
    for row in expected:
        got = actual[row["golden_id"]]
        got_cause = got.cause.value if got.cause is not None else None
        if got.outcome != row["outcome"] or got_cause != row["cause"]:
            disagreements.append(
                f"{row['golden_id']}: expected {row['outcome']}/{row['cause']}, "
                f"got {got.outcome}/{got_cause}"
            )
    assert disagreements == [], "\n  ".join(disagreements)
