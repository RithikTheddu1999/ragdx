"""Golden synthesis, import, versioned storage and the judge cache."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ragdx.cache import ContentCache, content_key
from ragdx.corpus import Document, corpus_hash
from ragdx.goldens import (
    GoldenSetManifest,
    RejectReason,
    SynthesisConfig,
    available_versions,
    check_corpus,
    import_goldens,
    load,
    next_version,
    save,
    synthesize,
)
from ragdx.judge.base import CachedJudge, JudgeVerdict, StubJudge
from ragdx.judge.loader import JudgeNotFoundError, load_judge
from ragdx.schema import Chunk, Golden
from support import (
    PRODUCTION_CHUNKER,
    ScriptedJudge,
    load_fixture_corpus,
)


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return load_fixture_corpus()


@pytest.fixture(scope="module")
def chunks(docs: list[Document]) -> list[Chunk]:
    return PRODUCTION_CHUNKER.chunk_all(docs)


class TestSynthesis:
    def test_produces_verified_goldens(self, docs: list[Document], chunks: list[Chunk]) -> None:
        batch = synthesize(docs, chunks, ScriptedJudge(), SynthesisConfig(n=8, seed=1))
        assert len(batch.goldens) == 8
        assert all(g.origin == "synthetic" for g in batch.goldens)
        assert all(g.synth_confidence is not None for g in batch.goldens)

    def test_spans_point_at_real_document_text(
        self, docs: list[Document], chunks: list[Chunk]
    ) -> None:
        by_id = {d.doc_id: d for d in docs}
        batch = synthesize(docs, chunks, ScriptedJudge(), SynthesisConfig(n=8, seed=1))
        for g in batch.goldens:
            quoted = by_id[g.gold_doc_id].text[g.gold_char_start : g.gold_char_end]
            assert quoted.strip()
            assert quoted in by_id[g.gold_doc_id].text

    def test_is_deterministic(self, docs: list[Document], chunks: list[Chunk]) -> None:
        config = SynthesisConfig(n=6, seed=7)
        first = synthesize(docs, chunks, ScriptedJudge(), config)
        second = synthesize(docs, chunks, ScriptedJudge(), config)
        assert [g.model_dump_json() for g in first.goldens] == [
            g.model_dump_json() for g in second.goldens
        ]

    def test_seed_changes_the_sample(self, docs: list[Document], chunks: list[Chunk]) -> None:
        a = synthesize(docs, chunks, ScriptedJudge(), SynthesisConfig(n=6, seed=1))
        b = synthesize(docs, chunks, ScriptedJudge(), SynthesisConfig(n=6, seed=2))
        assert [g.query for g in a.goldens] != [g.query for g in b.goldens]

    def test_rejects_candidates_answerable_from_a_distractor(
        self, docs: list[Document], chunks: list[Chunk]
    ) -> None:
        """The rejection that matters: a question answerable anywhere is noise."""
        batch = synthesize(
            docs, chunks, ScriptedJudge(always_answerable=True), SynthesisConfig(n=5, seed=1)
        )
        assert batch.goldens == []
        assert set(batch.counts_by_reason()) == {RejectReason.ANSWERABLE_FROM_DISTRACTOR.value}
        assert batch.rejection_rate == 1.0

    def test_abstention_is_a_rejection_not_a_pass(
        self, docs: list[Document], chunks: list[Chunk]
    ) -> None:
        batch = synthesize(docs, chunks, ScriptedJudge(abstain=True), SynthesisConfig(n=5, seed=1))
        assert batch.goldens == []
        assert set(batch.counts_by_reason()) == {RejectReason.JUDGE_ABSTAINED.value}

    def test_stub_judge_yields_nothing(self, docs: list[Document], chunks: list[Chunk]) -> None:
        """The canned stub answers nothing, so nothing may survive."""
        batch = synthesize(docs, chunks, StubJudge(), SynthesisConfig(n=5, seed=1))
        assert batch.goldens == []
        assert set(batch.counts_by_reason()) == {RejectReason.MALFORMED.value}

    def test_rejects_a_hallucinated_evidence_quote(
        self, docs: list[Document], chunks: list[Chunk]
    ) -> None:
        class Hallucinating(ScriptedJudge):
            def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
                return json.dumps(
                    {
                        "question": "What is the policy?",
                        "evidence": "a sentence that appears nowhere in the corpus",
                        "answer": "unknown",
                    }
                )

        batch = synthesize(docs, chunks, Hallucinating(), SynthesisConfig(n=3, seed=1))
        assert batch.goldens == []
        assert set(batch.counts_by_reason()) == {RejectReason.EVIDENCE_NOT_FOUND.value}

    def test_short_chunks_are_skipped(self, docs: list[Document], chunks: list[Chunk]) -> None:
        batch = synthesize(
            docs, chunks, ScriptedJudge(), SynthesisConfig(n=3, seed=1, min_tokens=10_000)
        )
        assert batch.n_considered == 0


class TestImporter:
    def _corpus(self) -> list[Document]:
        return [Document(doc_id="policy", text="Refunds are paid within ten business days.")]

    def test_imports_jsonl_with_quoted_evidence(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        path.write_text(
            json.dumps(
                {
                    "golden_id": "g1",
                    "query": "How fast are refunds paid?",
                    "doc_id": "policy",
                    "evidence": "within ten business days",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        batch = import_goldens(path, self._corpus())
        assert len(batch.goldens) == 1
        golden = batch.goldens[0]
        assert golden.origin == "human"
        assert self._corpus()[0].text[golden.gold_char_start : golden.gold_char_end] == (
            "within ten business days"
        )

    def test_imports_csv_with_explicit_spans(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.csv"
        path.write_text(
            "query,doc_id,gold_char_start,gold_char_end\nHow fast are refunds paid?,policy,17,41\n",
            encoding="utf-8",
        )
        batch = import_goldens(path, self._corpus())
        assert len(batch.goldens) == 1

    def _reject(self, tmp_path: Path, row: Mapping[str, object]) -> RejectReason:
        path = tmp_path / "labels.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        batch = import_goldens(path, self._corpus())
        assert batch.goldens == []
        return batch.rejections[0].reason

    def test_rejects_unknown_document(self, tmp_path: Path) -> None:
        row = {"query": "q", "doc_id": "nope", "evidence": "Refunds"}
        assert self._reject(tmp_path, row) is RejectReason.UNKNOWN_DOCUMENT

    def test_rejects_missing_evidence_and_span(self, tmp_path: Path) -> None:
        assert self._reject(tmp_path, {"query": "q", "doc_id": "policy"}) is RejectReason.MALFORMED

    def test_rejects_missing_query(self, tmp_path: Path) -> None:
        row = {"doc_id": "policy", "evidence": "Refunds"}
        assert self._reject(tmp_path, row) is RejectReason.MALFORMED

    def test_rejects_unlocatable_evidence(self, tmp_path: Path) -> None:
        row = {"query": "q", "doc_id": "policy", "evidence": "not in the document"}
        assert self._reject(tmp_path, row) is RejectReason.EVIDENCE_NOT_FOUND

    def test_rejects_span_outside_the_document(self, tmp_path: Path) -> None:
        row = {"query": "q", "doc_id": "policy", "gold_char_start": 0, "gold_char_end": 9999}
        assert self._reject(tmp_path, row) is RejectReason.SPAN_OUT_OF_RANGE

    def test_rejects_ambiguous_evidence(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        path.write_text(
            json.dumps({"query": "q", "doc_id": "d", "evidence": "the same"}) + "\n",
            encoding="utf-8",
        )
        docs = [Document(doc_id="d", text="the same and the same again")]
        batch = import_goldens(path, docs)
        assert batch.rejections[0].reason is RejectReason.EVIDENCE_AMBIGUOUS


def _golden(golden_id: str = "g1") -> Golden:
    return Golden(
        golden_id=golden_id,
        query="q",
        gold_doc_id="d",
        gold_char_start=0,
        gold_char_end=5,
        origin="human",
    )


class TestStore:
    def _manifest(self, version: int = 1, digest: str = "abc") -> GoldenSetManifest:
        return GoldenSetManifest(version=version, corpus_hash=digest, n_goldens=1, n_rejected=1)

    def test_round_trips(self, tmp_path: Path) -> None:
        save(tmp_path, [_golden()], self._manifest())
        goldens, manifest = load(tmp_path)
        assert goldens == [_golden()]
        assert manifest.version == 1
        assert manifest.rejection_rate == pytest.approx(0.5)

    def test_versions_increment(self, tmp_path: Path) -> None:
        assert next_version(tmp_path) == 1
        save(tmp_path, [_golden()], self._manifest(version=1))
        assert next_version(tmp_path) == 2
        save(tmp_path, [_golden()], self._manifest(version=2))
        assert available_versions(tmp_path) == [1, 2]

    def test_loads_the_latest_by_default(self, tmp_path: Path) -> None:
        save(tmp_path, [_golden("old")], self._manifest(version=1))
        save(tmp_path, [_golden("new")], self._manifest(version=2))
        assert load(tmp_path)[0][0].golden_id == "new"
        assert load(tmp_path, version=1)[0][0].golden_id == "old"

    def test_refuses_to_overwrite(self, tmp_path: Path) -> None:
        save(tmp_path, [_golden()], self._manifest())
        with pytest.raises(FileExistsError):
            save(tmp_path, [_golden()], self._manifest())

    def test_missing_set_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load(tmp_path)

    def test_file_is_byte_identical_for_the_same_goldens(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        save(a, [_golden("g2"), _golden("g1")], self._manifest())
        save(b, [_golden("g1"), _golden("g2")], self._manifest())
        assert (a / "v1.jsonl").read_bytes() == (b / "v1.jsonl").read_bytes()

    def test_corpus_drift_is_reported(self) -> None:
        docs = [Document(doc_id="d", text="hello")]
        assert check_corpus(self._manifest(digest=corpus_hash(docs)), docs) is None
        message = check_corpus(self._manifest(digest="stale"), docs)
        assert message is not None
        assert "corpus has changed" in message
        assert "rebuild the golden set" in message


class TestCache:
    def test_memory_only_cache(self) -> None:
        cache = ContentCache(namespace="t")
        assert cache.get("k") is None
        cache.put("k", "v")
        assert cache.get("k") == "v"
        assert cache.hits == 1 and cache.misses == 1

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        ContentCache(tmp_path, "judge").put("k", "v")
        assert ContentCache(tmp_path, "judge").get("k") == "v"

    def test_namespaces_are_isolated(self, tmp_path: Path) -> None:
        ContentCache(tmp_path, "a").put("k", "v")
        assert ContentCache(tmp_path, "b").get("k") is None

    def test_content_key_is_stable_and_order_independent(self) -> None:
        assert content_key("a", {"x": 1, "y": 2}) == content_key("a", {"y": 2, "x": 1})
        assert content_key("a") != content_key("b")


class TestCachedJudge:
    def test_replays_instead_of_asking_twice(self, tmp_path: Path) -> None:
        inner = StubJudge()
        inner.set_completion("hello", "world")
        cached = CachedJudge(inner, ContentCache(tmp_path, "judge"))
        assert cached.complete("hello") == "world"
        calls_after_first = len(inner.calls)
        assert cached.complete("hello") == "world"
        assert len(inner.calls) == calls_after_first

    def test_verdicts_survive_a_new_process(self, tmp_path: Path) -> None:
        inner = StubJudge()
        inner.set_verdict("p", JudgeVerdict(label="grounded", confidence=0.7))
        CachedJudge(inner, ContentCache(tmp_path, "judge")).judge("p", ("grounded",))
        fresh = CachedJudge(StubJudge(), ContentCache(tmp_path, "judge"))
        replayed = fresh.judge("p", ("grounded",))
        assert replayed.label == "grounded"
        assert not replayed.abstained


class TestJudgeLoader:
    def test_stub_by_default(self) -> None:
        assert isinstance(load_judge(), StubJudge)

    def test_loads_a_factory_by_dotted_path(self) -> None:
        judge = load_judge("support:scripted_judge")
        assert judge.name == "scripted-judge-v1"

    @pytest.mark.parametrize(
        "spec", ["not-a-path", "ragdx.cli:nope", "no_such_module:thing", "ragdx.cli:app"]
    )
    def test_bad_specs_raise(self, spec: str) -> None:
        with pytest.raises(JudgeNotFoundError):
            load_judge(spec)
