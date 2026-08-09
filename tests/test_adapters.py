"""Adapters: the generic trace file, and the two framework wrappers.

The framework adapters are exercised against hand-built stand-ins that mimic the
LangChain and LlamaIndex return shapes, so these run offline. Tests against the
real libraries are marked ``network`` and deselected by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from ragdx.ablations import DiagnosisTarget, FiltersRemoved, RankCutoff, run_battery
from ragdx.adapters.base import Replayed, Retriever
from ragdx.adapters.langchain import LangChainRetrieverAdapter, OffsetsUnavailableError
from ragdx.adapters.llamaindex import LlamaIndexRetrieverAdapter
from ragdx.adapters.trace_file import (
    TraceFormatError,
    TraceReplayRetriever,
    answers_by_golden,
    load_traces,
    write_traces,
)
from ragdx.config import load_config
from ragdx.corpus import Document
from ragdx.diagnose.classifier import Classifier
from ragdx.index import DenseRetriever
from ragdx.runner import run as ragdx_run
from ragdx.schema import FailureCause, Trace
from support import (
    CORPUS_DIR,
    PRODUCTION_CHUNKER,
    PRODUCTION_FILTERS,
    PRODUCTION_K,
    load_fixture_corpus,
    load_fixture_goldens,
)


def _traces_for_fixture(depth: int = PRODUCTION_K) -> list[Trace]:
    """What the fixture's production retriever would have recorded."""
    docs = load_fixture_corpus()
    chunks = PRODUCTION_CHUNKER.chunk_all(docs)
    retriever = DenseRetriever(chunks)
    traces = []
    for fx in load_fixture_goldens(docs):
        traces.append(
            Trace(
                trace_id=f"t-{fx.golden.golden_id}",
                query=fx.golden.query,
                retrieved=retriever.retrieve(fx.golden.query, depth, PRODUCTION_FILTERS),
                answer=fx.answer,
                config_snapshot={
                    "k": PRODUCTION_K,
                    "retriever": "dense",
                    "filters": dict(PRODUCTION_FILTERS),
                },
            )
        )
    return traces


class TestTraceFile:
    def test_round_trips_through_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        traces = _traces_for_fixture()
        write_traces(path, traces)
        assert [t.trace_id for t in load_traces(path)] == [t.trace_id for t in traces]

    def test_rejects_malformed_lines_with_a_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text('{"trace_id": "a", "query": "q"}\n{"nope": 1}\n', encoding="utf-8")
        with pytest.raises(TraceFormatError, match=r"traces\.jsonl:2"):
            load_traces(path)

    def test_rejects_an_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(TraceFormatError, match="no traces"):
            load_traces(path)

    def test_replays_recorded_results_and_renumbers_ranks(self) -> None:
        replay = TraceReplayRetriever(_traces_for_fixture())
        golden = load_fixture_goldens()[0].golden
        results = replay.retrieve(golden.query, 3)
        assert [r.rank for r in results] == [0, 1, 2]

    def test_an_unrecorded_query_returns_nothing_rather_than_a_neighbour(self) -> None:
        replay = TraceReplayRetriever(_traces_for_fixture())
        assert replay.retrieve("a query nobody ever ran", 5) == []

    def test_satisfies_the_retriever_and_replayed_protocols(self) -> None:
        replay = TraceReplayRetriever(_traces_for_fixture())
        assert isinstance(replay, Retriever)
        assert isinstance(replay, Replayed)
        assert replay.recorded_depth == PRODUCTION_K
        assert replay.recorded_filters == dict(PRODUCTION_FILTERS)

    def test_recorded_answers_map_onto_golden_ids(self) -> None:
        replay = TraceReplayRetriever(_traces_for_fixture())
        fixtures = load_fixture_goldens()
        answers = answers_by_golden(replay, {f.golden.golden_id: f.golden.query for f in fixtures})
        assert set(answers) == {f.golden.golden_id for f in fixtures if f.answer}


class TestReplayIsHonestAboutItsLimits:
    def _target(self) -> DiagnosisTarget:
        docs = load_fixture_corpus()
        chunks = PRODUCTION_CHUNKER.chunk_all(docs)
        from ragdx.ablations import AblationConfig

        return DiagnosisTarget(
            retriever=TraceReplayRetriever(_traces_for_fixture()),
            k=PRODUCTION_K,
            filters=dict(PRODUCTION_FILTERS),
            chunks=chunks,
            docs=docs,
            config=AblationConfig(rank_cutoff_k=20),
        )

    def test_rank_cutoff_refuses_to_guess_beyond_the_recording(self) -> None:
        golden = next(f.golden for f in load_fixture_goldens() if f.planted == "rank_cutoff")
        result = RankCutoff().run(self._target(), golden)
        assert result.skipped
        assert "replayed from a recording" in result.detail
        assert not result.recovered

    def test_filters_removed_cannot_be_re_run(self) -> None:
        golden = next(f.golden for f in load_fixture_goldens() if f.planted == "metadata_filter")
        result = FiltersRemoved().run(self._target(), golden)
        assert result.skipped
        assert "cannot be re-run" in result.detail

    def test_corpus_only_ablations_still_work(self) -> None:
        """BM25 and re-chunking ask about the corpus, not the retriever."""
        target = self._target()
        for planted, ablation in (
            ("vocabulary_mismatch", "lexical_only"),
            ("chunk_boundary", "alternate_chunking"),
        ):
            golden = next(f.golden for f in load_fixture_goldens() if f.planted == planted)
            results = run_battery(target, golden)
            recovered = [r for r in results if r.recovered]
            assert [r.ablation_name for r in recovered] == [ablation], planted

    def test_filter_exclusion_is_still_diagnosed_from_metadata(self) -> None:
        """The ablation cannot run, but the metadata settles it anyway."""
        golden = next(f.golden for f in load_fixture_goldens() if f.planted == "metadata_filter")
        diagnosis = Classifier(self._target()).classify(golden)
        assert diagnosis.cause is FailureCause.METADATA_FILTER
        assert diagnosis.confidence < 1.0, "unverified ranking must not claim certainty"
        assert "could not re-run" in diagnosis.evidence

    def test_rank_cutoff_failures_degrade_to_unclassified_not_a_wrong_label(self) -> None:
        """Better to say nothing than to blame the embedding for a shallow k."""
        golden = next(f.golden for f in load_fixture_goldens() if f.planted == "rank_cutoff")
        diagnosis = Classifier(self._target()).classify(golden)
        assert diagnosis.cause in {FailureCause.UNCLASSIFIED, FailureCause.EMBEDDING_BLIND_SPOT}


class TestRunFromTraces:
    def test_end_to_end(self, tmp_path: Path) -> None:
        fixtures = load_fixture_goldens()
        write_traces(tmp_path / "traces.jsonl", _traces_for_fixture())
        (tmp_path / "goldens.jsonl").write_text(
            "\n".join(f.golden.model_dump_json() for f in fixtures) + "\n", encoding="utf-8"
        )
        (tmp_path / "ragdx.yaml").write_text(
            yaml.safe_dump(
                {
                    "corpus": str(CORPUS_DIR),
                    "goldens": str(tmp_path / "goldens.jsonl"),
                    "traces": str(tmp_path / "traces.jsonl"),
                    "retrieval": {"k": 5, "filters": {"status": "current"}},
                    "chunking": {"size": 240, "overlap": 60},
                    "judge": "support:scripted_faithfulness_judge",
                }
            ),
            encoding="utf-8",
        )
        report = ragdx_run(load_config(tmp_path / "ragdx.yaml")).report
        assert report.summary.n_goldens == len(fixtures)
        assert report.config_snapshot["source"] == "traces"
        # Recorded answers came along with the traces, no answers file needed.
        assert report.summary.n_generation_failures == 2

    def test_goldens_with_no_matching_trace_are_warned_about(self, tmp_path: Path) -> None:
        write_traces(tmp_path / "traces.jsonl", _traces_for_fixture()[:3])
        fixtures = load_fixture_goldens()
        (tmp_path / "goldens.jsonl").write_text(
            "\n".join(f.golden.model_dump_json() for f in fixtures) + "\n", encoding="utf-8"
        )
        (tmp_path / "ragdx.yaml").write_text(
            yaml.safe_dump(
                {
                    "corpus": str(CORPUS_DIR),
                    "goldens": str(tmp_path / "goldens.jsonl"),
                    "traces": str(tmp_path / "traces.jsonl"),
                }
            ),
            encoding="utf-8",
        )
        report = ragdx_run(load_config(tmp_path / "ragdx.yaml")).report
        assert any("no matching trace" in w for w in report.warnings)


@dataclass
class FakeLangChainDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeLangChainRetriever:
    """Mimics BaseRetriever closely enough to exercise the adapter offline."""

    def __init__(self, documents: list[FakeLangChainDocument]) -> None:
        self.documents = documents
        self.search_kwargs: dict[str, Any] = {"k": 4}
        self.seen: list[dict[str, Any]] = []

    def invoke(self, input: str, config: Any = None, **kwargs: Any) -> list[Any]:
        self.seen.append(dict(self.search_kwargs))
        return self.documents[: self.search_kwargs.get("k", 4)]


class TestLangChainAdapter:
    def _documents(self) -> list[FakeLangChainDocument]:
        return [
            FakeLangChainDocument(
                page_content="Refunds are paid within ten business days.",
                metadata={"doc_id": "policy", "char_start": 0, "char_end": 41},
            ),
            FakeLangChainDocument(
                page_content="Returns travel on Standard Ground.",
                metadata={"doc_id": "returns", "char_start": 10, "char_end": 44},
            ),
        ]

    def test_maps_documents_to_retrieved_chunks(self) -> None:
        adapter = LangChainRetrieverAdapter(FakeLangChainRetriever(self._documents()))
        results = adapter.retrieve("how fast are refunds paid?", k=2)
        assert isinstance(adapter, Retriever)
        assert [r.rank for r in results] == [0, 1]
        assert results[0].chunk.doc_id == "policy"
        assert (results[0].chunk.char_start, results[0].chunk.char_end) == (0, 41)
        assert results[0].score > results[1].score

    def test_passes_k_and_filters_through_search_kwargs(self) -> None:
        retriever = FakeLangChainRetriever(self._documents())
        adapter = LangChainRetrieverAdapter(retriever)
        adapter.retrieve("q", k=1, filters={"status": "current"})
        assert retriever.seen[-1]["k"] == 1
        assert retriever.seen[-1]["filter"] == {"status": "current"}

    def test_restores_search_kwargs_afterwards(self) -> None:
        retriever = FakeLangChainRetriever(self._documents())
        LangChainRetrieverAdapter(retriever).retrieve("q", k=1, filters={"a": "b"})
        assert retriever.search_kwargs == {"k": 4}

    def test_falls_back_to_locating_text_in_the_source_document(self) -> None:
        text = "Preamble. Refunds are paid within ten business days. Postscript."
        documents = [
            FakeLangChainDocument(
                page_content="Refunds are paid within ten business days.",
                metadata={"doc_id": "policy"},
            )
        ]
        adapter = LangChainRetrieverAdapter(
            FakeLangChainRetriever(documents), docs=[Document(doc_id="policy", text=text)]
        )
        chunk = adapter.retrieve("q", k=1)[0].chunk
        assert text[chunk.char_start : chunk.char_end] == documents[0].page_content

    def test_missing_offsets_fail_loudly_with_an_actionable_message(self) -> None:
        documents = [
            FakeLangChainDocument(page_content="orphan text", metadata={"doc_id": "policy"})
        ]
        adapter = LangChainRetrieverAdapter(FakeLangChainRetriever(documents))
        with pytest.raises(OffsetsUnavailableError, match="char_start"):
            adapter.retrieve("q", k=1)


@dataclass
class FakeNode:
    text: str
    start_char_idx: int | None = None
    end_char_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    node_id: str = "n0"

    def get_content(self) -> str:
        return self.text


@dataclass
class FakeScoredNode:
    node: FakeNode
    score: float


class FakeLlamaRetriever:
    def __init__(self, nodes: list[FakeScoredNode]) -> None:
        self.nodes = nodes
        self.similarity_top_k = 4
        self.seen: list[int] = []

    def retrieve(self, str_or_query_bundle: Any) -> list[Any]:
        self.seen.append(self.similarity_top_k)
        return self.nodes[: self.similarity_top_k]


class TestLlamaIndexAdapter:
    def _nodes(self) -> list[FakeScoredNode]:
        return [
            FakeScoredNode(
                FakeNode(
                    "Refunds are paid within ten business days.", 0, 41, {"doc_id": "policy"}, "n0"
                ),
                score=0.82,
            ),
            FakeScoredNode(
                FakeNode("Returns travel on Standard Ground.", 10, 44, {"doc_id": "returns"}, "n1"),
                score=0.41,
            ),
        ]

    def test_maps_nodes_and_keeps_their_scores(self) -> None:
        adapter = LlamaIndexRetrieverAdapter(FakeLlamaRetriever(self._nodes()))
        results = adapter.retrieve("q", k=2)
        assert isinstance(adapter, Retriever)
        assert results[0].score == pytest.approx(0.82)
        assert results[0].chunk.chunk_id == "n0"
        assert (results[0].chunk.char_start, results[0].chunk.char_end) == (0, 41)

    def test_sets_similarity_top_k_and_restores_it(self) -> None:
        retriever = FakeLlamaRetriever(self._nodes())
        LlamaIndexRetrieverAdapter(retriever).retrieve("q", k=1)
        assert retriever.seen[-1] == 1
        assert retriever.similarity_top_k == 4

    def test_falls_back_to_locating_node_text(self) -> None:
        text = "Preamble. Refunds are paid within ten business days."
        nodes = [
            FakeScoredNode(
                FakeNode(
                    "Refunds are paid within ten business days.", None, None, {"doc_id": "policy"}
                ),
                score=0.5,
            )
        ]
        adapter = LlamaIndexRetrieverAdapter(
            FakeLlamaRetriever(nodes), docs=[Document(doc_id="policy", text=text)]
        )
        chunk = adapter.retrieve("q", k=1)[0].chunk
        assert chunk.char_start == 10


@pytest.mark.network
class TestRealFrameworks:
    """Deselected by default; run with `pytest -m network`."""

    def test_langchain_in_memory_retriever(self) -> None:
        langchain_core = pytest.importorskip("langchain_core")
        from langchain_core.documents import Document as LCDocument
        from langchain_core.retrievers import BaseRetriever

        assert langchain_core is not None

        class Fixed(BaseRetriever):  # type: ignore[misc]
            def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[LCDocument]:
                return [
                    LCDocument(
                        page_content="Refunds are paid within ten business days.",
                        metadata={"doc_id": "policy", "char_start": 0, "char_end": 41},
                    )
                ]

        results = LangChainRetrieverAdapter(Fixed()).retrieve("refunds", k=1)
        assert results[0].chunk.doc_id == "policy"

    def test_llamaindex_node_shapes(self) -> None:
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import NodeWithScore, TextNode

        node = NodeWithScore(
            node=TextNode(
                text="Refunds are paid within ten business days.",
                start_char_idx=0,
                end_char_idx=41,
                metadata={"doc_id": "policy"},
            ),
            score=0.9,
        )

        class Fixed:
            similarity_top_k = 1

            def retrieve(self, query: Any) -> list[Any]:
                return [node]

        results = LlamaIndexRetrieverAdapter(Fixed()).retrieve("refunds", k=1)
        assert results[0].chunk.char_end == 41


def test_adapter_modules_do_not_import_their_frameworks() -> None:
    """PLAN.md §4: the core package must import and run with neither installed.

    Both adapter modules are imported here; if either pulled its framework in at
    module level, that framework would appear in ``sys.modules`` afterwards.
    """
    import subprocess
    import sys

    probe = (
        "import sys, ragdx, ragdx.adapters.langchain, ragdx.adapters.llamaindex, ragdx.cli; "
        "leaked = [m for m in ('langchain_core', 'langchain', 'llama_index') "
        "if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"adapter modules leaked imports: {result.stdout}"
