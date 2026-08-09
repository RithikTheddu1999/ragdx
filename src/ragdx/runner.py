"""Wiring a ``ragdx.yaml`` into a finished report.

Everything here is assembly: load the corpus, chunk it, build the retriever the
config asks for, classify every golden, cluster, recommend, render. The
interesting decisions all live further down in `diagnose/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragdx.ablations.base import DiagnosisTarget
from ragdx.adapters.base import Retriever
from ragdx.adapters.trace_file import (
    TraceReplayRetriever,
    answers_by_golden,
    load_traces,
)
from ragdx.chunking import FixedSizeChunker
from ragdx.config import RagdxConfig
from ragdx.corpus import Document, corpus_hash, load_corpus
from ragdx.diagnose.classifier import Classifier, ClassifierConfig
from ragdx.diagnose.cluster import build_clusters
from ragdx.diagnose.recommend import recommend
from ragdx.embedding import load_embedder
from ragdx.goldens import store
from ragdx.index import DenseRetriever, HybridRetriever, LexicalRetriever
from ragdx.judge.loader import load_judge
from ragdx.report.render import Report, build_summary
from ragdx.schema import Chunk, Diagnosis, Golden


@dataclass
class RunResult:
    """Everything a run produced, before it is written anywhere."""

    report: Report
    diagnoses: list[Diagnosis]
    goldens: list[Golden]
    docs: list[Document]
    chunks: list[Chunk]


def load_goldens(path: Path) -> tuple[list[Golden], str | None]:
    """Load goldens from a versioned store directory or a JSONL file.

    Returns the goldens and the corpus hash they were built against, when the
    source records one.
    """
    if path.is_dir():
        goldens, manifest = store.load(path)
        return goldens, manifest.corpus_hash
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Golden.model_validate_json(line) for line in lines if line.strip()], None


def load_answers(path: Path | None) -> dict[str, str]:
    """Load ``{"golden_id": ..., "answer": ...}`` records, if any."""
    if path is None:
        return {}
    answers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        golden_id, answer = row.get("golden_id"), row.get("answer")
        if golden_id and answer:
            answers[str(golden_id)] = str(answer)
    return answers


def build_retriever(config: RagdxConfig, chunks: list[Chunk]) -> Retriever:
    embedder = load_embedder(config.embedder)
    if config.retrieval.plane == "lexical":
        return LexicalRetriever(chunks)
    if config.retrieval.plane == "hybrid":
        return HybridRetriever(chunks, embedder)
    return DenseRetriever(chunks, embedder)


def build_target(
    config: RagdxConfig,
    docs: list[Document],
    chunks: list[Chunk],
    replay: TraceReplayRetriever | None = None,
) -> DiagnosisTarget:
    """The production setup under diagnosis.

    ``chunks`` and ``docs`` are supplied even in replay mode: the corpus-only
    counterfactuals (BM25, a dense index, re-chunking) ask what a *different
    retrieval strategy over your documents* would have found, which needs no
    access to the retriever that produced the traces.
    """
    return DiagnosisTarget(
        retriever=replay if replay is not None else build_retriever(config, chunks),
        k=config.retrieval.k,
        filters=dict(config.retrieval.filters) or None,
        plane=replay.plane if replay is not None else config.retrieval.plane,
        chunks=chunks,
        docs=docs,
        embedder=load_embedder(config.embedder),
        config=config.ablation_config(),
    )


def _config_snapshot(config: RagdxConfig) -> dict[str, object]:
    return {
        "plane": config.retrieval.plane,
        "source": "traces" if config.traces else "built-in index",
        "k": config.retrieval.k,
        "filters": dict(config.retrieval.filters),
        "chunking": f"{config.chunking.size}/{config.chunking.overlap}",
        "rank_cutoff_k": config.ablations.rank_cutoff_k,
        "alternate_chunking": (
            f"{config.ablations.alternate_chunking.size}/"
            f"{config.ablations.alternate_chunking.overlap}"
        ),
        "embedder": config.embedder,
        "judge": config.judge,
        "coverage_threshold": config.coverage_threshold,
    }


def run(config: RagdxConfig) -> RunResult:
    """Diagnose everything the config points at."""
    docs = load_corpus(config.corpus)
    chunks = FixedSizeChunker(size=config.chunking.size, overlap=config.chunking.overlap).chunk_all(
        docs
    )
    goldens, built_against = load_goldens(config.goldens)
    answers = load_answers(config.answers)

    warnings: list[str] = []
    replay: TraceReplayRetriever | None = None
    if config.traces is not None:
        replay = TraceReplayRetriever(load_traces(config.traces))
        # Recorded answers fill in for any golden the answers file did not cover.
        recorded = answers_by_golden(replay, {g.golden_id: g.query for g in goldens})
        answers = {**recorded, **answers}
        unseen = [g.golden_id for g in goldens if not replay.retrieve(g.query, 1)]
        if unseen:
            warnings.append(
                f"{len(unseen)} of {len(goldens)} goldens have no matching trace "
                f"(matched on exact query text) and will be scored as retrieving "
                f"nothing: {', '.join(unseen[:5])}"
                f"{'…' if len(unseen) > 5 else ''}"
            )
    digest = corpus_hash(docs)
    if built_against and built_against != digest:
        warnings.append(
            f"The corpus has changed since this golden set was built "
            f"(golden set {built_against[:12]}, corpus {digest[:12]}). Evidence "
            f"spans may no longer point at the right text — rebuild the golden "
            f"set before trusting these numbers."
        )

    target = build_target(config, docs, chunks, replay)
    judge = load_judge(config.judge) if answers else None
    classifier = Classifier(
        target,
        config=ClassifierConfig(coverage_threshold=config.coverage_threshold),
        judge=judge,
    )
    diagnoses = classifier.classify_all(goldens, answers)

    # Only failures need their retrieved context shown in the report.
    failed = {d.golden_id for d in diagnoses if d.outcome != "hit"}
    retrieved_by_golden = {
        g.golden_id: [
            item.chunk.text for item in target.retriever.retrieve(g.query, target.k, target.filters)
        ]
        for g in goldens
        if g.golden_id in failed
    }

    clusters = build_clusters(
        diagnoses,
        goldens,
        docs,
        chunks,
        retrieved_by_golden=retrieved_by_golden,
        coverage=config.coverage_threshold,
    )
    report = Report(
        summary=build_summary(diagnoses, config.retrieval.k),
        clusters=clusters,
        recommendations=recommend(clusters, diagnoses, config),
        corpus_hash=digest,
        n_documents=len(docs),
        n_chunks=len(chunks),
        config_snapshot=_config_snapshot(config),
        warnings=warnings,
    )
    return RunResult(report=report, diagnoses=diagnoses, goldens=goldens, docs=docs, chunks=chunks)
