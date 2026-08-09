"""Grouping diagnoses by cause, with worked examples.

A list of fifty individual verdicts is not actionable. Fifty verdicts collapsed
into five causes, each carrying the queries that prove it, is.

Within ``vocabulary_mismatch`` the cluster is sub-grouped by the **offending
term** where one is detectable: the rarest word the query and the gold chunk
share. That is the word BM25 found and the embedding lost, and naming it is the
difference between "you have a vocabulary problem" and "your users search by
part number".
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from ragdx.corpus import Document
from ragdx.matching import chunk_satisfies
from ragdx.schema import Chunk, Diagnosis, FailureCause, Golden
from ragdx.text import tokenize

MAX_EXAMPLES = 5
PREVIEW_CHARS = 160


class ClusterExample(BaseModel):
    """One failing query, with enough context to act on without reading code."""

    golden_id: str
    query: str
    evidence: str
    expected_text: str
    retrieved_preview: list[str] = Field(default_factory=list)
    confidence: float
    subgroup: str | None = None


class Cluster(BaseModel):
    """Every diagnosis sharing one root cause."""

    cause: FailureCause
    count: int
    mean_confidence: float
    subgroups: dict[str, int] = Field(default_factory=dict)
    examples: list[ClusterExample] = Field(default_factory=list)
    golden_ids: list[str] = Field(default_factory=list)


def document_frequency(chunks: list[Chunk]) -> dict[str, int]:
    """How many chunks each token appears in. Used to find the rarest term."""
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(set(tokenize(chunk.text)))
    return dict(df)


def offending_term(
    golden: Golden, chunks: list[Chunk], df: dict[str, int], coverage: float
) -> str | None:
    """The rarest token the query shares with its gold chunk.

    ``None`` when the query and the gold chunk share no vocabulary at all — in
    which case there is no offending term and saying otherwise would be a guess.
    """
    query_tokens = set(tokenize(golden.query))
    if not query_tokens:
        return None
    gold_tokens: set[str] = set()
    for chunk in chunks:
        if chunk_satisfies(chunk, golden, coverage):
            gold_tokens |= set(tokenize(chunk.text))
    shared = query_tokens & gold_tokens
    if not shared:
        return None
    return min(sorted(shared), key=lambda t: df.get(t, 0))


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


def build_clusters(
    diagnoses: list[Diagnosis],
    goldens: list[Golden],
    docs: list[Document],
    chunks: list[Chunk],
    retrieved_by_golden: dict[str, list[str]] | None = None,
    coverage: float = 0.75,
    max_examples: int = MAX_EXAMPLES,
) -> list[Cluster]:
    """Group failing diagnoses by cause, largest cluster first."""
    by_id = {g.golden_id: g for g in goldens}
    by_doc = {d.doc_id: d for d in docs}
    retrieved_by_golden = retrieved_by_golden or {}
    df = document_frequency(chunks)

    grouped: dict[FailureCause, list[Diagnosis]] = {}
    for diagnosis in diagnoses:
        if diagnosis.outcome == "hit" or diagnosis.cause is None:
            continue
        grouped.setdefault(diagnosis.cause, []).append(diagnosis)

    clusters: list[Cluster] = []
    for cause, members in grouped.items():
        members = sorted(members, key=lambda d: d.golden_id)
        subgroups: Counter[str] = Counter()
        examples: list[ClusterExample] = []

        for diagnosis in members:
            golden = by_id.get(diagnosis.golden_id)
            if golden is None:
                continue
            subgroup = None
            if cause is FailureCause.VOCABULARY_MISMATCH:
                subgroup = offending_term(golden, chunks, df, coverage)
                if subgroup:
                    subgroups[subgroup] += 1
            if len(examples) < max_examples:
                doc = by_doc.get(golden.gold_doc_id)
                expected = doc.text[golden.gold_char_start : golden.gold_char_end] if doc else ""
                examples.append(
                    ClusterExample(
                        golden_id=golden.golden_id,
                        query=golden.query,
                        evidence=diagnosis.evidence,
                        expected_text=_preview(expected),
                        retrieved_preview=[
                            _preview(t) for t in retrieved_by_golden.get(golden.golden_id, [])[:3]
                        ],
                        confidence=diagnosis.confidence,
                        subgroup=subgroup,
                    )
                )

        clusters.append(
            Cluster(
                cause=cause,
                count=len(members),
                mean_confidence=round(sum(d.confidence for d in members) / len(members), 4),
                subgroups=dict(sorted(subgroups.items(), key=lambda kv: (-kv[1], kv[0]))),
                examples=examples,
                golden_ids=[d.golden_id for d in members],
            )
        )

    # Largest first, then by cause name so equal-sized clusters never reorder.
    return sorted(clusters, key=lambda c: (-c.count, c.cause.value))
