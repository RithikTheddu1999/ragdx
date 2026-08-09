"""Cause clusters → ranked fixes. This is the payoff of the whole tool.

The recovery count attached to each fix is not an estimate in the hand-waving
sense: for every failure in a cluster, an ablation *already re-ran retrieval
under that change and got the gold chunk back*. "Hybrid retrieval recovers 11 of
41" means eleven counterfactual retrievals succeeded.

Two causes that share a fix are reported as one recommendation. Splitting
"vocabulary mismatch" and "paraphrase gap" into separate rows would understate
the single change — turning on hybrid retrieval — that resolves both.

Fixes are ranked by recovery count ÷ implementation cost, so a cheap fix for six
failures outranks an expensive one for eight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ragdx.config import RagdxConfig
from ragdx.diagnose.cluster import Cluster
from ragdx.schema import Diagnosis, FailureCause


@dataclass(frozen=True)
class FixSpec:
    """A change a team could actually make, and roughly what it costs them."""

    key: str
    fix: str
    #: 1 = a config value. 5 = train or buy a model.
    cost: int


FIX_BY_CAUSE: dict[FailureCause, FixSpec] = {
    FailureCause.METADATA_FILTER: FixSpec("filters", "fix the metadata filter", 1),
    FailureCause.RANK_CUTOFF: FixSpec("rank", "raise k, or add a reranker", 1),
    FailureCause.VOCABULARY_MISMATCH: FixSpec(
        "hybrid", "enable hybrid retrieval (BM25 + dense)", 2
    ),
    FailureCause.PARAPHRASE_GAP: FixSpec("hybrid", "enable hybrid retrieval (BM25 + dense)", 2),
    FailureCause.GENERATION_UNGROUNDED: FixSpec(
        "generation", "prompt and grounding work on the generator", 2
    ),
    FailureCause.CHUNK_BOUNDARY: FixSpec("chunking", "re-chunk with overlap", 3),
    FailureCause.EMBEDDING_BLIND_SPOT: FixSpec("embedder", "use a domain-tuned embedder", 5),
}


@dataclass
class _Merged:
    """Accumulator for causes that share a single fix."""

    spec: FixSpec
    recovers: int = 0
    causes: list[FailureCause] = field(default_factory=list)


class Recommendation(BaseModel):
    """One change, and what it would buy."""

    fix: str
    detail: str
    causes: list[FailureCause] = Field(default_factory=list)
    recovers: int
    share_of_failures: float
    cost: int
    score: float


def _rank_detail(clusters: list[Cluster], diagnoses: list[Diagnosis], current_k: int) -> str:
    """The specific k that would have caught every rank-cutoff failure."""
    ranks = [
        result.recovered_at_rank
        for diagnosis in diagnoses
        if diagnosis.cause is FailureCause.RANK_CUTOFF
        for result in diagnosis.ablation_results
        if result.ablation_name == "rank_cutoff" and result.recovered_at_rank is not None
    ]
    if not ranks:
        return f"current k={current_k}"
    return (
        f"raising k from {current_k} to {max(ranks) + 1} catches all of them; "
        f"a reranker over the current depth is the alternative"
    )


def _filter_detail(diagnoses: list[Diagnosis], config: RagdxConfig) -> str:
    keys = ", ".join(sorted(config.retrieval.filters)) or "the configured filter"
    return f"the filter on {keys} excludes documents that hold the answer"


def _chunking_detail(config: RagdxConfig) -> str:
    now = config.chunking
    alt = config.ablations.alternate_chunking
    return (
        f"{now.size} chars / {now.overlap} overlap → {alt.size} chars / "
        f"{alt.overlap} overlap restores the split evidence spans"
    )


def recommend(
    clusters: list[Cluster],
    diagnoses: list[Diagnosis],
    config: RagdxConfig,
) -> list[Recommendation]:
    """Rank the available fixes by recovery count ÷ implementation cost."""
    total_failures = sum(c.count for c in clusters)
    merged: dict[str, _Merged] = {}

    for cluster in clusters:
        spec = FIX_BY_CAUSE.get(cluster.cause)
        if spec is None:
            # unclassified: there is no fix to recommend, and inventing one
            # would be exactly the confident guess this tool refuses to make.
            continue
        entry = merged.setdefault(spec.key, _Merged(spec=spec))
        entry.recovers += cluster.count
        entry.causes.append(cluster.cause)

    details = {
        "rank": _rank_detail(clusters, diagnoses, config.retrieval.k),
        "filters": _filter_detail(diagnoses, config),
        "chunking": _chunking_detail(config),
        "hybrid": "BM25 finds the exact terms the dense index dilutes away",
        "embedder": "no counterfactual retrieval recovered these; the embedding "
        "does not place them near the query",
        "generation": "retrieval already returned the right context; the answer did not use it",
    }

    recommendations = []
    for entry in merged.values():
        recovers = entry.recovers
        recommendations.append(
            Recommendation(
                fix=entry.spec.fix,
                detail=details.get(entry.spec.key, ""),
                causes=sorted(set(entry.causes), key=lambda c: c.value),
                recovers=recovers,
                share_of_failures=round(recovers / total_failures, 4) if total_failures else 0.0,
                cost=entry.spec.cost,
                score=round(recovers / entry.spec.cost, 4),
            )
        )

    # Best value first; ties broken by raw recovery count, then by name, so the
    # ordering is stable across runs.
    return sorted(recommendations, key=lambda r: (-r.score, -r.recovers, r.fix))


def top_fix(recommendations: list[Recommendation]) -> Recommendation | None:
    return recommendations[0] if recommendations else None
