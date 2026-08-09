"""Filters-removed ablation.

Cheapest and most specific: one retrieval with the metadata filter dropped. If
the gold chunk appears, the retriever was never allowed to see it — the ranking
was fine and the filter was wrong.
"""

from __future__ import annotations

from ragdx.ablations.base import Ablation, DiagnosisTarget, skipped
from ragdx.adapters.base import Replayed
from ragdx.index import matches_filters
from ragdx.matching import chunk_satisfies, gold_rank
from ragdx.schema import AblationResult, Golden

NAME = "filters_removed"


def excluded_by_filters(target: DiagnosisTarget, golden: Golden) -> bool:
    """True when the production filter makes the gold chunk unreturnable.

    Pure arithmetic over metadata: if no chunk that covers the evidence span
    satisfies the filter, the retriever was never permitted to return one, and
    no amount of reranking or re-embedding changes that. Used when the
    production retriever cannot be re-run.
    """
    if not target.filters or not target.chunks:
        return False
    covering = [
        c for c in target.chunks if chunk_satisfies(c, golden, target.config.coverage_threshold)
    ]
    return bool(covering) and not any(matches_filters(c, target.filters) for c in covering)


class FiltersRemoved(Ablation):
    """Re-run production retrieval with no metadata filter."""

    @property
    def name(self) -> str:
        return NAME

    @property
    def cost(self) -> int:
        return 1

    def applicable(self, target: DiagnosisTarget, golden: Golden) -> bool:
        # Nothing to remove, and nothing a filter change could fix if the
        # chunker cannot produce a satisfying chunk in the first place.
        return (
            bool(target.filters)
            and not isinstance(target.retriever, Replayed)
            and target.satisfiable(golden)
        )

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        if not target.filters:
            return skipped(NAME, "production retrieval applies no filters")
        if isinstance(target.retriever, Replayed):
            # The classifier can still reach `metadata_filter` from the gold
            # document's metadata alone — see excluded_by_filters() — but that
            # is an inference, not a retrieval that was re-run, and it is not
            # this ablation's job to blur the two.
            return skipped(
                NAME,
                "retrieval is replayed from a recording; the production retriever "
                "cannot be re-run with the filter removed",
            )
        if not target.satisfiable(golden):
            return skipped(NAME, "no chunk covers the evidence span under this chunking")

        results = target.retriever.retrieve(golden.query, target.k, None)
        rank = gold_rank(results, golden, target.config.coverage_threshold)
        keys = ", ".join(sorted(target.filters))
        if rank is None:
            return AblationResult(
                ablation_name=NAME,
                recovered=False,
                detail=f"still missing with filters ({keys}) removed",
            )
        return AblationResult(
            ablation_name=NAME,
            recovered=True,
            recovered_at_rank=rank,
            detail=f"recovered at rank {rank} once the filter on {keys} was dropped",
        )
