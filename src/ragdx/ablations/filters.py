"""Filters-removed ablation.

Cheapest and most specific: one retrieval with the metadata filter dropped. If
the gold chunk appears, the retriever was never allowed to see it — the ranking
was fine and the filter was wrong.
"""

from __future__ import annotations

from ragdx.ablations.base import Ablation, DiagnosisTarget, skipped
from ragdx.matching import gold_rank
from ragdx.schema import AblationResult, Golden

NAME = "filters_removed"


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
        return bool(target.filters) and target.satisfiable(golden)

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        if not target.filters:
            return skipped(NAME, "production retrieval applies no filters")
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
