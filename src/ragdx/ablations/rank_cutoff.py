"""Rank-cutoff ablation.

One retrieval at a much larger `k`. If the gold chunk appears, it was ranked all
along and simply fell below the production cutoff — a reranker or a larger `k`
recovers it, and nothing about the embedding or the chunking needs to change.

The ablation refuses to run when the deeper `k` is not meaningfully smaller than
the index, because "retrieve almost the whole corpus" is not a fix anybody can
deploy, and reporting it as one would be actively misleading.
"""

from __future__ import annotations

from ragdx.ablations.base import Ablation, DiagnosisTarget, skipped
from ragdx.matching import gold_rank
from ragdx.schema import AblationResult, Golden

NAME = "rank_cutoff"

#: The deeper k must leave at least this fraction of the index unretrieved.
MIN_INDEX_HEADROOM = 2.0


class RankCutoff(Ablation):
    """Re-run production retrieval at ``config.rank_cutoff_k``."""

    @property
    def name(self) -> str:
        return NAME

    @property
    def cost(self) -> int:
        return 2

    def _degenerate(self, target: DiagnosisTarget) -> bool:
        depth = target.config.rank_cutoff_k
        return target.n_chunks > 0 and depth * MIN_INDEX_HEADROOM > target.n_chunks

    def applicable(self, target: DiagnosisTarget, golden: Golden) -> bool:
        return (
            target.config.rank_cutoff_k > target.k
            and not self._degenerate(target)
            and target.satisfiable(golden)
        )

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        depth = target.config.rank_cutoff_k
        if depth <= target.k:
            return skipped(NAME, f"rank_cutoff_k ({depth}) is not deeper than k ({target.k})")
        if self._degenerate(target):
            return skipped(
                NAME,
                f"k={depth} would return most of a {target.n_chunks}-chunk index, "
                f"which is not a shippable fix",
            )
        if not target.satisfiable(golden):
            return skipped(NAME, "no chunk covers the evidence span under this chunking")

        results = target.retriever.retrieve(golden.query, depth, target.filters)
        rank = gold_rank(results, golden, target.config.coverage_threshold)
        if rank is None:
            return AblationResult(
                ablation_name=NAME,
                recovered=False,
                detail=f"still missing at k={depth}",
            )
        return AblationResult(
            ablation_name=NAME,
            recovered=True,
            recovered_at_rank=rank,
            detail=f"gold chunk recovered at rank {rank} with k={depth}; current k={target.k}",
        )
