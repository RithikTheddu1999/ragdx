"""Retrieval-plane ablations: lexical-only and dense-only.

These are mirror images, and only one of them applies to any given setup.

* A dense production retriever that misses a chunk BM25 finds at the same `k`
  has a **vocabulary mismatch**: the exact term was there, and mean-pooled
  embeddings washed it out.
* A lexical production retriever that misses a chunk a dense index finds has a
  **paraphrase gap**: the meaning was there, and the wording was not.

Both are re-run at the *production* `k`. Comparing BM25-at-100 against
dense-at-5 would conflate two changes and name the wrong cause.
"""

from __future__ import annotations

from ragdx.ablations.base import Ablation, DiagnosisTarget, skipped
from ragdx.matching import gold_rank
from ragdx.schema import AblationResult, Golden

LEXICAL_NAME = "lexical_only"
DENSE_NAME = "dense_only"


class LexicalOnly(Ablation):
    """Retrieve with BM25 instead of the dense index, at the same k."""

    @property
    def name(self) -> str:
        return LEXICAL_NAME

    @property
    def cost(self) -> int:
        return 3

    def applicable(self, target: DiagnosisTarget, golden: Golden) -> bool:
        return (
            target.plane != "lexical" and target.chunks is not None and target.satisfiable(golden)
        )

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        if target.plane == "lexical":
            return skipped(LEXICAL_NAME, "production retrieval is already lexical")
        index = target.lexical_index()
        if index is None:
            return skipped(LEXICAL_NAME, "corpus chunks were not supplied, cannot build BM25")
        if not target.satisfiable(golden):
            return skipped(LEXICAL_NAME, "no chunk covers the evidence span under this chunking")

        rank = gold_rank(
            index.retrieve(golden.query, target.k, target.filters),
            golden,
            target.config.coverage_threshold,
        )
        if rank is None:
            return AblationResult(
                ablation_name=LEXICAL_NAME,
                recovered=False,
                detail=f"BM25 also misses it at k={target.k}",
            )
        return AblationResult(
            ablation_name=LEXICAL_NAME,
            recovered=True,
            recovered_at_rank=rank,
            detail=(
                f"BM25 finds the gold chunk at rank {rank} with the same k={target.k}; "
                f"the dense plane does not"
            ),
        )


class DenseOnly(Ablation):
    """Retrieve with a dense index instead of BM25, at the same k."""

    @property
    def name(self) -> str:
        return DENSE_NAME

    @property
    def cost(self) -> int:
        return 3

    def applicable(self, target: DiagnosisTarget, golden: Golden) -> bool:
        return (
            target.plane in {"lexical", "hybrid"}
            and target.chunks is not None
            and target.satisfiable(golden)
        )

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        if target.plane not in {"lexical", "hybrid"}:
            return skipped(DENSE_NAME, "production retrieval is already dense")
        index = target.dense_index()
        if index is None:
            return skipped(
                DENSE_NAME, "corpus chunks were not supplied, cannot build a dense index"
            )
        if not target.satisfiable(golden):
            return skipped(DENSE_NAME, "no chunk covers the evidence span under this chunking")

        rank = gold_rank(
            index.retrieve(golden.query, target.k, target.filters),
            golden,
            target.config.coverage_threshold,
        )
        if rank is None:
            return AblationResult(
                ablation_name=DENSE_NAME,
                recovered=False,
                detail=f"a dense index also misses it at k={target.k}",
            )
        return AblationResult(
            ablation_name=DENSE_NAME,
            recovered=True,
            recovered_at_rank=rank,
            detail=(
                f"a dense index finds the gold chunk at rank {rank} with the same "
                f"k={target.k}; the lexical plane does not"
            ),
        )
