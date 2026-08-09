"""Alternate-chunking ablation.

Re-chunk the corpus larger and with overlap, re-index, retrieve again. The
expensive one, so it runs last and its index is built once per run.

The guard is what makes this ablation mean anything: it only runs when the
production chunking **cannot** produce a chunk covering the evidence span at
all. Larger chunks contain more words and therefore score better on almost
every query, so without that guard this ablation would "recover" gold chunks
for queries whose real problem was ranking, and every diagnosis would collapse
into "re-chunk your corpus".
"""

from __future__ import annotations

from ragdx.ablations.base import Ablation, DiagnosisTarget, skipped
from ragdx.matching import best_coverage, gold_rank
from ragdx.schema import AblationResult, Golden

NAME = "alternate_chunking"


class AlternateChunking(Ablation):
    """Re-chunk with larger, overlapping chunks and retrieve again."""

    @property
    def name(self) -> str:
        return NAME

    @property
    def cost(self) -> int:
        return 10

    def applicable(self, target: DiagnosisTarget, golden: Golden) -> bool:
        return target.docs is not None and not target.satisfiable(golden)

    def run(self, target: DiagnosisTarget, golden: Golden) -> AblationResult:
        if target.satisfiable(golden):
            return skipped(
                NAME,
                "production chunking already yields a chunk covering the evidence "
                "span, so re-chunking would only change ranking",
            )
        rechunked = target.rechunked_index()
        if rechunked is None:
            return skipped(NAME, "source documents were not supplied, cannot re-chunk")

        chunks, index = rechunked
        cfg = target.config
        before = best_coverage(target.chunks or [], golden)
        after = best_coverage(chunks, golden)
        rank = gold_rank(
            index.retrieve(golden.query, target.k, target.filters),
            golden,
            cfg.coverage_threshold,
        )
        shape = f"{cfg.alternate_chunk_size} chars, {cfg.alternate_chunk_overlap} overlap"
        if rank is None:
            return AblationResult(
                ablation_name=NAME,
                recovered=False,
                detail=(
                    f"re-chunking to {shape} raised span coverage from {before:.0%} "
                    f"to {after:.0%} but the chunk still does not rank in the top {target.k}"
                ),
            )
        return AblationResult(
            ablation_name=NAME,
            recovered=True,
            recovered_at_rank=rank,
            detail=(
                f"evidence span is split across chunks (best coverage {before:.0%}); "
                f"re-chunking to {shape} covers {after:.0%} of it and the chunk "
                f"ranks {rank}"
            ),
        )
