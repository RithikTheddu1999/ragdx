"""Does a retrieval satisfy a golden?

One definition, used by the runner, every ablation and the classifier — if these
ever disagree the diagnosis is nonsense.

A retrieved chunk satisfies a golden when it covers at least
``COVERAGE_THRESHOLD`` of the golden's evidence span. Coverage, not mere
overlap: an evidence span split across a chunk boundary leaves *every* chunk
holding only a fraction of the answer, and calling that a hit would hide the
single most actionable failure mode in the tool.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ragdx.schema import Chunk, Golden, RetrievedChunk

COVERAGE_THRESHOLD = 0.75


def chunk_satisfies(chunk: Chunk, golden: Golden, threshold: float = COVERAGE_THRESHOLD) -> bool:
    """True if ``chunk`` covers enough of the golden's evidence span."""
    return (
        chunk.overlap_fraction(golden.gold_doc_id, golden.gold_char_start, golden.gold_char_end)
        >= threshold
    )


def best_coverage(chunks: Iterable[Chunk], golden: Golden) -> float:
    """Highest span coverage achieved by any single chunk."""
    return max(
        (
            c.overlap_fraction(golden.gold_doc_id, golden.gold_char_start, golden.gold_char_end)
            for c in chunks
        ),
        default=0.0,
    )


def gold_rank(
    retrieved: Sequence[RetrievedChunk],
    golden: Golden,
    threshold: float = COVERAGE_THRESHOLD,
) -> int | None:
    """Rank of the first retrieved chunk satisfying ``golden``, else ``None``."""
    for item in retrieved:
        if chunk_satisfies(item.chunk, golden, threshold):
            return item.rank
    return None


def satisfiable(
    chunks: Iterable[Chunk], golden: Golden, threshold: float = COVERAGE_THRESHOLD
) -> bool:
    """True if *any* chunk in this chunking could satisfy the golden at all.

    False means the chunker, not the ranker, is the problem: no amount of
    reranking can surface a chunk that does not exist.
    """
    return any(chunk_satisfies(c, golden, threshold) for c in chunks)
