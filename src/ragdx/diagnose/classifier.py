"""Ablation results → a diagnosis.

This is the core, and it is deliberately almost all arithmetic. A gold chunk is
either inside the top `k` or it is not; an ablation either recovered it or it
did not. There is no judgement call and no LLM anywhere in this module, which is
the entire trust argument for the tool (PLAN.md §2).

Exactly one decision is not arithmetic — deciding, when *nothing* recovers the
gold chunk, whether the embedding simply cannot see it. That call is made from
the score distribution, carries a confidence below 1.0, and abstains to
``unclassified`` unless two independent signals agree.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragdx.ablations.base import Ablation, DiagnosisTarget
from ragdx.ablations.registry import first_recovery, run_battery
from ragdx.matching import COVERAGE_THRESHOLD, best_coverage, chunk_satisfies, gold_rank
from ragdx.schema import AblationResult, Diagnosis, FailureCause, Golden

#: Which ablation names which cause. One ablation, one cause, no overlap.
CAUSE_BY_ABLATION: dict[str, FailureCause] = {
    "filters_removed": FailureCause.METADATA_FILTER,
    "rank_cutoff": FailureCause.RANK_CUTOFF,
    "lexical_only": FailureCause.VOCABULARY_MISMATCH,
    "dense_only": FailureCause.PARAPHRASE_GAP,
    "alternate_chunking": FailureCause.CHUNK_BOUNDARY,
}


def cause_for_ablation(name: str) -> FailureCause | None:
    """The cause a recovering ablation names, or ``None`` if unrecognised."""
    return CAUSE_BY_ABLATION.get(name)


@dataclass(frozen=True)
class ClassifierConfig:
    """Thresholds. Every one of them errs towards abstaining."""

    coverage_threshold: float = COVERAGE_THRESHOLD
    #: The gold chunk's score must be at most this fraction of the score of the
    #: chunk sitting at position k, before "the embedding cannot see it" is on
    #: the table at all.
    blind_spot_score_ratio: float = 0.6
    #: ...and it must also sit in the bottom this-fraction of the whole index by
    #: rank. Two independent signals, because either alone is noisy.
    blind_spot_rank_fraction: float = 0.5
    #: Confidence assigned to that distribution-based call. Never 1.0.
    blind_spot_confidence: float = 0.6
    #: Confidence for a split evidence span that ragdx could not prove a
    #: re-chunk would fix, because no source documents were supplied.
    unverified_boundary_confidence: float = 0.7
    #: Confidence when re-chunking did fix the coverage but the chunk still did
    #: not rank — the chunking is genuinely wrong, but so is something else.
    partial_boundary_confidence: float = 0.5


@dataclass(frozen=True)
class ScoreProfile:
    """Where the gold chunk sits in the full, unfiltered score distribution."""

    gold_score: float
    cutoff_score: float
    gold_rank: int
    index_size: int

    @property
    def rank_fraction(self) -> float:
        return self.gold_rank / self.index_size if self.index_size else 0.0

    @property
    def score_ratio(self) -> float:
        if self.cutoff_score <= 0.0:
            return 1.0
        return self.gold_score / self.cutoff_score


class Classifier:
    """Turns a golden plus a retrieval setup into a `Diagnosis`."""

    def __init__(
        self,
        target: DiagnosisTarget,
        config: ClassifierConfig | None = None,
        battery: list[Ablation] | None = None,
    ) -> None:
        self.target = target
        self.config = config or ClassifierConfig()
        self.battery = battery

    def classify_all(self, goldens: list[Golden]) -> list[Diagnosis]:
        return [self.classify(g) for g in goldens]

    def classify(self, golden: Golden) -> Diagnosis:
        target, cfg = self.target, self.config

        retrieved = target.retriever.retrieve(golden.query, target.k, target.filters)
        rank = gold_rank(retrieved, golden, cfg.coverage_threshold)
        if rank is not None:
            # Fast path. Most queries in a healthy set land here, and none of
            # them pays for an ablation.
            return Diagnosis(
                golden_id=golden.golden_id,
                outcome="hit",
                cause=None,
                confidence=1.0,
                ablation_results=[],
                evidence=f"gold chunk retrieved at rank {rank} with k={target.k}",
                gold_rank=rank,
            )

        results = run_battery(target, golden, self.battery)
        recovery = first_recovery(results)
        if recovery is not None:
            cause = cause_for_ablation(recovery.ablation_name)
            if cause is not None:
                return Diagnosis(
                    golden_id=golden.golden_id,
                    outcome="retrieval_failure",
                    cause=cause,
                    confidence=1.0,
                    ablation_results=results,
                    evidence=recovery.detail or f"recovered by {recovery.ablation_name}",
                )

        return self._nothing_recovered(golden, results)

    def _nothing_recovered(self, golden: Golden, results: list[AblationResult]) -> Diagnosis:
        target, cfg = self.target, self.config

        if not target.satisfiable(golden):
            return self._split_span(golden, results)

        profile = self._score_profile(golden)
        if profile is None:
            return self._unclassified(
                golden,
                results,
                "no ablation recovered the gold chunk, and ragdx could not measure "
                "the score distribution to say whether the embedding can see it",
            )

        if (
            profile.score_ratio <= cfg.blind_spot_score_ratio
            and profile.rank_fraction >= cfg.blind_spot_rank_fraction
        ):
            return Diagnosis(
                golden_id=golden.golden_id,
                outcome="retrieval_failure",
                cause=FailureCause.EMBEDDING_BLIND_SPOT,
                confidence=cfg.blind_spot_confidence,
                ablation_results=results,
                evidence=(
                    f"no ablation recovered it; the gold chunk scores "
                    f"{profile.score_ratio:.0%} of the chunk at rank {target.k} and "
                    f"ranks {profile.gold_rank} of {profile.index_size}, so the "
                    f"embedding does not place it near this query"
                ),
            )

        return self._unclassified(
            golden,
            results,
            (
                f"no ablation recovered the gold chunk, but it scores "
                f"{profile.score_ratio:.0%} of the chunk at rank {target.k} and ranks "
                f"{profile.gold_rank} of {profile.index_size} — too close to the "
                f"retrieved field to blame the embedding"
            ),
        )

    def _split_span(self, golden: Golden, results: list[AblationResult]) -> Diagnosis:
        """The evidence span does not fit in any chunk. That much is arithmetic.

        Whether re-chunking *fixes* it is not, so the confidence depends on
        whether ragdx was able to try.
        """
        cfg = self.config
        coverage = best_coverage(self.target.chunks or [], golden)
        tried = next(
            (r for r in results if r.ablation_name == "alternate_chunking" and not r.skipped),
            None,
        )
        if tried is None:
            evidence = (
                f"no chunk covers more than {coverage:.0%} of the evidence span "
                f"(threshold {cfg.coverage_threshold:.0%}); ragdx had no source "
                f"documents to confirm that re-chunking would fix it"
            )
            confidence = cfg.unverified_boundary_confidence
        else:
            evidence = (
                f"the evidence span is split across chunks (best coverage "
                f"{coverage:.0%}); re-chunking covers it but the chunk still does "
                f"not rank in the top {self.target.k}, so the ranking is wrong too"
            )
            confidence = cfg.partial_boundary_confidence
        return Diagnosis(
            golden_id=golden.golden_id,
            outcome="retrieval_failure",
            cause=FailureCause.CHUNK_BOUNDARY,
            confidence=confidence,
            ablation_results=results,
            evidence=evidence,
        )

    def _unclassified(
        self, golden: Golden, results: list[AblationResult], evidence: str
    ) -> Diagnosis:
        return Diagnosis(
            golden_id=golden.golden_id,
            outcome="retrieval_failure",
            cause=FailureCause.UNCLASSIFIED,
            confidence=0.0,
            ablation_results=results,
            evidence=evidence,
        )

    def _score_profile(self, golden: Golden) -> ScoreProfile | None:
        """Rank and score the gold chunk against the whole index, unfiltered.

        Unfiltered on purpose: this measures what the *embedding* does, and a
        filter problem would already have been caught by an earlier ablation.
        """
        index = self.target.dense_index()
        chunks = self.target.chunks
        if index is None or not chunks:
            return None
        ranked = index.retrieve(golden.query, len(chunks), None)
        if len(ranked) <= self.target.k:
            return None
        position = next(
            (
                i
                for i, item in enumerate(ranked)
                if chunk_satisfies(item.chunk, golden, self.config.coverage_threshold)
            ),
            None,
        )
        if position is None:
            return None
        return ScoreProfile(
            gold_score=ranked[position].score,
            cutoff_score=ranked[self.target.k - 1].score,
            gold_rank=position,
            index_size=len(ranked),
        )
