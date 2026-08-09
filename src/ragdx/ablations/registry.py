"""The ordered battery.

Order is from most specific to least, so the *first* ablation that recovers the
gold chunk is the one that names the failure. Once one recovers, the rest are
not run: they would only tell us that a second, more invasive change also
happens to work.

    filters_removed      one retrieval, no filter
    rank_cutoff          one retrieval, deeper k
    lexical_only         BM25 at the same k          (dense/hybrid production)
    dense_only           dense at the same k         (lexical/hybrid production)
    alternate_chunking   re-chunk and re-index       (expensive, runs last)
"""

from __future__ import annotations

from ragdx.ablations.base import Ablation, DiagnosisTarget
from ragdx.ablations.chunking import AlternateChunking
from ragdx.ablations.filters import FiltersRemoved
from ragdx.ablations.lexical import DenseOnly, LexicalOnly
from ragdx.ablations.rank_cutoff import RankCutoff
from ragdx.schema import AblationResult, Golden


def default_battery() -> list[Ablation]:
    """The battery in run order."""
    return [FiltersRemoved(), RankCutoff(), LexicalOnly(), DenseOnly(), AlternateChunking()]


def run_battery(
    target: DiagnosisTarget,
    golden: Golden,
    battery: list[Ablation] | None = None,
    *,
    record_skipped: bool = True,
) -> list[AblationResult]:
    """Run ablations in order, stopping at the first recovery.

    Inapplicable ablations are still recorded (as ``skipped``) so the report can
    distinguish "we tested this and it did not help" from "we could not test
    this" — the classifier must never rule out a cause it did not examine.
    """
    results: list[AblationResult] = []
    for ablation in battery if battery is not None else default_battery():
        if not ablation.applicable(target, golden):
            if record_skipped:
                results.append(ablation.run(target, golden))
            continue
        result = ablation.run(target, golden)
        results.append(result)
        if result.recovered:
            break
    return results


def first_recovery(results: list[AblationResult]) -> AblationResult | None:
    """The ablation that named the failure, if any."""
    return next((r for r in results if r.recovered), None)
