"""Turning ablation results into diagnoses. No LLM lives here — see PLAN.md §2."""

from ragdx.diagnose.classifier import (
    CAUSE_BY_ABLATION,
    Classifier,
    ClassifierConfig,
    ScoreProfile,
    cause_for_ablation,
)

__all__ = [
    "CAUSE_BY_ABLATION",
    "Classifier",
    "ClassifierConfig",
    "ScoreProfile",
    "cause_for_ablation",
]
