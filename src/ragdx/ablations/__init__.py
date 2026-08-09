"""Counterfactual retrievals. The ablation that recovers the gold chunk names
the failure."""

from ragdx.ablations.base import Ablation, AblationConfig, DiagnosisTarget
from ragdx.ablations.chunking import AlternateChunking
from ragdx.ablations.filters import FiltersRemoved
from ragdx.ablations.lexical import DenseOnly, LexicalOnly
from ragdx.ablations.rank_cutoff import RankCutoff
from ragdx.ablations.registry import default_battery, first_recovery, run_battery

__all__ = [
    "Ablation",
    "AblationConfig",
    "AlternateChunking",
    "DenseOnly",
    "DiagnosisTarget",
    "FiltersRemoved",
    "LexicalOnly",
    "RankCutoff",
    "default_battery",
    "first_recovery",
    "run_battery",
]
