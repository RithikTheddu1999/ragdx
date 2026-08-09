"""``ragdx.yaml`` — the one file that describes the setup being diagnosed.

Paths inside the config resolve relative to the config file itself, so a config
committed next to a corpus works from any working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ragdx.ablations.base import AblationConfig
from ragdx.matching import COVERAGE_THRESHOLD

Plane = Literal["dense", "lexical", "hybrid"]


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int = 240
    overlap: int = 60


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plane: Plane = "dense"
    k: int = Field(default=5, ge=1)
    filters: dict[str, Any] = Field(default_factory=dict)


class AblationsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank_cutoff_k: int = 100
    alternate_chunking: ChunkingConfig = Field(
        default_factory=lambda: ChunkingConfig(size=960, overlap=480)
    )


class RagdxConfig(BaseModel):
    """A whole diagnostic run, described declaratively."""

    model_config = ConfigDict(extra="forbid")

    corpus: Path
    goldens: Path
    #: Optional JSONL of ``{"golden_id": ..., "answer": ...}`` — supply it and
    #: the generation plane is diagnosed too.
    answers: Path | None = None
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    ablations: AblationsConfig = Field(default_factory=AblationsConfig)
    judge: str = "stub"
    embedder: str = "stub"
    coverage_threshold: float = Field(default=COVERAGE_THRESHOLD, gt=0.0, le=1.0)

    def ablation_config(self) -> AblationConfig:
        return AblationConfig(
            rank_cutoff_k=self.ablations.rank_cutoff_k,
            alternate_chunk_size=self.ablations.alternate_chunking.size,
            alternate_chunk_overlap=self.ablations.alternate_chunking.overlap,
            coverage_threshold=self.coverage_threshold,
        )


def load_config(path: Path) -> RagdxConfig:
    """Parse ``ragdx.yaml``, resolving paths relative to the file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    config = RagdxConfig.model_validate(raw)
    root = path.parent
    resolved = config.model_copy(
        update={
            "corpus": (root / config.corpus).resolve(),
            "goldens": (root / config.goldens).resolve(),
            "answers": (root / config.answers).resolve() if config.answers else None,
        }
    )
    return resolved
