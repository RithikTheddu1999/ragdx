"""Versioned golden sets on disk.

A golden set is only meaningful against the corpus it was built from. Each
version is written alongside a manifest pinning the corpus hash, the generator
that produced it and how many candidates were rejected; loading a set against a
different corpus warns loudly rather than silently comparing apples to oranges.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ragdx.corpus import Document, corpus_hash
from ragdx.schema import Golden

_VERSION_RE = re.compile(r"^v(\d+)\.jsonl$")


class CorpusDriftWarning(UserWarning):
    """The corpus has changed since this golden set was built."""


class GoldenSetManifest(BaseModel):
    """Everything needed to decide whether a golden set is still valid."""

    version: int = Field(ge=1)
    corpus_hash: str
    n_goldens: int = Field(ge=0)
    n_rejected: int = Field(ge=0)
    generator: str = "human"
    created_at: str = ""
    notes: str = ""

    @property
    def rejection_rate(self) -> float:
        total = self.n_goldens + self.n_rejected
        return self.n_rejected / total if total else 0.0


def _goldens_path(directory: Path, version: int) -> Path:
    return directory / f"v{version}.jsonl"


def _manifest_path(directory: Path, version: int) -> Path:
    return directory / f"v{version}.manifest.json"


def available_versions(directory: Path) -> list[int]:
    if not directory.is_dir():
        return []
    versions = []
    for path in directory.iterdir():
        match = _VERSION_RE.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    return sorted(versions)


def next_version(directory: Path) -> int:
    versions = available_versions(directory)
    return (versions[-1] + 1) if versions else 1


def save(
    directory: Path,
    goldens: list[Golden],
    manifest: GoldenSetManifest,
) -> tuple[Path, Path]:
    """Write ``vN.jsonl`` and ``vN.manifest.json``. Refuses to overwrite."""
    directory.mkdir(parents=True, exist_ok=True)
    goldens_path = _goldens_path(directory, manifest.version)
    manifest_path = _manifest_path(directory, manifest.version)
    if goldens_path.exists():
        raise FileExistsError(f"golden set v{manifest.version} already exists at {goldens_path}")
    # Sorted by id so the file is byte-identical for the same set of goldens.
    lines = [g.model_dump_json() for g in sorted(goldens, key=lambda g: g.golden_id)]
    goldens_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return goldens_path, manifest_path


def load(directory: Path, version: int | None = None) -> tuple[list[Golden], GoldenSetManifest]:
    """Load a golden set version, defaulting to the latest."""
    versions = available_versions(directory)
    if not versions:
        raise FileNotFoundError(f"no golden sets found in {directory}")
    chosen = versions[-1] if version is None else version
    goldens_path = _goldens_path(directory, chosen)
    if not goldens_path.is_file():
        raise FileNotFoundError(f"golden set v{chosen} not found in {directory}")
    goldens = [
        Golden.model_validate_json(line)
        for line in goldens_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = GoldenSetManifest.model_validate_json(
        _manifest_path(directory, chosen).read_text(encoding="utf-8")
    )
    return goldens, manifest


def check_corpus(manifest: GoldenSetManifest, docs: list[Document]) -> str | None:
    """Return a warning message if the corpus has drifted, else ``None``.

    Returned rather than raised: a drifted corpus is usually still worth
    diagnosing, but the user has to be told the goldens may no longer line up.
    """
    current = corpus_hash(docs)
    if current == manifest.corpus_hash:
        return None
    return (
        f"corpus has changed since golden set v{manifest.version} was built "
        f"(manifest {manifest.corpus_hash[:12]}, current {current[:12]}). "
        f"Evidence spans may no longer point at the right text — rebuild the "
        f"golden set, or pin the corpus to the revision it was built from."
    )
