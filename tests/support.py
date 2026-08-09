"""Shared helpers for loading the planted-failure fixture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ragdx.chunking import FixedSizeChunker
from ragdx.corpus import Document, load_corpus
from ragdx.schema import Chunk, Golden
from ragdx.spans import find_span

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURE_DIR / "corpus"
GOLDENS_PATH = FIXTURE_DIR / "goldens.yaml"
EXPECTED_PATH = FIXTURE_DIR / "expected" / "diagnoses.json"

# The production configuration the fixture is diagnosed against.
PRODUCTION_K = 5
PRODUCTION_FILTERS: dict[str, Any] = {"status": "current"}
PRODUCTION_CHUNKER = FixedSizeChunker(size=240, overlap=60)
ALTERNATE_CHUNKER = FixedSizeChunker(size=960, overlap=480)
# Depth a reranker would realistically be fed: 4x the production k. The library
# default is 100 (PLAN.md §2); the fixture corpus is only ~200 chunks, and at
# k=100 the "ablation" degenerates into returning half the index.
RANK_CUTOFF_K = 20


@dataclass(frozen=True)
class FixtureGolden:
    """A golden plus the ground truth that was planted for it."""

    golden: Golden
    planted: str
    answer: str | None


def load_fixture_corpus() -> list[Document]:
    return load_corpus(CORPUS_DIR)


def load_fixture_chunks(chunker: FixedSizeChunker | None = None) -> list[Chunk]:
    return (chunker or PRODUCTION_CHUNKER).chunk_all(load_fixture_corpus())


def load_fixture_goldens(docs: list[Document] | None = None) -> list[FixtureGolden]:
    """Parse goldens.yaml, resolving each quoted evidence passage to a span."""
    by_id = {d.doc_id: d for d in (docs if docs is not None else load_fixture_corpus())}
    raw = yaml.safe_load(GOLDENS_PATH.read_text(encoding="utf-8"))
    out: list[FixtureGolden] = []
    for entry in raw:
        doc = by_id.get(entry["doc_id"])
        if doc is None:
            raise KeyError(f"golden {entry['golden_id']} names unknown doc {entry['doc_id']}")
        start, end = find_span(doc.text, entry["evidence"])
        out.append(
            FixtureGolden(
                golden=Golden(
                    golden_id=entry["golden_id"],
                    query=entry["query"],
                    gold_doc_id=doc.doc_id,
                    gold_char_start=start,
                    gold_char_end=end,
                    expected_answer=None,
                    origin="human",
                ),
                planted=entry["planted"],
                answer=entry.get("answer"),
            )
        )
    return out
