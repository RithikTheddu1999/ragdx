"""Loading source documents and hashing a corpus.

A document is text plus metadata. Metadata is set in optional YAML front matter
and is inherited by every chunk cut from the document, which is what makes the
metadata-filter ablation possible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

DOC_SUFFIXES = (".md", ".txt")
_FRONT_MATTER = "---\n"


class Document(BaseModel):
    """A source document, before chunking."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Split optional leading ``---`` YAML front matter from the body.

    The returned body keeps the offsets it will have on disk-free reading, i.e.
    character 0 of the body is character 0 for every golden evidence span. Gold
    spans are always relative to the *body*, never the raw file.
    """
    if not raw.startswith(_FRONT_MATTER):
        return {}, raw
    end = raw.find("\n---\n", len(_FRONT_MATTER) - 1)
    if end == -1:
        return {}, raw
    header = raw[len(_FRONT_MATTER) : end + 1]
    body = raw[end + len("\n---\n") :]
    loaded = yaml.safe_load(header)
    metadata: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    return metadata, body


def load_corpus(path: Path) -> list[Document]:
    """Load every ``.md`` / ``.txt`` file under ``path``, sorted by doc id.

    Sorted order is part of the determinism contract: chunk ids, index order and
    tie-breaks all derive from it.
    """
    if not path.is_dir():
        raise NotADirectoryError(f"corpus path is not a directory: {path}")
    docs: list[Document] = []
    for file in sorted(path.rglob("*")):
        if file.suffix not in DOC_SUFFIXES or not file.is_file():
            continue
        metadata, body = parse_front_matter(file.read_text(encoding="utf-8"))
        docs.append(
            Document(
                doc_id=file.relative_to(path).with_suffix("").as_posix(),
                text=body,
                metadata=metadata,
            )
        )
    if not docs:
        raise ValueError(f"no documents found under {path}")
    return docs


def corpus_hash(docs: list[Document]) -> str:
    """Stable content hash over doc ids, bodies and metadata.

    Golden sets and baselines pin themselves to this value so ragdx can shout
    when the corpus has moved underneath them.
    """
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        h.update(doc.doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(doc.text.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(doc.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()
