"""Import human-labelled goldens from CSV or JSONL.

Evidence may be given either as an exact character span or as quoted text. Most
human labellers quote; the quote is resolved against the source document, and a
quote that cannot be located — or that appears twice — is rejected rather than
guessed at, because a wrong span silently corrupts every diagnosis built on it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ragdx.corpus import Document
from ragdx.goldens.base import GoldenBatch, Rejection, RejectReason
from ragdx.schema import Golden
from ragdx.spans import AmbiguousSpanError, SpanNotFoundError, find_span

REQUIRED = ("query", "doc_id")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(text.splitlines()))
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def import_goldens(path: Path, docs: list[Document]) -> GoldenBatch:
    """Load goldens from ``path``, resolving and validating every span."""
    by_id = {d.doc_id: d for d in docs}
    goldens: list[Golden] = []
    rejections: list[Rejection] = []

    for index, row in enumerate(_read_rows(path)):
        source = str(row.get("golden_id") or f"{path.name}:{index + 1}")

        if any(not str(row.get(field, "")).strip() for field in REQUIRED):
            rejections.append(
                Rejection(
                    reason=RejectReason.MALFORMED,
                    detail=f"missing one of {REQUIRED}",
                    source=source,
                )
            )
            continue

        doc = by_id.get(str(row["doc_id"]))
        if doc is None:
            rejections.append(
                Rejection(
                    reason=RejectReason.UNKNOWN_DOCUMENT,
                    detail=f"no document {row['doc_id']!r} in corpus",
                    source=source,
                )
            )
            continue

        start = _as_int(row.get("gold_char_start", row.get("char_start")))
        end = _as_int(row.get("gold_char_end", row.get("char_end")))
        evidence = str(row.get("evidence", "") or "")

        if start is None or end is None:
            if not evidence.strip():
                rejections.append(
                    Rejection(
                        reason=RejectReason.MALFORMED,
                        detail="needs either a char span or quoted evidence",
                        source=source,
                    )
                )
                continue
            try:
                start, end = find_span(doc.text, evidence)
            except SpanNotFoundError as exc:
                rejections.append(
                    Rejection(
                        reason=RejectReason.EVIDENCE_NOT_FOUND, detail=str(exc), source=source
                    )
                )
                continue
            except AmbiguousSpanError as exc:
                rejections.append(
                    Rejection(
                        reason=RejectReason.EVIDENCE_AMBIGUOUS, detail=str(exc), source=source
                    )
                )
                continue

        if not 0 <= start < end <= len(doc.text):
            rejections.append(
                Rejection(
                    reason=RejectReason.SPAN_OUT_OF_RANGE,
                    detail=f"span {start}-{end} outside document of {len(doc.text)} chars",
                    source=source,
                )
            )
            continue

        expected = str(row.get("expected_answer", "") or "") or None
        goldens.append(
            Golden(
                golden_id=str(row.get("golden_id") or f"human-{index + 1:04d}"),
                query=str(row["query"]).strip(),
                gold_doc_id=doc.doc_id,
                gold_char_start=start,
                gold_char_end=end,
                expected_answer=expected,
                origin="human",
            )
        )

    return GoldenBatch(goldens=goldens, rejections=rejections)
