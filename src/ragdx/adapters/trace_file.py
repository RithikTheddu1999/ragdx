"""Generic trace ingest: a JSONL file of ``Trace`` records.

This is the real answer to "does it work with my stack". Any system that can
write down what it retrieved can be diagnosed, with no adapter and no access to
the retriever itself. Two framework adapters exist for convenience; this is the
one that covers everything else.

What a recording can and cannot support:

* **Can**: deciding hit or miss, clustering, the generation plane, and the
  counterfactuals that only need the corpus — BM25, a dense index, re-chunking.
  Those ask "would a different retrieval strategy over your documents have found
  this", which is answerable without touching your retriever.
* **Cannot**: retrieving deeper than the recording goes, or re-running your
  retriever with the filter off. Those ablations report themselves skipped.

The one thing a trace must carry is character offsets on every chunk. Gold
evidence is a span in the source document (PLAN.md §6), so a chunk with no
offsets cannot be matched against it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragdx.schema import RetrievedChunk, Trace


class TraceFormatError(ValueError):
    """A trace file could not be read as ``Trace`` records."""


def load_traces(path: Path) -> list[Trace]:
    """Read a JSONL file of ``Trace`` records, newest-format errors named loudly."""
    traces: list[Trace] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            traces.append(Trace.model_validate_json(line))
        except ValueError as exc:
            raise TraceFormatError(f"{path}:{number}: {exc}") from exc
    if not traces:
        raise TraceFormatError(f"{path} contains no traces")
    return traces


class TraceReplayRetriever:
    """Serves recorded retrievals, and is honest about their limits.

    Ranks are renumbered from 0 on truncation so the contract of the retriever
    protocol still holds, and a query with no recording returns nothing rather
    than the results of some other query.
    """

    name = "trace-replay"

    def __init__(self, traces: list[Trace]) -> None:
        self.traces = sorted(traces, key=lambda t: t.trace_id)
        # First trace wins for a repeated query, deterministically by trace_id.
        self._by_query: dict[str, Trace] = {}
        for trace in self.traces:
            self._by_query.setdefault(trace.query, trace)

    @property
    def recorded_depth(self) -> int:
        """The shallowest recording in the file — the depth we can rely on."""
        return min((len(t.retrieved) for t in self.traces), default=0)

    @property
    def recorded_filters(self) -> dict[str, Any] | None:
        filters = self.traces[0].config_snapshot.get("filters") if self.traces else None
        return filters if isinstance(filters, dict) else None

    @property
    def recorded_k(self) -> int | None:
        k = self.traces[0].config_snapshot.get("k") if self.traces else None
        return int(k) if isinstance(k, int) else None

    @property
    def plane(self) -> str:
        plane = self.traces[0].config_snapshot.get("retriever") if self.traces else None
        return str(plane) if isinstance(plane, str) else "dense"

    def answers(self) -> dict[str, str]:
        """Recorded answers, keyed by query, for the generation plane."""
        return {t.query: t.answer for t in self.traces if t.answer}

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        trace = self._by_query.get(query)
        if trace is None:
            return []
        return [
            RetrievedChunk(chunk=item.chunk, score=item.score, rank=rank)
            for rank, item in enumerate(trace.retrieved[:k])
        ]


def answers_by_golden(
    retriever: TraceReplayRetriever, queries_by_golden: dict[str, str]
) -> dict[str, str]:
    """Map recorded answers onto golden ids by matching on the query text."""
    recorded = retriever.answers()
    return {
        golden_id: recorded[query]
        for golden_id, query in queries_by_golden.items()
        if query in recorded
    }


def write_traces(path: Path, traces: list[Trace]) -> None:
    """Write traces back out — used by tests and by anyone building a fixture."""
    path.write_text(
        "\n".join(json.dumps(json.loads(t.model_dump_json()), sort_keys=True) for t in traces)
        + "\n",
        encoding="utf-8",
    )
