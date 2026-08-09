"""Assembling and rendering the report.

Two artifacts come out of a run:

* ``report.html`` — self-contained, inline CSS, no network requests. Something a
  stranger can open and act on without reading the source.
* ``report.json`` — the machine artifact Phase 2 consumes. It carries **no
  timestamp**, so two runs over the same corpus produce byte-identical bytes and
  a baseline diff shows only what actually changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from ragdx.diagnose.cluster import Cluster
from ragdx.diagnose.recommend import Recommendation
from ragdx.schema import Diagnosis, FailureCause

TEMPLATE_DIR = Path(__file__).parent

CAUSE_LABELS: dict[str, str] = {
    FailureCause.RANK_CUTOFF.value: "Rank cutoff (gold below k)",
    FailureCause.VOCABULARY_MISMATCH.value: "Vocabulary mismatch",
    FailureCause.PARAPHRASE_GAP.value: "Paraphrase gap",
    FailureCause.CHUNK_BOUNDARY.value: "Chunk boundary split",
    FailureCause.METADATA_FILTER.value: "Metadata filter over-exclusion",
    FailureCause.EMBEDDING_BLIND_SPOT.value: "Embedding blind spot",
    FailureCause.GENERATION_UNGROUNDED.value: "Generation failure (answer ungrounded)",
    FailureCause.UNCLASSIFIED.value: "Unclassified (low confidence)",
}

RECOVERABLE_BY: dict[str, str] = {
    FailureCause.RANK_CUTOFF.value: "raising k, or a reranker",
    FailureCause.VOCABULARY_MISMATCH.value: "hybrid retrieval (BM25 + dense)",
    FailureCause.PARAPHRASE_GAP.value: "hybrid retrieval (BM25 + dense)",
    FailureCause.CHUNK_BOUNDARY.value: "re-chunking with overlap",
    FailureCause.METADATA_FILTER.value: "fixing the metadata filter",
    FailureCause.EMBEDDING_BLIND_SPOT.value: "a domain-tuned embedder",
    FailureCause.GENERATION_UNGROUNDED.value: "prompt / grounding work",
    FailureCause.UNCLASSIFIED.value: "— nothing recovered these; investigate by hand",
}


class ReportSummary(BaseModel):
    """The numbers at the top of the page."""

    n_goldens: int
    n_hits: int
    n_retrieval_failures: int
    n_generation_failures: int
    n_unclassified: int
    recall_at_k: float


class Report(BaseModel):
    """The whole diagnosis, ready to render or diff."""

    summary: ReportSummary
    clusters: list[Cluster] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    corpus_hash: str = ""
    n_documents: int = 0
    n_chunks: int = 0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_summary(diagnoses: list[Diagnosis], k: int) -> ReportSummary:
    hits = sum(1 for d in diagnoses if d.outcome == "hit")
    retrieval = sum(1 for d in diagnoses if d.outcome == "retrieval_failure")
    generation = sum(1 for d in diagnoses if d.outcome == "generation_failure")
    unclassified = sum(1 for d in diagnoses if d.cause is FailureCause.UNCLASSIFIED)
    total = len(diagnoses)
    # Retrieval recall: a generation failure means retrieval *worked*, so it
    # counts towards recall. Conflating the two is exactly what this tool exists
    # to stop people doing.
    retrieved_ok = hits + generation
    return ReportSummary(
        n_goldens=total,
        n_hits=hits,
        n_retrieval_failures=retrieval,
        n_generation_failures=generation,
        n_unclassified=unclassified,
        recall_at_k=round(retrieved_ok / total, 4) if total else 0.0,
    )


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(report: Report, generated_at: str = "") -> str:
    """Render the self-contained HTML page."""
    template = _environment().get_template("template.html.j2")
    return template.render(
        report=report,
        generated_at=generated_at,
        cause_labels=CAUSE_LABELS,
        recoverable_by=RECOVERABLE_BY,
        top_fix=report.recommendations[0] if report.recommendations else None,
    )


def write_report(report: Report, out_dir: Path, generated_at: str = "") -> tuple[Path, Path]:
    """Write ``report.html`` and ``report.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    json_path = out_dir / "report.json"
    html_path.write_text(render_html(report, generated_at), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return html_path, json_path
