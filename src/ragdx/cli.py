"""ragdx command line."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import ValidationError

from ragdx.cache import ContentCache
from ragdx.chunking import FixedSizeChunker
from ragdx.config import load_config
from ragdx.corpus import corpus_hash, load_corpus
from ragdx.goldens import (
    GoldenBatch,
    GoldenSetManifest,
    SynthesisConfig,
    import_goldens,
    next_version,
    save,
    synthesize,
)
from ragdx.judge.base import CachedJudge
from ragdx.judge.loader import JudgeNotFoundError, load_judge
from ragdx.plugins import PluginNotFoundError
from ragdx.report.render import CAUSE_LABELS, RECOVERABLE_BY, write_report
from ragdx.runner import run as ragdx_run

app = typer.Typer(
    name="ragdx",
    help="Diagnose *why* RAG retrieval failed, not just how often.",
    no_args_is_help=True,
    add_completion=False,
)

goldens_app = typer.Typer(
    name="goldens", help="Build and manage golden sets.", no_args_is_help=True
)
app.add_typer(goldens_app)


def _not_implemented(what: str) -> None:
    typer.secho(f"{what}: not implemented", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _report_batch(batch: GoldenBatch) -> None:
    typer.echo(f"  kept       {len(batch.goldens)}")
    typer.echo(f"  rejected   {len(batch.rejections)} ({batch.rejection_rate:.0%})")
    for reason, count in batch.counts_by_reason().items():
        typer.echo(f"    {reason:<28} {count}")


def _write(directory: Path, batch: GoldenBatch, corpus_digest: str, generator: str) -> None:
    manifest = GoldenSetManifest(
        version=next_version(directory),
        corpus_hash=corpus_digest,
        n_goldens=len(batch.goldens),
        n_rejected=len(batch.rejections),
        generator=generator,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    goldens_path, manifest_path = save(directory, batch.goldens, manifest)
    typer.echo(f"  wrote      {goldens_path}")
    typer.echo(f"             {manifest_path}")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to ragdx.yaml."),
    out: Path = typer.Option(Path("./ragdx-report"), "--out", "-o", help="Output directory."),
) -> None:
    """Evaluate a golden set and write report.html + report.json."""
    try:
        settings = load_config(config)
    except (OSError, ValueError, ValidationError) as exc:
        typer.secho(f"could not read {config}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    try:
        result = ragdx_run(settings)
    except (OSError, ValueError, PluginNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    report = result.report
    for warning in report.warnings:
        typer.secho(warning, fg=typer.colors.YELLOW, err=True)

    typer.echo(
        f"{report.summary.n_goldens} queries evaluated · "
        f"{report.summary.n_retrieval_failures} retrieval failures · "
        f"recall@{settings.retrieval.k} {report.summary.recall_at_k:.0%}"
    )
    if report.clusters:
        typer.echo("")
        width = max(
            len("ROOT CAUSE"),
            *(len(CAUSE_LABELS.get(c.cause.value, c.cause.value)) for c in report.clusters),
        )
        typer.echo(f"  {'ROOT CAUSE':<{width}}{'FAILURES':>10}   RECOVERABLE BY")
        typer.echo("  " + "─" * (width + 45))
        for cluster in report.clusters:
            label = CAUSE_LABELS.get(cluster.cause.value, cluster.cause.value)
            fix = RECOVERABLE_BY.get(cluster.cause.value, "—")
            typer.echo(f"  {label:<{width}}{cluster.count:>10}   {fix}")
    if report.recommendations:
        best = report.recommendations[0]
        total = report.summary.n_retrieval_failures + report.summary.n_generation_failures
        typer.echo("")
        typer.echo(
            f"  Top single fix: {best.fix} → est. recovery "
            f"{best.recovers}/{total} ({best.share_of_failures:.0%})"
        )

    html_path, json_path = write_report(
        report, out, generated_at=datetime.now(UTC).isoformat(timespec="seconds")
    )
    typer.echo("")
    typer.echo(f"  wrote {html_path}")
    typer.echo(f"        {json_path}")


@goldens_app.command("build")
def goldens_build(
    corpus: Path = typer.Option(..., "--corpus", help="Directory of source documents."),
    n: int = typer.Option(50, "--n", help="Number of goldens to synthesize."),
    out: Path = typer.Option(Path("./goldens"), "--out", help="Golden set directory."),
    judge_spec: str = typer.Option(
        "stub", "--judge", help="'module:attribute' resolving to a Judge, or 'stub'."
    ),
    chunk_size: int = typer.Option(240, "--chunk-size", help="Chunk size in characters."),
    chunk_overlap: int = typer.Option(60, "--chunk-overlap", help="Chunk overlap in characters."),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
    cache_dir: Path = typer.Option(
        Path(".ragdx/cache"), "--cache-dir", help="Where judge answers are cached."
    ),
) -> None:
    """Synthesize and verify a golden set from a corpus."""
    try:
        judge = CachedJudge(load_judge(judge_spec), ContentCache(cache_dir, "judge"))
    except JudgeNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    docs = load_corpus(corpus)
    chunks = FixedSizeChunker(size=chunk_size, overlap=chunk_overlap).chunk_all(docs)
    typer.echo(f"corpus     {len(docs)} documents, {len(chunks)} chunks")

    batch = synthesize(docs, chunks, judge, SynthesisConfig(n=n, seed=seed))
    _report_batch(batch)
    if not batch.goldens:
        typer.secho(
            "no goldens survived verification — check the --judge wiring",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    _write(out, batch, corpus_hash(docs), judge.name)


@goldens_app.command("import")
def goldens_import(
    path: Path = typer.Option(..., "--path", help="CSV or JSONL of human-labeled goldens."),
    corpus: Path = typer.Option(..., "--corpus", help="Directory of source documents."),
    out: Path = typer.Option(Path("./goldens"), "--out", help="Golden set directory."),
) -> None:
    """Import human-labeled goldens, resolving and validating every span."""
    docs = load_corpus(corpus)
    batch = import_goldens(path, docs)
    typer.echo(f"corpus     {len(docs)} documents")
    _report_batch(batch)
    if not batch.goldens:
        typer.secho("no goldens imported", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    _write(out, batch, corpus_hash(docs), "human")


@app.command()
def ci(
    config: str = typer.Option(..., "--config", "-c", help="Path to ragdx.yaml."),
    baseline: str = typer.Option(..., "--baseline", help="Path to baseline.json."),
) -> None:
    """Run the regression gate. Exit 0 pass, 1 regression, 2 error."""
    _not_implemented("ci")


if __name__ == "__main__":  # pragma: no cover
    app()
