"""ragdx command line."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from ragdx.cache import ContentCache
from ragdx.chunking import FixedSizeChunker
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
    config: str = typer.Option(..., "--config", "-c", help="Path to ragdx.yaml."),
    out: str = typer.Option("./ragdx-report", "--out", "-o", help="Output directory."),
) -> None:
    """Evaluate a golden set and write report.html + report.json."""
    _not_implemented("run")


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
