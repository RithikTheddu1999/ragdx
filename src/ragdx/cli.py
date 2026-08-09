"""ragdx command line."""

from __future__ import annotations

import typer

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


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to ragdx.yaml."),
    out: str = typer.Option("./ragdx-report", "--out", "-o", help="Output directory."),
) -> None:
    """Evaluate a golden set and write report.html + report.json."""
    _not_implemented("run")


@goldens_app.command("build")
def goldens_build(
    corpus: str = typer.Option(..., "--corpus", help="Directory of source documents."),
    n: int = typer.Option(50, "--n", help="Number of goldens to synthesize."),
) -> None:
    """Synthesize and verify a golden set from a corpus."""
    _not_implemented("goldens build")


@goldens_app.command("import")
def goldens_import(
    path: str = typer.Option(..., "--path", help="CSV or JSONL of human-labeled goldens."),
) -> None:
    """Import human-labeled goldens."""
    _not_implemented("goldens import")


@app.command()
def ci(
    config: str = typer.Option(..., "--config", "-c", help="Path to ragdx.yaml."),
    baseline: str = typer.Option(..., "--baseline", help="Path to baseline.json."),
) -> None:
    """Run the regression gate. Exit 0 pass, 1 regression, 2 error."""
    _not_implemented("ci")


if __name__ == "__main__":  # pragma: no cover
    app()
