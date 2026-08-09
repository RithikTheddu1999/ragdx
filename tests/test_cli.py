"""CLI surface tests. Subcommands are stubs until their milestone lands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ragdx.cli import app

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "goldens", "ci"):
        assert name in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--config", "ragdx.yaml"],
        ["goldens", "build", "--corpus", "./docs", "--n", "50"],
        ["goldens", "import", "--path", "labels.jsonl"],
        ["ci", "--config", "ragdx.yaml", "--baseline", ".ragdx/baseline.json"],
    ],
)
def test_stubs_exit_nonzero(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 1
