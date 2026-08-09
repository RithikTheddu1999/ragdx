"""CLI surface. `run` and `ci` stay stubs until their milestones land."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragdx.cli import app
from support import CORPUS_DIR

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
        ["ci", "--config", "ragdx.yaml", "--baseline", ".ragdx/baseline.json"],
    ],
)
def test_unimplemented_commands_exit_one(argv: list[str]) -> None:
    assert runner.invoke(app, argv).exit_code == 1


class TestGoldensBuild:
    def test_builds_a_versioned_verified_set(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "goldens", "build",
                "--corpus", str(CORPUS_DIR),
                "--n", "5",
                "--out", str(tmp_path / "goldens"),
                "--judge", "support:scripted_judge",
                "--cache-dir", str(tmp_path / "cache"),
            ],
        )  # fmt: skip
        assert result.exit_code == 0, result.stdout
        assert "kept       5" in result.stdout
        assert "rejected" in result.stdout

        goldens = (tmp_path / "goldens" / "v1.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(goldens) == 5
        manifest = json.loads((tmp_path / "goldens" / "v1.manifest.json").read_text())
        assert manifest["version"] == 1
        assert manifest["generator"] == "scripted-judge-v1"
        assert manifest["corpus_hash"]

    def test_second_build_writes_v2(self, tmp_path: Path) -> None:
        argv = [
            "goldens", "build",
            "--corpus", str(CORPUS_DIR),
            "--n", "2",
            "--out", str(tmp_path / "goldens"),
            "--judge", "support:scripted_judge",
            "--cache-dir", str(tmp_path / "cache"),
        ]  # fmt: skip
        assert runner.invoke(app, argv).exit_code == 0
        assert runner.invoke(app, argv).exit_code == 0
        assert (tmp_path / "goldens" / "v2.jsonl").is_file()

    def test_stub_judge_fails_loudly_rather_than_writing_an_empty_set(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "goldens", "build",
                "--corpus", str(CORPUS_DIR),
                "--n", "3",
                "--out", str(tmp_path / "goldens"),
                "--cache-dir", str(tmp_path / "cache"),
            ],
        )  # fmt: skip
        assert result.exit_code == 2
        assert not (tmp_path / "goldens").exists()

    def test_unresolvable_judge_exits_two(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "goldens", "build",
                "--corpus", str(CORPUS_DIR),
                "--out", str(tmp_path / "goldens"),
                "--judge", "no_such_module:judge",
            ],
        )  # fmt: skip
        assert result.exit_code == 2


class TestGoldensImport:
    def test_imports_and_reports_rejections(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "golden_id": "g1",
                            "query": "How long is a return label valid?",
                            "doc_id": "returns-policy",
                            "evidence": "valid for twenty eight days",
                        }
                    ),
                    json.dumps({"query": "q", "doc_id": "does-not-exist", "evidence": "x"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "goldens", "import",
                "--path", str(labels),
                "--corpus", str(CORPUS_DIR),
                "--out", str(tmp_path / "goldens"),
            ],
        )  # fmt: skip
        assert result.exit_code == 0, result.stdout
        assert "kept       1" in result.stdout
        assert "unknown_document" in result.stdout
        assert (tmp_path / "goldens" / "v1.jsonl").is_file()

    def test_all_rejected_exits_two(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            json.dumps({"query": "q", "doc_id": "nope", "evidence": "x"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "goldens", "import",
                "--path", str(labels),
                "--corpus", str(CORPUS_DIR),
                "--out", str(tmp_path / "goldens"),
            ],
        )  # fmt: skip
        assert result.exit_code == 2
