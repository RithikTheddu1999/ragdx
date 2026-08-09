"""Milestone 7: clustering, ranked fixes, and the report a stranger can act on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ragdx.config import RagdxConfig, load_config
from ragdx.diagnose.cluster import build_clusters, document_frequency, offending_term
from ragdx.diagnose.recommend import recommend
from ragdx.report.render import build_summary, render_html, write_report
from ragdx.runner import RunResult, load_answers, load_goldens
from ragdx.runner import run as ragdx_run
from ragdx.schema import Diagnosis, FailureCause
from support import CORPUS_DIR, load_fixture_goldens


def _write_run_inputs(tmp_path: Path, **overrides: object) -> Path:
    """A ragdx.yaml over the fixture corpus, with its goldens and answers."""
    fixtures = load_fixture_goldens()
    goldens_path = tmp_path / "goldens.jsonl"
    goldens_path.write_text(
        "\n".join(f.golden.model_dump_json() for f in fixtures) + "\n", encoding="utf-8"
    )
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text(
        "\n".join(
            json.dumps({"golden_id": f.golden.golden_id, "answer": f.answer})
            for f in fixtures
            if f.answer
        )
        + "\n",
        encoding="utf-8",
    )
    config: dict[str, object] = {
        "corpus": str(CORPUS_DIR),
        "goldens": str(goldens_path),
        "answers": str(answers_path),
        "retrieval": {"plane": "dense", "k": 5, "filters": {"status": "current"}},
        "chunking": {"size": 240, "overlap": 60},
        "ablations": {"rank_cutoff_k": 20, "alternate_chunking": {"size": 960, "overlap": 480}},
        "judge": "support:scripted_faithfulness_judge",
    }
    config.update(overrides)
    config_path = tmp_path / "ragdx.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    path = _write_run_inputs(tmp_path_factory.mktemp("run"))
    return ragdx_run(load_config(path))


class TestConfig:
    def test_paths_resolve_relative_to_the_config_file(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("hello", encoding="utf-8")
        (tmp_path / "g.jsonl").write_text("", encoding="utf-8")
        path = tmp_path / "ragdx.yaml"
        path.write_text(yaml.safe_dump({"corpus": "./docs", "goldens": "./g.jsonl"}))
        config = load_config(path)
        assert config.corpus == (tmp_path / "docs").resolve()
        assert config.goldens == (tmp_path / "g.jsonl").resolve()
        assert config.answers is None

    def test_unknown_keys_are_rejected_rather_than_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "ragdx.yaml"
        path.write_text(yaml.safe_dump({"corpus": ".", "goldens": ".", "retreival": {"k": 5}}))
        with pytest.raises(Exception, match=r"retreival|extra"):
            load_config(path)

    def test_defaults_match_the_documented_ones(self) -> None:
        config = RagdxConfig(corpus=Path("."), goldens=Path("."))
        assert config.retrieval.plane == "dense"
        assert config.retrieval.k == 5
        assert config.ablations.rank_cutoff_k == 100
        assert config.judge == "stub"


class TestClustering:
    def test_groups_by_cause_largest_first(self, result: RunResult) -> None:
        counts = [c.count for c in result.report.clusters]
        assert counts == sorted(counts, reverse=True)
        assert {c.cause for c in result.report.clusters} == {
            FailureCause.METADATA_FILTER,
            FailureCause.RANK_CUTOFF,
            FailureCause.VOCABULARY_MISMATCH,
            FailureCause.CHUNK_BOUNDARY,
            FailureCause.EMBEDDING_BLIND_SPOT,
            FailureCause.GENERATION_UNGROUNDED,
        }

    def test_hits_are_never_clustered(self, result: RunResult) -> None:
        clustered = {gid for c in result.report.clusters for gid in c.golden_ids}
        hits = {d.golden_id for d in result.diagnoses if d.outcome == "hit"}
        assert clustered & hits == set()

    def test_examples_are_capped_and_carry_context(self, result: RunResult) -> None:
        for cluster in result.report.clusters:
            assert len(cluster.examples) <= 5
            for example in cluster.examples:
                assert example.query
                assert example.evidence
                assert example.expected_text

    def test_vocabulary_mismatch_names_the_offending_term(self, result: RunResult) -> None:
        cluster = next(
            c for c in result.report.clusters if c.cause is FailureCause.VOCABULARY_MISMATCH
        )
        assert set(cluster.subgroups) == {"rollover", "deadhead"}

    def test_offending_term_abstains_when_nothing_is_shared(self, result: RunResult) -> None:
        df = document_frequency(result.chunks)
        blind = next(d for d in result.diagnoses if d.cause is FailureCause.EMBEDDING_BLIND_SPOT)
        golden = next(g for g in result.goldens if g.golden_id == blind.golden_id)
        assert offending_term(golden, result.chunks, df, 0.75) is None

    def test_clustering_is_stable(self, result: RunResult) -> None:
        again = build_clusters(
            result.diagnoses, result.goldens, result.docs, result.chunks, coverage=0.75
        )
        assert [c.model_dump_json() for c in again] == [
            c.model_dump_json()
            for c in build_clusters(
                result.diagnoses, result.goldens, result.docs, result.chunks, coverage=0.75
            )
        ]


class TestRecommendations:
    def test_recovery_counts_add_up_to_the_failures_they_explain(self, result: RunResult) -> None:
        explained = sum(r.recovers for r in result.report.recommendations)
        unclassified = sum(
            c.count for c in result.report.clusters if c.cause is FailureCause.UNCLASSIFIED
        )
        assert explained == sum(c.count for c in result.report.clusters) - unclassified

    def test_ranked_by_value_for_money(self, result: RunResult) -> None:
        scores = [r.score for r in result.report.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_causes_sharing_a_fix_are_merged(self, result: RunResult) -> None:
        """Vocabulary mismatch and paraphrase gap are one change, not two."""
        hybrid = [r for r in result.report.recommendations if "hybrid" in r.fix]
        assert len(hybrid) == 1

    def test_rank_cutoff_names_the_k_that_would_work(self, result: RunResult) -> None:
        rec = next(r for r in result.report.recommendations if "raise k" in r.fix)
        assert "raising k from 5 to" in rec.detail

    def test_filter_fix_names_the_filter_key(self, result: RunResult) -> None:
        rec = next(r for r in result.report.recommendations if "filter" in r.fix)
        assert "status" in rec.detail

    def test_unclassified_gets_no_invented_fix(self, result: RunResult) -> None:
        config = RagdxConfig(corpus=Path("."), goldens=Path("."))
        diagnoses = [
            Diagnosis(
                golden_id="u1",
                outcome="retrieval_failure",
                cause=FailureCause.UNCLASSIFIED,
                confidence=0.0,
                evidence="nothing recovered it",
            )
        ]
        clusters = build_clusters(diagnoses, [], [], [])
        assert recommend(clusters, diagnoses, config) == []


class TestSummary:
    def test_generation_failures_count_towards_retrieval_recall(self) -> None:
        """Retrieval worked for those; conflating the planes is what this
        tool exists to stop."""
        diagnoses = [
            Diagnosis(golden_id="a", outcome="hit", confidence=1.0, evidence="x"),
            Diagnosis(
                golden_id="b",
                outcome="generation_failure",
                cause=FailureCause.GENERATION_UNGROUNDED,
                confidence=0.9,
                evidence="x",
            ),
            Diagnosis(
                golden_id="c",
                outcome="retrieval_failure",
                cause=FailureCause.RANK_CUTOFF,
                confidence=1.0,
                evidence="x",
            ),
        ]
        summary = build_summary(diagnoses, k=5)
        assert summary.recall_at_k == pytest.approx(2 / 3, abs=1e-4)
        assert summary.n_generation_failures == 1


class TestReportArtifacts:
    def test_html_is_self_contained(self, result: RunResult) -> None:
        html = render_html(result.report, generated_at="2026-01-01T00:00:00Z")
        assert "<style>" in html
        for forbidden in ("http://", "https://", "<script"):
            assert forbidden not in html, f"report reaches outside itself: {forbidden}"

    def test_html_shows_causes_counts_and_the_top_fix(self, result: RunResult) -> None:
        html = render_html(result.report)
        assert "Rank cutoff (gold below k)" in html
        assert "Top single fix" in html
        assert "Vocabulary mismatch" in html

    def test_html_explains_how_causes_were_assigned(self, result: RunResult) -> None:
        """A stranger should learn the diagnoses are counterfactual, not guessed."""
        html = render_html(result.report)
        assert "counterfactual retrieval" in html
        assert "unclassified" in html

    def test_html_escapes_content(self, result: RunResult) -> None:
        report = result.report.model_copy(deep=True)
        report.clusters[0].examples[0].query = "<script>alert(1)</script>"
        assert "<script>alert(1)</script>" not in render_html(report)

    def test_json_carries_no_timestamp_so_runs_are_diffable(
        self, result: RunResult, tmp_path: Path
    ) -> None:
        _, json_path = write_report(result.report, tmp_path, generated_at="2026-01-01")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert "generated_at" not in payload
        assert payload["summary"]["n_goldens"] == 51
        assert payload["corpus_hash"]

    def test_two_runs_produce_byte_identical_json(self, tmp_path: Path) -> None:
        config = load_config(_write_run_inputs(tmp_path))
        first = ragdx_run(config).report.model_dump_json(indent=2)
        second = ragdx_run(config).report.model_dump_json(indent=2)
        assert first == second

    def test_write_report_creates_both_artifacts(self, result: RunResult, tmp_path: Path) -> None:
        html_path, json_path = write_report(result.report, tmp_path / "nested")
        assert html_path.is_file() and json_path.is_file()


class TestRunnerInputs:
    def test_goldens_load_from_a_versioned_directory(self, tmp_path: Path) -> None:
        from ragdx.goldens import GoldenSetManifest, save

        fixtures = load_fixture_goldens()[:3]
        save(
            tmp_path,
            [f.golden for f in fixtures],
            GoldenSetManifest(version=1, corpus_hash="abc123", n_goldens=3, n_rejected=0),
        )
        goldens, built_against = load_goldens(tmp_path)
        assert len(goldens) == 3
        assert built_against == "abc123"

    def test_corpus_drift_is_surfaced_as_a_warning(self, tmp_path: Path) -> None:
        from ragdx.goldens import GoldenSetManifest, save

        goldens_dir = tmp_path / "goldens"
        save(
            goldens_dir,
            [f.golden for f in load_fixture_goldens()[:3]],
            GoldenSetManifest(version=1, corpus_hash="stale-hash", n_goldens=3, n_rejected=0),
        )
        config_path = _write_run_inputs(tmp_path, goldens=str(goldens_dir), answers=None)
        report = ragdx_run(load_config(config_path)).report
        assert report.warnings
        assert "corpus has changed" in report.warnings[0]

    def test_answers_without_a_file_are_empty(self) -> None:
        assert load_answers(None) == {}

    def test_answers_skip_incomplete_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        path.write_text(
            '{"golden_id": "g1", "answer": "yes"}\n{"golden_id": "g2"}\n\n', encoding="utf-8"
        )
        assert load_answers(path) == {"g1": "yes"}
