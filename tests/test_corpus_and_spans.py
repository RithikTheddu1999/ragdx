"""Front matter parsing, corpus hashing and evidence-span resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdx.corpus import Document, corpus_hash, load_corpus, parse_front_matter
from ragdx.spans import AmbiguousSpanError, SpanNotFoundError, find_span, normalize


class TestFrontMatter:
    def test_parses_and_strips(self) -> None:
        meta, body = parse_front_matter("---\ndepartment: legal\nstatus: current\n---\nBody text.")
        assert meta == {"department": "legal", "status": "current"}
        assert body == "Body text."

    def test_absent_front_matter_leaves_body_untouched(self) -> None:
        meta, body = parse_front_matter("Just a document.")
        assert meta == {}
        assert body == "Just a document."

    def test_unterminated_front_matter_is_not_metadata(self) -> None:
        meta, body = parse_front_matter("---\ndepartment: legal\nno terminator")
        assert meta == {}
        assert body.startswith("---")


class TestCorpusHash:
    def _docs(self) -> list[Document]:
        return [
            Document(doc_id="a", text="alpha", metadata={"status": "current"}),
            Document(doc_id="b", text="beta", metadata={}),
        ]

    def test_is_stable_and_order_independent(self) -> None:
        docs = self._docs()
        assert corpus_hash(docs) == corpus_hash(list(reversed(docs)))

    def test_changes_with_text(self) -> None:
        docs = self._docs()
        changed = [Document(doc_id="a", text="alpha!", metadata={"status": "current"}), docs[1]]
        assert corpus_hash(docs) != corpus_hash(changed)

    def test_changes_with_metadata(self) -> None:
        docs = self._docs()
        changed = [Document(doc_id="a", text="alpha", metadata={"status": "archived"}), docs[1]]
        assert corpus_hash(docs) != corpus_hash(changed)


class TestLoadCorpus:
    def test_rejects_a_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            load_corpus(tmp_path / "nope")

    def test_rejects_an_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no documents"):
            load_corpus(tmp_path)

    def test_doc_id_is_the_relative_path_without_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "page.md").write_text("hello", encoding="utf-8")
        assert load_corpus(tmp_path)[0].doc_id == "sub/page"


class TestSpans:
    def test_normalize_maps_back_to_original_offsets(self) -> None:
        text = "one   two\n\nthree"
        norm, origin = normalize(text)
        assert norm == "one two three"
        assert text[origin[norm.index("three")]] == "t"

    def test_find_span_ignores_line_wrapping(self) -> None:
        text = "The quick brown\nfox jumps over the lazy dog."
        start, end = find_span(text, "quick brown fox jumps")
        assert text[start:end] == "quick brown\nfox jumps"

    def test_missing_evidence_raises(self) -> None:
        with pytest.raises(SpanNotFoundError):
            find_span("hello world", "goodbye")

    def test_empty_evidence_raises(self) -> None:
        with pytest.raises(SpanNotFoundError):
            find_span("hello world", "   ")

    def test_ambiguous_evidence_raises_rather_than_guessing(self) -> None:
        with pytest.raises(AmbiguousSpanError):
            find_span("repeat me. repeat me.", "repeat me")
