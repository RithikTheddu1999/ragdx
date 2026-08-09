"""Chunk offsets have to be exact — every ablation is arithmetic over them."""

from __future__ import annotations

import itertools

import pytest

from ragdx.chunking import FixedSizeChunker
from ragdx.corpus import Document

TEXT = " ".join(f"word{i:03d}" for i in range(200))


def _doc(text: str = TEXT) -> Document:
    return Document(doc_id="d", text=text, metadata={"status": "current"})


class TestFixedSizeChunker:
    def test_offsets_round_trip_to_the_source(self) -> None:
        doc = _doc()
        for chunk in FixedSizeChunker(size=120).chunk(doc):
            assert doc.text[chunk.char_start : chunk.char_end] == chunk.text

    def test_chunks_respect_the_size_limit(self) -> None:
        for chunk in FixedSizeChunker(size=120).chunk(_doc()):
            assert len(chunk.text) <= 120

    def test_no_word_is_split(self) -> None:
        for chunk in FixedSizeChunker(size=37).chunk(_doc()):
            for word in chunk.text.split():
                assert word.startswith("word")
                assert len(word) == 7

    def test_without_overlap_chunks_are_contiguous(self) -> None:
        chunks = FixedSizeChunker(size=120, overlap=0).chunk(_doc())
        for previous, following in itertools.pairwise(chunks):
            assert following.char_start >= previous.char_end

    def test_overlap_repeats_content(self) -> None:
        chunks = FixedSizeChunker(size=120, overlap=40).chunk(_doc())
        assert len(chunks) > 1
        for previous, following in itertools.pairwise(chunks):
            assert following.char_start < previous.char_end

    def test_overlap_still_terminates(self) -> None:
        assert len(FixedSizeChunker(size=30, overlap=29).chunk(_doc())) < 400

    def test_covers_the_whole_document(self) -> None:
        chunks = FixedSizeChunker(size=120).chunk(_doc())
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == len(TEXT)

    def test_metadata_is_inherited(self) -> None:
        chunk = FixedSizeChunker(size=120).chunk(_doc())[0]
        assert chunk.metadata == {"status": "current"}

    def test_word_longer_than_size_is_emitted_alone(self) -> None:
        chunks = FixedSizeChunker(size=5).chunk(_doc("supercalifragilistic tiny"))
        assert chunks[0].text == "supercalifragilistic"

    def test_empty_document_yields_nothing(self) -> None:
        assert FixedSizeChunker(size=50).chunk(_doc("   \n  ")) == []

    def test_name_encodes_the_parameters(self) -> None:
        assert FixedSizeChunker(size=240, overlap=60).name == "fixed240x60"

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
    def test_rejects_nonsense_parameters(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match=r"size|overlap"):
            FixedSizeChunker(size=size, overlap=overlap)
