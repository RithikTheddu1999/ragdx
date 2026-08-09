"""Shared helpers for loading the planted-failure fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ragdx.ablations.base import AblationConfig, DiagnosisTarget
from ragdx.chunking import FixedSizeChunker
from ragdx.corpus import Document, load_corpus
from ragdx.index import DenseRetriever
from ragdx.judge.base import JudgeVerdict
from ragdx.schema import Chunk, Golden
from ragdx.spans import find_span
from ragdx.text import tokenize

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURE_DIR / "corpus"
GOLDENS_PATH = FIXTURE_DIR / "goldens.yaml"
EXPECTED_PATH = FIXTURE_DIR / "expected" / "diagnoses.json"

# The production configuration the fixture is diagnosed against.
PRODUCTION_K = 5
PRODUCTION_FILTERS: dict[str, Any] = {"status": "current"}
PRODUCTION_CHUNKER = FixedSizeChunker(size=240, overlap=60)
ALTERNATE_CHUNKER = FixedSizeChunker(size=960, overlap=480)
# Depth a reranker would realistically be fed: 4x the production k. The library
# default is 100 (PLAN.md §2); the fixture corpus is only ~200 chunks, and at
# k=100 the "ablation" degenerates into returning half the index.
RANK_CUTOFF_K = 20


@dataclass(frozen=True)
class FixtureGolden:
    """A golden plus the ground truth that was planted for it."""

    golden: Golden
    planted: str
    answer: str | None


def load_fixture_corpus() -> list[Document]:
    return load_corpus(CORPUS_DIR)


def load_fixture_chunks(chunker: FixedSizeChunker | None = None) -> list[Chunk]:
    return (chunker or PRODUCTION_CHUNKER).chunk_all(load_fixture_corpus())


def load_fixture_goldens(docs: list[Document] | None = None) -> list[FixtureGolden]:
    """Parse goldens.yaml, resolving each quoted evidence passage to a span."""
    by_id = {d.doc_id: d for d in (docs if docs is not None else load_fixture_corpus())}
    raw = yaml.safe_load(GOLDENS_PATH.read_text(encoding="utf-8"))
    out: list[FixtureGolden] = []
    for entry in raw:
        doc = by_id.get(entry["doc_id"])
        if doc is None:
            raise KeyError(f"golden {entry['golden_id']} names unknown doc {entry['doc_id']}")
        start, end = find_span(doc.text, entry["evidence"])
        out.append(
            FixtureGolden(
                golden=Golden(
                    golden_id=entry["golden_id"],
                    query=entry["query"],
                    gold_doc_id=doc.doc_id,
                    gold_char_start=start,
                    gold_char_end=end,
                    expected_answer=None,
                    origin="human",
                ),
                planted=entry["planted"],
                answer=entry.get("answer"),
            )
        )
    return out


class ScriptedJudge:
    """A judge whose answers are derived from the text, not from a model.

    It plants a rare marker token from the passage into the question it writes,
    then answers "answerable" exactly when that marker appears in the context it
    is shown. That gives synthesis a deterministic, offline judge whose verdicts
    are actually *about* the text — so the verification step is genuinely
    exercised rather than rubber-stamped.
    """

    name = "scripted-judge-v1"

    def __init__(self, abstain: bool = False, always_answerable: bool = False) -> None:
        self.abstain = abstain
        self.always_answerable = always_answerable

    @staticmethod
    def _marker(text: str) -> str:
        tokens = [t for t in tokenize(text) if len(t) > 3]
        return max(tokens, key=lambda t: (len(t), t)) if tokens else "passage"

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        passage = prompt.split("PASSAGE:\n", 1)[-1].strip()
        sentence = passage.split(". ", 1)[0].strip()
        if not sentence:
            return ""
        marker = self._marker(sentence)
        return json.dumps(
            {
                "question": f"What does the handbook say about {marker}?",
                "evidence": sentence,
                "answer": sentence,
            }
        )

    def judge(self, prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
        if self.abstain:
            return JudgeVerdict(label=labels[0], confidence=0.0, abstained=True)
        question = prompt.split("QUESTION:\n", 1)[-1].split("\n\nCONTEXT:\n", 1)[0]
        context = prompt.split("CONTEXT:\n", 1)[-1]
        marker = question.rstrip("?").split()[-1].lower()
        answerable = self.always_answerable or marker in context.lower()
        return JudgeVerdict(
            label="answerable" if answerable else "not_answerable",
            confidence=0.9 if answerable else 0.8,
            rationale=f"marker {marker!r} {'in' if answerable else 'not in'} context",
        )


def scripted_judge() -> ScriptedJudge:
    """Factory used by the CLI tests via `--judge support:scripted_judge`."""
    return ScriptedJudge()


def fixture_target(docs: list[Document] | None = None) -> DiagnosisTarget:
    """The fixture corpus wired up as a production retrieval setup to diagnose."""
    docs = docs if docs is not None else load_fixture_corpus()
    chunks = PRODUCTION_CHUNKER.chunk_all(docs)
    return DiagnosisTarget(
        retriever=DenseRetriever(chunks),
        k=PRODUCTION_K,
        filters=dict(PRODUCTION_FILTERS),
        plane="dense",
        chunks=chunks,
        docs=docs,
        config=AblationConfig(
            rank_cutoff_k=RANK_CUTOFF_K,
            alternate_chunk_size=ALTERNATE_CHUNKER.size,
            alternate_chunk_overlap=ALTERNATE_CHUNKER.overlap,
        ),
    )


NUMBER_WORDS = frozenset(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "half",
        "twice",
        "double",
    ]
)


def numeric_claims(text: str) -> set[str]:
    """Digits and number words — where unfaithful answers usually give themselves away."""
    return {t for t in tokenize(text) if t.isdigit() or t in NUMBER_WORDS}


class ScriptedFaithfulnessJudge:
    """A faithfulness judge that checks numeric claims instead of guessing.

    Deterministic and offline: an answer is ungrounded when it asserts a number
    the retrieved context never states. That is exactly the failure planted in
    the fixture (48 hours vs the next payroll run, five times vs twice), and it
    means the generation-plane tests exercise a real decision rather than a
    hardcoded verdict.
    """

    name = "scripted-faithfulness-v1"

    def __init__(self, abstain: bool = False) -> None:
        self.abstain = abstain

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        return ""

    def judge(self, prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
        if self.abstain:
            return JudgeVerdict(label=labels[0], confidence=0.0, abstained=True)
        answer = prompt.split("ANSWER:\n", 1)[-1].split("\n\nCONTEXT:\n", 1)[0]
        context = prompt.split("CONTEXT:\n", 1)[-1]
        unsupported = numeric_claims(answer) - numeric_claims(context)
        if unsupported:
            return JudgeVerdict(
                label="ungrounded",
                confidence=0.9,
                rationale=f"context never states {', '.join(sorted(unsupported))}",
            )
        return JudgeVerdict(label="grounded", confidence=0.9, rationale="every figure is supported")


def fixture_answers() -> dict[str, str]:
    """Recorded answers for the goldens that have them."""
    return {f.golden.golden_id: f.answer for f in load_fixture_goldens() if f.answer}
