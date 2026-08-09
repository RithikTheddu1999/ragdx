"""Synthesize golden questions from corpus chunks, then verify them.

The verification step is the whole point and is not optional. Left unverified,
roughly a third of generated questions turn out to be answerable from anywhere
in the corpus — "what is the escalation window?" is answerable from six
different pages — and a golden like that does not test retrieval. It just adds
noise to the failure rate, which then makes every diagnosis downstream noise
too.

So each candidate is checked twice:

1. Can it be answered from the chunk it was generated from? If not, the
   generator asked about something it only half saw.
2. Can it *also* be answered from an unrelated chunk? If so, retrieving the
   "wrong" chunk would still produce the right answer, and the question cannot
   distinguish a working retriever from a broken one.

A candidate is kept only when the first succeeds and the second fails. The
judge may abstain at either step, and an abstention is a rejection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ragdx.corpus import Document
from ragdx.goldens.base import GoldenBatch, Rejection, RejectReason
from ragdx.judge.base import Judge
from ragdx.schema import Chunk, Golden
from ragdx.spans import AmbiguousSpanError, SpanNotFoundError, find_span
from ragdx.text import tokenize

ANSWERABLE = "answerable"
NOT_ANSWERABLE = "not_answerable"
LABELS = (ANSWERABLE, NOT_ANSWERABLE)


def generation_prompt(chunk: Chunk) -> str:
    return (
        "You are building an evaluation set for a retrieval system.\n"
        "Read the passage and write ONE question that can be answered *only* "
        "from this passage. Avoid pronouns and avoid referring to 'the passage' "
        "— the question must stand alone.\n\n"
        "Reply with JSON only, using exactly these keys:\n"
        '  {"question": "...", "evidence": "...", "answer": "..."}\n'
        "where `evidence` is the shortest verbatim quote from the passage that "
        "answers the question.\n\n"
        f"PASSAGE:\n{chunk.text}\n"
    )


def answerability_prompt(question: str, context: str) -> str:
    return (
        "Decide whether the QUESTION can be answered using only the CONTEXT.\n"
        f"Answer '{ANSWERABLE}' or '{NOT_ANSWERABLE}'. If the context is "
        "merely on a similar topic but does not contain the answer, the correct "
        f"reply is '{NOT_ANSWERABLE}'.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
    )


def _shuffle_key(seed: int, chunk_id: str) -> str:
    """Deterministic ordering that does not depend on the PRNG implementation."""
    return hashlib.blake2b(f"{seed}:{chunk_id}".encode(), digest_size=16).hexdigest()


def _parse_generation(raw: str) -> dict[str, str] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    question = str(parsed.get("question", "")).strip()
    evidence = str(parsed.get("evidence", "")).strip()
    if not question or not evidence:
        return None
    return {
        "question": question,
        "evidence": evidence,
        "answer": str(parsed.get("answer", "")).strip(),
    }


@dataclass(frozen=True)
class SynthesisConfig:
    """Knobs for golden synthesis."""

    n: int = 50
    seed: int = 0
    #: Chunks shorter than this carry too little to ask a standalone question about.
    min_tokens: int = 15
    #: How many candidates to attempt per golden wanted, before giving up.
    attempts_per_golden: int = 3


def _distractor_for(chunk: Chunk, ordered: list[Chunk]) -> Chunk | None:
    """A chunk from a different document, else any non-overlapping chunk."""
    for candidate in ordered:
        if candidate.doc_id != chunk.doc_id:
            return candidate
    for candidate in ordered:
        if candidate.chunk_id != chunk.chunk_id:
            return candidate
    return None


def synthesize(
    docs: list[Document],
    chunks: list[Chunk],
    judge: Judge,
    config: SynthesisConfig | None = None,
) -> GoldenBatch:
    """Generate and verify up to ``config.n`` goldens from ``chunks``."""
    cfg = config or SynthesisConfig()
    by_doc = {d.doc_id: d for d in docs}
    ordered = sorted(chunks, key=lambda c: _shuffle_key(cfg.seed, c.chunk_id))
    eligible = [c for c in ordered if len(tokenize(c.text)) >= cfg.min_tokens]

    goldens: list[Golden] = []
    rejections: list[Rejection] = []
    budget = cfg.n * cfg.attempts_per_golden

    for chunk in eligible[:budget]:
        if len(goldens) >= cfg.n:
            break
        source = chunk.chunk_id

        candidate = _parse_generation(judge.complete(generation_prompt(chunk)))
        if candidate is None:
            rejections.append(
                Rejection(
                    reason=RejectReason.MALFORMED,
                    detail="generator did not return usable JSON",
                    source=source,
                )
            )
            continue

        doc = by_doc.get(chunk.doc_id)
        if doc is None:
            rejections.append(
                Rejection(
                    reason=RejectReason.UNKNOWN_DOCUMENT, detail=chunk.doc_id, source=source
                )
            )
            continue

        # Resolve against the chunk, then offset into the document, so a quote
        # that appears twice in the corpus but once in the chunk still resolves.
        try:
            local_start, local_end = find_span(chunk.text, candidate["evidence"])
        except SpanNotFoundError as exc:
            rejections.append(
                Rejection(reason=RejectReason.EVIDENCE_NOT_FOUND, detail=str(exc), source=source)
            )
            continue
        except AmbiguousSpanError as exc:
            rejections.append(
                Rejection(reason=RejectReason.EVIDENCE_AMBIGUOUS, detail=str(exc), source=source)
            )
            continue
        start = chunk.char_start + local_start
        end = chunk.char_start + local_end

        positive = judge.judge(answerability_prompt(candidate["question"], chunk.text), LABELS)
        if positive.abstained:
            rejections.append(
                Rejection(
                    reason=RejectReason.JUDGE_ABSTAINED,
                    detail="abstained on the gold chunk",
                    source=source,
                )
            )
            continue
        if positive.label != ANSWERABLE:
            rejections.append(
                Rejection(
                    reason=RejectReason.NOT_ANSWERABLE_FROM_GOLD,
                    detail=positive.rationale,
                    source=source,
                )
            )
            continue

        distractor = _distractor_for(chunk, ordered)
        if distractor is None:
            rejections.append(Rejection(reason=RejectReason.NO_DISTRACTOR, source=source))
            continue

        negative = judge.judge(
            answerability_prompt(candidate["question"], distractor.text), LABELS
        )
        if negative.abstained:
            rejections.append(
                Rejection(
                    reason=RejectReason.JUDGE_ABSTAINED,
                    detail="abstained on the distractor chunk",
                    source=source,
                )
            )
            continue
        if negative.label != NOT_ANSWERABLE:
            rejections.append(
                Rejection(
                    reason=RejectReason.ANSWERABLE_FROM_DISTRACTOR,
                    detail=f"also answerable from {distractor.chunk_id}",
                    source=source,
                )
            )
            continue

        goldens.append(
            Golden(
                golden_id=f"syn-{len(goldens) + 1:04d}",
                query=candidate["question"],
                gold_doc_id=chunk.doc_id,
                gold_char_start=start,
                gold_char_end=end,
                expected_answer=candidate["answer"] or None,
                origin="synthetic",
                synth_confidence=positive.confidence,
            )
        )

    return GoldenBatch(goldens=goldens, rejections=rejections)
