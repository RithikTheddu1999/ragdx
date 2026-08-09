"""Faithfulness on the generation plane.

The second — and last — sanctioned use of an LLM judge (PLAN.md §2). It runs
only for queries where the gold chunk **was** retrieved. If retrieval already
failed, the answer being wrong tells you nothing about the generator, and asking
would just add a noisy label on top of a diagnosis that is already certain.

That narrowness is the point: it cleanly separates "the retriever's fault" from
"the generator's fault", which is one of the most useful things the report says.
This is not a general eval metric suite and must not grow into one.
"""

from __future__ import annotations

from ragdx.judge.base import Judge, JudgeVerdict
from ragdx.schema import RetrievedChunk

GROUNDED = "grounded"
UNGROUNDED = "ungrounded"
FAITHFULNESS_LABELS = (GROUNDED, UNGROUNDED)


def faithfulness_prompt(query: str, answer: str, contexts: list[str]) -> str:
    joined = "\n\n---\n\n".join(contexts)
    return (
        "Decide whether the ANSWER is fully supported by the CONTEXT.\n"
        f"Reply '{GROUNDED}' if every factual claim in the answer is stated in "
        f"the context. Reply '{UNGROUNDED}' if the answer asserts anything the "
        "context does not support, including numbers, dates or conditions that "
        "differ from the context.\n"
        "Judge only support by the context. Do not use outside knowledge, and do "
        "not penalise an answer for leaving things out.\n\n"
        f"QUESTION:\n{query}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{joined}\n"
    )


def assess_faithfulness(
    judge: Judge,
    query: str,
    answer: str,
    retrieved: list[RetrievedChunk],
) -> JudgeVerdict:
    """Ask whether ``answer`` is grounded in what was actually retrieved.

    An empty context is not evidence of an ungrounded answer — it is evidence
    that there was nothing to ground it in — so the judge is not consulted and
    the verdict abstains.
    """
    contexts = [item.chunk.text for item in retrieved]
    if not contexts or not answer.strip():
        return JudgeVerdict(
            label=GROUNDED,
            confidence=0.0,
            abstained=True,
            rationale="no retrieved context or no answer to judge",
        )
    return judge.judge(faithfulness_prompt(query, answer, contexts), FAITHFULNESS_LABELS)
