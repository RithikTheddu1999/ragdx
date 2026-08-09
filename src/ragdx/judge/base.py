"""LLM judge protocol, plus the offline stub.

Per PLAN.md §2 the judge is allowed in exactly two places: synthesizing goldens
(Milestone 3) and faithfulness scoring (Milestone 6). Nothing in ``diagnose/``
may call it. Every verdict carries a confidence and may abstain.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class JudgeVerdict(BaseModel):
    """A single judgement. ``abstained`` beats a confident guess."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    rationale: str = ""


class Judge(Protocol):
    """Minimal LLM surface: one prompt in, one text or verdict out."""

    @property
    def name(self) -> str:
        """Model identifier, recorded in golden-set manifests."""
        ...

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Free-form completion. Used only by golden synthesis."""
        ...

    def judge(self, prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
        """Constrained judgement over ``labels``."""
        ...


def prompt_key(prompt: str, extra: object = None) -> str:
    """Content hash of a prompt, for the on-disk judge cache."""
    payload = json.dumps({"prompt": prompt, "extra": extra}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StubJudge:
    """Canned, deterministic judge so the suite runs with no API key.

    Responses are looked up by exact prompt hash. An unknown prompt abstains
    rather than inventing an answer — a stub that guesses would let a broken
    call site pass its tests.
    """

    def __init__(
        self,
        completions: dict[str, str] | None = None,
        verdicts: dict[str, JudgeVerdict] | None = None,
        name: str = "stub-judge-v1",
    ) -> None:
        self._completions = dict(completions or {})
        self._verdicts = dict(verdicts or {})
        self._name = name
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def set_completion(self, prompt: str, response: str) -> None:
        self._completions[prompt_key(prompt)] = response

    def set_verdict(self, prompt: str, verdict: JudgeVerdict) -> None:
        self._verdicts[prompt_key(prompt)] = verdict

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        key = prompt_key(prompt)
        self.calls.append(key)
        return self._completions.get(key, "")

    def judge(self, prompt: str, labels: tuple[str, ...]) -> JudgeVerdict:
        key = prompt_key(prompt)
        self.calls.append(key)
        verdict = self._verdicts.get(key)
        if verdict is None:
            return JudgeVerdict(
                label=labels[0] if labels else "",
                confidence=0.0,
                abstained=True,
                rationale="stub judge has no canned response for this prompt",
            )
        return verdict
