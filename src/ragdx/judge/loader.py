"""Resolving a judge from a config string.

ragdx deliberately bundles no LLM SDK — PLAN.md §4 lists the dependencies, and
none of them is a model client. Users point ragdx at their own judge with a
``module:attribute`` string; the attribute is either a ``Judge`` instance or a
zero-argument callable returning one. ``stub`` selects the offline stub, which
abstains on everything and is only useful for wiring tests.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from ragdx.judge.base import Judge, StubJudge

STUB_SPEC = "stub"


class JudgeNotFoundError(ValueError):
    """The configured judge could not be resolved."""


def load_judge(spec: str = STUB_SPEC) -> Judge:
    """Resolve ``module:attribute`` (or ``stub``) into a Judge."""
    if spec == STUB_SPEC:
        return StubJudge()
    if ":" not in spec:
        raise JudgeNotFoundError(
            f"judge {spec!r} is not 'module:attribute' (or the literal {STUB_SPEC!r})"
        )
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise JudgeNotFoundError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        target: Any = getattr(module, attribute)
    except AttributeError as exc:
        raise JudgeNotFoundError(f"{module_name!r} has no attribute {attribute!r}") from exc

    # Only functions and classes are *called* to produce a judge. An arbitrary
    # callable object is treated as the judge itself — calling it blind would
    # happily invoke something like a CLI app object as a side effect.
    candidate = target() if (inspect.isfunction(target) or inspect.isclass(target)) else target
    if not isinstance(candidate, Judge):
        raise JudgeNotFoundError(
            f"{spec!r} resolved to {type(candidate).__name__}, which does not "
            f"implement complete() and judge()"
        )
    return candidate
