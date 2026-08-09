"""Resolving a judge from a config string.

ragdx deliberately bundles no LLM SDK — PLAN.md §4 lists the dependencies, and
none of them is a model client. Users point ragdx at their own judge with a
``module:attribute`` string; the attribute is either a ``Judge`` instance or a
zero-argument callable returning one. ``stub`` selects the offline stub, which
abstains on everything and is only useful for wiring tests.
"""

from __future__ import annotations

from ragdx.judge.base import Judge, StubJudge
from ragdx.plugins import PluginNotFoundError, load_plugin

STUB_SPEC = "stub"

#: Kept as its own name so existing callers and error handling do not change.
JudgeNotFoundError = PluginNotFoundError


def load_judge(spec: str = STUB_SPEC) -> Judge:
    """Resolve ``module:attribute`` (or ``stub``) into a Judge."""
    return load_plugin(spec, Judge, "judge", StubJudge())
