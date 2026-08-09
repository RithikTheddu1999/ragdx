"""Resolving user-supplied objects from ``module:attribute`` strings.

ragdx bundles no model client and no embedding model — PLAN.md §4 lists the
dependencies and none of them is either. Users point config at their own, and
this is the one place that indirection is implemented.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, TypeVar, cast

T = TypeVar("T")


class PluginNotFoundError(ValueError):
    """A configured plugin could not be resolved."""


def load_plugin(spec: str, protocol: Any, kind: str, default: T) -> T:
    """Resolve ``module:attribute`` into an object satisfying ``protocol``.

    The literal ``"stub"`` returns ``default``. Only functions and classes are
    *called* to produce the object — calling an arbitrary callable blind would
    happily invoke something like a CLI app object as a side effect.
    """
    if spec == "stub":
        return default
    if ":" not in spec:
        raise PluginNotFoundError(
            f"{kind} {spec!r} is not 'module:attribute' (or the literal 'stub')"
        )
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PluginNotFoundError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        target: Any = getattr(module, attribute)
    except AttributeError as exc:
        raise PluginNotFoundError(f"{module_name!r} has no attribute {attribute!r}") from exc

    candidate = target() if (inspect.isfunction(target) or inspect.isclass(target)) else target
    if not isinstance(candidate, protocol):
        raise PluginNotFoundError(
            f"{spec!r} resolved to {type(candidate).__name__}, which is not a {kind}"
        )
    return cast(T, candidate)
