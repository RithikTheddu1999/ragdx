"""Resolving quoted evidence text to character offsets.

Humans label evidence by quoting a passage, not by counting characters. This
maps a quote back to an exact ``(start, end)`` span in the document body, with
whitespace normalized so a re-wrapped source document does not invalidate every
label.
"""

from __future__ import annotations


class SpanNotFoundError(ValueError):
    """The quoted evidence does not appear in the document."""


class AmbiguousSpanError(ValueError):
    """The quoted evidence appears more than once; the label is not a span."""


def normalize(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space.

    Returns the normalized text and, for each normalized character, the index it
    came from in ``text``.
    """
    out: list[str] = []
    origin: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space and out:
                out.append(" ")
                origin.append(i)
            in_space = True
        else:
            out.append(ch)
            origin.append(i)
            in_space = False
    while out and out[-1] == " ":
        out.pop()
        origin.pop()
    return "".join(out), origin


def find_span(text: str, quote: str) -> tuple[int, int]:
    """Locate ``quote`` in ``text``, returning offsets into the original text.

    Raises rather than guessing: an evidence label that matches nothing, or
    matches twice, is a broken label and silently picking an occurrence would
    put a wrong span into the ground truth.
    """
    normalized, origin = normalize(text)
    needle, _ = normalize(quote)
    if not needle:
        raise SpanNotFoundError("evidence quote is empty")
    first = normalized.find(needle)
    if first == -1:
        raise SpanNotFoundError(f"evidence not found in document: {needle[:80]!r}")
    if normalized.find(needle, first + 1) != -1:
        raise AmbiguousSpanError(f"evidence occurs more than once: {needle[:80]!r}")
    return origin[first], origin[first + len(needle) - 1] + 1
