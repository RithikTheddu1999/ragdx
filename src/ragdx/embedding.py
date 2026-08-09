"""Embedders.

``StubEmbedder`` is what lets the whole test suite run offline. It is not a toy
random projection: it reproduces the two behaviours that matter for differential
diagnosis on a real dense retriever.

1. **Rare terms get diluted.** A chunk's vector is the mean of its token vectors,
   so a single rare identifier buried in thirty other words barely moves it,
   and repeats are damped sublinearly rather than counted. That is exactly the
   signal BM25's IDF weighting keeps and a mean-pooled embedding loses, and it
   is what makes ``vocabulary_mismatch`` a real, reproducible condition offline.
2. **Some paraphrases bridge.** A small synonym table maps related words onto a
   shared token vector, so a query can match a chunk it shares no surface form
   with — the behaviour lexical search lacks, which is what ``paraphrase_gap``
   is about.

Vectors come from hashing, never from a PRNG, so they are identical across
platforms, numpy versions and runs.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ragdx.text import tokenize

Vectors = npt.NDArray[np.float64]

# Words that a competent dense embedder would place near each other. Each group
# collapses to its first member's vector.
DEFAULT_SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("refund", "reimbursement", "reimburse", "refunded", "repayment", "moneyback"),
    ("shipment", "shipping", "consignment", "delivery", "parcel", "package"),
    ("employee", "staff", "colleague", "worker", "personnel"),
    ("policy", "rule", "guideline", "standard"),
    ("vehicle", "truck", "van", "lorry", "fleet"),
    ("delay", "late", "overdue", "slippage"),
    ("customer", "client", "shipper", "account"),
    ("approve", "approval", "authorise", "authorisation", "signoff"),
    ("cost", "price", "fee", "charge", "surcharge"),
    ("document", "paperwork", "form", "record"),
)


class Embedder(Protocol):
    """Turns text into L2-normalized vectors."""

    @property
    def name(self) -> str:
        """Stable identifier, used in cache keys."""
        ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Return an ``(len(texts), dim)`` array of unit vectors."""
        ...


class StubEmbedder:
    """Deterministic hash-based embedder. No network, no model, no API key."""

    def __init__(
        self,
        dim: int = 768,
        synonyms: tuple[tuple[str, ...], ...] = DEFAULT_SYNONYMS,
        salt: str = "ragdx-stub-v1",
    ) -> None:
        self._dim = dim
        self._salt = salt
        self._canonical: dict[str, str] = {word: group[0] for group in synonyms for word in group}
        self._cache: dict[str, Vectors] = {}

    @property
    def name(self) -> str:
        return f"stub-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _token_vector(self, token: str) -> Vectors:
        canonical = self._canonical.get(token, token)
        cached = self._cache.get(canonical)
        if cached is not None:
            return cached
        vec = self._hash_vector(canonical)
        self._cache[canonical] = vec
        return vec

    def _hash_vector(self, canonical: str) -> Vectors:
        raw = bytearray()
        counter = 0
        while len(raw) < self._dim * 4:
            raw += hashlib.blake2b(
                f"{self._salt}:{canonical}:{counter}".encode(), digest_size=64
            ).digest()
            counter += 1
        ints = np.frombuffer(bytes(raw[: self._dim * 4]), dtype=">u4").astype(np.float64)
        vec = ints / 2147483647.5 - 1.0
        vec /= np.linalg.norm(vec)
        return vec

    def embed(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self._dim), dtype=np.float64)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            # Unweighted pooling, deliberately: this is the dilution described
            # above. Term frequency is sublinear because repeating a token four
            # times does not make a transformer embedding four times as sure.
            counts = Counter(tokens)
            acc = np.zeros(self._dim, dtype=np.float64)
            for token, tf in counts.items():
                acc += (1.0 + math.log(tf)) * self._token_vector(token)
            norm = float(np.linalg.norm(acc))
            if norm > 0.0:
                out[row] = acc / norm
        return out
