"""Tokenization shared by the stub embedder and the BM25 lexical index.

Both planes must see the same tokens, otherwise a "lexical vs dense" ablation is
comparing two different notions of a word and the diagnosis is meaningless.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short. A big stopword list starts making topical decisions, which
# is the retriever's job, not the tokenizer's.
_STOPWORD_TEXT = """
    a an the and or but if then than that this these those of to in on at for from by with
    as is are was were be been being do does did doing have has had having it its i you we
    they he she them his her our your their what which who whom when where why how
    can could should would may might must will shall not no nor so such about into over
    under again further once here there all any both each few more most other some only
    own same too very s t
"""

STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokens, optionally minus stopwords.

    ``"NW-4417"`` becomes ``["nw", "4417"]`` — splitting identifiers is what makes
    a rare numeric component available to BM25's IDF weighting.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens
