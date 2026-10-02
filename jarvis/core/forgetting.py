"""Which of a device's facts "forget about the park" means.

Two scores, the higher one counts: the share of the query's words found among
the fact's (on crude stems, so "park" finds "parked" and "parking" — MiniLM
puts "the park" at 0.37 from "I parked on level 2"), and, when an embedder is
given, their cosine similarity (so "my car" can find a fact about parking).
Every fact that clears ``MIN_SCORE`` is offered, best first, up to
``MAX_OFFERED`` — the speaker hears them all in one question before anything
goes.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

MIN_SCORE = 0.5
MAX_OFFERED = 3

_STOP = frozenset(
    "a an the my me i you your to of on in at for about that this it is was what "
    "all any every everything note notes thing things".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def matching(
    query: str,
    facts: Sequence[str],
    *,
    embed: Callable[[str], np.ndarray] | None = None,
) -> list[str]:
    """The facts ``query`` is about, best first; ``[]`` when none is."""
    wanted = _words(query)
    if not wanted or not facts:
        return []
    q = embed(query) if embed is not None else None
    scored = []
    for order, fact in enumerate(facts):
        score = len(wanted & _words(fact)) / len(wanted)
        if q is not None:
            score = max(score, float(np.dot(q, embed(fact))))
        if score >= MIN_SCORE:
            scored.append((-score, order, fact))
    return [fact for _s, _o, fact in sorted(scored)[:MAX_OFFERED]]
