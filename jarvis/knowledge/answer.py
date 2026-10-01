"""Turning retrieved chunks into the line JARVIS speaks.

Two tiers, per the PRD's degradation chain, and neither of them is Claude:

- a local reasoner (Ollama) is up → it composes the answer from the top chunks,
  told to use nothing else;
- otherwise → the single best snippet, verbatim, in a canned frame with its
  source: *"From your notes, sir: … — from <source>."*
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from jarvis.knowledge.store import FACT, Hit

log = logging.getLogger(__name__)

#: about twenty seconds of speech; a chunk can be several times that
MAX_SNIPPET_CHARS = 320
#: scores this close are the same match; the newer one is the answer, because
#: yesterday's "I parked on level three" is not where the car is today
_NEAR_TIE = 0.05

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING_MARK = re.compile(r"(?:^|(?<=\s))#{1,6}\s+")

#: what the reasoner replies when the excerpts do not answer the question — a
#: chunk can clear the score bar without holding the answer
NO_ANSWER = "NO_ANSWER"

_SYSTEM = (
    "You answer a question using only the numbered excerpts from the user's own "
    "notes. Do not add anything that is not in them. If the excerpts do not "
    f"answer the question, reply with exactly {NO_ANSWER} and nothing else. If "
    "two excerpts disagree, trust the most recent one. Reply in one or two short "
    "sentences meant to be spoken aloud, and say which source the answer came "
    "from. No markdown, no lists."
)


def source_name(hit: Hit) -> str:
    """How a source is said out loud: a file by its name, a fact by its date."""
    if hit.kind == FACT:
        return f"what you told me on {hit.when.day} {hit.when:%B}"
    return re.sub(r"[-_]+", " ", Path(hit.source).stem).strip() or "your documents"


def best(hits: list[Hit]) -> Hit:
    """The top hit — or, among hits scoring practically the same, the newest."""
    near = [h for h in hits if hits[0].score - h.score <= _NEAR_TIE]
    return max(near, key=lambda h: h.when)


def snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """``text`` as something worth saying: no heading marks, and cut at the end
    of a sentence once it runs past ``max_chars``."""
    text = " ".join(_HEADING_MARK.sub("", text).split())
    if len(text) <= max_chars:
        return text
    kept = ""
    for sentence in _SENTENCE.split(text):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    return kept or text[:max_chars].rsplit(" ", 1)[0]


def frame(hit: Hit) -> str:
    """The CPU-only tier: the best snippet, verbatim, with where it came from."""
    return f"From your notes, sir: {snippet(hit.text).rstrip('.!? ')} — from {source_name(hit)}."


def _prompt(question: str, hits: list[Hit]) -> str:
    excerpts = "\n".join(
        f"[{i}] (source: {source_name(hit)}, {hit.when.day} {hit.when:%B %Y}) "
        f"{snippet(hit.text, 600)}"
        for i, hit in enumerate(hits, start=1)
    )
    return f"Excerpts:\n{excerpts}\n\nQuestion: {question}"


def compose(question: str, hits: list[Hit], llm=None) -> str | None:
    """The spoken answer to ``question`` from ``hits`` (best first, not empty),
    or ``None`` when the reasoner read them and found no answer there."""
    top = best(hits)
    if llm is not None and getattr(llm, "available", False):
        try:
            reply = " ".join((llm.generate(_SYSTEM, _prompt(question, hits)) or "").split())
        except Exception as exc:  # noqa: BLE001 - degrade to the snippet
            log.warning("knowledge: reasoner failed, reading the snippet instead: %s", exc)
            reply = ""
        if NO_ANSWER.lower() in reply.lower():
            return None
        if reply:
            sources = {source_name(hit).lower() for hit in hits}
            if not any(name in reply.lower() for name in sources):
                reply = f"{reply.rstrip('.!? ')} — from {source_name(top)}."
            return reply
    return frame(top)
