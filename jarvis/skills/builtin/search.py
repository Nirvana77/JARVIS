"""Look something up on Wikipedia. Ported from ``actions/search.py``."""

from __future__ import annotations

import logging
import re

from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)


def _spoken_summary(text: str, max_sentences: int = 2) -> str:
    """MediaWiki's ``exsentences`` is unreliable, so trim here: drop everything
    from the first ``== section ==`` header and keep the first N sentences."""
    text = re.split(r"\n=+\s", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()

MANIFEST = SkillManifest(
    name="search",
    description="Look something up on Wikipedia and read a short summary.",
    examples=[
        "search black holes",
        "search for the speed of sound",
        "look up the eiffel tower",
        "what is photosynthesis",
        "tell me about jupiter",
        "who is ada lovelace",
    ],
    params={"query": {"type": "string", "required": True}},
    permissions=frozenset({"pure", "net"}),
)


def run(ctx, query: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "What would you like me to look up?"

    import wikipedia

    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)
        return f"According to Wikipedia: {_spoken_summary(summary)}"
    except wikipedia.exceptions.DisambiguationError as exc:
        options = ", ".join(exc.options[:3])
        return f"That could mean a few things: {options}. Which did you mean?"
    except wikipedia.exceptions.PageError:
        return f"I couldn't find a Wikipedia page for {query}."
    except Exception as exc:  # noqa: BLE001 - network/parse trouble
        log.warning("wikipedia search for %r failed: %s", query, exc)
        return "I had trouble reaching Wikipedia."
