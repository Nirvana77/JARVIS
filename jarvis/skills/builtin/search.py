"""Look something up on Wikipedia. Ported from ``actions/search.py``."""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

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
        return f"According to Wikipedia: {summary}"
    except wikipedia.exceptions.DisambiguationError as exc:
        options = ", ".join(exc.options[:3])
        return f"That could mean a few things: {options}. Which did you mean?"
    except wikipedia.exceptions.PageError:
        return f"I couldn't find a Wikipedia page for {query}."
    except Exception:  # noqa: BLE001 - network/parse trouble
        return "I had trouble reaching Wikipedia."
