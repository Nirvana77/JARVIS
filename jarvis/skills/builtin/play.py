"""Search YouTube for something to play. Ported from ``actions/play.py``
(orphaned in the legacy code — it now has a ``play`` intent)."""

from __future__ import annotations

import urllib.parse
import webbrowser

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="play",
    description="Open a YouTube search for something to play.",
    examples=[
        "play lofi beats",
        "play some jazz",
        "put on the beatles",
        "play relaxing piano music",
        "play the news",
    ],
    params={"query": {"type": "string", "required": True}},
    permissions=frozenset({"net"}),
)


def run(ctx, query: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "What would you like me to play?"
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    webbrowser.open(url)
    return f"Playing {query} on YouTube."
