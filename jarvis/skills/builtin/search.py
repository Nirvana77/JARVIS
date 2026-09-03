"""Look something up on Wikipedia.

Ported from ``actions/search.py`` but no longer uses the abandoned ``wikipedia``
package (2014, unmaintained) — it hits the MediaWiki API directly with ``requests``
and a descriptive ``User-Agent``. Wikipedia now returns an empty body to
unidentified clients, which is what surfaced as
``Expecting value: line 1 column 1 (char 0)``.
"""

from __future__ import annotations

import logging
import re

import requests

from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)

_USER_AGENT = "JARVIS-voice-assistant/0.1 (personal local use)"
_API = "https://en.wikipedia.org/w/api.php"
_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_TIMEOUT = 8

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


def _spoken_summary(text: str, max_sentences: int = 2) -> str:
    """Trim to something worth speaking: drop any ``== section ==`` tail and keep
    the first couple of sentences."""
    text = re.split(r"\n=+\s", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers["User-Agent"] = _USER_AGENT
    return sess


def _search_title(sess: requests.Session, query: str) -> str | None:
    resp = sess.get(
        _API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
            "formatversion": 2,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _summary(sess: requests.Session, title: str) -> dict:
    slug = requests.utils.quote(title.replace(" ", "_"), safe="")
    resp = sess.get(_REST_SUMMARY + slug, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def run(ctx, query: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "What would you like me to look up?"

    try:
        sess = _session()
        title = _search_title(sess, query)
        if not title:
            return f"I couldn't find anything on Wikipedia for {query}."
        data = _summary(sess, title)
        if data.get("type") == "disambiguation":
            return f"{title} could mean several things. Can you be more specific?"
        extract = (data.get("extract") or "").strip()
        if not extract:
            return f"I found a page for {title} but no summary."
        return f"According to Wikipedia: {_spoken_summary(extract)}"
    except Exception as exc:  # noqa: BLE001 - network / parse trouble
        log.warning("wikipedia search for %r failed: %s", query, exc)
        return "I had trouble reaching Wikipedia."
