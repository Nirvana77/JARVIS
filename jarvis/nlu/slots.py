"""Rule-based slot extraction.

The classifier gives an intent label; this pulls the argument out of the raw
transcript. Deliberately small and deterministic — durations/times/richer slots
can grow here later. Intents with no argument return ``{}``.
"""

from __future__ import annotations

import re

#: filler words dropped from the front of an utterance before slot parsing
_LEADING_FILLERS = {"jarvis", "hey", "ok", "okay", "please", "could", "you", "can"}

#: verb phrases that introduce the slot value, per intent (longest matched first)
_VERB_PREFIXES: dict[str, list[str]] = {
    "search": [
        "search for",
        "search",
        "look up",
        "google",
        "find",
        "tell me about",
        "what is",
        "what's",
        "who is",
        "who's",
    ],
    "play": ["play", "put on", "start playing"],
    "open_app": ["open up", "open", "launch", "start", "go to"],
    "note": [
        "make a note that",
        "take a note that",
        "note that",
        "note",
        "remember that",
        "remember",
        "write down that",
        "write down",
        "write",
    ],
}

#: intent label -> the slot key its value is stored under
_SLOT_KEY: dict[str, str] = {
    "search": "query",
    "play": "query",
    "open_app": "app",
    "note": "text",
}

_TRAILING = re.compile(r"\b(please|jarvis|for me|now|thanks|thank you)\b", re.IGNORECASE)


def _strip_leading_fillers(tokens: list[str]) -> list[str]:
    while tokens and tokens[0].strip(",.!?").lower() in _LEADING_FILLERS:
        tokens.pop(0)
    return tokens


def extract(label: str, text: str) -> dict[str, str]:
    if label not in _SLOT_KEY:
        return {}

    phrase = " ".join(_strip_leading_fillers(text.split())).strip()
    lowered = phrase.lower()

    for prefix in sorted(_VERB_PREFIXES.get(label, []), key=len, reverse=True):
        if lowered == prefix:
            phrase = ""
            break
        if lowered.startswith(prefix + " "):
            phrase = phrase[len(prefix) + 1 :]
            break

    phrase = _TRAILING.sub("", phrase)
    phrase = " ".join(phrase.split()).strip(" ,.!?")
    return {_SLOT_KEY[label]: phrase}
