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


# -- typed params (edge tools) -------------------------------------------------
#
# An edge tool declares its params with a type instead of an intent-specific
# verb list (``jarvis/skills/edge.py``), so they are pulled out by type:
#
#   duration  "5 minutes", "an hour and a half", "1 hour and 30 minutes" -> seconds
#   number    "40", "seventy" -> 40, 70
#   text      what follows "to" / "that" / "saying" / "about", once any
#             duration phrase is taken out: "remind me in 20 minutes to take
#             the pizza out" -> "take the pizza out"

PARAM_TYPES = ("duration", "number", "text")

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}
#: what introduces a duration ("in 5 minutes", "for an hour"): taken out with it
_DURATION_LEADS = {"in", "for", "after", "within"}
#: what introduces a text param, first one wins
_TEXT_MARKERS = ("to", "that", "saying", "about")


def _number_at(norm: list[str], i: int, *, articles: bool) -> tuple[float, int] | None:
    """The number starting at word ``i``, and the index after it."""
    word = norm[i]
    if re.fullmatch(r"\d+(\.\d+)?", word):
        return float(word), i + 1
    if articles and word in ("a", "an"):
        return 1.0, i + 1
    if word in _TENS:
        if i + 1 < len(norm) and _ONES.get(norm[i + 1], 10) < 10:
            return float(_TENS[word] + _ONES[norm[i + 1]]), i + 2
        return float(_TENS[word]), i + 1
    if word in _ONES:
        return float(_ONES[word]), i + 1
    return None


def _and_a_half(norm: list[str], j: int) -> bool:
    return norm[j : j + 3] == ["and", "a", "half"]


def _durations(norm: list[str]) -> tuple[float, list[tuple[int, int]]]:
    """Total seconds of every "<number> <unit>" in the words, and their spans."""
    total, spans, i = 0.0, [], 0
    while i < len(norm):
        if norm[i] == "half" and i + 2 < len(norm) and norm[i + 1] in ("a", "an") \
                and norm[i + 2] in _UNITS:
            total += 0.5 * _UNITS[norm[i + 2]]
            spans.append((i, i + 3))
            i += 3
            continue
        found = _number_at(norm, i, articles=True)
        if found is None:
            i += 1
            continue
        value, j = found
        if _and_a_half(norm, j):  # "one and a half hours"
            value, j = value + 0.5, j + 3
        if j >= len(norm) or norm[j] not in _UNITS:
            i += 1
            continue
        unit = _UNITS[norm[j]]
        j += 1
        if _and_a_half(norm, j):  # "an hour and a half"
            value, j = value + 0.5, j + 3
        total += value * unit
        spans.append((i, j))
        i = j
    return total, spans


def extract_typed(spec: dict, text: str) -> dict:
    """Pull the params ``spec`` declares (``{name: {"type": ...}}``) out of an
    utterance. A param that is not there is left out, so the skill can ask."""
    words = [w for w in re.split(r"[\s\-]+", text.strip()) if w]
    norm = [w.lower().strip(",.!?;:'\"") for w in words]
    seconds, spans = _durations(norm)

    out: dict = {}
    for name, param in spec.items():
        kind = param.get("type") if isinstance(param, dict) else None
        if kind == "duration" and spans:
            out[name] = int(round(seconds))
        elif kind == "number":
            for i in range(len(norm)):
                found = _number_at(norm, i, articles=False)
                if found is not None:
                    value = found[0]
                    out[name] = int(value) if value.is_integer() else value
                    break
        elif kind == "text":
            # Take the durations out, with the word that led into them.
            drop: set[int] = set()
            for start, end in spans:
                drop.update(range(start, end))
                if start > 0 and norm[start - 1] in _DURATION_LEADS:
                    drop.add(start - 1)
            kept = [(w, n) for k, (w, n) in enumerate(zip(words, norm)) if k not in drop]
            at = next((k for k, (_w, n) in enumerate(kept) if n in _TEXT_MARKERS), None)
            if at is not None:
                phrase = " ".join(w for w, _n in kept[at + 1 :])
                phrase = _TRAILING.sub("", phrase)
                phrase = " ".join(phrase.split()).strip(" ,.!?")
                if phrase:
                    out[name] = phrase
    return out
