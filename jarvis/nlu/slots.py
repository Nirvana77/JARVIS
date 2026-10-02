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
        "make a note of",
        "take a note that",
        "add a note that",
        "take note that",
        "note down that",
        "note that",
        "take a note",
        "note",
        "write down that",
        "write down",
        "jot down",
        "write",
    ],
    # M5. A recall question with none of these lead-ins ("where did i park") is
    # its own best query, so it is passed whole.
    "recall": [
        "what do my notes say about",
        "what did i tell you about",
        "what did i say about",
        "what do you know about",
        "what do you remember about",
        "check my notes for",
        "look in my notes for",
        "do i have any notes on",
        "do i have any notes about",
        "recall",
    ],
    "remember": [
        "remember that",
        "remember this",
        "remember",
        "don't forget that",
        "do not forget that",
        "keep in mind that",
        "i want you to remember",
        # a reminder is kept like any fact; nothing alerts at a time (yet)
        "remind me to",
        "remind me about",
        "remind me that",
        "remind me",
    ],
    # What a "forget …" is about: "forget about the park" -> "the park",
    # "remove the park note" -> "the park" (a trailing "note" is dropped too).
    "forget_fact": [
        "forget what i told you about",
        "forget what i said about",
        "forget everything about",
        "you can forget about",
        "forget about",
        "forget that",
        "forget",
        "stop remembering",
        "remove the note about",
        "remove my note about",
        "delete the note about",
        "delete my note about",
        "remove",
        "delete",
        "erase",
    ],
}

#: M4.5 + M5: skills that *store* what was said. They act only on an utterance
#: that has one of their own lead-ins — without it the slot filler keeps the
#: whole sentence, and "forget about the park", misheard as `remember`, would
#: be remembered (`has_lead_in`).
STORING = frozenset({"note", "remember"})

#: intent label -> the slot key its value is stored under
_SLOT_KEY: dict[str, str] = {
    "search": "query",
    "play": "query",
    "open_app": "app",
    "note": "text",
    "recall": "query",
    "remember": "text",
    "forget_fact": "query",
}

_TRAILING_NOTE = re.compile(r"\s+notes?$", re.IGNORECASE)

_TRAILING = re.compile(r"\b(please|jarvis|for me|now|thanks|thank you)\b", re.IGNORECASE)


def _strip_leading_fillers(tokens: list[str]) -> list[str]:
    while tokens and tokens[0].strip(",.!?").lower() in _LEADING_FILLERS:
        tokens.pop(0)
    return tokens


def _lead_in(label: str, text: str) -> tuple[str | None, str]:
    """The lead-in ``text`` starts with (longest first), and the rest."""
    phrase = " ".join(_strip_leading_fillers(text.split())).strip().rstrip(" ,.!?")
    lowered = phrase.lower()
    for prefix in sorted(_VERB_PREFIXES.get(label, []), key=len, reverse=True):
        if lowered == prefix:
            return prefix, ""
        if lowered.startswith(prefix + " "):
            return prefix, phrase[len(prefix) + 1 :]
    return None, phrase


def has_lead_in(label: str, text: str) -> bool:
    """Does ``text`` begin the way ``label`` is said ("remember that …")?"""
    return _lead_in(label, text)[0] is not None


def extract(label: str, text: str) -> dict[str, str]:
    if label not in _SLOT_KEY:
        return {}

    _prefix, phrase = _lead_in(label, text)
    phrase = _TRAILING.sub("", phrase)
    phrase = " ".join(phrase.split()).strip(" ,.!?")
    if label == "forget_fact":
        phrase = _TRAILING_NOTE.sub("", phrase)
    return {_SLOT_KEY[label]: phrase}


# -- typed params (edge tools) -------------------------------------------------
#
# An edge tool declares its params with a type instead of an intent-specific
# verb list (``jarvis/skills/edge.py``), so they are pulled out by type:
#
#   duration  "5 minutes", "an hour and a half", "1 hour and 30 minutes" -> seconds
#   number    "40", "seventy" -> 40, 70
#   text      what follows "to" / "that" / "saying" / "about" / "for", once any
#             duration phrase is taken out: "remind me in 20 minutes to take
#             the pizza out" -> "take the pizza out"
#   name      what a thing is called: after "called" / "named", or the words in
#             front of "timer" / "reminder" / "alarm": "set a pizza timer for
#             10 minutes", "cancel the pizza timer" -> "pizza"

PARAM_TYPES = ("duration", "number", "text", "name")

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
_TEXT_MARKERS = ("to", "that", "saying", "about", "for")
#: what a name param names, and what introduces one outright
_NAME_NOUNS = {"timer", "timers", "reminder", "reminders", "alarm", "alarms"}
_NAME_LEADS = ("called", "named")
#: words that end a name, walking back from the noun ("cancel the | pizza timer")
_NAME_STOPS = {
    "a", "an", "the", "my", "this", "that", "your", "all", "every", "any", "new", "other",
    "set", "start", "make", "create", "cancel", "stop", "delete", "clear", "remove", "end",
    "for", "in", "of", "please",
}


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


def _name(kept: list[tuple[str, str]]) -> str | None:
    """"a timer called pizza" / "the pizza timer" -> "pizza"."""
    norms = [n for _w, n in kept]
    for k, n in enumerate(norms):
        if n in _NAME_LEADS:
            taken = []
            for w, n2 in kept[k + 1 :]:
                if n2 in _TEXT_MARKERS or n2 in _NAME_NOUNS:
                    break
                taken.append(n2)
            return " ".join(taken).strip(" ,.!?") or None
    at = next((k for k, n in enumerate(norms) if n in _NAME_NOUNS), None)
    if at is None:
        return None
    taken = []
    for k in range(at - 1, -1, -1):
        n = norms[k]
        if n in _NAME_STOPS or _number_at(norms, k, articles=False) is not None or n in _UNITS:
            break
        taken.insert(0, n)
    return " ".join(taken).strip(" ,.!?") or None


def extract_typed(spec: dict, text: str) -> dict:
    """Pull the params ``spec`` declares (``{name: {"type": ...}}``) out of an
    utterance. A param that is not there is left out, so the skill can ask."""
    words = [w for w in re.split(r"[\s\-]+", text.strip()) if w]
    norm = [w.lower().strip(",.!?;:'\"") for w in words]
    seconds, spans = _durations(norm)

    # The words left once the durations are out, with the word that led into
    # each ("in 5 minutes", "for an hour"): what text and name are read from.
    drop: set[int] = set()
    for start, end in spans:
        drop.update(range(start, end))
        if start > 0 and norm[start - 1] in _DURATION_LEADS:
            drop.add(start - 1)
    kept = [(w, n) for k, (w, n) in enumerate(zip(words, norm)) if k not in drop]

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
        elif kind == "name":
            found = _name(kept)
            if found:
                out[name] = found
        elif kind == "text":
            at = next((k for k, (_w, n) in enumerate(kept) if n in _TEXT_MARKERS), None)
            if at is not None:
                phrase = " ".join(w for w, _n in kept[at + 1 :])
                phrase = _TRAILING.sub("", phrase)
                phrase = " ".join(phrase.split()).strip(" ,.!?")
                if phrase:
                    out[name] = phrase
    return out
