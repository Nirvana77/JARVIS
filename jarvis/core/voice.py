"""Did a persona rewrite keep the facts? (The owner's option C, 2026-10-03.)

When a line is not already in the persona's voice, ``Persona.phrase`` has the
local LLM rewrite it. Measured with ``qwen3:8b``, it sometimes dropped or
changed content: "1 sheep, 2 sheep, 3 sheep." came back as "One, sir. Two,
sir. Three, sir." A rewrite is kept only when

- it holds exactly the same numbers, read as digits or words ("5" = "five",
  "21" = "twenty-one"), and
- every content word of the original (four letters or more, not a function
  word, compared by a crude stem) is still in it.

Strict on purpose: when in doubt the original is spoken. That is plainer,
never wrong. Lines written in voice to begin with (``SkillManifest.voice``)
are never rewritten at all.
"""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES)

#: words of four letters or more that carry no fact of their own
_FUNCTION_WORDS = frozenset("""
    about above after again against also although always another anything
    around because been before being below between both cannot could does
    doing done down during each either else enough even ever every from
    further have having here hers herself himself however indeed into itself
    just least less like made make many maybe more most much must myself
    near need never none nothing okay once only other ought ours ourselves
    over perhaps please quite rather really right same shall should since
    some something still such sure than that their theirs them themselves
    then there these they this those though through thus till under until
    upon very want was were what whatever when where whether which while
    whom whose will with within without would your yours yourself
    yourselves certainly indeed apologies afraid sorry kindly
""".split())

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.:]\d+)*")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("-", " "))


def numbers(text: str) -> set[float]:
    """Every number in ``text``: digits ("3.5", the parts of "7:45") and
    number words ("twenty five", "a hundred")."""
    found: set[float] = set()
    current: float | None = None
    for token in _tokens(text) + ["."]:
        if token[0].isdigit():
            if current is not None:
                found.add(current)
                current = None
            for part in token.split(":"):
                found.add(float(part) if "." in part else int(part))
            continue
        word = token
        if word in _UNITS or word in _TENS:
            value = _UNITS.get(word, _TENS.get(word))
            current = value if current is None else current + value
        elif word in _SCALES:
            current = (current or 1) * _SCALES[word]
        elif word == "and" and current is not None:
            continue  # "a hundred and five"
        elif word in ("a", "an") and current is None:
            continue  # maybe "a hundred"; harmless otherwise
        else:
            if current is not None:
                found.add(current)
                current = None
    return found


def _stem(word: str) -> str:
    word = word.split("'")[0]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    words = (t.split("'")[0] for t in _tokens(text))  # "today's" is "today"
    return {
        _stem(w)
        for w in words
        if w.isalpha() and len(w) >= 4 and w not in _FUNCTION_WORDS and w not in _NUMBER_WORDS
    }


def facts_kept(original: str, rewrite: str) -> bool:
    """May ``rewrite`` be spoken instead of ``original``?"""
    if numbers(original) != numbers(rewrite):
        return False
    return content_words(original) <= content_words(rewrite)
