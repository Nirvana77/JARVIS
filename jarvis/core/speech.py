"""Making an answer speakable, and cutting it into sentences (M3 decision 8).

Text written to be read has things in it that are unbearable said out loud. A
fenced code block is thirty seconds of punctuation; a path is a spelling test;
a URL with a query string is worse. So the voder is given the *spoken* version
of an answer, and the log keeps the whole of it.

The split into sentences is the other half of the latency budget: the first
sentence is synthesized and sent while the rest is still being made, so the
first sound out of the Pi's speaker comes within the budget even when the answer
is a paragraph.
"""

from __future__ import annotations

import re

#: Past this, an answer is cut at a sentence end and the user is told where the
#: rest is. About a minute of speech.
MAX_SPOKEN_CHARS = 600

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\((?:[^)]*)\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", re.M)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_RULE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", re.M)
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.S)
#: Struck-through text is text the writer took back. Emphasis keeps its words
#: and drops its markers; this drops both, because saying a retracted sentence
#: out loud says the opposite of what was meant.
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)
_URL = re.compile(r"\b(?:https?|ftp)://([^\s/:]+)(?::\d+)?(?:/\S*)?")
#: A path: at least two segments, one of them a directory, and no spaces. The
#: leading boundary keeps "and/or" (one slash, two plain words) out of it.
_PATH = re.compile(r"(?<![\w/])(?:~|\.{0,2}/|[\w.-]+/)[\w.-]*(?:/[\w.-]+)+")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{2,}")

#: Said once, however many blocks there were: the point is that there is code to
#: look at, and saying it three times is just as useless as reading it out.
CODE_PHRASE = "code on the screen"
REST_IN_LOG = "The rest is in the log."

#: Abbreviations and decimals that end in a full stop without ending a sentence.
_SENTENCE_END = re.compile(
    r"""
    (?<![A-Z][a-z]\.)          # "Mr." / "Dr." / "St."
    (?<!\b[A-Z]\.)             # an initial
    (?<!\d\.)                  # 3.5
    (?<=[.!?])                 # the end itself
    (?=["')\]]*\s)             # followed by whitespace, past any closing quote
    """,
    re.X,
)

#: A sentence longer than this is broken at a word boundary, so speech starts
#: sooner. Piper is fast, but a 60-word run is still a wait with no sound in it.
_MAX_PART_CHARS = 240


def speakable(text: str | None) -> str:
    """The version of ``text`` that is worth saying out loud."""
    if not text:
        return ""

    out = str(text)
    had_code = bool(_CODE_BLOCK.search(out))
    out = _CODE_BLOCK.sub(" ", out)
    out = _IMAGE.sub(r"\1", out)
    out = _LINK.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _RULE.sub("", out)
    out = _HEADING.sub("", out)
    out = _BULLET.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _STRIKE.sub("", out)
    out = _EMPHASIS.sub(r"\2", out)
    # A URL is its host; the path and query are unreadable out loud.
    out = _URL.sub(lambda m: m.group(1), out)
    # A path is its last part: the directories are context the listener has.
    out = _PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], out)

    # Line breaks become sentence breaks, so a list reads as sentences rather
    # than one long run-on.
    out = _BLANK_LINES.sub("\n", out)
    lines = [ln.strip() for ln in out.split("\n")]
    joined = ""
    for line in lines:
        if not line:
            continue
        if joined and not joined.endswith((".", "!", "?", ":", ";", ",")):
            joined += "."
        joined = f"{joined} {line}" if joined else line
    out = _SPACES.sub(" ", joined).strip()

    if had_code:
        out = f"{out} — {CODE_PHRASE}." if out else f"{CODE_PHRASE}."
        out = _SPACES.sub(" ", out).strip()

    return _truncate(out)


def _truncate(text: str) -> str:
    if len(text) <= MAX_SPOKEN_CHARS:
        return text
    head = text[:MAX_SPOKEN_CHARS]
    # Cut at the last sentence end that fits; failing that, the last word.
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut > MAX_SPOKEN_CHARS // 3:
        head = head[: cut + 1]
    else:
        space = head.rfind(" ")
        head = head[:space] if space > 0 else head
    return f"{head.rstrip()} {REST_IN_LOG}"


def split_sentences(text: str | None) -> list[str]:
    """Cut an answer into the parts the voder synthesizes one at a time.

    Never loses a word: the parts joined with a space are the original text's
    words in order.
    """
    if not text or not text.strip():
        return []
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if sentence:
            parts.extend(_chunk(sentence))
    return parts


def _chunk(sentence: str) -> list[str]:
    """Break one over-long sentence at word boundaries."""
    if len(sentence) <= _MAX_PART_CHARS:
        return [sentence]
    out: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) > _MAX_PART_CHARS and current:
            out.append(current)
            current = word
        else:
            current = candidate
    if current:
        out.append(current)
    return out
