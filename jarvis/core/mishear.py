"""M4: is this transcript a mishearing of something JARVIS knows?

STT is at its worst on the words JARVIS most needs right — a freshly-taught
skill's name has no language-model prior behind it, so whisper reaches for the
nearest real word ("flip a coin" -> "flip a corn"). When the NLU gives up on a
transcript, the orchestrator asks the local reasoner for the command that was
most likely *said*, and this module is the pure part of that: what goes into
the prompt, and what is accepted back out of the reply.

Nothing here acts on a guess. The orchestrator confirms it by voice and runs
it through the real classifier (``Orchestrator._reason_about_unclear``).
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Mapping

#: "a few" of a skill's ``MANIFEST.examples`` — enough to show how it is
#: phrased without the prompt growing with every skill that is taught
EXAMPLES_PER_COMMAND = 6

#: A mishearing keeps most of the sounds, so the guess has to resemble what
#: was heard ("flip of corn" / "flip a coin" is 0.74; an unrelated command is
#: around 0.3). A small model would often rather answer than say "none", and
#: this is what keeps a stray sentence from becoming a "did you mean".
MIN_SOUNDALIKE = 0.5

SYSTEM = (
    "You correct speech-recognition mistakes for a voice assistant. You are "
    "given the commands the assistant knows and what the recogniser heard. "
    "If what was heard sounds like one of those commands spoken aloud, with "
    "similar-sounding words swapped in, reply with the command the speaker "
    "most likely said, keeping their own details (names, numbers, what to "
    "search for). If nothing fits, reply with exactly: none\n"
    "Reply with the command alone on one line: no explanation, no quotes."
)

_NONE = {"none", "n/a", "na", "no", "nothing", "unknown", "null"}
_LABEL = re.compile(
    r"^(?:the\s+)?(?:intended\s+command|command|correction|corrected|guess|answer)\s*:\s*",
    re.IGNORECASE,
)
_QUOTES = "\"'“”‘’`"


def vocabulary(manifests: Iterable, intent_meta: Mapping) -> list[str]:
    """One line per thing JARVIS can be asked for: the seed intents' patterns,
    and each registered skill's name with a few of its examples. A builtin
    skill is both, and gets one line."""
    known: dict[str, list[str]] = {}

    def add(label: str, phrases: Iterable[str]) -> None:
        seen = known.setdefault(label, [])
        for phrase in phrases:
            phrase = " ".join(phrase.replace("{intent}", "<something>").split())
            if phrase and phrase.lower() not in (s.lower() for s in seen):
                seen.append(phrase)

    for tag, seed in intent_meta.items():
        add(tag, seed.patterns)
    for manifest in manifests:
        add(manifest.name, list(manifest.examples)[:EXAMPLES_PER_COMMAND])
    return [f"{label}: {' | '.join(phrases)}" for label, phrases in known.items() if phrases]


def build_prompt(heard: str, known: list[str]) -> str:
    return (
        "Commands the assistant knows (name: ways of saying it):\n"
        + "\n".join(f"- {line}" for line in known)
        + f'\n\nThe recogniser heard: "{heard.strip()}"\n'
        "The command the speaker most likely said, or none:"
    )


def _sounds(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def same_words(a: str, b: str) -> bool:
    """The same thing said, give or take case, spacing and punctuation."""
    return _sounds(a) == _sounds(b)


def soundalike(a: str, b: str) -> float:
    """0..1, on letters and digits only — word boundaries are exactly what a
    mishearing moves ("philip acorn" / "flip a coin")."""
    return difflib.SequenceMatcher(None, _sounds(a), _sounds(b)).ratio()


def parse_guess(reply: str, heard: str) -> str | None:
    """The corrected phrase out of the reasoner's reply, or ``None`` when it
    declined, rambled, or only repeated what was heard."""
    line = next((ln.strip() for ln in (reply or "").splitlines() if ln.strip()), "")
    line = _LABEL.sub("", line).strip(_QUOTES + " ").rstrip(".!?").strip(_QUOTES + " ")
    if not line:
        return None
    first_word = re.split(r"[^a-z/]+", line.lower(), maxsplit=1)[0]
    if first_word in _NONE:
        return None
    # a command, not a sentence about one
    if len(line.split()) > 2 * len(heard.split()) + 4:
        return None
    if same_words(line, heard):
        return None
    return line
