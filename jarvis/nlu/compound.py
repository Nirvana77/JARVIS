"""M4.5: "set a timer for five minutes and find my watch" is two commands.

The classifier sees one sentence and one label; a sentence that is really two
commands lands on whichever half is louder, or on ``unknown``. ``split`` cuts
a transcript at "and" / "then" / a comma. Whether the pieces *are* commands
is not decided here: the orchestrator classifies each one and only treats the
sentence as a chain when every piece stands on its own
(``Orchestrator._compound``) — "rock and roll" and "5 minutes and 30 seconds"
have an "and" in them too.
"""

from __future__ import annotations

import re

#: more pieces than this is a sentence with commas in it, not a list of commands
MAX_CLAUSES = 4

#: Each clause must clear both, well above the classifier's own cutoffs: on
#: the live model real clauses score >= 0.77 / >= 0.86, and stray halves
#: ("garfunkel", "30 seconds", "dogs") never reach both.
MIN_CONFIDENCE = 0.6
MIN_SIMILARITY = 0.6

_JOIN = re.compile(
    r"(?:\s*[,.;]\s*|\s+)(?:and\s+then|then|and)\s+"   # "... and ...", "..., then ...", "... . Then ..."
    r"|\s*[,;]\s+",                                     # "..., ..."
    re.IGNORECASE,
)


def split(text: str) -> list[str]:
    """The clauses of a transcript that might be several commands, or ``[]``
    when it is not one (no joining word, an empty piece, too many pieces)."""
    clauses = [c.strip(" .,;!?") for c in _JOIN.split(text.strip())]
    if not 2 <= len(clauses) <= MAX_CLAUSES or not all(clauses):
        return []
    return clauses
