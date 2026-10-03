"""Append a timestamped note. Ported from ``actions/write.py`` — but it takes
its text from a slot and touches only ``ctx``, so the old
``write.py`` -> ``command_helper`` import cycle is gone.

M4.5: a note is also what the device it was said to is asked to remember
(``ctx.memory``), so "where did I park?" can be answered later. The notes file
is written as before.

M1 only handles an inline note ("note that the wifi code is 1234"). A spoken
"what should I write?" dictation follow-up is a later polish (it needs the
orchestrator to re-open the capture window).

M5: the note is also mirrored into the knowledge base's docs dir, one paragraph
per note, and indexed there and then — so "what is the wifi code" finds it on
the next turn. ("Remember that ..." is the `remember` skill's now.)
"""

from __future__ import annotations

import logging
from datetime import datetime

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="note",
    description="Append a timestamped note to your notes file.",
    examples=[
        "note that the wifi code is 1234",
        "make a note that the meeting moved to friday",
        "write down buy milk on the way home",
        "note that call the dentist tomorrow",
    ],
    params={"text": {"type": "string", "required": False}},
    permissions=frozenset({"fs_write"}),
    voice="jarvis",
)

NOTES_FILE = "notes.txt"
#: the mirror in `[knowledge] docs_dir`
MIRROR_FILE = "dictated-notes.md"

log = logging.getLogger(__name__)


def _mirror(knowledge, text: str, now: datetime) -> None:
    """Best effort: the note itself is already safely in the notes file."""
    try:
        knowledge.docs_dir.mkdir(parents=True, exist_ok=True)
        path = knowledge.docs_dir / MIRROR_FILE
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{text} (noted {now.day} {now:%B %Y})\n\n")
        knowledge.index_file(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not mirror the note into the knowledge base: %s", exc)


def run(ctx, text: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return "What would you like me to note, sir?"
    now = datetime.now()
    path = ctx.data_dir / NOTES_FILE
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{now.isoformat(timespec='seconds')}  {text}\n")
    if ctx.knowledge is not None:
        _mirror(ctx.knowledge, text, now)
    memory = getattr(ctx, "memory", None)
    if memory is not None:
        memory.remember(text)
    return "Noted, sir."
