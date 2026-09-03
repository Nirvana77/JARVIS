"""Append a timestamped note. Ported from ``actions/write.py`` — but it takes
its text from a slot and touches only ``ctx``, so the old
``write.py`` -> ``command_helper`` import cycle is gone.

M1 only handles an inline note ("note that the wifi code is 1234"). A spoken
"what should I write?" dictation follow-up is a later polish (it needs the
orchestrator to re-open the capture window).
"""

from __future__ import annotations

from datetime import datetime

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="note",
    description="Append a timestamped note to your notes file.",
    examples=[
        "note that the wifi code is 1234",
        "remember that i parked on level three",
        "make a note that the meeting moved to friday",
        "write down buy milk on the way home",
        "note that call the dentist tomorrow",
    ],
    params={"text": {"type": "string", "required": False}},
    permissions=frozenset({"fs_write"}),
)

NOTES_FILE = "notes.txt"


def run(ctx, text: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return "What would you like me to note?"
    path = ctx.data_dir / NOTES_FILE
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {text}\n")
    return "Noted."
