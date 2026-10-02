"""Remember a spoken fact (M5).

"Remember that I parked on level three" becomes a timestamped one-chunk entry
in the knowledge base — no file behind it — which `recall` (and "what is ...")
can then answer from.

M4.5: the device it was said to keeps it as well (``ctx.memory``), with the
id of the knowledge-base fact, so "what do you remember?" can read it back
and "forget everything I told you" can take it out of both. With the
knowledge base off, the device's memory still keeps it.
"""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="remember",
    description="Remember a fact you tell it, so it can be recalled later.",
    examples=[
        "remember that i parked on level three",
        "remember that the gate code is 7731",
        "remember my locker number is 52",
        "remember this the spare key is under the blue pot",
        "remember to call the dentist tomorrow",
        "don't forget that the meeting moved to friday",
        "don't forget the bins go out on thursday",
        "keep in mind that my sister's birthday is in march",
    ],
    params={"text": {"type": "string", "required": False}},
    permissions=frozenset({"fs_write"}),
)


def run(ctx, text: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return "What would you like me to remember?"
    memory = getattr(ctx, "memory", None)
    if ctx.knowledge is None and memory is None:
        return "My memory is switched off, sir."
    ref = ctx.knowledge.remember(text) if ctx.knowledge is not None else None
    if memory is not None:
        memory.remember(text, knowledge_ref=ref)
    return "I'll remember that."
