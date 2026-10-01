"""Remember a spoken fact (M5).

"Remember that I parked on level three" becomes a timestamped one-chunk entry
in the knowledge base — no file behind it — which `recall` (and "what is ...")
can then answer from.
"""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="remember",
    description="Remember a fact you tell it, so it can be recalled later.",
    examples=[
        "remember that i parked on level three",
        "remember that the wifi code is 1234",
        "remember my locker number is 52",
        "remember this the spare key is under the blue pot",
        "don't forget that the meeting moved to friday",
        "keep in mind that the bins go out on thursday",
    ],
    params={"text": {"type": "string", "required": False}},
    permissions=frozenset({"fs_write"}),
)


def run(ctx, text: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return "What would you like me to remember?"
    if ctx.knowledge is None:
        return "My memory is switched off, sir."
    ctx.knowledge.remember(text)
    return "I'll remember that."
