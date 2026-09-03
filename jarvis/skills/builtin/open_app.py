"""Open a website / web app in the browser. Ported from ``actions/openApp.py``
(the module the legacy ``open`` intent never resolved because of the name
mismatch)."""

from __future__ import annotations

import re
import webbrowser

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="open_app",
    description="Open a website or web app in the default browser.",
    examples=[
        "open github",
        "open youtube",
        "launch gmail",
        "open twitter for me",
        "go to wikipedia",
        "open up spotify",
    ],
    params={"app": {"type": "string", "required": True}},
    permissions=frozenset({"shell"}),
)


def run(ctx, app: str = "") -> str:
    name = re.sub(r"[^a-z0-9.-]", "", (app or "").strip().lower())
    if not name:
        return "Which app should I open?"
    url = name if "." in name else f"https://www.{name}.com"
    if not url.startswith("http"):
        url = f"https://{url}"
    webbrowser.open(url)
    return f"Opening {name}."
