"""The time and the date, where the speaker is (``ctx.now()``,
``[general] timezone``). There was none until the owner asked the watch
"What is today's date?" and was told he had not asked JARVIS to remember
anything (2026-10-03)."""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="clock",
    description="Tell the current time and date.",
    examples=[
        "what time is it",
        "what's the time",
        "tell me the time",
        "what is today's date",
        "what's the date today",
        "what is the current date",
        "what day is it today",
        "what's the date and time",
        "what day of the week is it",
    ],
    params={},
    permissions=frozenset({"pure"}),
    voice="jarvis",
)


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def run(ctx, **_params) -> str:
    now = ctx.now()
    return f"It's {now:%H:%M} on {now:%A} the {ordinal(now.day)} of {now:%B}, sir."
