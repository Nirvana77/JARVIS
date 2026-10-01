"""How the watch's battery is doing: fetch its power log and say what it shows.

Only under ``python -m jarvis serve`` (``ctx.edges``). The watch sends up the
part of its SD-card power log the brain doesn't have yet, and the brain reads
the day — see ``jarvis/remote/powerlog.py``. When the watch can't be reached,
what the brain already has is still worth saying.
"""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="watch_power",
    description="Fetch the watch's power log and say how its battery is doing.",
    examples=[
        "how was the watch battery today",
        "how is the watch battery",
        "how much battery did the watch use",
        "how is my watch doing on battery",
        "fetch the power log",
        "get the power data from the watch",
        "what did the watch do today",
        "how much power did the watch use today",
        "check the watch battery",
    ],
    params={},
    permissions=frozenset({"net"}),
)


def run(ctx, **_params) -> str:
    if ctx.edges is None:
        return "The watch isn't connected to me."
    result = ctx.edges.power_report()
    if result.status == "ok":
        return result.spoken or "I fetched the power log, but there's nothing in it for today yet."
    if result.status == "unsupported":
        return "The watch can't send its power log; its firmware is too old."
    lead = (
        "The watch didn't send its power log in time."
        if result.status == "timeout"
        else "The watch isn't connected to me."
    )
    # What the brain already has is still worth hearing.
    if result.spoken and "nothing" not in result.spoken:
        return f"{lead} From what I have: {result.spoken}"
    return lead
