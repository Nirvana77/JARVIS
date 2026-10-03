"""Force update: send the staged firmware to the watch whatever it runs now.

"Update the watch" (``update_watch``) won't offer an image over a development
build (``-dirty``), the same version or a newer one, and the watch itself
refuses those too. This is the override: the brain offers it anyway and marks
the offer ``force``, so the watch also skips its version checks. Not its
battery check (at least 30 % or charging) or the image's checksum: those are
what keep it from being bricked.
"""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="force_update_watch",
    description="Force the staged firmware onto the watch, past its version checks.",
    examples=[
        "force update",
        "force update the watch",
        "force the watch to update",
        "force an update on my watch",
        "install the firmware anyway",
        "update the watch anyway",
        "force install the new firmware",
        "reinstall the watch firmware",
    ],
    params={},
    permissions=frozenset({"net"}),
    voice="jarvis",
)


def run(ctx, **_params) -> str:
    if ctx.edges is None:
        return "The watch isn't connected to me, sir."
    result = ctx.edges.update_firmware(force=True)
    if result.status == "sent":
        return (
            f"Forcing the watch onto {result.version}. If its battery is low it will wait for a "
            f"charger; otherwise it restarts when it's done."
        )
    if result.status == "none":
        return "I have no firmware for the watch to force onto it, sir."
    if result.status == "unsupported":
        return "That device can't update itself, sir."
    return "The watch isn't connected to me, sir."
