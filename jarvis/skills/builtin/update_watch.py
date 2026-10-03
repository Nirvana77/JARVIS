"""Update the watch: send the firmware staged in ``data/firmware/<device>.bin``
to the connected edge, now rather than on its next connect.

Only under ``python -m jarvis serve`` (``ctx.edges``). The brain announces the
image once this reply has finished playing, the watch downloads and checks it
itself, and restarts into it — see ``jarvis/remote/firmware.py``.
"""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="update_watch",
    description="Update the connected watch to the firmware staged on the brain.",
    examples=[
        "update the watch",
        "update my watch",
        "update the watch firmware",
        "install the new firmware on the watch",
        "flash the new firmware to the watch",
        "push the update to the watch",
        "upgrade the watch",
        "send the new firmware to my watch",
    ],
    params={},
    permissions=frozenset({"net"}),
    voice="jarvis",
)


def run(ctx, **_params) -> str:
    if ctx.edges is None:
        return "The watch isn't connected to me, sir."
    result = ctx.edges.update_firmware()
    if result.status == "sent":
        return f"Updating the watch to {result.version}, sir. It will restart when it's done."
    if result.status == "current":
        return f"The watch is already running {result.version}, sir."
    if result.status == "dev":
        return (
            f"The watch is running a development build, {result.running}. "
            f"It won't take {result.version} over that."
        )
    if result.status == "newer":
        return f"The watch is already running {result.running}, sir, newer than the {result.version} I have."
    if result.status == "none":
        return "I have no new firmware for the watch, sir."
    if result.status == "unsupported":
        return "That device can't update itself, sir."
    return "The watch isn't connected to me, sir."
