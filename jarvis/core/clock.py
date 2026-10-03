"""What time it is for the person JARVIS talks to.

The owner's brain and pod both run on UTC; the owner does not. Everything that
says or reasons about "now" goes through here, in ``[general] timezone``.
"""

from __future__ import annotations

import datetime as _dt


def local(config, base: _dt.datetime | None = None) -> _dt.datetime:
    """``base`` (default: now) as an aware time in the configured zone, or in
    the machine's own when none is set."""
    base = base or _dt.datetime.now(_dt.timezone.utc)
    if base.tzinfo is None:
        base = base.astimezone()
    zone = getattr(getattr(config, "general", None), "timezone", "")
    if zone:
        from zoneinfo import ZoneInfo

        return base.astimezone(ZoneInfo(zone))
    return base.astimezone()
