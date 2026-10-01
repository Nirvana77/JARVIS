"""The watch's power log, read on the brain.

The watch writes a CSV per day to its SD card (``jarvis-edge/main/powerlog.c``)
and sends it up when asked (``RemoteServer.fetch_power_log``); the copies live
in ``data/remote/power/<device_id>/``. A row every 10 s ("periodic": it has a
``sleep_pct``) and one per event. This turns them into what is worth knowing:
how long in each mode, how much of that asleep, how fast the battery fell in
each mode, the restarts and their reasons, and the drops.

Drain is the battery voltage's fall per hour between consecutive rows on
battery (not charging, no USB) in the same mode: a voltage, not a current, but
comparable between modes and days, and it needs no meter.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

HEADER = (
    "time,uptime_s,mode,screen,wifi,rssi,ps,link,batt_pct,batt_mv,charging,usb,"
    "sleep_pct,sleeps,heap_kb,event\n"
)
_COLUMNS = HEADER.strip().split(",")
MODES = ("awake", "watch", "offline")
_MODE_NAME = {"awake": "awake", "watch": "watch mode", "offline": "offline"}
#: two rows further apart than this are not one stretch (a reboot, a gap)
_MAX_STEP_S = 60
_BOOT_RE = re.compile(r"^boot \(([^)]*)\)")


@dataclass(frozen=True)
class Row:
    time: str          # "2026-10-01 08:00:00", or "-" before the clock was set
    uptime_s: int
    mode: str
    wifi: str
    link: bool
    batt_pct: int | None
    batt_mv: int | None
    charging: bool
    usb: bool
    sleep_pct: float | None  # periodic rows only
    event: str


@dataclass
class ModeStats:
    seconds: float = 0.0
    asleep_s: float = 0.0     # sleep_pct-weighted
    drop_mv: float = 0.0      # battery voltage lost while on battery
    battery_s: float = 0.0    # and over how long

    @property
    def sleep_pct(self) -> float | None:
        return 100.0 * self.asleep_s / self.seconds if self.seconds else None

    @property
    def mv_per_h(self) -> float | None:
        return 3600.0 * self.drop_mv / self.battery_s if self.battery_s >= 60 else None


@dataclass(frozen=True)
class Boot:
    time: str
    reason: str


@dataclass
class Summary:
    modes: dict[str, ModeStats] = field(default_factory=lambda: {m: ModeStats() for m in MODES})
    first_pct: int | None = None
    last_pct: int | None = None
    last_mv: int | None = None
    charging_now: bool = False
    boots: list[Boot] = field(default_factory=list)
    wifi_drops: int = 0
    link_drops: int = 0
    first_time: str = ""
    last_time: str = ""

    @property
    def seconds(self) -> float:
        return sum(m.seconds for m in self.modes.values())


def _int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(paths: Iterable[str | Path]) -> list[Row]:
    """Every parseable row of these files, in file order. A torn or foreign
    line is skipped, not fatal: the log is written by a watch that may lose
    power mid-line."""
    rows: list[Row] = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for fields in csv.reader(text.splitlines()):
            if len(fields) != len(_COLUMNS) or fields[0] == "time":
                continue
            rec = dict(zip(_COLUMNS, fields))
            uptime = _int(rec["uptime_s"])
            if uptime is None or rec["mode"] not in MODES:
                continue
            rows.append(
                Row(
                    time=rec["time"],
                    uptime_s=uptime,
                    mode=rec["mode"],
                    wifi=rec["wifi"],
                    link=rec["link"] == "1",
                    batt_pct=_int(rec["batt_pct"]),
                    batt_mv=_int(rec["batt_mv"]),
                    charging=rec["charging"] == "1",
                    usb=rec["usb"] == "1",
                    sleep_pct=_float(rec["sleep_pct"]),
                    event=rec["event"],
                )
            )
    return rows


def summarize(rows: list[Row]) -> Summary:
    s = Summary()
    prev: Row | None = None
    for row in rows:
        boot = _BOOT_RE.match(row.event)
        if boot:
            s.boots.append(Boot(row.time, boot.group(1)))
        elif row.event.startswith("Wi-Fi: dropped"):
            s.wifi_drops += 1
        elif row.event.startswith("link: dropped") or row.event.startswith("link: closed"):
            s.link_drops += 1
        if row.batt_pct is not None and row.batt_pct >= 0:
            if s.first_pct is None:
                s.first_pct = row.batt_pct
            s.last_pct = row.batt_pct
        if row.batt_mv:
            s.last_mv = row.batt_mv
        s.charging_now = row.charging or row.usb
        if row.time != "-":
            s.first_time = s.first_time or row.time
            s.last_time = row.time

        step = row.uptime_s - prev.uptime_s if prev is not None else 0
        if row.sleep_pct is not None and prev is not None and 0 < step <= _MAX_STEP_S:
            stats = s.modes[row.mode]
            stats.seconds += step
            stats.asleep_s += step * row.sleep_pct / 100.0
            on_battery = not (row.charging or row.usb or prev.charging or prev.usb)
            if on_battery and prev.mode == row.mode and row.batt_mv and prev.batt_mv:
                stats.drop_mv += prev.batt_mv - row.batt_mv
                stats.battery_s += step
        prev = row
    return s


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, m = seconds // 3600, seconds // 60 % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {seconds % 60:02d}s"


def report(s: Summary, title: str = "") -> str:
    """The long form, for ``python -m jarvis power``."""
    total = s.seconds
    lines = []
    if title:
        lines.append(f"{title} ({_duration(total)} logged)")
    lines.append(f"{'mode':<9}{'time':>9}{'share':>7}{'asleep':>8}{'drain':>12}")
    for mode in MODES:
        m = s.modes[mode]
        if not m.seconds:
            lines.append(f"{mode:<9}{'-':>9}")
            continue
        share = 100.0 * m.seconds / total
        sleep = f"{m.sleep_pct:.0f}%"
        drain = f"{m.mv_per_h:.0f} mV/h" if m.mv_per_h is not None else "-"
        lines.append(f"{mode:<9}{_duration(m.seconds):>9}{share:>6.0f}%{sleep:>8}{drain:>12}")
    battery = "battery: "
    if s.first_pct is not None:
        battery += f"{s.first_pct}% -> {s.last_pct}%"
    if s.last_mv:
        battery += f", {s.last_mv} mV"
    battery += ", charging" if s.charging_now else ""
    lines.append(battery)
    if s.boots:
        lines.append("restarts: " + ", ".join(f"{b.time.split(' ')[-1]} {b.reason}" for b in s.boots))
    lines.append(f"Wi-Fi drops: {s.wifi_drops}, link drops: {s.link_drops}")
    return "\n".join(lines)


def spoken(s: Summary) -> str:
    """A few sentences, for "how was the watch battery today?"."""
    total = s.seconds
    if not total:
        return "There's nothing in the watch's power log for today yet."
    parts = [
        f"{_MODE_NAME[mode]} {100.0 * s.modes[mode].seconds / total:.0f}%"
        for mode in MODES
        if s.modes[mode].seconds
    ]
    lines = ["Today the watch was " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + " of the time."]
    watch = s.modes["watch"]
    if watch.seconds and watch.sleep_pct is not None:
        lines.append(f"In watch mode it slept {watch.sleep_pct:.0f}% of the time.")
    if s.last_pct is not None:
        lines.append(
            f"The battery is at {s.last_pct}%" + (" and charging." if s.charging_now else ".")
        )
    unplanned = [b.reason for b in s.boots if b.reason not in ("power on", "restart", "usb")]
    if unplanned:
        lines.append(f"It restarted unexpectedly {len(unplanned)} time{'s' if len(unplanned) > 1 else ''}: {', '.join(unplanned)}.")
    return " ".join(lines)
