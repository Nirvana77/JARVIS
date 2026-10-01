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

Current is an estimate too: the watch's PMU (AXP2101) measures no current, so
it is the fuel gauge's fall on battery times the cell's capacity (BATTERY_MAH).
One percent is 4 mAh, so it needs a while in a mode to mean anything: under
10 minutes or under 1 % it is not given at all.
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
#: newer firmware appends its own estimate of the current: a 17th column, so
#: rows (and files) from before it still read
_COLUMNS_NEW = _COLUMNS + ["ma_est"]
MODES = ("awake", "watch", "offline")
_MODE_NAME = {"awake": "awake", "watch": "watch mode", "offline": "offline"}
#: two rows further apart than this are not one stretch (a reboot, a gap)
_MAX_STEP_S = 60
#: the watch's cell (3.7 V LiPo); what turns the gauge's % into mAh
BATTERY_MAH = 400
#: a current estimate needs at least this much on battery in the mode...
_MIN_CURRENT_S = 600
#: ...and the gauge to have moved this many percent
_MIN_CURRENT_PCT = 1
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
    ma_est: float | None = None  # the watch's own: its gauge, regressed over 30 min


@dataclass
class ModeStats:
    seconds: float = 0.0
    asleep_s: float = 0.0     # sleep_pct-weighted
    drop_mv: float = 0.0      # battery voltage lost while on battery
    battery_s: float = 0.0    # and over how long
    drop_pct: float = 0.0     # the gauge's fall while on battery
    gauge_s: float = 0.0      # and over how long
    est_ma_s: float = 0.0     # the watch's own ma_est, time-weighted
    est_s: float = 0.0

    @property
    def watch_ma(self) -> float | None:
        """The watch's own estimate (ma_est), averaged over the mode."""
        return self.est_ma_s / self.est_s if self.est_s else None

    @property
    def sleep_pct(self) -> float | None:
        return 100.0 * self.asleep_s / self.seconds if self.seconds else None

    @property
    def mv_per_h(self) -> float | None:
        return 3600.0 * self.drop_mv / self.battery_s if self.battery_s >= 60 else None

    def ma(self, capacity_mah: float = BATTERY_MAH) -> float | None:
        """Average current in this mode, estimated from the gauge; None when
        there is too little to go on."""
        if self.gauge_s < _MIN_CURRENT_S or self.drop_pct < _MIN_CURRENT_PCT:
            return None
        return self.drop_pct / 100.0 * capacity_mah / (self.gauge_s / 3600.0)


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
            if len(fields) not in (len(_COLUMNS), len(_COLUMNS_NEW)) or fields[0] == "time":
                continue
            rec = dict(zip(_COLUMNS_NEW, fields))
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
                    ma_est=_float(rec.get("ma_est", "")),
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
            if row.ma_est is not None:
                stats.est_ma_s += step * row.ma_est
                stats.est_s += step
            on_battery = not (row.charging or row.usb or prev.charging or prev.usb)
            if on_battery and prev.mode == row.mode and row.batt_mv and prev.batt_mv:
                stats.drop_mv += prev.batt_mv - row.batt_mv
                stats.battery_s += step
            if (on_battery and prev.mode == row.mode and row.batt_pct is not None
                    and prev.batt_pct is not None and row.batt_pct >= 0 and prev.batt_pct >= 0):
                stats.drop_pct += prev.batt_pct - row.batt_pct
                stats.gauge_s += step
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
    lines.append(f"{'mode':<9}{'time':>9}{'share':>7}{'asleep':>8}{'drain':>12}{'~current':>10}")
    for mode in MODES:
        m = s.modes[mode]
        if not m.seconds:
            lines.append(f"{mode:<9}{'-':>9}")
            continue
        share = 100.0 * m.seconds / total
        sleep = f"{m.sleep_pct:.0f}%"
        drain = f"{m.mv_per_h:.0f} mV/h" if m.mv_per_h is not None else "-"
        ma = m.ma() if m.ma() is not None else m.watch_ma  # ours, else the watch's own
        current = f"{ma:.0f} mA" if ma is not None else "-"
        lines.append(f"{mode:<9}{_duration(m.seconds):>9}{share:>6.0f}%{sleep:>8}{drain:>12}{current:>10}")
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
    lines.append(f"~current: from the gauge's fall on battery ({BATTERY_MAH} mAh cell), "
                 f"once a mode has {_MIN_CURRENT_S // 60} min and {_MIN_CURRENT_PCT}% of it")
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
        ma = watch.ma() if watch.ma() is not None else watch.watch_ma
        drawing = f", drawing about {ma:.0f} mA" if ma is not None else ""
        lines.append(f"In watch mode it slept {watch.sleep_pct:.0f}% of the time{drawing}.")
    if s.last_pct is not None:
        lines.append(
            f"The battery is at {s.last_pct}%" + (" and charging." if s.charging_now else ".")
        )
    unplanned = [b.reason for b in s.boots if b.reason not in ("power on", "restart", "usb")]
    if unplanned:
        lines.append(f"It restarted unexpectedly {len(unplanned)} time{'s' if len(unplanned) > 1 else ''}: {', '.join(unplanned)}.")
    return " ".join(lines)
