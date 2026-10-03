"""M7: the learning's own bookkeeping, ``<data>/learning/state.json``.

- **Daily counters**: skills built autonomously today (``max_builds_per_day``)
  and repairs per skill today (``max_repairs_per_skill_per_day``). They cap
  what JARVIS may spend on Claude by itself.
- **Build requests**: what was asked for today and what became of it, so the
  same request is not built twice in a day.
- **Disabled skills**: a learned skill that kept failing after its repairs.
  It stays off until the owner enables it again (``jarvis learning enable``,
  or teaching / editing / reverting it).
- **Events**: what was learned, for "what have you learned today?".
- **Confusions**: label pairs that were corrected, which keep the record of
  what gets mixed up.

``builtin-failures.jsonl`` beside it is a builtin skill that raised: builtins
are repo code, and a person fixes those.
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path
from typing import Callable

from jarvis.learning._files import locked, read_json, write_json

#: daily counters kept this long
_KEEP_DAYS = 30
#: events kept, newest last
_MAX_EVENTS = 2000


class LearningState:
    def __init__(
        self,
        path: Path | None,
        *,
        now: Callable[[], _dt.datetime] = _dt.datetime.now,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._now = now
        self._data: dict = {}
        self._lock = threading.Lock()

    # -- storage --------------------------------------------------------------

    def _load(self) -> dict:
        data = self._data if self.path is None else read_json(self.path, None)
        if not isinstance(data, dict):
            data = {}
        for name, empty in (
            ("builds", {}), ("repairs", {}), ("disabled", {}),
            ("requests", []), ("events", []), ("confusions", {}),
        ):
            data.setdefault(name, empty)
        return data

    def _read(self) -> dict:
        if self.path is None:
            with self._lock:
                return self._load()
        return self._load()

    def _update(self, change):
        if self.path is None:
            with self._lock:
                self._data = self._load()
                return change(self._data)
        with locked(self.path):
            data = self._load()
            result = change(data)
            self._prune(data)
            write_json(self.path, data)
            return result

    def _today(self) -> str:
        return self._now().date().isoformat()

    def _prune(self, data: dict) -> None:
        cutoff = (self._now() - _dt.timedelta(days=_KEEP_DAYS)).date().isoformat()
        data["builds"] = {d: n for d, n in data["builds"].items() if d >= cutoff}
        data["repairs"] = {d: n for d, n in data["repairs"].items() if d >= cutoff}
        data["requests"] = [r for r in data["requests"] if r.get("day", "") >= cutoff]
        data["events"] = data["events"][-_MAX_EVENTS:]

    # -- daily counters -------------------------------------------------------

    def builds_today(self) -> int:
        return int(self._read()["builds"].get(self._today(), 0))

    def count_build(self) -> None:
        day = self._today()

        def change(data):
            data["builds"][day] = int(data["builds"].get(day, 0)) + 1

        self._update(change)

    def repairs_today(self, skill: str) -> int:
        return int(self._read()["repairs"].get(self._today(), {}).get(skill, 0))

    def count_repair(self, skill: str) -> None:
        day = self._today()

        def change(data):
            today = data["repairs"].setdefault(day, {})
            today[skill] = int(today.get(skill, 0)) + 1

        self._update(change)

    # -- disabled skills ------------------------------------------------------

    def disable(self, skill: str, reason: str) -> None:
        stamp = self._now().isoformat(timespec="seconds")

        def change(data):
            data["disabled"][skill] = {"reason": reason, "ts": stamp}

        self._update(change)

    def enable(self, skill: str) -> bool:
        def change(data):
            return data["disabled"].pop(skill, None) is not None

        return self._update(change)

    def is_disabled(self, skill: str) -> bool:
        return skill in self._read()["disabled"]

    def disabled(self) -> dict[str, dict]:
        return dict(self._read()["disabled"])

    # -- build requests -------------------------------------------------------

    def add_request(self, text: str, description: str, *, name: str, status: str) -> None:
        day = self._today()

        def change(data):
            data["requests"].append(
                {"text": text, "description": description, "name": name,
                 "status": status, "day": day}
            )

        self._update(change)

    def set_request_status(self, name: str, status: str) -> None:
        def change(data):
            for r in reversed(data["requests"]):
                if r["name"] == name:
                    r["status"] = status
                    return

        self._update(change)

    def requests_today(self) -> list[dict]:
        today = self._today()
        return [r for r in self._read()["requests"] if r.get("day") == today]

    # -- events and confusions -------------------------------------------------

    def add_event(self, kind: str, text: str) -> None:
        stamp = self._now().isoformat(timespec="seconds")

        def change(data):
            data["events"].append({"ts": stamp, "kind": kind, "text": text})

        self._update(change)

    def events(self, since: _dt.date | None = None) -> list[dict]:
        events = self._read()["events"]
        if since is None:
            return list(events)
        start = since.isoformat()
        return [e for e in events if e["ts"][:10] >= start]

    def confusion(self, heard_as: str, meant: str) -> int:
        """One more correction from ``heard_as`` to ``meant``; the new count."""
        pair = f"{heard_as} -> {meant}"

        def change(data):
            data["confusions"][pair] = int(data["confusions"].get(pair, 0)) + 1
            return data["confusions"][pair]

        return self._update(change)

    def confusions(self) -> dict[str, int]:
        return dict(self._read()["confusions"])

    # -- builtins, for a person -----------------------------------------------

    def builtin_failure(self, skill: str, heard: str, params: dict, error: str) -> None:
        if self.path is None:
            return
        line = {
            "ts": self._now().isoformat(timespec="seconds"), "skill": skill,
            "heard": heard, "params": params, "error": error,
        }
        target = self.path.with_name("builtin-failures.jsonl")
        with locked(target):
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
