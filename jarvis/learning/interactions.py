"""M7 part B: the interaction log, one JSON line per turn.

``<data>/interactions/<device>/<YYYY-MM-DD>.jsonl``. A line is what was heard,
what the classifier made of it, which path the turn took and how it ended::

    {"id": "…", "ts": "2026-10-03T12:00:00", "device": "watch",
     "heard": "flip a corn", "label": "unknown", "confidence": 0.31,
     "runner_up": ["flip_a_coin", 0.28], "path": "mishear",
     "skill": "flip_a_coin", "params": {}, "outcome": "ok",
     "said": "Heads, sir.", "model": 29, "learned": "<phrasing id>"}

``path`` is one of ``direct | compound | confirmed | mishear | plan | answer |
unknown | correction | build`` or a session/meta action (``greeting``,
``teach``, …); ``outcome`` one of ``ok | error | declined | cancelled``.

It is what the learning reads back (recent ``ok`` turns are the regression
probes for a retrain) and what "what have you learned today?" summarises.
Audio is never logged, and only turns that reached the orchestrator, i.e.
that passed addressing, are logged.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable

from jarvis.core.memory import LOCAL, is_device_id

log = logging.getLogger(__name__)


class InteractionLog:
    def __init__(
        self,
        directory: Path | None,
        *,
        keep_days: int = 365,
        now: Callable[[], _dt.datetime] = _dt.datetime.now,
    ) -> None:
        #: ``None``: records in RAM only (tests, a log that is turned off)
        self.directory = Path(directory) if directory is not None else None
        self.keep_days = max(1, int(keep_days))
        self._now = now
        self._ram: list[dict] = []
        # the loop appends; the CLI and a worker thread may read
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def append(self, record: dict) -> dict:
        """Write one turn. ``id`` and ``ts`` are filled in when missing; the
        device falls back to ``local`` when it is not a valid id (it becomes a
        folder name). Never raises: a full disk costs the log, not the turn."""
        rec = dict(record)
        rec.setdefault("id", self.new_id())
        rec.setdefault("ts", self._now().isoformat(timespec="seconds"))
        device = rec.get("device")
        rec["device"] = device if is_device_id(device) else LOCAL
        with self._lock:
            if self.directory is None:
                self._ram.append(rec)
                return rec
            path = self.directory / rec["device"] / f"{rec['ts'][:10]}.jsonl"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as exc:
                log.error("could not write the interaction log (%s): %s", path, exc)
        return rec

    def records(
        self, device: str | None = None, since: _dt.datetime | _dt.date | None = None
    ) -> list[dict]:
        """Every logged turn, oldest first; one device's, and/or from ``since``."""
        since_ts = _since_ts(since)
        with self._lock:
            if self.directory is None:
                found = list(self._ram)
            else:
                found = self._read_files(device, since_ts)
        found = [
            r for r in found
            if (device is None or r.get("device") == device)
            and (since_ts is None or r.get("ts", "") >= since_ts)
        ]
        found.sort(key=lambda r: r.get("ts", ""))
        return found

    def _read_files(self, device: str | None, since_ts: str | None) -> list[dict]:
        if not self.directory.is_dir():
            return []
        out: list[dict] = []
        folders = [self.directory / device] if device else sorted(self.directory.iterdir())
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.jsonl")):
                if since_ts is not None and path.stem < since_ts[:10]:
                    continue
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue  # a line cut short by a crash
        return out

    def forget(self, device: str) -> int:
        """Delete everything logged for ``device`` ("forget everything I told
        you"). Returns how many turns went."""
        with self._lock:
            if self.directory is None:
                gone = [r for r in self._ram if r.get("device") == device]
                self._ram = [r for r in self._ram if r.get("device") != device]
                return len(gone)
            if not is_device_id(device):
                return 0
            folder = self.directory / device
            if not folder.is_dir():
                return 0
            count = 0
            for path in folder.glob("*.jsonl"):
                try:
                    count += sum(1 for _ in path.open(encoding="utf-8"))
                except OSError:
                    pass
            shutil.rmtree(folder, ignore_errors=True)
            return count

    def rotate(self) -> int:
        """Drop days older than ``keep_days``. Returns how many files went."""
        cutoff = (self._now() - _dt.timedelta(days=self.keep_days)).date().isoformat()
        with self._lock:
            if self.directory is None:
                before = len(self._ram)
                self._ram = [r for r in self._ram if r.get("ts", "")[:10] >= cutoff]
                return before - len(self._ram)
            if not self.directory.is_dir():
                return 0
            gone = 0
            for path in self.directory.glob("*/*.jsonl"):
                if path.stem < cutoff:
                    path.unlink(missing_ok=True)
                    gone += 1
            return gone


def _since_ts(since) -> str | None:
    if since is None:
        return None
    if isinstance(since, _dt.datetime):
        return since.isoformat(timespec="seconds")
    return since.isoformat()
