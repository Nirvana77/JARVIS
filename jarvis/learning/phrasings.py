"""M7 part C: phrasings learned from use, the corpus's third source.

``<data>/learning/phrasings.json`` holds (text, label) pairs JARVIS picked up
from confirmed turns: the words that were actually *heard* when "Did you
mean 'flip a coin'?" got a yes, a one-step plan that was confirmed, an unsure
turn nobody corrected. ``build_corpus(learned=…)`` trains on them beside the
``intents.json`` seeds and the skills' own examples.

A pair that was undone (a correction, ``jarvis learning undo``, a retrain that
failed the regression gate) is *rejected*: kept in the same file so the same
lesson is never learned again.
"""

from __future__ import annotations

import datetime as _dt
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from jarvis.learning._files import key, locked, read_json, write_json


@dataclass
class Phrasing:
    id: str
    text: str
    label: str
    #: how it was learned: mishear | plan | confirmed | unsure | correction | build
    why: str
    ts: str
    #: the interaction-log id of the turn it came from
    turn: str | None = None
    #: in a model that was trained and swapped in
    trained: bool = False


class Phrasings:
    def __init__(
        self,
        path: Path | None,
        *,
        max_per_label: int = 50,
        now: Callable[[], _dt.datetime] = _dt.datetime.now,
    ) -> None:
        #: ``None``: RAM only
        self.path = Path(path) if path is not None else None
        self.max_per_label = max(1, int(max_per_label))
        self._now = now
        self._data = {"phrasings": [], "rejected": []}
        self._lock = threading.Lock()

    # -- storage --------------------------------------------------------------

    def _load(self) -> dict:
        if self.path is None:
            return self._data
        data = read_json(self.path, None)
        if not isinstance(data, dict):
            data = {}
        data.setdefault("phrasings", [])
        data.setdefault("rejected", [])
        return data

    def _update(self, change):
        """Read, change, write, all under the lock: ``change(data)`` mutates
        ``data`` in place and returns the caller's result."""
        if self.path is None:
            with self._lock:
                return change(self._data)
        with locked(self.path):
            data = self._load()
            result = change(data)
            write_json(self.path, data)
            return result

    # -- reading --------------------------------------------------------------

    def entries(self) -> list[Phrasing]:
        if self.path is None:
            with self._lock:
                rows = list(self._data["phrasings"])
        else:
            rows = self._load()["phrasings"]
        return [Phrasing(**row) for row in rows]

    def examples(self) -> list[tuple[str, str]]:
        """``(text, label)`` for the corpus."""
        return [(p.text, p.label) for p in self.entries()]

    def pending(self) -> list[Phrasing]:
        """Learned, not yet in a model that was swapped in."""
        return [p for p in self.entries() if not p.trained]

    def find(self, phrasing_id: str) -> Phrasing | None:
        return next((p for p in self.entries() if p.id == phrasing_id), None)

    def is_rejected(self, text: str, label: str) -> bool:
        rejected = self._load()["rejected"] if self.path is not None else self._data["rejected"]
        k = key(text)
        return any(r["label"] == label and key(r["text"]) == k for r in rejected)

    # -- changing -------------------------------------------------------------

    def add(
        self, text: str, label: str, *, why: str, turn: str | None = None
    ) -> Phrasing | None:
        """Learn that ``text`` means ``label``. ``None`` when it is already
        known or was rejected before. Past ``max_per_label`` the oldest
        learned phrasing of that label goes."""
        text = " ".join(text.split())
        k = key(text)
        if not k or not label:
            return None
        entry = Phrasing(
            id=uuid.uuid4().hex[:10], text=text, label=label, why=why,
            ts=self._now().isoformat(timespec="seconds"), turn=turn,
        )

        def change(data):
            if any(r["label"] == label and key(r["text"]) == k for r in data["rejected"]):
                return None
            if any(p["label"] == label and key(p["text"]) == k for p in data["phrasings"]):
                return None
            data["phrasings"].append(asdict(entry))
            same = [p for p in data["phrasings"] if p["label"] == label]
            if len(same) > self.max_per_label:
                same.sort(key=lambda p: p["ts"])
                drop = {p["id"] for p in same[: len(same) - self.max_per_label]}
                data["phrasings"] = [p for p in data["phrasings"] if p["id"] not in drop]
            return entry

        return self._update(change)

    def remove(self, phrasing_id: str, *, reason: str = "undone") -> Phrasing | None:
        """Unlearn one phrasing, and never learn it again."""
        found = self._take(lambda p: p["id"] == phrasing_id, reason)
        return found[0] if found else None

    def quarantine(self, ids, *, reason: str = "regression") -> list[Phrasing]:
        """A retrain with these in it got worse: take them out, reject them."""
        wanted = set(ids)
        return self._take(lambda p: p["id"] in wanted, reason)

    def _take(self, match, reason: str) -> list[Phrasing]:
        stamp = self._now().isoformat(timespec="seconds")

        def change(data):
            gone = [p for p in data["phrasings"] if match(p)]
            data["phrasings"] = [p for p in data["phrasings"] if not match(p)]
            for p in gone:
                data["rejected"].append(
                    {"text": p["text"], "label": p["label"], "why": reason, "ts": stamp}
                )
            return [Phrasing(**p) for p in gone]

        return self._update(change)

    def mark_trained(self, ids) -> None:
        wanted = set(ids)

        def change(data):
            for p in data["phrasings"]:
                if p["id"] in wanted:
                    p["trained"] = True

        self._update(change)
