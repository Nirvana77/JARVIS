"""M4.5: what JARVIS remembers, per device.

Two kinds, kept apart because they live differently:

  the recent conversation   what was heard and what was said back, the last
                            few turns — RAM only, gone after a quiet spell
  lasting facts             what a device was asked to note or remember —
                            ``data/memory/<device>.json``, until forgotten

What was said to ``remember`` also goes into the M5 knowledge base, which is
shared by every device; the device keeps the ids of the facts it put there
(``knowledge_refs``), so that forgetting takes them out again.

Both are handed to the reasoner when a turn needs thinking about
(``Orchestrator._think``). A device is an edge's ``device_id`` (``watch``,
``livingroom``) or ``local`` for the all-in-one and text modes. The
orchestrator serves one turn at a time, so "the device of this turn" is a
single value (``focus``) — which is how a skill gets the right one through
``ctx.memory`` without knowing devices exist.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

LOCAL = "local"

#: the id becomes a filename; the same rule ``jarvis.remote.protocol`` holds an
#: edge's ``hello`` to (not imported: this module stays free of the remote stack)
_DEVICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: a fact is a sentence somebody said, not a document
MAX_FACT_CHARS = 500


def is_device_id(value) -> bool:
    # fullmatch: `$` alone would let "watch\n" through
    return isinstance(value, str) and bool(_DEVICE_RE.fullmatch(value))


def _key(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def same_text(a: str, b: str) -> bool:
    """The same fact, give or take case, spacing and punctuation."""
    return _key(a) == _key(b)


class Memory:
    def __init__(
        self,
        directory: Path | None = None,
        *,
        turns: int = 8,
        idle_forget_s: float = 900.0,
        max_facts: int = 200,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        #: ``None`` keeps facts in RAM only (tests, a throwaway orchestrator)
        self.directory = Path(directory) if directory is not None else None
        self.max_turns = max(1, int(turns))
        self.idle_forget_s = float(idle_forget_s)
        self.max_facts = max(1, int(max_facts))
        self._clock = clock
        #: the device whose turn it is
        self.current = LOCAL
        # skills run on worker threads (`note` -> ctx.memory) while the loop reads
        self._lock = threading.Lock()
        self._turns: dict[str, deque[tuple[str, str]]] = {}
        self._last_turn: dict[str, float] = {}
        self._facts: dict[str, list[dict]] = {}
        #: the knowledge-base facts each device put there, until forgotten
        self._refs: dict[str, list[str]] = {}

    @classmethod
    def from_config(cls, config) -> "Memory":
        return cls(
            config.memory_dir,
            turns=config.memory.turns,
            idle_forget_s=config.memory.idle_forget_s,
            max_facts=config.memory.max_facts,
        )

    def focus(self, device: str) -> None:
        self.current = self._checked(device)

    def device(self, device: str | None = None) -> "DeviceMemory":
        """One device's memory; the one in focus when ``device`` is not given."""
        return DeviceMemory(self, self._checked(self.current if device is None else device))

    @staticmethod
    def _checked(device: str) -> str:
        if not is_device_id(device):
            raise ValueError(f"not a device id: {device!r}")
        return device

    # -- facts (disk) ------------------------------------------------------

    def _path(self, device: str) -> Path | None:
        return self.directory / f"{device}.json" if self.directory is not None else None

    def _load(self, device: str) -> list[dict]:
        if device in self._facts:
            return self._facts[device]
        facts: list[dict] = []
        refs: list[str] = []
        path = self._path(device)
        if path is not None and path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                facts = [
                    {"text": f["text"], "at": str(f.get("at", ""))}
                    for f in raw.get("facts", [])
                    if isinstance(f, dict) and isinstance(f.get("text"), str) and f["text"].strip()
                ]
                refs = [r for r in raw.get("knowledge_refs", []) if isinstance(r, str)]
            except OSError as exc:
                log.warning("memory for %s cannot be read (%s); starting it empty", device, exc)
            except (ValueError, AttributeError) as exc:
                # Set it aside rather than write over it: a typo from a hand
                # edit must not cost every fact in the file.
                aside = path.with_name(f"{path.name}.unreadable-{datetime.now():%Y%m%dT%H%M%S}")
                log.warning("memory for %s is damaged (%s); kept as %s", device, exc, aside.name)
                try:
                    path.replace(aside)
                except OSError as move_exc:
                    log.warning("could not set it aside: %s", move_exc)
        self._facts[device] = facts
        self._refs[device] = refs
        return facts

    def _save(self, device: str, *, strict: bool = False) -> None:
        """Write a device's facts. A failure is logged — or raised, with
        ``strict``, for the caller that is about to tell someone it is gone."""
        path = self._path(device)
        if path is None:
            return
        facts = self._facts.get(device, [])
        refs = self._refs.get(device, [])
        try:
            if not facts and not refs:
                path.unlink(missing_ok=True)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            body = {"facts": facts, "knowledge_refs": refs} if refs else {"facts": facts}
            tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)  # never a half-written file
        except OSError as exc:
            if strict:
                raise
            log.warning("could not write the memory for %s: %s", device, exc)


class DeviceMemory:
    """A view of :class:`Memory` for one device. Cheap; make one per use."""

    def __init__(self, memory: Memory, device: str) -> None:
        self._m = memory
        self.device = device

    # -- lasting facts -----------------------------------------------------

    def remember(self, text: str, knowledge_ref: str | None = None) -> bool:
        """Keep ``text``. ``knowledge_ref``: the id of the same fact in the
        knowledge base, kept until :meth:`forget` — even when the fact itself
        is later said again or pushed out by newer ones, since the knowledge
        base still has it."""
        text = " ".join((text or "").split())[:MAX_FACT_CHARS].strip()
        if not text:
            return False
        m = self._m
        with m._lock:
            facts = m._load(self.device)
            if knowledge_ref:
                m._refs.setdefault(self.device, []).append(knowledge_ref)
            # said again: one fact, and the newest
            facts[:] = [f for f in facts if _key(f["text"]) != _key(text)]
            facts.append({"text": text, "at": datetime.now().isoformat(timespec="seconds")})
            del facts[: max(0, len(facts) - m.max_facts)]
            m._save(self.device)
        return True

    def knowledge_refs(self) -> list[str]:
        with self._m._lock:
            self._m._load(self.device)
            return list(self._m._refs.get(self.device, []))

    def entries(self) -> list[tuple[str, str]]:
        """``(text, when)`` per fact, oldest first; ``when`` is ISO 8601."""
        with self._m._lock:
            return [(f["text"], f.get("at", "")) for f in self._m._load(self.device)]

    def facts(self) -> list[str]:
        with self._m._lock:
            return [f["text"] for f in self._m._load(self.device)]

    def forget_facts(self, texts, knowledge_refs=()) -> list[str]:
        """Drop the facts whose text is one of ``texts`` and the knowledge
        refs in ``knowledge_refs``; the conversation stays. Returns what was
        dropped. Raises ``OSError`` (and drops nothing) if it cannot be
        written."""
        drop = {_key(t) for t in texts}
        gone_refs = set(knowledge_refs)
        m = self._m
        with m._lock:
            kept = m._load(self.device)
            kept_refs = m._refs.get(self.device, [])
            m._facts[self.device] = [f for f in kept if _key(f["text"]) not in drop]
            m._refs[self.device] = [r for r in kept_refs if r not in gone_refs]
            try:
                m._save(self.device, strict=True)
            except OSError:
                m._facts[self.device], m._refs[self.device] = kept, kept_refs
                raise
            return [f["text"] for f in kept if _key(f["text"]) in drop]

    def forget(self) -> int:
        """Drop this device's facts and its conversation. Returns how many
        facts there were. Raises ``OSError`` if the file could not be removed
        — and then nothing is forgotten, because it would be back after a
        restart."""
        m = self._m
        with m._lock:
            kept = m._load(self.device)
            kept_refs = m._refs.get(self.device, [])
            count = len(kept)
            m._facts[self.device] = []
            m._refs[self.device] = []
            try:
                m._save(self.device, strict=True)
            except OSError:
                m._facts[self.device] = kept
                m._refs[self.device] = kept_refs
                raise
            m._turns.pop(self.device, None)
            m._last_turn.pop(self.device, None)
        return count

    # -- the recent conversation ------------------------------------------

    def add_turn(self, heard: str, said: str) -> None:
        heard = " ".join((heard or "").split())
        if not heard:
            return
        m = self._m
        with m._lock:
            self._expire()
            turns = m._turns.setdefault(self.device, deque(maxlen=m.max_turns))
            turns.append((heard, " ".join((said or "").split())))
            m._last_turn[self.device] = m._clock()

    def turns(self) -> list[tuple[str, str]]:
        with self._m._lock:
            self._expire()
            return list(self._m._turns.get(self.device, ()))

    def _expire(self) -> None:
        m = self._m
        last = m._last_turn.get(self.device)
        if last is not None and m._clock() - last > m.idle_forget_s:
            m._turns.pop(self.device, None)
            m._last_turn.pop(self.device, None)
