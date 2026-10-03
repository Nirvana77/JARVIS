"""``python -m jarvis learning [status|log|phrasings|undo ID|enable SKILL]``.

Reads the learning stores straight from ``data/`` — no model, no brain — so
it works while the brain (or the pod, which shares ``data/``) is running. An
undone phrasing leaves the corpus, so the next start retrains without it
(the corpus digest changes).
"""

from __future__ import annotations

import datetime as _dt
import re

from jarvis.learning import Learning

_DAYS = re.compile(r"^(\d+)d$")


def parse_since(value: str | None, now: _dt.datetime | None = None) -> _dt.datetime | None:
    """``"2d"`` (two days back from now) or ``"2026-09-30"``; ``None`` for all."""
    if not value:
        return None
    now = now or _dt.datetime.now()
    found = _DAYS.match(value.strip())
    if found:
        return now - _dt.timedelta(days=int(found.group(1)))
    return _dt.datetime.fromisoformat(value.strip())


def run(config, op: str = "status", *, arg: str | None = None, since: str | None = None,
        device: str | None = None) -> int:
    learning = Learning.from_config(config)
    if op == "status":
        return _status(config, learning)
    if op == "log":
        return _log(learning, parse_since(since or "1d"), device)
    if op == "phrasings":
        return _phrasings(learning)
    if op == "undo":
        found = learning.phrasings.remove(arg or "", reason="undone by hand")
        if found is None:
            print(f"no learned phrasing with id {arg!r} (see: jarvis learning phrasings)")
            return 1
        print(f"undone: {found.text!r} -> {found.label}; the next start retrains without it")
        return 0
    if op == "enable":
        if not learning.state.enable(arg or ""):
            print(f"{arg!r} is not switched off")
            return 1
        print(f"{arg} is switched back on")
        return 0
    print(f"unknown op {op!r}")
    return 2


def _status(config, learning: Learning) -> int:
    s = config.learning
    entries = learning.phrasings.entries()
    pending = [p for p in entries if not p.trained]
    print(f"learning: {'on' if s.enabled else 'off (logging only)'} · log: {'on' if s.log else 'off'}")
    print(f"  {len(entries)} learned phrasing(s), {len(pending)} waiting for a retrain")
    print(f"  skills built by itself today: {learning.state.builds_today()} of {s.max_builds_per_day}")
    disabled = learning.state.disabled()
    if disabled:
        print("  switched off:")
        for name, info in sorted(disabled.items()):
            print(f"    {name}  ({info.get('reason', '')}, {info.get('ts', '')})")
    confusions = learning.state.confusions()
    if confusions:
        print("  corrected mix-ups:")
        for pair, n in sorted(confusions.items(), key=lambda kv: -kv[1]):
            print(f"    {pair}  ×{n}")
    failures = learning.state.builtin_failures()
    if failures:
        print(f"  builtin failures for a person to fix: {len(failures)} "
              f"({config.learning_dir / 'builtin-failures.jsonl'})")
    return 0


def _log(learning: Learning, since, device) -> int:
    records = learning.log.records(device=device, since=since)
    for r in records:
        target = f" -> {r['skill']}" if r.get("skill") else ""
        learned = "  [learned]" if r.get("learned") else ""
        print(
            f"{r.get('ts', '')}  {r.get('device', ''):<10} {r.get('path', ''):<11} "
            f"{r.get('outcome', ''):<9} {r.get('heard', '')!r}{target}{learned}"
        )
    print(f"({len(records)} turn(s))")
    return 0


def _phrasings(learning: Learning) -> int:
    entries = learning.phrasings.entries()
    for p in entries:
        state = "trained" if p.trained else "waiting"
        print(f"{p.id}  {p.ts}  {p.why:<10} {state:<8} {p.text!r} -> {p.label}")
    print(f"({len(entries)} phrasing(s))")
    return 0
