"""Small, safe file writes for the learning stores.

The dev brain and the production pod share ``data/`` (the pod mounts it as a
hostPath), so two processes may update the same JSON file. Every
read-modify-write takes an ``flock`` on a sibling ``.lock`` file and replaces
the file atomically; a reader never sees half a file, and one writer's change
is never overwritten by the other's stale copy.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

#: flock is per open file description, so threads in one process need a lock
#: of their own as well
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(str(path), threading.Lock())


@contextlib.contextmanager
def locked(path: Path):
    """Exclusive access to ``path`` across threads and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with _thread_lock(path):
        with open(lock_path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def read_json(path: Path, default):
    """The file's JSON, or ``default`` when it is missing. A corrupt file is
    renamed aside (``<name>.corrupt-<time>``) rather than read as empty and
    then overwritten: what was learned may still be recoverable by hand."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(text)
    except ValueError:
        aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        log.error("%s is not valid JSON; moved aside to %s", path, aside.name)
        os.replace(path, aside)
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def key(text: str) -> str:
    """The same words, give or take case, spacing and punctuation."""
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())
