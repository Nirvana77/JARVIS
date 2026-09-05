"""M2: retrain off the event loop in a separate process, so a `teach`/
`edit_skill`/`revert_skill` retrain never blocks the voice loop. Wraps
`jarvis.nlu.train.train()` (unchanged) around a `multiprocessing.Process` +
`multiprocessing.Queue`; the orchestrator polls (via `asyncio.to_thread`) and
only ever swaps `self.nlu` through the merge gate, never from here directly.
"""

from __future__ import annotations

import multiprocessing
import queue as _queue
from pathlib import Path
from typing import Sequence

from jarvis.nlu.corpus import Example
from jarvis.nlu.train import TrainResult, train


def _worker_entry(
    examples: list[tuple[str, str, str]],
    embedding_model: str,
    out_dir: str,
    result_queue: "multiprocessing.Queue",
) -> None:
    try:
        parsed = [Example(text=t, label=l, source=s) for t, l, s in examples]
        result = train(parsed, embedding_model, Path(out_dir))
        result_queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 — reported to the parent, not raised here
        result_queue.put(("error", repr(exc)))


class RetrainWorker:
    """One retrain job. `start()` once, `poll()` (non-blocking or with a
    timeout) until it returns a result."""

    def __init__(self) -> None:
        self._proc: multiprocessing.Process | None = None
        self._queue: "multiprocessing.Queue | None" = None

    def start(
        self, examples: Sequence[Example], embedding_model: str, out_dir: Path
    ) -> None:
        # Example is a plain (text, label, source) dataclass — pickle it as a
        # tuple so the worker process doesn't need to import jarvis.nlu.corpus
        # before jarvis itself is on its sys.path.
        payload = [(e.text, e.label, e.source) for e in examples]
        self._queue = multiprocessing.Queue()
        self._proc = multiprocessing.Process(
            target=_worker_entry,
            args=(payload, embedding_model, str(out_dir), self._queue),
            daemon=True,
        )
        self._proc.start()

    def poll(self, timeout: float = 0.0) -> tuple[str, TrainResult | str] | None:
        """Blocks up to `timeout` seconds. Returns `("ok", TrainResult)`,
        `("error", message)`, or `None` if nothing has arrived yet."""
        if self._queue is None:
            return None
        try:
            result = self._queue.get(timeout=timeout)
        except _queue.Empty:
            return None
        if self._proc is not None:
            self._proc.join(timeout=1.0)
        return result
