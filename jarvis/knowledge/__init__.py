"""M5: the local knowledge base (RAG).

``Knowledge`` is what the rest of JARVIS holds: the store, the folder it
ingests, and the settings that decide what counts as an answer. Skills reach it
as ``ctx.knowledge``; the orchestrator runs ``watch()`` as its interval task.

Nothing here talks to Claude — see PRD/jarvis-2026-rebuild.md § "Knowledge
base (RAG)": *Claude is never invoked for knowledge answers.*
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import threading

from jarvis.knowledge import answer as _answer
from jarvis.knowledge import ingest
from jarvis.knowledge.ingest import ScanResult
from jarvis.knowledge.store import Hit, KnowledgeStore, fastembed_embedder

log = logging.getLogger(__name__)

__all__ = ["Hit", "Knowledge", "KnowledgeStore", "ScanResult", "build"]


class Knowledge:
    def __init__(self, store: KnowledgeStore, settings) -> None:
        self.store = store
        self.settings = settings
        #: one scan at a time: the interval task, `note`, and the CLI can all ask
        self._scan_lock = threading.Lock()
        self._stop = threading.Event()

    @property
    def docs_dir(self):
        return self.settings.docs_path

    # -- asking ---------------------------------------------------------------

    def search(self, query: str, *, min_score: float | None = None) -> list[Hit]:
        """The best chunks for ``query``, best first — only those that clear
        ``min_score`` (default ``[knowledge] min_score``)."""
        floor = self.settings.min_score if min_score is None else min_score
        hits = self.store.search(query, k=self.settings.top_k)
        return [h for h in hits if h.score >= floor]

    def answer(self, question: str, llm=None, *, min_score: float | None = None) -> str | None:
        """The line to speak in answer to ``question``, or ``None`` when the
        knowledge base has nothing that clears the bar. ``llm`` is the local
        reasoner, if any — never Claude."""
        hits = self.search(question, min_score=min_score)
        if not hits:
            return None
        return _answer.compose(question, hits, llm)

    # -- telling --------------------------------------------------------------

    def remember(self, text: str, when: dt.datetime | None = None) -> None:
        """A spoken fact, straight into the store — no file behind it."""
        self.store.add_fact(text.strip(), when)

    def index_file(self, path) -> None:
        """Index one file in ``docs_dir`` now, without waiting for a scan —
        `note` uses it so a dictated note is recallable on the next turn."""
        ingest.index_file(
            self.store,
            path,
            chunk_chars=self.settings.chunk_chars,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def scan(self) -> ScanResult:
        """Bring the store in line with ``docs_dir``. Blocking; call it from a
        worker thread."""
        with self._scan_lock:
            return ingest.scan(
                self.store,
                self.docs_dir,
                chunk_chars=self.settings.chunk_chars,
                chunk_overlap=self.settings.chunk_overlap,
                stop=self._stop,
            )

    async def watch(self) -> None:
        """The interval task: scan now, then every ``scan_interval_s``. The scan
        itself runs in a worker thread — embedding must not stall a turn.

        Cancelling stops the scan at the next file and waits for the worker,
        because a thread cannot be killed and must not outlive the loop."""
        while True:
            scan = asyncio.ensure_future(asyncio.to_thread(self.scan))
            try:
                result = await asyncio.shield(scan)
                if result.changed:
                    log.info("knowledge: %s", result.summary())
            except asyncio.CancelledError:
                self._stop.set()
                await asyncio.wait([scan])
                self._stop.clear()
                raise
            except Exception:  # noqa: BLE001 - one bad scan must not end the watch
                log.exception("knowledge scan failed")
            await asyncio.sleep(self.settings.scan_interval_s)

    def close(self) -> None:
        self.store.close()


def build(config) -> Knowledge | None:
    """The knowledge base for ``config``, or ``None`` when it is switched off.
    Cheap: no model is loaded and no file is created until something is
    written or found."""
    if not config.knowledge.enabled:
        return None
    store = KnowledgeStore(
        config.knowledge_db_path,
        fastembed_embedder(config.nlu.embedding_model),
        model_name=config.nlu.embedding_model,
    )
    return Knowledge(store, config.knowledge)
