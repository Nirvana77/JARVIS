"""Document ingestion: load a file, chunk it, keep the store in line with a folder.

``scan`` is the whole of "folder watch": list ``docs_dir``, compare each file's
``mtime_ns:size`` digest with what the store last saw, re-chunk what is new or
changed, and drop what is gone. It is the old ``intents.json`` mtime poll, for a
folder — driven by an asyncio interval task (``Knowledge.watch``) instead of a
thread of its own. Spoken facts have no file and are never touched here.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: what a scan picks up; everything else in the folder is ignored
SUFFIXES = frozenset({".txt", ".md", ".pdf"})

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


# -- loading ------------------------------------------------------------------

def load_text(path: Path) -> str:
    """The text of ``path``. Raises for a file that cannot be read."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # only a PDF pays for the import

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


# -- chunking -----------------------------------------------------------------

def _split_long(piece: str, max_chars: int) -> list[str]:
    """``piece`` as runs of whole words, each within ``max_chars``."""
    out, current = [], ""
    for word in piece.split(" "):
        while len(word) > max_chars:  # one unbroken run: nothing to split on
            if current:
                out.append(current)
                current = ""
            out.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            out.append(current)
            current = word
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _pack(units: list[str], max_chars: int, overlap: int) -> list[str]:
    """Join ``units`` (sentences) into chunks within ``max_chars``; each chunk
    after the first opens with the trailing units of the one before it, up to
    ``overlap`` characters, so a fact split across a boundary is in one of them."""
    chunks: list[str] = []
    current: list[str] = []
    fresh = 0  # units in `current` that are not carried-over overlap

    def length(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + max(len(parts) - 1, 0)

    for unit in units:
        if current and length(current + [unit]) > max_chars:
            if fresh:
                chunks.append(" ".join(current))
                carried: list[str] = []
                for prev in reversed(current):
                    if length([prev] + carried) > overlap:
                        break
                    carried.insert(0, prev)
                current, fresh = carried, 0
            if current and length(current + [unit]) > max_chars:
                current = []  # the overlap itself leaves no room: drop it
        current.append(unit)
        fresh += 1
    if fresh:
        chunks.append(" ".join(current))
    return chunks


def chunk(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split ``text`` into chunks worth retrieving on their own.

    A paragraph is a chunk: one note per paragraph comes back as that note. A
    markdown heading is kept with the paragraph under it. Only a paragraph
    longer than ``max_chars`` is split, on sentence boundaries, with
    ``overlap`` characters carried between neighbours.
    """
    paragraphs = [" ".join(p.split()) for p in _PARAGRAPH.split(text)]
    paragraphs = [p for p in paragraphs if p]

    merged: list[str] = []
    heading = ""
    for paragraph in paragraphs:
        if paragraph.startswith("#") and len(paragraph) < max_chars // 2:
            heading = f"{heading} {paragraph}".strip()
            continue
        merged.append(f"{heading} {paragraph}".strip())
        heading = ""
    if heading:
        merged.append(heading)

    chunks: list[str] = []
    for paragraph in merged:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        units: list[str] = []
        for sentence in _SENTENCE.split(paragraph):
            units.extend(
                [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
            )
        chunks.extend(_pack(units, max_chars, overlap))
    return chunks


# -- scanning -----------------------------------------------------------------

@dataclass
class ScanResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    #: could not be read; indexed as empty until the file changes
    failed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.failed)

    def summary(self) -> str:
        parts = [
            f"{len(names)} {label}"
            for label, names in (
                ("added", self.added), ("updated", self.updated),
                ("removed", self.removed), ("unreadable", self.failed),
            )
            if names
        ]
        return ", ".join(parts) or "no changes"


def digest(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _documents(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        return []
    return sorted(
        p for p in docs_dir.rglob("*")
        if p.suffix.lower() in SUFFIXES
        and p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(docs_dir).parts)
    )


def index_file(store, path: Path, *, chunk_chars: int = 800, chunk_overlap: int = 100) -> None:
    """(Re)index one file. Raises if it cannot be read."""
    path = Path(path)
    current = digest(path)
    store.replace_source(str(path), current, chunk(load_text(path), chunk_chars, chunk_overlap))


def scan(
    store,
    docs_dir: Path,
    *,
    chunk_chars: int = 800,
    chunk_overlap: int = 100,
    stop: threading.Event | None = None,
) -> ScanResult:
    """Bring ``store`` in line with ``docs_dir``. ``stop`` (set from another
    thread) ends it before the next file; what was not reached is picked up by
    the next scan."""
    result = ScanResult()
    docs_dir = Path(docs_dir)
    known = store.file_digests()
    documents = _documents(docs_dir)
    present = {str(p) for p in documents}

    for path in sorted(set(known) - present):
        if store.remove_source(path):
            result.removed.append(path)

    for document in documents:
        if stop is not None and stop.is_set():
            break
        key = str(document)
        try:
            current = digest(document)
        except OSError:
            continue  # gone between the listing and now; the next scan drops it
        if known.get(key) == current:
            continue
        try:
            chunks = chunk(load_text(document), chunk_chars, chunk_overlap)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            log.warning("knowledge: cannot read %s: %s", document, exc)
            # Remembered with its digest and no chunks, so it is retried when
            # the file changes rather than complained about every interval.
            store.replace_source(key, current, [])
            result.failed.append(key)
            continue
        store.replace_source(key, current, chunks)
        (result.updated if key in known else result.added).append(key)
    return result
