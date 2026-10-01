"""The knowledge store: chunks of text, their embeddings, and search over them.

One sqlite file (``data/knowledge/kb.sqlite``):

- ``sources`` — one row per ingested file, one per spoken fact
- ``chunks``  — the text, a hash of it, and its embedding as a float32 BLOB
- ``meta``    — the embedding model and dimension, and a write generation

The BLOB is the source of truth. When the ``sqlite-vec`` extension loads, a
``vec0`` table indexes the same rows by cosine distance; it is rebuilt from the
BLOBs whenever it is missing or behind. When the extension does not load,
search is brute-force cosine over a numpy matrix read from the BLOBs — the same
ranking, and a database written by either backend is readable by the other.

The embedder is injected (``embed(texts) -> float32 matrix``, rows
L2-normalised), so nothing here imports a model. The store starts no threads;
it is called from ``asyncio.to_thread`` workers, so one connection is shared
behind a lock.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)

Embed = Callable[[list[str]], np.ndarray]

FILE = "file"
FACT = "fact"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    digest TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_id, ord);
"""

_VEC_TABLE = "vec_chunks"


@dataclass(frozen=True)
class Hit:
    text: str
    #: the file's path, or an opaque id for a spoken fact
    source: str
    kind: str
    score: float
    #: when the file was last indexed / when the fact was said
    when: dt.datetime


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into ``conn``. Raises if it cannot."""
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def fastembed_embedder(model_name: str) -> Embed:
    """The production embedder: fastembed's model, loaded on first use — so a
    JARVIS with nothing in its knowledge base never loads it."""
    lock = threading.Lock()
    model = None

    def embed(texts: list[str]) -> np.ndarray:
        nonlocal model
        with lock:
            if model is None:
                from fastembed import TextEmbedding

                model = TextEmbedding(model_name=model_name)
            return np.asarray(list(model.embed(list(texts))), dtype=np.float32)

    return embed


class KnowledgeStore:
    def __init__(
        self,
        path: Path,
        embed: Embed,
        *,
        model_name: str = "",
        use_sqlite_vec: bool = True,
    ) -> None:
        self.path = Path(path)
        self._embed = embed
        self._model_name = model_name
        self._use_sqlite_vec = use_sqlite_vec
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._vec: bool | None = None  # None until the extension has been tried
        #: (generation, chunk ids, matrix) for the brute-force path
        self._matrix: tuple[str, list[int], np.ndarray] | None = None

    # -- connection -----------------------------------------------------------

    @property
    def backend(self) -> str:
        with self._lock:
            if self._vec is None:
                if self._conn is not None or self.path.exists():
                    self._open(create=False)
                else:
                    self._try_vec(sqlite3.connect(":memory:"), close=True)
            return "sqlite-vec" if self._vec else "numpy"

    def _try_vec(self, conn: sqlite3.Connection, *, close: bool = False) -> None:
        if not self._use_sqlite_vec:
            self._vec = False
        else:
            try:
                _load_sqlite_vec(conn)
                self._vec = True
            except Exception as exc:  # noqa: BLE001 - degrading is the point
                log.warning(
                    "sqlite-vec unavailable (%s) - knowledge search falls back to "
                    "brute-force cosine", exc,
                )
                self._vec = False
        if close:
            conn.close()

    def _open(self, *, create: bool) -> sqlite3.Connection | None:
        """The connection, or ``None`` when there is no database yet and this
        call is not allowed to create one (reads never do)."""
        if self._conn is not None:
            return self._conn
        if not create and not self.path.exists():
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        self._try_vec(conn)
        with conn:
            conn.executescript(_SCHEMA)
        self._conn = conn
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._matrix = None

    # -- meta -----------------------------------------------------------------

    @staticmethod
    def _meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _bump(self, conn: sqlite3.Connection, *, vec_in_step: bool) -> None:
        """Mark a write. The index records the generation it last matched, so a
        write made without it (the numpy fallback) is noticed on the next open."""
        generation = uuid.uuid4().hex
        self._set_meta(conn, "generation", generation)
        if vec_in_step:
            self._set_meta(conn, "vec_generation", generation)

    # -- keeping the model and the index in step ------------------------------

    def _ready(self, conn: sqlite3.Connection) -> None:
        """Before any search or write: vectors from another embedding model are
        re-embedded, and a missing or stale index is rebuilt."""
        stored = self._meta(conn, "model")
        if stored != self._model_name:
            rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
            if rows:
                log.info(
                    "knowledge: embedding model changed (%s -> %s), re-embedding %d chunks",
                    stored or "?", self._model_name or "?", len(rows),
                )
                vectors = self._embed([text for _id, text in rows])
                with conn:
                    conn.executemany(
                        "UPDATE chunks SET embedding = ? WHERE id = ?",
                        [(_blob(v), cid) for (cid, _t), v in zip(rows, vectors)],
                    )
                    self._set_meta(conn, "dim", str(vectors.shape[1]))
                    self._set_meta(conn, "model", self._model_name)
                    self._bump(conn, vec_in_step=False)
            else:
                with conn:
                    self._set_meta(conn, "model", self._model_name)
        if self._vec and self._meta(conn, "vec_generation") != self._meta(conn, "generation"):
            self._rebuild_index(conn)

    def _rebuild_index(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        with conn:
            conn.execute(f"DROP TABLE IF EXISTS {_VEC_TABLE}")
            if rows:
                self._create_index(conn, len(rows[0][1]) // 4)
                conn.executemany(
                    f"INSERT INTO {_VEC_TABLE}(rowid, embedding) VALUES(?, ?)", rows
                )
            self._set_meta(conn, "vec_generation", self._meta(conn, "generation"))

    @staticmethod
    def _create_index(conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE} "
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )

    def _has_index(self, conn: sqlite3.Connection) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (_VEC_TABLE,)
        ).fetchone() is not None

    # -- writing --------------------------------------------------------------

    def replace_source(
        self,
        path: str,
        digest: str,
        chunks: Sequence[str],
        *,
        kind: str = FILE,
        when: dt.datetime | None = None,
    ) -> int:
        """Make ``path``'s chunks exactly ``chunks``. A chunk whose text was
        already stored for this source keeps its embedding; returns how many
        had to be embedded."""
        chunks = list(chunks)
        when = when or dt.datetime.now()
        with self._lock:
            conn = self._open(create=True)
            self._ready(conn)
            known = dict(
                conn.execute(
                    "SELECT c.text_hash, c.embedding FROM chunks c "
                    "JOIN sources s ON s.id = c.source_id WHERE s.path = ?",
                    (path,),
                ).fetchall()
            )
        hashes = [_hash(text) for text in chunks]
        missing: dict[str, str] = {}  # hash -> text, each new text once
        for h, text in zip(hashes, chunks):
            if h not in known:
                missing.setdefault(h, text)
        if missing:
            # Outside the lock: embedding a large file must not stall a search.
            for h, vector in zip(missing, self._embed(list(missing.values()))):
                known[h] = _blob(vector)

        with self._lock:
            conn = self._open(create=True)
            with conn:
                self._delete_chunks(conn, path)
                conn.execute(
                    "INSERT INTO sources(path, kind, digest, updated_at) VALUES(?, ?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET kind = excluded.kind, "
                    "digest = excluded.digest, updated_at = excluded.updated_at",
                    (path, kind, digest, when.isoformat(timespec="seconds")),
                )
                (source_id,) = conn.execute(
                    "SELECT id FROM sources WHERE path = ?", (path,)
                ).fetchone()
                if chunks:
                    dim = len(known[hashes[0]]) // 4
                    self._set_meta(conn, "dim", str(dim))
                    if self._vec:
                        self._create_index(conn, dim)
                for ord_, (text, h) in enumerate(zip(chunks, hashes)):
                    cursor = conn.execute(
                        "INSERT INTO chunks(source_id, ord, text, text_hash, embedding) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (source_id, ord_, text, h, known[h]),
                    )
                    if self._vec:
                        conn.execute(
                            f"INSERT INTO {_VEC_TABLE}(rowid, embedding) VALUES(?, ?)",
                            (cursor.lastrowid, known[h]),
                        )
                self._bump(conn, vec_in_step=bool(self._vec))
        return len(missing)

    def _delete_chunks(self, conn: sqlite3.Connection, path: str) -> None:
        if self._vec and self._has_index(conn):
            conn.execute(
                f"DELETE FROM {_VEC_TABLE} WHERE rowid IN (SELECT c.id FROM chunks c "
                "JOIN sources s ON s.id = c.source_id WHERE s.path = ?)",
                (path,),
            )
        conn.execute(
            "DELETE FROM chunks WHERE source_id IN (SELECT id FROM sources WHERE path = ?)",
            (path,),
        )

    def remove_source(self, path: str) -> bool:
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return False
            self._ready(conn)
            with conn:
                self._delete_chunks(conn, path)
                removed = conn.execute(
                    "DELETE FROM sources WHERE path = ?", (path,)
                ).rowcount > 0
                if removed:
                    self._bump(conn, vec_in_step=bool(self._vec))
            return removed

    def add_fact(self, text: str, when: dt.datetime | None = None) -> str:
        """A spoken fact: one chunk, its own source, no file behind it."""
        source = f"fact:{uuid.uuid4().hex}"
        self.replace_source(source, "", [text], kind=FACT, when=when)
        return source

    # -- reading --------------------------------------------------------------

    def file_digests(self) -> dict[str, str]:
        """``path -> digest`` for every ingested *file* — what a folder scan
        compares against. Spoken facts are not in it, so a scan never drops them."""
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return {}
            return dict(
                conn.execute(
                    "SELECT path, digest FROM sources WHERE kind = ?", (FILE,)
                ).fetchall()
            )

    def sources(self) -> list[tuple[str, str, int, dt.datetime]]:
        """``(path, kind, chunk count, when)`` per source, oldest first."""
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT s.path, s.kind, COUNT(c.id), s.updated_at FROM sources s "
                "LEFT JOIN chunks c ON c.source_id = s.id GROUP BY s.id "
                "ORDER BY s.updated_at, s.id"
            ).fetchall()
        return [(p, k, n, dt.datetime.fromisoformat(w)) for p, k, n, w in rows]

    def stats(self) -> dict:
        with self._lock:
            backend = self.backend
            conn = self._open(create=False)
            if conn is None:
                return {"files": 0, "facts": 0, "chunks": 0, "backend": backend}
            kinds = dict(conn.execute("SELECT kind, COUNT(*) FROM sources GROUP BY kind"))
            (chunks,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return {
            "files": kinds.get(FILE, 0),
            "facts": kinds.get(FACT, 0),
            "chunks": chunks,
            "backend": backend,
        }

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """The ``k`` chunks nearest ``query`` by cosine similarity, best first."""
        query = (query or "").strip()
        if not query or k <= 0:
            return []
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return []
            if conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is None:
                return []
            self._ready(conn)
        vector = np.asarray(self._embed([query])[0], dtype=np.float32)
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return []
            if self._vec and self._has_index(conn):
                scored = [
                    (cid, 1.0 - distance)
                    for cid, distance in conn.execute(
                        f"SELECT rowid, distance FROM {_VEC_TABLE} "
                        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                        (_blob(vector), k),
                    )
                ]
            else:
                scored = self._brute_force(conn, vector, k)
            # Equal scores in insertion order, whichever backend found them.
            scored.sort(key=lambda pair: (-round(pair[1], 6), pair[0]))
            hits = []
            for cid, score in scored:
                row = conn.execute(
                    "SELECT c.text, s.path, s.kind, s.updated_at FROM chunks c "
                    "JOIN sources s ON s.id = c.source_id WHERE c.id = ?",
                    (cid,),
                ).fetchone()
                if row is not None:
                    text, path, kind, when = row
                    hits.append(
                        Hit(text, path, kind, float(score), dt.datetime.fromisoformat(when))
                    )
        return hits

    def _brute_force(
        self, conn: sqlite3.Connection, vector: np.ndarray, k: int
    ) -> list[tuple[int, float]]:
        generation = self._meta(conn, "generation")
        if self._matrix is None or self._matrix[0] != generation:
            rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
            matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _id, blob in rows])
            self._matrix = (generation, [cid for cid, _blob_ in rows], matrix)
        _generation, ids, matrix = self._matrix
        norms = np.linalg.norm(matrix, axis=1) * (float(np.linalg.norm(vector)) or 1.0)
        scores = (matrix @ vector) / np.where(norms == 0.0, 1.0, norms)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(ids[i], float(scores[i])) for i in order]
