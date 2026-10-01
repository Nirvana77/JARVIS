# Milestone 5 — knowledge base (RAG): plan

**Status:** In progress
**Date:** 2026-10-01
**Owner:** Kevin Lundell
**Builds on:** Milestones 1–3. Milestone 4 (misheard-command reasoning) is not
started; nothing here depends on it, and this milestone is taken out of order.
**Spec:** `PRD/jarvis-2026-rebuild.md` § "Knowledge base (RAG)", § "Milestone 5",
and the M5 line of § "Verification".

---

## Problem

JARVIS cannot recall anything. "Remember that I parked on level three" appends
a line to `data/skills/note/notes.txt` that nothing ever reads back, and "what
is …" always goes to Wikipedia, even when the answer is in the user's own
notes. The PRD's answer is a local knowledge base: documents in a folder and
spoken facts are embedded, and recall questions are answered from them.

## Goal

- Files dropped in `knowledge.docs_dir` (`.txt`, `.md`, `.pdf`) and facts said
  aloud ("remember that …") become searchable.
- A recall question gets a **grounded answer with its source**: composed by
  Ollama when it is up, otherwise the best snippet read verbatim in a canned
  frame (*"From your notes, sir: …"*).
- **Claude is never invoked for a knowledge answer.**

---

## Key design decisions

### 1. The embedding is stored once; sqlite-vec is an index over it

`data/knowledge/kb.sqlite` holds `sources` (one row per file, one per spoken
fact), `chunks` (text, a hash of the text, and the embedding as a BLOB) and a
small `meta` table (embedding model, dimension). The BLOB is the source of
truth. When the `sqlite-vec` extension loads, a `vec0` table (cosine distance)
indexes the same rows and is rebuilt from the BLOBs whenever it is missing or
out of step. When it does not load, search is brute-force cosine over a numpy
matrix read from the BLOBs — the PRD's fallback, with the same ranking. A
database written by one backend is readable by the other.

### 2. The embedder is injected, and is the NLU's model

`KnowledgeStore` takes `embed: Callable[[list[str]], np.ndarray]`. Production
passes a lazy fastembed wrapper for `[nlu] embedding_model` (the model is
already on disk for the NLU); it is loaded on the first write or the first
search of a non-empty store, so a JARVIS with no knowledge pays nothing. Tests
pass a deterministic hashed bag-of-words fake, so the store, ingestion and
skill tests need no model. A changed `embedding_model` re-embeds every chunk
in place rather than serving vectors from the wrong space.

### 3. Re-ingesting a file only embeds what changed

`replace_source` keeps the embedding of every chunk whose text hash is already
stored for that source. `note` appends one line to a growing file on every
dictation, and that must cost one embedding, not the whole file again.

### 4. Ingestion is an asyncio interval task; the work is off the loop

The PRD moves the old `intents.json` poll "onto the loop". `Knowledge.watch()`
is that task: started and cancelled by `Orchestrator.run()` beside the idle
loop. Each scan runs in `asyncio.to_thread`, because embedding is CPU work and
must not stall a turn. A file's digest is `mtime_ns:size`; new and changed
files are re-chunked, removed files are dropped, spoken facts are never
touched by a scan. Cancelling the task stops the scan at the next file and
waits for the worker, so no thread outlives the orchestrator.

Nothing is created at startup: a missing `docs_dir` scans as empty and
`kb.sqlite` is only created by the first write.

### 5. Skills reach the knowledge base through `ctx.knowledge`

`Context` gains an optional `knowledge`, the same shape as `llm` and `edges`.
A skill still imports nothing from the core stack.

| Skill | Change |
|---|---|
| `recall` (new) | "what do my notes say about …", "what did I tell you about …", "where did I park". Retrieves top-k, answers grounded. |
| `remember` (new) | "remember that …" → one timestamped fact, straight into `kb.sqlite`. |
| `note` | Also mirrors the line into `<docs_dir>/dictated-notes.md` and scans, so it is recallable on the next turn. |
| `search` | **Notes first, then Wikipedia**: a hit above the bar is answered from the notes; anything else goes to Wikipedia as before. |

"Remember that …" moves from `note` to `remember` (examples, slot prefixes and
the `intents.json` seed pattern), as the PRD specifies.

### 6. One answer helper, two tiers, no Claude

`jarvis/knowledge/answer.py` is shared by `recall` and `search`:

- **Ollama up** → one `ctx.llm.generate` call over the top-k chunks, told to
  answer only from the excerpts and name the source. An error or an empty reply
  falls through to the next tier.
- **Otherwise** → `From your notes, sir: <best snippet> — from <source>.` The
  source is the file name, or "what you told me on <date>" for a spoken fact.

The orchestrator's existing `persona.phrase` then voices the line, as it does
for every skill. Nothing in `jarvis/knowledge/` or these skills imports
`anthropic`.

### 7. Config

```toml
[knowledge]
enabled = true
docs_dir = "~/jarvis/knowledge"
scan_interval_s = 60
top_k = 4
min_score = 0.35
chunk_chars = 800
chunk_overlap = 100
```

`[knowledge] enabled = false` leaves `ctx.knowledge` as `None`; the skills say
so instead of failing.

### 8. A command for the impatient

`python -m jarvis knowledge scan` ingests now and prints what changed;
`python -m jarvis knowledge status` lists the sources, chunk counts and which
backend is in use.

---

## Acceptance criteria

1. A sample doc in `docs_dir` is ingested on startup; asking *"what does the
   manual say about …"* in text mode gives an answer containing the doc's text
   and naming the file.
2. That turn makes **no call to Claude**: a recording fake client sees zero
   calls, importing the knowledge package and its skills never loads
   `anthropic`, and the dry run works with a blocked API key.
3. "Remember that I parked on level three", then "where did I park" → the fact
   comes back with its date, without restarting.
4. "Note that the door code is 4821", then "what is the door code" → answered
   from the notes. "What is photosynthesis" still reaches Wikipedia.
5. Changing a file re-indexes it; deleting it removes its chunks; spoken facts
   survive both.
6. With `sqlite-vec` unavailable the same queries return the same ranking.
7. With Ollama down the verbatim-snippet frame is spoken; with it up the
   composed answer is, and an Ollama failure degrades to the frame.
8. The edge's import graph is unchanged (`tests/test_edge_imports.py`), and
   `sqlite_vec` / `pypdf` are added to its forbidden list.
9. No thread is left behind by a scan, a cancelled watch task, or a turn.
10. `python check_setup.py` and `python -m jarvis --selftest` stay green.

## Out of scope

- Conversation history as a retrieval source.
- Forgetting or editing a single fact by voice.
- Re-ranking, hybrid keyword search, or any index tuning beyond top-k cosine.
- Fixing `Persona.phrase` blocking the event loop (recorded in
  `PRD/known-issues.md`).
