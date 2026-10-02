# Milestone 5 — knowledge base (RAG): outcome

**Date:** 2026-10-01
**Branch:** `knowledge-base` (off `develop`; committed per stage, not pushed, not merged)
**Result:** ✅ JARVIS answers from the user's own documents and from facts said
aloud. Files in `[knowledge] docs_dir` and "remember that …" facts are embedded
into `data/knowledge/kb.sqlite`; `recall` (and "what is …", notes first)
answers from them with the source named. Claude is never called for it.
**Not verified:** the Ollama-composed tier. There is no Ollama on the dev
machine, so it is covered by tests against a fake reasoner only — see
"What was not checked".

Plan and design decisions: `PRD/milestone-5-knowledge-base.md`. Taken before
Milestone 4, which is still not started.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Store | `jarvis/knowledge/store.py` (new) | `sources` / `chunks` / `meta` in one sqlite file. Each chunk's embedding is a float32 BLOB — the source of truth. `sqlite-vec` (`vec0`, cosine) indexes the same rows when the extension loads and is rebuilt from the BLOBs when it is missing or behind; otherwise brute-force cosine over a numpy matrix, same ranking. The embedder is injected. A changed embedding model re-embeds in place. One connection behind a lock; no threads of its own; nothing on disk until the first write. |
| Ingestion | `jarvis/knowledge/ingest.py` (new) | `.txt` / `.md` direct, `.pdf` via `pypdf`. A paragraph is a chunk; a heading is folded in as "Heading: text"; only a paragraph over `chunk_chars` is split, on sentences, with overlap. `scan` compares `mtime_ns:size`, re-indexes what changed, drops what is gone. Re-indexing embeds only chunks whose text is new. |
| Facade + interval task | `jarvis/knowledge/__init__.py` (new) | `Knowledge.search / answer / remember / index_file / scan / watch`. `watch` is the asyncio interval task; each scan runs in `asyncio.to_thread`, and cancelling waits for that worker. |
| Answer | `jarvis/knowledge/answer.py` (new) | Reasoner up → one grounded `generate` over the top-k chunks (`NO_ANSWER` = nothing there; a reply that forgot its source is given one). Otherwise *"From your notes, sir: … — from <source>."* Of near-equal hits the newest is the answer. |
| Skills | `skills/builtin/recall.py`, `remember.py` (new); `note.py`, `search.py` | `note` mirrors into `<docs_dir>/dictated-notes.md` and indexes it at once. `search` asks the notes first at `search_min_score`, then Wikipedia. "Remember that …" moved from `note` to `remember`. |
| Wiring | `core/context.py`, `skills/registry.py`, `core/orchestrator.py`, `app.py`, `__main__.py` | `ctx.knowledge`; all three builders scan at startup and hand one `Knowledge` to registry and orchestrator; `run()` owns the re-scan task. `python -m jarvis knowledge scan\|status`; a knowledge row in `--selftest`. |
| NLU | `intents.json`, `nlu/slots.py`, `nlu/corpus.py` | Seed patterns for `recall` and `remember`, more for `note`, slot prefixes, two shared subjects in the placeholder fillers. |
| Config / setup | `config.py`, `config.example.toml`, `check_setup.py`, requirements | `[knowledge]` (8 keys, nonsense values clamped); `sqlite-vec`, `pypdf` required; a "Knowledge base" section in the setup report. |
| Tests | 5 new files + `tests/knowledge_harness.py` | **742 passed, 2 skipped** (`develop`: 625 passed, 2 skipped). 114 of the new ones are in the knowledge base's own files. |

## Test changes (said explicitly, per CLAUDE.md)

- **`tests/test_nlu.py`**: `test_seed_intents_classify_above_threshold` expected
  `"remember that i parked on level three"` → `note`. The PRD's knowledge base
  gives that phrase to `remember`, so the expectation was out of date, not the
  code. The seed test got a `note` phrasing in its place, and a new test,
  trained on seed + builtin skills as a real start is, expects `remember`.
- **`tests/test_requirements.py`**: the name-map check skipped `sqlite_vec` as
  "not imported by jarvis/ yet". It is now, so the skip is gone.
- **`tests/test_text_io.py`**, **`tests/test_orchestrator.py`**: three tests
  that build JARVIS from the real config now point the knowledge base at a tmp
  docs dir (or stub `build_knowledge`). Their assertions are unchanged; without
  this they would index whatever is in the developer's `~/jarvis/knowledge`.
- **`tests/test_skills.py`**, **`tests/test_config_example.py`**,
  **`tests/test_edge_imports.py`**: the builtin roster, the `[knowledge]`
  section, and `sqlite_vec` / `pypdf` added to what the edge must never import.

## What the tests caught (and the code, not the test, was fixed)

1. **Equal scores came back in a different order from the two backends.**
   `test_both_backends_rank_alike` — the results are now sorted by score, then
   insertion order, whichever backend found them.
2. **Gibberish classified as `note`.** A seed pattern `Note {intent}` added
   while tuning was a frame loose enough to take anything.
   `test_gibberish_is_unknown` caught it; the pattern was removed.

## Found by the dry run, not by the tests

1. **The index was not ready for the first question.** The startup scan was
   the interval task's first tick, so a scripted first line raced the embedding
   model loading. The builders now scan before saying "ready"; the task waits
   one interval first.
2. **"Note that …" was `unknown`, and "what's my locker number" too.** The two
   new skills had only their manifest examples, while `note` and `search` have
   seed patterns expanded over ten fillers — and the examples shared subjects
   ("the wifi code") across intents, so the classifier learned the subject, not
   the frame. Fixed with seed patterns for both, and shared subjects in the
   fillers. Measured after: `remember that i parked on level three` 0.95,
   `where did i park` 0.84, `what's my locker number` → recall 0.86,
   `what is photosynthesis` → search 0.87.
3. **A wrong answer, read out with a source.** With `min_score = 0.3`, "what's
   my locker number" (not in the notes) was answered with the door code (0.45).
   24 questions were then scored against the dry-run store: answerable ones
   0.50–0.85, unanswerable ones up to 0.45. `min_score` is now 0.48.
4. **A stale model.** The dry run kept classifying with examples from an hour
   earlier: `ensure_nlu` only retrains for a new label. Recorded as known
   issue 15, not fixed here.

## Found by an independent review

A second model reviewed `jarvis/knowledge/` cold and reported eleven defects the
tests did not cover, ten of them reproduced with a script. Each was fixed with a
test first (commit "fixes from an
independent review"):

- the embedder held its lock for a whole batch, so a question queued behind a
  scan embedding a large file;
- a scan that had read the notes file could store that older text after `note`
  appended — files are now read and stored under a per-source lock;
- a heading with no blank line under it swallowed its section, and several of
  them became one chunk;
- a reasoner saying "I have nothing on that" was spoken as an answer with a
  source, and blocked `search`'s Wikipedia fallback (now `NO_ANSWER`);
- the recency tie-break used indexing time, so every old line of a re-indexed
  notes file was "newest" — a chunk now keeps the date its text was first seen,
  taken from the file's mtime;
- an unmounted or unreadable docs folder emptied the index; an unreadable file
  lost its indexed content and was never retried after a `chmod`;
- `chunk_chars = 0` hung the startup scan;
- two crashes on rare paths (numpy search racing the last removal; a zero
  vector's NULL distance in sqlite-vec);
- a second cancel abandoned the scan worker and left the stop flag set.

What it found and was left is known issue 17.

## Verification

### Automated

`python -m pytest`: **742 passed, 2 skipped** in 109 s. The two skips are not
new: no `unshare --user --net` on this machine, and `yaml` in the requirements
name map.

The PRD's M5 line — *ingest a sample doc, ask "what does X say about Y",
verify a grounded answer with a source and no network call to Claude* — is
`tests/test_knowledge_e2e.py`: the real text pipeline (real NLU, registry,
persona, MiniLM) over a tmp docs dir, with the API key replaced by a blocked
one and `ClaudeClient.generate_skill` recording calls. The answer is the
document's sentence with "from coffee manual"; the call list is empty. A
subprocess check confirms that importing the knowledge package and its four
skills never loads `anthropic`.

`python check_setup.py` passes and reports the knowledge base;
`python -m jarvis --selftest` passes and prints its row.

### Threads (CLAUDE.md step 5)

- Tests: the autouse `no_leaked_threads` fixture runs on all 742. The watch,
  cancel-mid-scan, double-cancel and end-to-end tests also compare
  `threading.active_count()` before and after themselves: equal in each.
- Dry run (below), sampled four times a second with `ps -T`: **baseline 32**
  once the models are loaded, **33–34 during a turn or an interval scan, back
  to 32 after each** — 11 turns and about 17 interval scans, no growth.

### Dry run

`ANTHROPIC_API_KEY=blocked python -m jarvis -v text`, lines piped in three
seconds apart, against a scratch config (tmp data dir and docs dir holding a
manual and a note file; `scan_interval_s = 2`). Reasoner: none.

```
you> what does the coffee manual say about descaling
Jarvis: From your notes, sir: Coffee machine: Descale the coffee machine every two months with
        citric acid. Run two full tanks of clean water through it afterwards — from coffee manual.
you> remember that i parked on level three
Jarvis: I'll remember that.
you> where did i park
Jarvis: From your notes, sir: i parked on level three — from what you told me on 1 October.
you> note that the door code is 4821
Jarvis: Noted.
you> what is the door code
Jarvis: From your notes, sir: the door code is 4821 (noted 1 October 2026) — from dictated notes.
you> what is photosynthesis
Jarvis: According to Wikipedia: Photosynthesis is a system of biological processes …
you> when do the bins go out
Jarvis: From your notes, sir: The bins go out on Thursday night. Recycling is collected every
        other week — from bins.
you> what's my locker number
Jarvis: I have nothing on that in your notes, sir.
you> how big is the water tank
Jarvis: Could you rephrase that, sir?
```

The only lines in the log naming Claude are the startup banner's
`factory: claude:claude-opus-5`. The last turn is known issue 14: the answer is
in the manual, and the question never reached it.

`python -m jarvis knowledge scan` reported `2 added`; `status` listed both
files, 4 chunks, `index: sqlite-vec`.

## What was not checked

- **The composed tier against a real Ollama.** Prompt, `NO_ANSWER`, the
  appended source and the fall-through on error are tested against a fake
  reasoner. Whether a small local model actually follows the prompt, and what
  the persona rewrite then does to its answer (known issue 16), is untested.
- **The voice path and `serve`.** The dry run was text mode. `build_orchestrator`
  and `build_server_orchestrator` get the same three lines of wiring, and
  nothing in the knowledge base depends on audio, but neither was run.
- **A large corpus.** The biggest store exercised held a few dozen chunks.
  Startup-scan time and search latency on hundreds of PDFs are unmeasured.
- **A real PDF.** The loader is tested with a one-page PDF built in the test.

## Known behaviour worth stating plainly

- **A new JARVIS start retrains the NLU once**, because `recall` and
  `remember` are new labels. After editing `intents.json` by hand, run
  `python -m jarvis nlu rebuild` (known issue 15).
- **"Remember that …" no longer writes to `notes.txt`.** It is a fact in
  `kb.sqlite`. "Note that …" still writes the notes file, and now the mirror.
- **Without a reasoner, a recall answer is a quotation, not an answer.** It
  reads the best chunk, up to 320 characters, cut at a sentence.
- **The bars are set on a small sample** (24 questions, one small store). They
  are config keys for that reason.
