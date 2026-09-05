# JARVIS — 2026 Rebuild PRD

**Status:** Approved
**Date:** 2026-09-02
**Owner:** Kevin Lundell

This document is canonical and version-controlled with the code. Milestones that
change scope update this file.

---

## Context

`JARVIS/` is a 2024 voice-assistant proof of concept: a flat set of scripts
(`main.py`, `libs/*.py`, `actions/*.py`) doing wake-less Google-cloud STT →
bag-of-words Keras intent classifier → dynamic `importlib` action dispatch →
`pyttsx3` TTS, with an unused OpenAI Q&A path (now swapped to Anthropic in
`libs/anthropic_helper.py`).

The intended product is bigger than the POC:

- An **always-on local daemon**. A spoken wake word (**"Jarvis …"**) opens a
  listening window; everything after the wake word is treated as a command.
- **All routine work is local** — recognition, dispatch, phrasing, speech.
  No network on the hot path.
- JARVIS **learns new skills over time**: when it hits something it can't do it
  offers to learn it; **Claude (API) writes a new Python skill module**, JARVIS
  validates and sandboxes it, keeps it on the user's confirmation, retrains its
  local intent model, and can then do it itself. **Claude is a teacher invoked
  rarely, never a runtime dependency.**
- Retraining **never blocks the main loop**. It runs on a separate worker and is
  merged back in only while JARVIS is idle, then announced — no cancelled
  actions, no "do I have the new skill yet?" ambiguity.
- JARVIS speaks in a **selectable persona** (env var / config). The default is
  JARVIS-from-Iron-Man, defined by a style description plus a corpus of
  in-character lines (the repo's `Movies/*.txt` transcripts) that condition how
  the model phrases replies; swapping the persona is dropping in another folder.
- A **local knowledge base (RAG)** is the source for recall / "what is …"
  answers. Claude is *not* wired in as a knowledge fallback.

Blocking reality: **`tensorflow` has no Python 3.14 wheels** and the checked-in
`JARVIS_model.keras` is an unloadable Keras-2 file. The rebuild **drops
TensorFlow entirely** (embeddings + a tiny sklearn head instead), which also
removes the 3.14 blocker — pending a wheel-availability check on the rest of the
new audio stack (Phase 0 below).

Blockers already known: the Anthropic key in `.env` is identity-linked and needs
`ANTHROPIC_WORKSPACE_ID`; `pyaudio` needs `portaudio-devel` (both are the user's
to resolve, tracked in `check_setup.py`).

---

## Target architecture

### Degradation chain (CPU-only is fully functional; GPU / Ollama only improve it)

| Layer | Always (CPU) | Better (GPU / Ollama) | If "better" is absent |
|---|---|---|---|
| Wake word | `openwakeword==0.4.0` (ONNX) "Jarvis", always-on | same | — |
| Capture | `sounddevice` ring buffer + VAD | same | — |
| STT | `faster-whisper base` int8 CPU | `small`/`medium` on GPU | — |
| NLU | `fastembed` MiniLM (ONNX, no torch) + sklearn `LogisticRegression` head + slot filler | same | — |
| Reasoning / persona phrasing | Ollama `qwen2.5:3b`/`llama3.2:3b` Q4 (**optional**) | 7–8B on GPU | fixed events use the persona's canned `responses.toml` verbatim (still in-voice); dynamic free-form text falls back to plain phrasing; free-form Q&A → *"I have no skill for that, sir. Shall I learn it?"* |
| TTS | Piper, British male voice (ONNX, CPU) | same | — |
| Knowledge | `fastembed` + `sqlite-vec` (brute-force cosine fallback) | Ollama composes answer from chunks | read best snippet verbatim with a canned frame + source |
| Skill factory | Claude API — fires only on `teach` / `edit_skill` | same | needs network only at that moment |

Startup probes for a running Ollama + model and for a sandbox backend; degrades
silently and logs what tier it got.

### Package layout (replaces the flat scripts)

```
jarvis/
  __main__.py              # `python -m jarvis`; thin main.py shim kept
  config.py                # loads config.toml + .env
  core/
    orchestrator.py        # async state machine, busy flag, merge gate
    context.py             # Context handed to every skill (say/schedule/data_dir/llm/http)
    persona.py             # loads the active persona; canned lines verbatim + optional Ollama style-rewrite
    reasoner.py            # optional local LLM client (Ollama HTTP), capability-probed
  audio/
    wake.py                # openWakeWord
    capture.py             # sounddevice + VAD
    stt.py                 # faster-whisper
    tts.py                 # Piper
  nlu/
    classifier.py          # fastembed embed + sklearn head + threshold/unknown
    slots.py               # entity/slot extraction (durations, times, app names, query tail)
    corpus.py              # build data/nlu/corpus.sqlite from seed + skill manifests + corrections
    train.py               # embed new examples, fit head, eval → versioned artifact (worker process)
  skills/
    contract.py            # SkillManifest + Skill protocol
    registry.py            # discovery, hot-reload, quarantine
    builtin/               # search.py, open_app.py, play.py, note.py, remember.py (M3), + meta: teach.py, edit_skill.py, revert_skill.py, sleep.py
    learned/               # Claude-generated skills land here after confirmation
  factory/
    build.py               # host-only: Claude API → module + manifest + test
    validate.py            # AST import/call allowlist, manifest completeness, signature check
    sandbox.py             # SubprocessSandbox (primary); DockerSandbox/PodmanSandbox (optional)
  knowledge/
    store.py               # fastembed + sqlite-vec vector store
    ingest.py              # document + fact ingestion
data/
  models/nlu/v<N>/{head.joblib, labels.json, meta.json}
  nlu/corpus.sqlite
  knowledge/kb.sqlite
  skills/<name>/           # per-skill fs_write scratch
intents.json               # hand-authored SEED only (see below)
config.toml                # non-secret config
.env                       # secrets only (ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID)
personas/
  <name>/
    persona.toml           # display name, style description + rules, address term ("sir"), preferred Piper voice
    style/                 # any number of .txt transcripts conditioning the LLM rewrite (seeded from every Movies/*.txt)
    responses.toml         # in-voice canned lines for fixed events (greeting, standby, error, skill-learned, disambiguation)
```

### The Skill contract (`jarvis/skills/contract.py`)

```python
@dataclass
class SkillManifest:
    name: str                       # "timer"
    description: str                 # human + used in disambiguation prompts
    examples: list[str]             # appended to the NLU training corpus
    params: dict                    # {"seconds": {"type": "integer", "required": True}}
    permissions: set[str]          # subset of {"pure","notify","fs_read","fs_write","net","shell"}
    version: int = 1
    origin: str = "builtin"        # or "learned"

class Skill(Protocol):
    MANIFEST: SkillManifest
    def run(self, ctx: Context, **params) -> str: ...   # returns the line to speak
```

- A skill never imports the audio/core stack; it only touches `ctx`
  (`ctx.say`, `ctx.schedule`, `ctx.data_dir`, `ctx.llm` if permitted,
  `ctx.http` if permitted). This removes the `actions/write.py` ↔
  `command_helper` import cycle.
- Learned skills default to `{"pure","notify"}`; anything more is granted
  explicitly at install time by voice.

### Concurrency & seamless hot-swap

Single asyncio event loop = the orchestrator. It owns `self.nlu` and
`self.registry` (swappable references) and:

```python
self.state: Literal["idle","listening","thinking","speaking","acting"]
self._staged: NLU | None            # a fully-loaded, gate-passed replacement
```

`busy = self.state != "idle"`.

- **Retrain runs in a `multiprocessing.Process`** (fast head-only path) or a
  container (rare full path). On completion it puts a result on a
  `multiprocessing.Queue`.
- An asyncio task `drain_retrain_results()` awaits the queue, **pre-loads** the
  new model + runs a self-check off the loop, and on success sets
  `self._staged`. **No swap yet.**
- **Merge gate** — in every turn's `finally` and on the idle tick:
  ```python
  if self._staged and self.state == "idle":
      old, self.nlu = self.nlu, self._staged
      self.registry = self.registry.rebuilt()     # picks up skills/learned/*.py
      self._staged = None
      old.close()
      self.say_quiet(f"I've learned '{name}', sir. My capabilities are updated.")
  ```
  Single assignment on the loop thread → no lock. A turn already holding the old
  `self.nlu` finishes on it; the next wake uses the new one. Mid-transcription /
  mid-action work is never cancelled.
- **Self-check before staging**: new model must load and still classify the seed
  examples sanely; on failure the skill is **quarantined** (file not promoted,
  logged, JARVIS says it set the skill aside). No elaborate regression gate —
  the busy flag + idle merge + quarantine is the mechanism.
- **Ordering**: the generated skill file stays in `staging/`; it is promoted to
  `skills/learned/` **at the same moment** as the model swap.
- **Rollback**: keep the last **3** model versions and skill-file versions
  (`timer.v1.py`, `timer.v2.py`, …). `revert_skill` un-promotes + swaps back +
  quarantines the newer one.
- **UX split** (orchestrator estimates from example count / container job):
  fast → *"One moment, sir… done. Try me."*; slow → *"I'll practice that and let
  you know."* then serve on the old model and announce on merge.

### Skill factory

Claude runs **on the host** (needs network + key); **all execution of generated
code is sandboxed** (no network, no key).

| Stage | Runs on | Net | Secrets |
|---|---|---|---|
| Clarify spec (dialog) | host — Ollama or 2-question script | — | — |
| Generate module + manifest + test | host `factory/build.py` → Claude API | Claude only | `ANTHROPIC_API_KEY` |
| Static validation (AST allowlist) | sandbox | none | none |
| Run generated tests | sandbox | none | none |
| Dry-run `run()` with sample params | sandbox | none | none |
| Retrain head + self-check | worker process (fast) / container (slow) | none | none |
| Present report → voice confirm | host | — | — |
| Promote file + atomic model swap | host orchestrator (idle merge) | — | — |

- **`SubprocessSandbox` is the primary path**: `python -I` in a temp cwd,
  `resource` rlimits (`RLIMIT_AS`, `RLIMIT_CPU`), wrapped in
  `unshare --user --map-root-user --net` where unprivileged user namespaces are
  available (default on Fedora), plus an **AST-enforced import/call ban**
  (`socket`, `http`, `urllib`, `requests`, `subprocess`, `ctypes`, `os.system`,
  `eval`, `exec`, `__import__`) unless the matching permission was granted.
  Completion reported via a **callback / queue drained by an asyncio task**.
- `DockerSandbox` / `PodmanSandbox` are optional stronger backends, selected in
  `config.toml` when present. When only the subprocess path is available,
  JARVIS warns once that isolation is weaker.

### Knowledge base (RAG)

- `jarvis/knowledge/store.py`: `fastembed` MiniLM embeddings + `sqlite-vec`
  index in `data/knowledge/kb.sqlite`; brute-force cosine over a numpy matrix as
  the fallback for small corpora.
- **Recall path**: knowledge-type intent → retrieve top-k chunks → compose the
  answer in persona via Ollama if present; **CPU-only fallback** = speak the
  single best snippet inside a canned frame (*"From your notes, sir: …"*) with
  its source.
- **Claude is never invoked for knowledge answers.**
- **Ingestion — folder watch + spoken facts**:
  - `knowledge.docs_dir` in `config.toml` (default `~/jarvis/knowledge/`).
    `jarvis/knowledge/ingest.py` scans it on startup and re-scans on an asyncio
    interval task (mtime+size digest per file — the same idea as the old
    `my_thread_function` poll, moved onto the loop). New/changed files are
    chunked and embedded; removed files are dropped from the index.
    Loaders: `.txt`/`.md` direct, `.pdf` via `pypdf`.
  - Spoken facts: a `remember` builtin skill (`"Jarvis, remember that …"`)
    appends a timestamped one-chunk note straight into `kb.sqlite`
    (no file needed). `note.py` (migrated `write.py`) also mirrors into the
    docs dir so dictated notes are recalled too.

### Personas (swappable, data-only)

A persona is a folder under `personas/`, never code — adding one is dropping a
directory, no sandboxing involved.

- **Selection**: `JARVIS_PERSONA` env var → else `persona.active` in
  `config.toml` → else `"jarvis"`. Ships with `jarvis` (seeded from every
  `Movies/*.txt`) and `plain` (neutral, no style corpus).
- `persona.toml`: display name, style description + rules, address term
  (`"sir"`), preferred Piper voice model.
- `style/*.txt`: transcripts of in-character lines / exchanges. `core/persona.py`
  embeds them once (reusing `fastembed`); when Ollama is present, each outgoing
  dynamic line is rewritten with a system prompt (description + rules) and a
  **few-shot block of the k nearest style lines to that response** — so the
  persona conditions phrasing without bloating the prompt. This is the concrete
  meaning of "the persona is learned from how JARVIS talks".
- `responses.toml`: fixed-event lines already in voice, spoken verbatim — no LLM
  needed, so fixed events stay in-character on a pure-CPU box.
- Natural later extension (out of scope now): a `teach`-style flow where Claude
  drafts a new persona folder from a description + sample lines.

### `intents.json` — narrowed role

- `intents.json` stays as the **hand-authored seed**: builtin intents,
  meta-skill intents (`teach`, `edit_skill`, `revert_skill`, `sleep`),
  persona-critical canned lines. Human-editable, version-controlled, the
  "boot this agent" file. The trainer **never writes to it**. Existing
  `patterns`/`responses`/`action`/`tag` schema is kept and read as examples.
- **Each skill module owns its `examples`** in its `MANIFEST`, versioned with
  the code. Learned skills never touch `intents.json`.
- `data/nlu/corpus.sqlite` (rebuildable via `jarvis nlu rebuild`) = seed ∪ every
  registered skill's examples ∪ user corrections. The trainer reads only this.

---

## Migration mapping

| Old | New |
|---|---|
| `main.py` | `jarvis/__main__.py` + thin `main.py` shim |
| `libs/command_helper.py` `run()` / `standby` / `takeCommand` | `jarvis/core/orchestrator.py` |
| `libs/command_helper.py` `my_thread_function` (intents.json mtime watcher) | **dropped** — retrains are event-driven on skill install |
| `takeCommand()` Google STT | `jarvis/audio/{wake,capture,stt}.py` |
| `libs/brain.py` (bag-of-words + Keras) | `jarvis/nlu/classifier.py` + `jarvis/nlu/slots.py` |
| `libs/training.py` `train_model()` | `jarvis/nlu/train.py` + `jarvis/nlu/corpus.py` (worker process) |
| `JARVIS_model.keras`, `words.pkl`, `classes.pkl` | `data/models/nlu/v<N>/…` (versioned, delete old files) |
| `libs/voice.py` (pyttsx3) | `jarvis/audio/tts.py` (Piper) + `jarvis/core/persona.py` |
| `Movies/*.txt` (all transcripts) | `personas/jarvis/style/*.txt` (persona style corpus; the `Irom Man 2.txt` typo is fixed in the copy; drop in more transcripts and `persona.py` picks them up) |
| `libs/anthropic_helper.py` / removed `libs/openai_helper.py` | `jarvis/factory/build.py` (Claude, host-only) + `jarvis/core/reasoner.py` (Ollama, optional) |
| `actions/search.py` | `jarvis/skills/builtin/search.py` (`MANIFEST` + `run(ctx, query)`) |
| `actions/openApp.py` | `jarvis/skills/builtin/open_app.py` |
| `actions/play.py` | `jarvis/skills/builtin/play.py` (was orphaned — give it a `play` intent) |
| `actions/write.py` | `jarvis/skills/builtin/note.py` (was orphaned; uses `ctx`, no import cycle) |
| `ask_chat_gpt` inline path | removed; superseded by `teach` + knowledge base |
| `.env` (`language`, `api_key`) | `.env` secrets only; `config.toml` for `language`, wake/STT/voice/Ollama/thresholds/paths and `persona.active` (env `JARVIS_PERSONA` overrides) |

---

## Phased delivery

### Phase 0 — dependency spike (no app code) — ✅ DONE (2026-09-03)
Confirm CPU wheels on the target Python (3.14 first; **fall back to a 3.12
`.venv` if any are missing**): `faster-whisper`/`ctranslate2`, `openwakeword`,
`piper-tts`/`piper-phonemize`, `fastembed`/`onnxruntime`, `sqlite-vec`,
`sounddevice`, `scikit-learn`, `pypdf`. Record the outcome; pick the
interpreter. Extend `check_setup.py` with these rows.

**Outcome (full detail in `PRD/phase-0-dependency-spike.md`):**
- **Interpreter: Python 3.14.** Every dependency has a cp314 / pure-Python CPU
  wheel and imports clean. The 3.12 fallback is not needed — dropped.
- **`piper-phonemize` removed** — `piper-tts` 1.7 bundles phonemization; it is
  not a separate package and is not on PyPI.
- **`openwakeword` pinned to `0.4.0`** — 0.5+ require `tflite-runtime`, which
  has no PyPI wheel; 0.4.0 runs the ONNX path (onnxruntime is already in the
  stack).
- `check_setup.py` gained a "2026 rebuild stack (PRD Phase 0)" section; all
  rows green.

### Milestone 1 — local voice loop (no learning, no RAG) — ✅ DONE (2026-09-03)
- `jarvis/` skeleton, `config.toml`, `Context`, `Skill` contract, `registry`.
- `audio/{wake,capture,stt,tts}.py`; `core/persona.py` with `personas/jarvis`
  (seeded from every `Movies/*.txt`) + `personas/plain`, selected by
  `JARVIS_PERSONA` / `config.toml`; canned `responses.toml` verbatim, LLM
  style-rewrite only when Ollama is present.
- `nlu/{corpus,classifier,slots,train}.py`; produce `v1` from `intents.json` +
  builtin manifests; threshold + `unknown` fallback.
- `core/orchestrator.py`: async state machine, busy flag, wake → window →
  `sleep`/timeout → standby.
- Port builtin skills: `search`, `open_app`, `play`, `note`.
- **Outcome**: *"Jarvis, search black holes"* runs end-to-end, offline except the
  skill's own web calls.

**Delivered (full detail in `PRD/milestone-1-voice-loop.md`):**
- `python -m jarvis` (run | `--selftest` | `models pull` | `nlu rebuild`).
  `main.py` / `libs/` / `actions/` left in place, untouched.
- Wake word is **"hey jarvis"** (openwakeword 0.4.0's bundled model, offline).
- NLU = fastembed MiniLM + `LogisticRegression`; `unknown` via a probability
  threshold **and** a nearest-neighbour similarity floor (LR is overconfident
  on out-of-domain junk). Held-out accuracy ~0.95.
- 49 automated tests (`python -m pytest`); NLU tests skip cleanly if the
  embedding model can't be fetched offline.
- Not verified here (no audio device): live mic/STT, Piper playback, the
  Ollama style-rewrite path, `models pull` downloads. Manual steps in the
  outcome doc.
- Deviation: no separate `sleep` intent — "go to sleep" folded into `goodbye`
  (action `exit`); `goodbye`/`shutdown` pattern overlap the PRD flagged is
  split.

### Milestone 2 — skill factory + seamless hot-swap — ✅ DONE (2026-09-05)
- `factory/{build,validate,sandbox}.py` (`SubprocessSandbox` primary).
- `teach` meta-skill: no-match → offer → clarify → build → sandbox → confirm →
  stage.
- Retrain worker process + result queue + `drain_retrain_results()` task.
- Merge gate (swap `nlu` + `registry` only at `state == idle`), announcement,
  3-version retention, quarantine-on-self-check-fail, fast/slow UX split.
- `edit_skill` / `revert_skill` meta-skills; `timer.v<N>.py` retention.

**Outcome (full detail in `PRD/milestone-2-skill-factory-outcome.md`;
plan/design decisions in `PRD/milestone-2-skill-factory.md`):**
- All of the above shipped and is exercised against real subprocesses (real
  `pytest`, real `unshare` network isolation, a real `multiprocessing`
  retrain) — 113 tests pass (was 67 before this milestone).
- Added a `python -m jarvis text` mode (`jarvis/audio/text_io.py`) — the real
  pipeline, text in/out instead of mic/wake-word/STT/speaker — specifically
  so `teach`/`edit_skill`/`revert_skill` could be driven and verified
  end-to-end without a mic. Now the required last step for any PRD work
  (see `CLAUDE.md`).
- Found and fixed via smoke-testing: the sandbox's default `RLIMIT_AS`
  (256MB) was too tight for `pytest.main()` to even start in this venv; the
  default is now 512MB.
- Found and fixed via the `text`-mode dry-run (after the full test suite was
  already green): a generated skill's manifest always defaulted to
  `origin="builtin"` (nothing ever set it to `"learned"`), which silently
  made `edit_skill`/`revert_skill` unable to find *any* real taught skill.
  Fixed at the one authoritative point — `Registry.discover()` now sets
  `origin` from the package a module was actually found in, regardless of
  what the module's own code claims.
- Deviation: skill-version history lives at `data/skills/_versions/<name>/
  vN.py` (mirrors the existing `data/models/nlu/v<N>/` convention) rather
  than the literal `timer.v1.py`, since a dotted filename isn't an
  importable module name.
- Not verified here: a live Claude call (blocked by the pre-existing
  `ANTHROPIC_WORKSPACE_ID` gap noted below and in `check_setup.py`) and the
  mic/wake-word/STT path itself (no audio device in this environment) —
  everything downstream of transcription was verified via text mode instead.

### Milestone 3 — knowledge base (RAG)
- `knowledge/{store,ingest}.py`; recall intent → retrieve → compose (Ollama) or
  verbatim-snippet fallback; no Claude.
- Ingestion: startup + interval scan of `knowledge.docs_dir`
  (`.txt`/`.md`/`.pdf`), plus a `remember` builtin for spoken facts; `note.py`
  mirrors dictated notes into the docs dir.

### Milestone 4 — polish
- Barge-in during TTS, systemd user service, packaging (`pyproject.toml`),
  tests, README/CLAUDE.md rewrite.

---

## Verification

- **Phase 0**: `check_setup.py` shows ✓ for every new dependency on the chosen
  interpreter.
- **M1**: unit tests for `nlu.train`/`classifier` (train v1 from a fixture
  corpus, assert seed intents classify above threshold, assert gibberish →
  `unknown`); `persona.py` test (loads `jarvis` and `plain`, `JARVIS_PERSONA`
  env beats `config.toml`, missing persona → clear error); manual: run
  `python -m jarvis`, say *"Jarvis, search black holes"*, *"Jarvis, play lofi"*,
  *"Jarvis, go to sleep"* — verify persona TTS and logged standby/window
  transitions; re-run with `JARVIS_PERSONA=plain` and confirm the phrasing of
  fixed responses changes.
- **M2**:
  - `factory/validate` unit tests — a module importing `socket` without `net`
    permission is rejected; a missing-`params` manifest is rejected.
  - `sandbox` test — a skill that opens a socket fails in the sandbox.
  - **merge-gate timing test** — set `state="acting"`, post a staged model,
    assert `self.nlu` is unchanged; set `state="idle"`, tick, assert the swap
    happened and the announcement fired.
  - manual: *"Jarvis, learn how to set a timer"* → full teach → confirm →
    hear *"I've learned 'timer'"* → *"Jarvis, set a timer for 10 seconds"* fires.
  - manual rollback: *"Jarvis, revert the timer skill"*.
- **M3**: ingest a sample doc, ask *"Jarvis, what does X say about Y"*, verify a
  grounded answer with a source and **no network call to Claude** (assert via
  logs / a blocked key).
- Regression: `python check_setup.py` stays green; `python -m jarvis --selftest`
  loads the current NLU model, lists registered skills, and exits 0.
