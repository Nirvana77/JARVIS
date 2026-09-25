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
  No network on the hot path *in the default all-in-one topology*; the
  optional remote-edge topology (below) deliberately puts a **self-hosted**
  link on the hot path — still no third-party cloud.
- JARVIS can **optionally be split in two** (Milestone 3). The **brain** runs
  on a server with a good GPU: NLU, skills and persona, plus warm local
  `jarvis-whisper` (STT) and `jarvis-voder` (TTS) services. A thin **edge**
  device (Raspberry Pi-class) in the room owns the mic and speaker. It cuts
  speech into segments and sends them to the brain over TLS, possibly across
  the internet. There is no wake word on the edge: the brain decides from the
  transcript whether JARVIS was addressed. The voice pipeline is ported from
  [SBRA-Dynamics/Mike](https://github.com/SBRA-Dynamics/Mike).
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
| Transport (M3) | in-process (all-in-one) | remote edge ↔ brain over `wss://` | edge unreachable → local "offline" earcon, reconnect with backoff |

Startup probes for a running Ollama + model and for a sandbox backend; degrades
silently and logs what tier it got.

### Deployment topologies

| Layer | All-in-one (default, `python -m jarvis`) | Remote: edge (`python -m jarvis edge`) | Remote: brain (`python -m jarvis serve`) |
|---|---|---|---|
| Mic capture | local | ✓ | — |
| Speech gating | openwakeword "hey jarvis" + VAD | segmenter (energy VAD, adaptive floor), PTT button | addressing modes on the transcript (ByName / Always / PushToTalk / Ignore) + hold window |
| STT (faster-whisper) | in-process | — | ✓ `jarvis-whisper` loopback service (GPU) |
| NLU, skills, persona, reasoner, factory, knowledge | local | — | ✓ |
| TTS (Piper) | in-process | — | ✓ `jarvis-voder` loopback service, sentence by sentence |
| Playback | local | ✓ (`speech` parts) | — |

The remote topology is opt-in. The orchestrator is the same in every mode. Only
the injected `wake`/`mic`/`stt`/`tts` objects differ: `RemoteLink` on the brain
feeds it gated, merged transcripts, the same way `TextIO` does in text mode.
The wire protocol, segmenter, addressing, services, security (TLS required,
per-device tokens) and resilience rules are in
`PRD/milestone-3-remote-edge.md`.

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
    speech.py              # speakable text + sentence split for the voder (M3)
  audio/
    wake.py                # openWakeWord
    capture.py             # sounddevice + VAD
    stt.py                 # faster-whisper
    tts.py                 # Piper
    player.py              # PCM playback with stop() (M3; shared by Speaker and the edge)
    segment.py             # pure frames→segments VAD, ported from Mike (M3)
    whisper_client.py      # client for the jarvis-whisper service + NullTranscriber (M3)
  nlu/
    classifier.py          # fastembed embed + sklearn head + threshold/unknown
    slots.py               # entity/slot extraction (durations, times, app names, query tail)
    corpus.py              # build data/nlu/corpus.sqlite from seed + skill manifests + corrections
    train.py               # embed new examples, fit head, eval → versioned artifact (worker process)
  skills/
    contract.py            # SkillManifest + Skill protocol
    registry.py            # discovery, hot-reload, quarantine
    builtin/               # search.py, open_app.py, play.py, note.py, remember.py (M5), + meta: teach.py, edit_skill.py, revert_skill.py, sleep.py
    learned/               # Claude-generated skills land here after confirmation
  factory/
    build.py               # host-only: Claude API → module + manifest + test
    validate.py            # AST import/call allowlist, manifest completeness, signature check
    sandbox.py             # SubprocessSandbox (primary); DockerSandbox/PodmanSandbox (optional)
  remote/                  # M3 — optional brain/edge split (pipeline ported from Mike)
    protocol.py            # versioned JSON schema (base64 PCM segments), validate_c2s, close codes
    intake.py              # decode_segment bounds, transcribe, `heard`
    addressing.py          # ByName/Always/PushToTalk/Ignore, always-live mode commands, hold window
    server.py              # `serve`: wss:// server, auth, RemoteLink (wake/mic/stt/tts roles)
    edge.py                # `edge`: mic → segmenter → audio; play speech; PTT; light imports only
  knowledge/
    store.py               # fastembed + sqlite-vec vector store
    ingest.py              # document + fact ingestion
services/                  # M3 — warm loopback services on the brain host, own venvs + systemd units
  whisper/serve.py         # POST /transcribe (PCM → text), GET /healthz
  voder/serve.py           # POST /speak (text → PCM), GET /voices, GET /healthz
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
- **Background learning** (M2.5; replaces M2's fast/slow UX split): after the
  teach/edit/revert dialog, the whole build → sandbox → retrain pipeline runs
  as a background job. JARVIS keeps serving commands on the old model. It asks
  the job's questions (permission grant, keep-confirm) and makes its
  announcements only at safe points between turns, then merges.

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

### Milestone 2.5 — background skill learning (addition to M2) — ✅ DONE (2026-09-21)

Full plan: `PRD/milestone-2.5-background-learning.md`; outcome:
`PRD/milestone-2.5-background-learning-outcome.md`.

**Motivation**: M2 keeps the *model swap* off the hot path, but everything
after the teach/edit/revert dialog still runs inside the turn: the Claude
build (10–60 s), sandbox, retrain and confirm. JARVIS can't take another
command until learning is done.

- Each flow splits into a **dialog** (foreground, the questions only the user
  can answer) and a **job** (background: build → validate → permission grant →
  sandbox → retrain → self-check → keep-confirm → stage).
- After the dialog, JARVIS says it will work on it and the session carries on.
  The user can ask for other things while the skill is learned.
- The job never speaks directly. Its announcements and yes/no questions
  (permission grant, "Shall I keep it?") are queued. They are spoken at
  **safe points**: after a turn, or before the drop to standby. They are
  never spoken mid-turn, and questions are never asked in standby. An unclear
  answer leaves the question pending instead of counting as "no".
- Generated code still never runs before a needed permission is granted.
- Jobs are serialized, and each job retrains on the latest registry, so
  back-to-back teaches can't erase each other.
- Replaces M2's fast/slow UX split. The slow path's auto-confirm is dropped
  and every new or edited skill gets an explicit keep-confirm.

**Verification**: see the plan doc. In short: control returns after the
dialog while a fake `generate` is still blocked, and a second command
dispatches. Questions are asked only at safe points. Permissions are granted
before the sandbox. Serialized jobs keep both skills. Shutdown cancels
cleanly. Text-mode dry run with a delayed fake Claude.

### Milestone 3 — remote edge (brain server + audio satellite) — ✅ DONE (2026-09-25)

Full plan and design decisions: `PRD/milestone-3-remote-edge.md`; outcome:
`PRD/milestone-3-remote-edge-outcome.md`. The voice
pipeline is **ported from [SBRA-Dynamics/Mike](https://github.com/SBRA-Dynamics/Mike)**
(its PRD 5a voice, PRD 6 hold window and PRD 8.3 voder), not reinvented.

**Motivation**: the best STT and the local LLM want a GPU, but the place
JARVIS needs to *hear and speak* is a room. A room is where a small, quiet,
cheap device belongs, not a GPU server. Splitting the two lets the brain live
on the strong machine while a Raspberry Pi-class edge sits where the user is,
even when the two are on different networks.

- **Two runnables and two services.** `python -m jarvis serve` is the brain.
  `python -m jarvis edge --server wss://…` is the satellite. `jarvis-whisper`
  (faster-whisper, GPU) and `jarvis-voder` (Piper) are warm services on
  127.0.0.1 next to the brain. They survive brain restarts, a hang means
  killing one process, and the brain degrades honestly when one is down
  ("Cannot hear you: …"). All-in-one `python -m jarvis` (with its local wake
  word) and `text` mode stay the default and are unchanged.
- **Thin edge, no wake word, no models.** A pure segmenter (Mike's
  `segment.ts`, ported) turns mic frames into speech segments. It uses energy
  VAD with a minimum-statistics noise floor, 300 ms pre-roll, 700 ms hangover,
  a 15 s cap cut at a pause, and per-segment diagnostics. The edge sends one
  `audio` message per segment, plus `speaking{on}` the moment speech starts or
  stops. The edge installs only numpy, sounddevice and websockets.
- **The addressing modes are the wake word.** The brain transcribes every
  segment, sends `heard` back *before* routing, and gates on the transcript:
  ByName ("Jarvis, …", the default), Always, PushToTalk (a button on the Pi,
  with the mic truly off otherwise) or Ignore. The mode commands ("pause
  input", "continue input", "change input to …", "turn off the mic") work in
  every mode. A conversation window after replies and `_ask` prompts skips the
  name. An empty transcription produces nothing.
- **Hold window.** Fragments within 2 s are merged into one turn. The countdown
  pauses while the edge reports `speaking`, and a 20 s cap releases what is
  held. A sentence with a pause in it is therefore one command, not two.
- **Speech out, sentence by sentence.** Text is made speakable, then the first
  sentence is synthesized and sent at once (< 500 ms budget). It goes as
  `speech` parts only to a connection that asked for them. A button press
  stops playback at once and drops the unsent sentences.
- **No orchestrator rewrite.** The brain-side `RemoteLink` plays the
  `wake`/`mic`/`stt`/`tts` roles over gated transcripts, as `TextIO` does in
  text mode. `Interrupter` gains a public `trigger_cancel()`.
- **Internet-safe.**
  - Transport: a versioned JSON protocol over `wss://`, with TLS required (own
    certificate or a reverse proxy), and the edge verifies the server cert.
  - Auth: per-device tokens in `.env`, compared in constant time; a hello
    timeout; close codes.
  - Input handling: base64 and size bounds checked before decoding, and
    validation that never raises.
  - Logging: no audio or tokens in logs.
  - Privacy, stated plainly in the README: with no wake word, all speech the
    Pi hears is transcribed on the self-hosted server. PushToTalk and "turn
    off the mic" are the ways to stop it.
- **Resilient**: ping keepalive, edge reconnect with backoff and an offline
  earcon, and a clean session end on disconnect. Mike's durable session replay
  is deliberately not adopted. There is one edge per brain for now, and
  `device_id` is in the protocol for multi-room later.

**Verification**: tests with no real network, mic or GPU:
- segmenter determinism and floor behaviour on synthetic and recorded PCM;
- protocol and intake bounds;
- every addressing mode and mode command;
- the hold window on a fake clock;
- loopback WebSocket round trips with fake whisper and fake voder (addressed
  command → `heard` → skill → `speech`; un-addressed speech → `heard` only;
  `_ask` without the name; interrupt; whisper down; disconnect);
- auth rejections;
- the whisper service on `--port 0`;
- an edge-import test.

Text-mode regression is unchanged. Manual checks follow Mike-style acceptance
criteria on a real GPU brain and Pi edge over the internet, with per-stage
latency logged against the budget: segment flush < 200 ms, transcription
< 800 ms, first speech < 500 ms.

### Milestone 4 — misheard-command reasoning

**Motivation**: STT is never perfect, and it's at its worst on exactly the
words JARVIS most needs right — a freshly-taught skill's name has no
language-model prior behind it, so faster-whisper reaches for the nearest
real word it knows (*"flip a coin"* → *"flip a corn"*; *"flip three coins"* →
*"flip three chords"*). Today, once NLU confidence lands below the "unknown"
cutoff (or, since the confidence-gate fix, below `meta_action_threshold` for
teach/edit_skill/revert_skill), JARVIS just gives up: *"I'm afraid I didn't
catch that, sir."* This milestone gives it one more move first: ask whether
the raw transcript is a plausible mishearing of something it actually knows
how to do — the same kind of correction a person would make silently
("he obviously means *coin*, not *corn*") — before admitting defeat.

- **Reasoner-only, never Claude.** This is exactly the "Reasoning / persona
  phrasing" row of the degradation-chain table doing a new job, not a new
  network dependency: `core/reasoner.py`'s existing Ollama client (optional,
  capability-probed, already used for persona style-rewriting). Consistent
  with "Claude is a teacher invoked rarely, never a runtime dependency" —
  when no Ollama is running, this whole step is skipped and behavior is
  identical to today.
- **New orchestrator step**, e.g. `_reason_about_unclear(text) -> str | None`,
  called exactly where `persona.line("unknown")` currently fires unconditionally
  — in the `UNKNOWN` branch of `handle()`, and in the low-confidence
  meta-action branch added for the STT-garbling fix. Only invoked when
  `self.reasoner.available`.
- **Prompt**: the raw heard text, plus the vocabulary to compare it against —
  every registered skill's name and a few of its `MANIFEST.examples`
  (`registry.manifests()`), and the seed intents' patterns from
  `intent_meta()`. Ask for its single best guess at the intended command, or
  an explicit "none" when nothing is plausible — this is a small, fast,
  local call (no code generation, no `max_tokens` pressure), not the skill
  factory's `build()` path.
- **Never act on a guess directly.** Compounding STT error + NLU uncertainty
  + an LLM's own guess into an unconfirmed action is worse than just asking
  again. The corrected phrase is either (a) confirmed by voice first — *"Did
  you mean 'flip a coin', sir?"* — before being re-run through the real NLU
  classifier and dispatched normally, or (b) if declined or nothing plausible
  came back, JARVIS falls back to today's plain "didn't catch that" line.
  One guess, one confirmation, per turn — it doesn't loop trying to guess
  again after a "no."
- Reuses the confirm-by-voice pattern already built for M2 (`_ask_yes_no`) —
  no new dialog machinery needed, just a new caller of it.

**Verification**: unit tests with an injected fake reasoner (deterministic
canned guess) covering — reasoner absent → behavior unchanged from today;
reasoner present with a plausible correction → confirms → the *corrected*
text is what gets classified and dispatched (not the reasoner's raw guess
verbatim); user declines the "did you mean" confirm → falls back to the
plain unknown line, nothing dispatched; reasoner returns "none" → same
fallback, no confirm asked. Manual: with Ollama running, say a command whose
skill name is likely to be misheard, verify the "did you mean" confirm fires
and a "yes" correctly dispatches; without Ollama running, verify the exact
same misheard command falls back to the current plain response.

### Milestone 5 — knowledge base (RAG)
- `knowledge/{store,ingest}.py`; recall intent → retrieve → compose (Ollama) or
  verbatim-snippet fallback; no Claude.
- Ingestion: startup + interval scan of `knowledge.docs_dir`
  (`.txt`/`.md`/`.pdf`), plus a `remember` builtin for spoken facts; `note.py`
  mirrors dictated notes into the docs dir.

### Milestone 6 — polish
- Voice barge-in during TTS: AEC on the all-in-one path and on the edge, so
  `speaking:on` can stop a reply without the button.
- Optionally, switch the all-in-one mic path from `Microphone.record_utterance`
  to the M3 segmenter.
- systemd user services for the all-in-one mode, `serve` and `edge`. The
  `jarvis-whisper` / `jarvis-voder` units already ship in M3.
- Packaging (`pyproject.toml`, including a light edge-only install), tests,
  README/CLAUDE.md rewrite.
- **Skill factory repair loop.** Today `factory/jobs.py` runs one pass of
  build → validate → sandbox tests → dry-run, and the first failure ends the
  learning job. Instead, run it as a loop: when a stage fails, send the failure
  back to Claude and ask it to fix the code, then re-run the whole gate on the
  new code.
  - **Continue the conversation.** A repair is a follow-up turn in the same
    `messages` list, not a fresh prompt. It carries Claude's previous reply,
    then a user turn that names the failed stage and includes its output. The
    output is the `ValidationError` message, or the sandbox's
    stdout/stderr/exit code for a failed test run or dry-run. It is truncated
    to a fixed size and sent as data. The turn asks for both code blocks again,
    in the same format.
  - **Bounded, not `while True`.** At most `factory.max_repair_attempts`
    repairs (default 3, in `config.toml`), so a stuck skill can't burn API
    spend forever. When the cap is reached, the job fails as it does today and
    the last failure is logged.
  - **What loops and what doesn't.** Code failures loop: a malformed reply
    (`BuildError` from a missing code block), a validation rejection, failing
    generated tests, and a failing dry-run. Everything else ends the job at
    once: no API key, a network or API error, the user declining a
    permission, or cancellation/shutdown. The permission set is fixed for the
    whole job. A repair can use a different approach to avoid a banned
    import, but it can never gain a permission the user didn't grant.
  - **Every attempt passes the full gate.** A repaired module is validated
    and sandboxed from scratch like the first one. Nothing from a failed
    attempt is staged. It still all runs in the background learning job
    (M2.5), and the only thing the user hears is the final "learned" or
    "couldn't learn" announcement.

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
- **M3**: tests per the milestone's own Verification paragraph above
  (segmenter, protocol/intake bounds, addressing modes and mode commands, hold
  window, loopback round trips with fake whisper/voder, auth, whisper service,
  edge imports). Manual: a GPU brain with both services and a Pi edge over the
  internet. Check "Jarvis, …" is answered aloud, un-addressed talk yields
  `heard` only, "pause/continue input" work by voice, PushToTalk keeps the mic
  off, and whisper-down says "Cannot hear you".
- **M4**: fake-reasoner unit tests per the milestone's own Verification
  paragraph above; manual, with and without Ollama running, on a real
  misheard command.
- **M5**: ingest a sample doc, ask *"Jarvis, what does X say about Y"*, verify a
  grounded answer with a source and **no network call to Claude** (assert via
  logs / a blocked key).
- **M6 repair loop**: fake `generate` unit tests, with no network:
  - a first attempt that fails its sandbox tests and then passes is accepted,
    and the repair turn carried the failure output;
  - a skill that keeps failing stops after exactly `max_repair_attempts`
    repairs with `accepted=False`;
  - a declined permission or an API error triggers no repair call;
  - a repair that adds a banned import is rejected by validation.
- Regression: `python check_setup.py` stays green; `python -m jarvis --selftest`
  loads the current NLU model, lists registered skills, and exits 0.
