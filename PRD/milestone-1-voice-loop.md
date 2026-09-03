# Milestone 1 — local voice loop: outcome

**Date:** 2026-09-03
**Branch:** `milestone-1-voice-loop` → merged to `develop`
**Result:** ✅ The rebuilt `jarvis/` package runs the full local loop — wake word →
faster-whisper STT → fastembed + sklearn NLU → skill dispatch → persona-phrased
Piper TTS — with no learning and no RAG (M2/M3). `python -m jarvis` is the entry
point; the legacy `main.py` / `libs/` / `actions/` are untouched.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Config | `jarvis/config.py`, `config.toml` | `tomllib` + `.env` (secrets only); `JARVIS_PERSONA` / `language` env overrides kept |
| Skill contract | `jarvis/skills/contract.py` | `SkillManifest` + `Skill` Protocol per the PRD; permission vocabulary declared, **not enforced** (M2 sandbox) |
| NLU | `jarvis/nlu/{corpus,slots,train,classifier}.py` | see below |
| Persona | `jarvis/core/persona.py`, `personas/jarvis`, `personas/plain` | see below |
| Reasoner | `jarvis/core/reasoner.py` | Ollama HTTP, capability-probed; `available=False` ⇒ plain phrasing |
| Audio | `jarvis/audio/{wake,capture,stt,tts}.py` | see below |
| Orchestrator | `jarvis/core/orchestrator.py` | single asyncio loop; blocking work via `asyncio.to_thread`; injected components; after a command it keeps listening `capture.follow_up_s` (default 10s) for a follow-up with no wake word, then speaks the standby line. **Press Enter** to cancel the current listen / transcription (`capture.allow_interrupt`, TTY only) — the turn is dropped and JARVIS keeps listening; an abandoned whisper call is left to finish so the model isn't hit concurrently |
| Skills | `jarvis/skills/registry.py`, `jarvis/skills/builtin/{search,open_app,play,note}.py` | ported from `actions/*`; each returns the line to speak |
| CLI | `jarvis/__main__.py`, `jarvis/app.py` | `run` \| `--selftest` \| `models pull` \| `nlu rebuild` |
| Tests | `tests/test_{nlu,persona,audio,orchestrator,skills}.py` | 49 tests, `python -m pytest` |

## NLU

- Corpus = `intents.json` seed patterns (with `{intent}` placeholders expanded
  against a filler list) **+** every registered skill's `MANIFEST.examples`,
  cached in `data/nlu/corpus.sqlite`. 336 examples, 12 intent labels.
- Head = `fastembed` `sentence-transformers/all-MiniLM-L6-v2` embeddings →
  `sklearn.linear_model.LogisticRegression(C=10)`. Held-out accuracy ~0.95.
- `unknown` fallback is **two gates**: max class probability `< nlu.threshold`
  (0.35) **or** cosine similarity to the nearest training phrase `<
  nlu.similarity_floor` (0.30). The similarity floor is necessary — the LR head
  is confidently wrong on out-of-domain junk ("blorp gnnn wibble frotz" → `note`
  at 0.51 without it). The training embedding matrix is saved alongside the head
  for this.
- Versioned artifacts in `data/models/nlu/v<N>/{head.joblib, labels.json,
  meta.json, train_embeddings.npy}`; last 3 kept. In-process training; the
  worker-process retrain + idle merge-gate is M2 (the seam — `_staged` +
  `_merge_gate()` — is in `orchestrator.py`, a no-op).
- `intents.json` changes: `open` → `open_app`; added `play`, `note`; the
  `goodbye`/`shutdown` pattern overlap the PRD flagged is split; **no separate
  `sleep` intent** — "go to sleep" lives in `goodbye` (action `exit`).

## Persona

- `Persona.load(name)` resolves `personas/<name>/`, raising `PersonaNotFound`
  (listing what's available) on a miss. `persona.toml` (display name, style
  description + rules, `address_term`, Piper `voice`) + `responses.toml`
  (fixed-event lines, spoken verbatim via `line()`).
- **All** `personas/jarvis/style/*.txt` are parsed, not one. The transcript
  parser keeps only lines J.A.R.V.I.S. himself speaks — `Spoken by` sections,
  vocative-"sir" lines, and the JARVIS turns of two-party `Dialogue:` blocks —
  dropping headers, `―…[src]` footers, `Spoken about` sections, and non-JARVIS
  turns. **67 in-voice lines pooled across all five films.**
- `phrase(text)` rewrites a dynamic line via the reasoner (system prompt =
  style + rules + k nearest style lines by embedding) when Ollama is up;
  otherwise returns the text unchanged. `plain` persona has no `style/`.

## Audio

- **wake.py** — openwakeword 0.4.0, bundled `hey_jarvis_v0.1.onnx` (+ shared
  melspec/embedding graphs). No download, fully offline. Wake phrase: **"hey
  jarvis"**. 0.4.0 pin matters — 0.5+ need `tflite-runtime` (no PyPI wheel).
- **capture.py** — `sounddevice` int16 frame stream; `record_utterance()` gated
  by an RMS energy VAD with an adaptive noise floor. silero VAD / barge-in are
  M4.
- **stt.py** — faster-whisper int8 CPU, lazy load, downloads to
  `data/models/whisper/`. Default **`small.en`** (English-only beats
  multilingual at the same size; `config.toml [stt].model`). Audio is
  peak-normalised; `condition_on_previous_text=False`, `vad_filter` with
  `speech_pad_ms=400`, and `hotwords` biased to the command vocabulary.
  `Transcriber.last_avg_logprob` (~0 confident, < -1 shaky) shows in the
  orchestrator's `heard:` debug line.
- **tts.py** — Piper ONNX; **degrades to `print("Jarvis: <text>")`** when the
  voice model is absent or playback fails (headless-safe, matches
  `libs/voice.py`).

## Verification

Automated (ran here):
```
python -m pytest                     # 49 passed
python -m jarvis --selftest          # PASS — persona jarvis, NLU v1, 4 skills, sanity probes ok
JARVIS_PERSONA=plain python -m jarvis --selftest   # persona plain, 0 style lines
python -m jarvis nlu rebuild         # v2..v4, 3-version retention confirmed
python check_setup.py                # unchanged; only the pre-existing ANTHROPIC_WORKSPACE_ID row fails
```

Not verifiable here (no audio device / large downloads); **for the user to run**:
```
python -m jarvis models pull         # faster-whisper base + Piper en_GB-alan-medium + warm fastembed
python -m jarvis                     # say "hey jarvis" -> "search black holes"
                                     #     "hey jarvis" -> "play some lofi"
                                     #     "hey jarvis" -> "go to sleep"
JARVIS_PERSONA=plain python -m jarvis # fixed-response wording changes
```
- If `en_GB-alan-medium` 404s from `rhasspy/piper-voices`, set another British
  male medium voice in `config.toml [tts].voice`.
- The Ollama style-rewrite path is only exercised when a local Ollama with the
  configured model is running (`[reasoner]` in `config.toml`).

## Dependencies

`pytest` added to the venv (dev/test only). All runtime deps were already
installed in Phase 0. No `pyproject.toml` yet (M4).

## Deferred to later milestones (unchanged from the PRD)

- M2: skill factory (Claude), `SubprocessSandbox`, `teach`/`edit_skill`/
  `revert_skill`, worker-process retrain + idle merge-gate, permission
  enforcement.
- M3: knowledge base (RAG), `remember` skill, doc-folder ingestion.
- M4: barge-in, silero VAD, systemd unit, `pyproject.toml`, README/CLAUDE.md
  rewrite, `main.py` → thin shim, delete the legacy TF path.
