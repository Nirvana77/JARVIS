# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**The rebuild is underway.** `PRD/jarvis-2026-rebuild.md` is the canonical
spec; `PRD/milestone-N-*.md` are per-milestone plans and
`PRD/milestone-N-*-outcome.md` their outcomes once shipped. The rebuilt
package lives in `jarvis/` (`python -m jarvis ...`) alongside the untouched
legacy `main.py`/`libs/`/`actions/` this file otherwise describes — see the
PRD for the target architecture and the migration mapping between the two.
There **is** now a test suite (`python -m pytest`, `tests/`) covering
`jarvis/` — the "There is no test suite" note further down is about the
legacy code only.

## Implementing PRD work

Follow this order for any PRD/milestone item — don't skip or reorder steps:

1. **Write the tests first**, derived from the PRD's "Verification" section
   and the milestone plan's acceptance criteria — before writing the
   implementation itself. Match the conventions already in `tests/`
   (dependency-injected fakes per component; `tests/conftest.py`'s
   `config`/`embedding_model`/`embedder` fixtures).
2. **Implement** the code the tests describe.
3. **Run the full suite**: `python -m pytest`. If anything fails, **fix the
   code — never the test** — the one exception is a test that turns out to
   be factually wrong about what the PRD/plan actually specified, and even
   then say so explicitly rather than quietly loosening it.
4. Once the suite is green, **dry-run the program** — actually converse with
   it — via `python -m jarvis text` (or `--script FILE` for a saved
   conversation; see `jarvis/audio/text_io.py`). This is a separate check
   from the unit tests passing: it confirms the *real* wiring (NLU, registry,
   persona, the skill factory) behaves the way the PRD describes end-to-end,
   not just that the pieces work in isolation behind fakes.

## Setup

Install dependencies:
```bash
pip install anthropic wolframalpha ecapture wikipedia pyttsx3 nltk python-dotenv SpeechRecognition openpyxl pyaudio tensorflow
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4')]"
```
(`wolframalpha` and `ecapture` are listed in the historical install instructions but are not imported anywhere in the current code. `punkt_tab` is required by NLTK ≥ 3.9's `word_tokenize`.)

**Python version:** `tensorflow` has no wheels for Python 3.14 — `libs/brain.py` and `libs/training.py` (the only TF consumers) will not import there. Use a **Python 3.12** venv if you need the intent classifier; the Claude Q&A path and the rest run on 3.14. The checked-in `JARVIS_model.keras` is a Keras-2 file from 2024 that current Keras cannot load, so it must be retrained regardless (which happens automatically — it is well past the 7-day staleness trigger).

**System package:** `pyaudio` needs PortAudio headers — `sudo dnf install portaudio-devel` (Fedora) / `sudo apt install portaudio19-dev` (Debian).

Create a `.env` file in the project root:
```
language="en"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
HF_TOKEN="hf_..."   # optional — faster / rate-limited Hugging Face model downloads
```
`libs/anthropic_helper.py` reads `ANTHROPIC_API_KEY` (the name the Anthropic SDK picks up on its own), falling back to the legacy `api_key` name if that's all an older `.env` has. Optional: `ANTHROPIC_MODEL` overrides the default model (`claude-opus-5`); `ANTHROPIC_WORKSPACE_ID` is sent as the `anthropic-workspace-id` header and is **required for identity-linked API keys** (otherwise calls 400 with `anthropic-workspace-id is required`). `language` overrides the default in `command_helper` and is passed to Google Web Speech for recognition. `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) is read by `jarvis.config` and exported so `fastembed` / `faster-whisper` / `huggingface_hub` download the NLU, STT, and Piper models authenticated instead of anonymously.

Run `python check_setup.py` to verify the environment: it prints a ✓/✗/– report for imports, NLTK data, the API key (masked), and whether the configured Claude model resolves.

## Running

```bash
python main.py           # Run JARVIS
python libs/training.py  # Train the NLP model manually
python libs/brain.py     # Type sentences to see raw intent classification
```

Always run from the repo root. All file paths are resolved relative to the current working directory: `intents.json`, `JARVIS_model.keras`, `words.pkl`, `classes.pkl`.

There is no test suite for this legacy code path. (`jarvis/` — the rebuild —
has one: `python -m pytest`, see the top of this file and the PRD.)

### Model (re)training triggers

- **Missing model**: if `JARVIS_model.keras` is absent on startup, `train_model()` runs before anything else.
- **Stale model**: on startup, if `getctime('JARVIS_model.keras')` is more than 7 days (604800s) old, it retrains.
- **`intents.json` changed**: a background thread (`my_thread_function` in `command_helper.py`) polls `intents.json`'s mtime every second and calls `train_model()` when it changes.
- **`train` intent**: saying a "train" pattern spawns `training.train_model()` on a thread.

## Network dependencies

Every turn of the main loop needs the internet:
- **Speech-to-text**: `takeCommand()` uses `speech_recognition.recognize_google` (Google Web Speech API).
- **`search` action**: `wikipedia.summary`.
- **`ask_chat_gpt` action**: Anthropic `claude-opus-5` via `messages.create` (`libs/anthropic_helper.py`).

## Architecture

JARVIS is a voice assistant with an intent-classification NLP pipeline.

1. **`main.py`** — Entry point. `llm.init()` → `commands.init()` → `commands.run()`.

2. **`libs/command_helper.py`** — Core loop and dispatcher.
   - `run()` starts the `intents.json` watcher thread, then loops: `takeCommand()` → `brain.predict_class()` → `brain.get_response()` → `runCommand()`.
   - **Standby state machine** (`standby` global, starts `True`):
     - action `start` → wake (`standby = False`); says "already awake" if not in standby.
     - action `exit` → if awake, go to standby; if already in standby, `shutdown()`.
     - action `shutdown` → always `shutdown()` (speaks goodbye, joins watcher thread, `exit()`).
     - While `standby is True`, every other action is ignored — JARVIS must be woken first.
   - Non-standby actions: `none` (just speak the response), `train` (retrain on a thread), `ask_chat_gpt` (two-step: prompts "What is your question?", takes a second voice input, calls Claude), or anything else → dynamic import of `actions/<action>.py` and call `run(userIntent)`.
   - `userIntent` is the **full lowercased query string, including the command verb** (e.g. `"search cats"`). Actions do not strip the verb.

3. **`libs/brain.py`** — Inference. `init()` loads `JARVIS_model.keras` + `words.pkl` + `classes.pkl`. `predict_class()` does bag-of-words encoding, runs the net, keeps intents above `ERROR_THRESHOLD = 0.25` sorted by probability. `get_response()` returns `{response, tag, action}` for the top intent, or an `action: 'none'` "could you rephrase" fallback when nothing clears the threshold.

4. **`libs/training.py`** — `train_model()` tokenizes/lemmatizes `intents.json` patterns, builds bag-of-words vectors, and trains a Sequential net: Dense(128)→ReLU→Dropout(0.5)→Dense(64)→ReLU→Dropout(0.5)→Dense(n_classes)→softmax, `SGD(lr=0.01, momentum=0.9, nesterov=True)`, 200 epochs, batch size 5. Writes `JARVIS_model.keras`, `words.pkl`, `classes.pkl` to the repo root.

5. **`libs/voice.py`** — TTS via `pyttsx3`; `speak()` also prints `Jarvis: <text>`. `init()` selects `voices[0]`.

6. **`libs/anthropic_helper.py`** — `ask_claude(prompt)` wraps `messages.create` (`claude-opus-5` by default, `ANTHROPIC_MODEL` to override; adaptive thinking, `max_tokens=1024`, voice-oriented system prompt). `init()` sets `client` and `model`, reading the key from `ANTHROPIC_API_KEY`/`api_key` and passing `anthropic-workspace-id` when `ANTHROPIC_WORKSPACE_ID` is set.

7. **`actions/`** — Plugin modules, each with `run(query)`. The intent's `action` field is the module name: `importlib.import_module(f'actions.{action}')`. `ImportError`/`AttributeError` degrade to "Sorry, I don't know how to do that yet."

### Intent ↔ action wiring (current state)

`intents.json` and `actions/` are **not fully in sync**:

| intent tag | `action` | module resolved | status |
|---|---|---|---|
| `search` | `search` | `actions/search.py` | works |
| `open` | `open` | `actions/open.py` | **broken** — module is named `openApp.py`, so this raises `ImportError` |
| (none) | — | `actions/play.py` | orphaned — no intent triggers it |
| (none) | — | `actions/write.py` | orphaned — no intent triggers it |
| `train` | `train` | handled inline in `command_helper` | works |
| `greeting`/`goodbye`/`shutdown` | `start`/`exit`/`shutdown` | handled inline | works |
| `ask_chat_gpt` | `ask_chat_gpt` | handled inline | code path exists but no intent uses it |

Also note `goodbye` and `shutdown` share several identical patterns (`"shutdown JARVIS"`, `"stop JARVIS"`, ...), which makes classification between them unreliable.

`actions/write.py` imports `takeCommand` from `libs.command_helper` at module load — keep that import cycle in mind when refactoring `command_helper`.

## Adding a New Intent / Action

1. Add an entry to `intents.json` with `tag`, `patterns`, `responses`, and `action`.
2. If the action is new, create `actions/<action>.py` with a `run(userIntent)` function. **The filename must exactly match the `action` string.**
3. The model retrains automatically (watcher thread on save, or on next run) — or run `python libs/training.py`.

## Generated Files (not committed)

- `JARVIS_model.keras` — trained Keras model
- `words.pkl` / `classes.pkl` — vocabulary and class pickles from training
