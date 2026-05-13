# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install dependencies:
```bash
pip install openai wolframalpha ecapture wikipedia pyttsx3 nltk python-dotenv SpeechRecognition openpyxl pyaudio tensorflow
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

Create a `.env` file in the project root:
```
language="en"
api_key="YOUR_OPENAI_API_KEY"
```

## Running

```bash
python main.py           # Run JARVIS
python libs/training.py  # Train the NLP model manually
python libs/brain.py     # Test intent classification interactively
```

On first run (or when `JARVIS_model.keras` is missing), the model is trained automatically. The model is also auto-retrained when `intents.json` is modified (watched by a background thread) or when it's older than 7 days.

## Architecture

JARVIS is a voice assistant with an intent-classification NLP pipeline:

1. **`main.py`** — Entry point. Initializes OpenAI client and command handler, then enters the main loop.

2. **`libs/command_helper.py`** — Core loop. Captures microphone audio via `SpeechRecognition`, classifies it through `brain.predict_class()`, and dispatches to the appropriate handler. Manages a `standby` state (JARVIS must be "woken up" before it responds to commands). A background thread watches `intents.json` for changes and triggers retraining.

3. **`libs/brain.py`** — Inference layer. Loads the trained Keras model and pickled vocabulary (`words.pkl`, `classes.pkl`) at startup. `predict_class()` does bag-of-words encoding and runs the neural net; `get_response()` maps the top intent tag to a random response from `intents.json`.

4. **`libs/training.py`** — Trains a 3-layer dense neural network (128→64→softmax) on the intent patterns in `intents.json`. Outputs `JARVIS_model.keras`, `words.pkl`, and `classes.pkl` to the project root.

5. **`libs/voice.py`** — Text-to-speech via `pyttsx3`.

6. **`libs/openai_helper.py`** — Wraps the OpenAI completion API for fallback free-form Q&A (`ask_chat_gpt` action).

7. **`actions/`** — Each file is a plugin module with a `run(query)` function. The `action` field in an intent maps directly to the module name (e.g., `"action": "search"` → `actions/search.py`). Actions are loaded dynamically via `importlib.import_module`.

## Adding a New Intent / Action

1. Add an entry to `intents.json` with a `tag`, `patterns`, `responses`, and `action` field.
2. If the action is new, create `actions/<action_name>.py` with a `run(userIntent)` function.
3. The model will retrain automatically on the next run (or trigger manually: `python libs/training.py`).

## Generated Files (not committed)

- `JARVIS_model.keras` — trained Keras model
- `words.pkl` / `classes.pkl` — vocabulary and class pickles from training
