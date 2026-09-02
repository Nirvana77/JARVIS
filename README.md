# J.A.R.V.I.S.

Just A Rather Very Intelligent System (J.A.R.V.I.S.) from Iron Man — a voice
assistant with an intent-classification NLP pipeline and a Claude-backed
free-form Q&A fallback.

## Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install anthropic wikipedia pyttsx3 nltk python-dotenv SpeechRecognition openpyxl pyaudio tensorflow
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4')]"
```

### System packages

`pyaudio` builds against PortAudio, so install the dev headers first:

```bash
sudo dnf install portaudio-devel      # Fedora
# sudo apt install portaudio19-dev    # Debian/Ubuntu
```

### Python version note

`tensorflow` (used by the intent classifier in `libs/brain.py` /
`libs/training.py`) has **no wheels for Python 3.14 yet**. Use a **Python 3.12**
virtualenv if you need the intent model. The Claude Q&A path and everything
else run fine on 3.14.

## Configure

Create a `.env` file in the project root:

```
language="en"
ANTHROPIC_API_KEY="sk-ant-..."
# ANTHROPIC_WORKSPACE_ID="wrkspc_..."   # required only for identity-linked API keys
# ANTHROPIC_MODEL="claude-sonnet-5"     # optional; default is claude-opus-5
```

If your key is **identity-linked** (calls fail with
`anthropic-workspace-id is required`), either add `ANTHROPIC_WORKSPACE_ID`
(find it in the Anthropic Console workspace URL) or issue a standard
workspace-scoped API key.

## Verify

```bash
python check_setup.py
```

Prints a ✓/✗/– report for dependencies, NLTK data, the API key (masked), and
whether the configured Claude model resolves.

## Run

```bash
python main.py           # Run JARVIS
python libs/training.py  # Train the NLP model manually
python libs/brain.py     # Type sentences to see raw intent classification
```

Always run from the repo root — `intents.json`, `JARVIS_model.keras`,
`words.pkl`, and `classes.pkl` are resolved relative to the working directory.
