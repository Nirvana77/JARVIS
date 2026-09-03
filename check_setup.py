"""Sanity check for a JARVIS environment.

Run after installing dependencies to confirm the pieces are in place:

    python check_setup.py

Checks imports, NLTK data, the Anthropic API key (masked) and that the
configured Claude model resolves, plus the TTS engine. Exits non-zero if any
*required* check fails.
"""

import importlib
import os
import sys

OK = "✓"      # check mark
BAD = "✗"     # cross
SKIP = "–"    # en dash

failures = 0


def line(symbol, label, detail=""):
    print(f"  {symbol}  {label}" + (f"  —  {detail}" if detail else ""))


def require(cond, label, ok_detail="", bad_detail=""):
    global failures
    if cond:
        line(OK, label, ok_detail)
    else:
        failures += 1
        line(BAD, label, bad_detail)
    return cond


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------
section("Environment")
line(OK, "Python", sys.version.split()[0])
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
line(OK if in_venv else SKIP, "Virtualenv", sys.prefix if in_venv else "not in a venv")

try:
    from dotenv import load_dotenv

    load_dotenv()
    line(OK, ".env loaded")
except Exception as e:  # pragma: no cover
    require(False, ".env loaded", bad_detail=repr(e))


# --------------------------------------------------------------------------
section("Dependencies")
REQUIRED = [
    ("anthropic", "anthropic"),
    ("dotenv", "python-dotenv"),
    ("speech_recognition", "SpeechRecognition"),
    ("wikipedia", "wikipedia"),
    ("pyttsx3", "pyttsx3"),
    ("nltk", "nltk"),
    ("numpy", "numpy"),
    ("openpyxl", "openpyxl"),
]
OPTIONAL = [
    ("pyaudio", "PyAudio (mic capture)"),
    ("tensorflow", "TensorFlow (intent model — no Python 3.14 wheels yet)"),
]

for mod, name in REQUIRED:
    try:
        m = importlib.import_module(mod)
        require(True, name, ok_detail=getattr(m, "__version__", ""))
    except Exception as e:
        require(False, name, bad_detail=repr(e))

for mod, name in OPTIONAL:
    try:
        m = importlib.import_module(mod)
        line(OK, name, getattr(m, "__version__", ""))
    except Exception:
        line(SKIP, name, "not installed")


# --------------------------------------------------------------------------
section("NLTK data")
DOWNLOAD_HINT = (
    "run: python -c \"import nltk; "
    "[nltk.download(p) for p in ('punkt','punkt_tab','wordnet','omw-1.4')]\""
)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    # Functional test — this is exactly what brain.py does per query.
    nltk.word_tokenize("waking jarvis up")
    line(OK, "tokenizer (punkt)")
    WordNetLemmatizer().lemmatize("running", "v")
    line(OK, "lemmatizer (wordnet)")
except LookupError as e:
    require(False, "nltk data", bad_detail=str(e).strip().splitlines()[0] + " — " + DOWNLOAD_HINT)
except Exception as e:
    line(SKIP, "nltk data", repr(e))


# --------------------------------------------------------------------------
section("Anthropic")
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("api_key")
model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

placeholder = api_key in (None, "", "YOUR_OPENAI_API_KEY", "YOUR_ANTHROPIC_API_KEY")
if placeholder:
    require(
        False,
        "API key",
        bad_detail="set ANTHROPIC_API_KEY in .env (currently unset or a placeholder)",
    )
else:
    masked = api_key[:8] + "…" + api_key[-4:] if len(api_key) > 14 else "set"
    line(OK, "API key", f"{masked}  (len {len(api_key)})")

    workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
    line(
        OK if workspace_id else SKIP,
        "Workspace id",
        workspace_id[:12] + "…" if workspace_id else "unset (only needed for identity-linked keys)",
    )

    try:
        import anthropic

        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        client = anthropic.Anthropic(api_key=api_key, default_headers=headers)
        info = client.models.retrieve(model)
        require(True, f"Model '{model}' resolves", ok_detail=getattr(info, "display_name", info.id))
    except Exception as e:
        require(False, f"Model '{model}' resolves", bad_detail=repr(e))

if placeholder:
    line(SKIP, f"Model '{model}' resolves", "skipped (no API key)")


# --------------------------------------------------------------------------
section("Text-to-speech")
try:
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices") or []
    line(OK, "pyttsx3 engine", f"{len(voices)} voice(s) available")
except Exception as e:
    line(SKIP, "pyttsx3 engine", f"{e!r} (needs an audio backend; fine on a headless box)")


# --------------------------------------------------------------------------
section("Intent model (brain.py / training.py)")
line(
    SKIP,
    "Keras model",
    "deferred — needs TensorFlow, which has no Python 3.14 wheels; "
    "JARVIS_model.keras also predates Keras 3 and must be retrained",
)


# --------------------------------------------------------------------------
# PRD Phase 0 — dependency spike for the 2026 rebuild (see
# PRD/jarvis-2026-rebuild.md and PRD/phase-0-dependency-spike.md).
# The rebuild drops TensorFlow; this is the replacement local stack. Every
# row below has a CPU wheel for Python 3.14 as of the spike, so a missing
# import here is a real setup failure, not a "no wheel yet" skip.
section("2026 rebuild stack (PRD Phase 0)")
REBUILD_STACK = [
    ("faster_whisper", "faster-whisper (STT)"),
    ("ctranslate2", "ctranslate2 (faster-whisper backend)"),
    ("openwakeword", "openwakeword (wake word)"),
    ("piper", "piper-tts (TTS)"),
    ("onnxruntime", "onnxruntime (fastembed / openwakeword / piper backend)"),
    ("fastembed", "fastembed (NLU + knowledge embeddings)"),
    ("sklearn", "scikit-learn (NLU classifier head)"),
    ("sqlite_vec", "sqlite-vec (knowledge vector index)"),
    ("sounddevice", "sounddevice (audio capture)"),
    ("pypdf", "pypdf (knowledge .pdf loader)"),
]
for mod, name in REBUILD_STACK:
    try:
        m = importlib.import_module(mod)
        require(True, name, ok_detail=getattr(m, "__version__", ""))
    except Exception as e:
        require(False, name, bad_detail=repr(e))

# Functional probes for the two rows with a system-level dependency.
try:
    import sounddevice as _sd

    _n = len(_sd.query_devices())
    line(OK, "PortAudio (sounddevice backend)", f"{_n} device(s)")
except Exception as e:
    line(
        SKIP,
        "PortAudio (sounddevice backend)",
        f"{e!r} — install portaudio-devel / portaudio19-dev (fine on a headless box)",
    )

try:
    import sqlite3 as _sqlite3

    import sqlite_vec as _sqlite_vec

    _db = _sqlite3.connect(":memory:")
    _db.enable_load_extension(True)
    _sqlite_vec.load(_db)
    (_vec_version,) = _db.execute("select vec_version()").fetchone()
    _db.close()
    require(True, "sqlite-vec extension loads", ok_detail=_vec_version)
except Exception as e:
    require(False, "sqlite-vec extension loads", bad_detail=repr(e))


# --------------------------------------------------------------------------
print()
if failures:
    print(f"{BAD}  {failures} required check(s) failed.")
    sys.exit(1)
print(f"{OK}  All required checks passed.")
