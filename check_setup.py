"""Sanity check for a JARVIS environment.

    python check_setup.py

**Required** means the 2026 rebuild (`python -m jarvis`, `requirements.txt`):
if a row there is ✗, the assistant will not run. Everything else — the legacy
`main.py` / `libs/` stack, the pieces later milestones will need, and the
optional remote-edge services — is reported as ✓/– and never fails the run,
because a rebuild-only install is the normal case and should exit 0.
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
section("Dependencies — the 2026 rebuild (requirements.txt)")
# Everything `jarvis/` actually imports. A ✗ here is a real failure.
REQUIRED = [
    ("numpy", "numpy"),
    ("sounddevice", "sounddevice (audio in and out)"),
    ("openwakeword", "openwakeword (wake word)"),
    ("faster_whisper", "faster-whisper (speech to text)"),
    ("piper", "piper-tts (speech out)"),
    ("fastembed", "fastembed (NLU embeddings)"),
    ("sklearn", "scikit-learn (NLU classifier head)"),
    ("joblib", "joblib (persists that head)"),
    ("anthropic", "anthropic (the skill factory)"),
    ("requests", "requests (reasoner + search skill)"),
    ("httpx", "httpx (the whisper/voder service clients)"),
    ("websockets", "websockets (the remote-edge link)"),
    ("dotenv", "python-dotenv (reads .env)"),
    ("huggingface_hub", "huggingface-hub (models pull)"),
]
for mod, name in REQUIRED:
    try:
        m = importlib.import_module(mod)
        require(True, name, ok_detail=getattr(m, "__version__", ""))
    except Exception as e:
        require(False, name, bad_detail=f"{e!r} — pip install -r requirements.txt")


# Pulled in by the packages above rather than required directly, but their
# versions are worth seeing: ctranslate2 decides which GPUs faster-whisper can
# use at all, and onnxruntime which execution providers exist.
for mod, name in [
    ("ctranslate2", "ctranslate2 (faster-whisper backend)"),
    ("onnxruntime", "onnxruntime (fastembed / openwakeword / piper backend)"),
]:
    try:
        m = importlib.import_module(mod)
        line(OK, name, getattr(m, "__version__", ""))
    except Exception as e:
        line(SKIP, name, f"{e!r} — installed with the packages above")


# --------------------------------------------------------------------------
# Approved in the Phase 0 spike and not imported yet: M5 is the knowledge base.
section("Later milestones (not needed yet)")
for mod, name in [
    ("sqlite_vec", "sqlite-vec (M5 knowledge vector index)"),
    ("pypdf", "pypdf (M5 .pdf loader)"),
]:
    try:
        m = importlib.import_module(mod)
        line(OK, name, getattr(m, "__version__", ""))
    except Exception:
        line(SKIP, name, "not installed — M5 will need it")


# --------------------------------------------------------------------------
# The original main.py / libs/ path. Kept working, but nothing the rebuild
# needs, and `requirements.txt` deliberately does not install it.
section("Legacy stack (main.py / libs/ — optional)")
for mod, name in [
    ("speech_recognition", "SpeechRecognition"),
    ("wikipedia", "wikipedia"),
    ("pyttsx3", "pyttsx3"),
    ("nltk", "nltk"),
    ("openpyxl", "openpyxl"),
    ("pyaudio", "PyAudio (legacy mic capture)"),
    ("tensorflow", "TensorFlow (legacy intent model; no 3.14 wheels)"),
]:
    try:
        m = importlib.import_module(mod)
        line(OK, name, getattr(m, "__version__", ""))
    except Exception:
        line(SKIP, name, "not installed (legacy path only)")


# --------------------------------------------------------------------------
section("NLTK data (legacy path only)")
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
except LookupError:
    # NLTK's own message opens with a 70-character banner of asterisks, which
    # is not what anyone needs to read here.
    line(SKIP, "nltk data", "corpora not downloaded — " + DOWNLOAD_HINT)
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
section("Intent model (legacy brain.py / training.py)")
line(
    SKIP,
    "Keras model",
    "deferred — needs TensorFlow (no Python 3.14 wheels), and the checked-in "
    "JARVIS_model.keras predates Keras 3 and must be retrained anyway",
)


# --------------------------------------------------------------------------
# The two rows above that depend on something outside pip: a system library,
# and an extension that has to load into sqlite. Neither is fatal — a brain
# behind a remote edge needs no sound card of its own.
section("Functional probes")
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
    line(OK, "sqlite-vec extension loads", _vec_version)
except Exception as e:
    line(SKIP, "sqlite-vec extension loads", f"{e!r} — M5 will need it")


# --------------------------------------------------------------------------
# M3 — the remote edge (PRD/milestone-3-remote-edge.md). Opt-in: an all-in-one
# install needs none of it, so nothing here is required. The two services are
# probed over their health endpoints, which is also how you tell "not running"
# from "running but cold".
section("Remote edge (M3, optional)")
try:
    import websockets as _websockets

    line(OK, "websockets (brain + edge link)", getattr(_websockets, "__version__", ""))
except Exception:
    line(SKIP, "websockets (brain + edge link)", "not installed — `pip install websockets`")

try:
    from jarvis.config import load_config as _load_config

    _config = _load_config()
except Exception as e:  # pragma: no cover
    _config = None
    line(SKIP, "config.toml", repr(e))

if _config is not None:
    for _label, _url, _describe in (
        (
            "jarvis-whisper",
            _config.whisper.url,
            lambda h: f"{h.get('model')} on {h.get('device')}"
            + ("" if h.get("warm") else " (cold)"),
        ),
        (
            "jarvis-voder",
            _config.voder.url,
            lambda h: f"{h.get('voice')} at {h.get('sample_rate')} Hz",
        ),
    ):
        if not _url or _url.strip().lower() == "off":
            line(SKIP, _label, "off in config.toml")
            continue
        try:
            import httpx as _httpx

            _health = _httpx.get(f"{_url.rstrip('/')}/healthz", timeout=2).json()
            line(OK if _health.get("ok") else BAD, _label, _describe(_health))
        except Exception:
            line(SKIP, _label, f"not running at {_url}")

    # The tokens are secrets: their presence is reported, never their value.
    _edge_token = _config.edge_token
    line(
        OK if _edge_token else SKIP,
        "JARVIS_EDGE_TOKEN (this edge)",
        f"set ({len(_edge_token)} chars)" if _edge_token else "unset — only the edge needs it",
    )
    _tokens = _config.edge_tokens
    line(
        OK if _tokens else SKIP,
        "JARVIS_EDGE_TOKENS (the brain's devices)",
        f"{len(_tokens)} device(s): {', '.join(sorted(_tokens))}"
        if _tokens
        else "unset — only the brain needs it",
    )
    # Only worth saying when the remote path is actually set up: an all-in-one
    # install has a [server] section it never uses.
    if _tokens and _config.server.host not in ("127.0.0.1", "::1", "localhost", ""):
        _tls = _config.server.tls_enabled or _config.server.allow_insecure
        line(
            OK if _config.server.tls_enabled else (BAD if not _tls else SKIP),
            "brain TLS",
            "cert + key set"
            if _config.server.tls_enabled
            else ("allow_insecure = true (LAN/dev only!)" if _tls else "MISSING — `serve` will refuse to start"),
        )


# --------------------------------------------------------------------------
print()
if failures:
    print(f"{BAD}  {failures} required check(s) failed.")
    sys.exit(1)
print(f"{OK}  All required checks passed.")
