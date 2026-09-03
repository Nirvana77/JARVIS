"""JARVIS — local voice assistant (2026 rebuild).

The package is the rebuilt architecture described in
``PRD/jarvis-2026-rebuild.md``. Milestone 1 is the local voice loop: wake word ->
local STT -> embedding + sklearn NLU -> skill dispatch -> persona-phrased Piper
TTS. The legacy flat scripts (``main.py``, ``libs/``, ``actions/``) still work
and are untouched; this package has its own entry point, ``python -m jarvis``.
"""

__version__ = "0.1.0"
