"""JARVIS — local voice assistant.

The architecture is described in ``PRD/jarvis-2026-rebuild.md``: wake word (or
a remote edge) -> local STT -> embedding + sklearn NLU -> skill dispatch ->
persona-phrased Piper TTS, with a Claude-backed factory that writes new skills.
Entry point: ``python -m jarvis`` (``python main.py`` is a shim for it).
"""

__version__ = "0.1.0"
