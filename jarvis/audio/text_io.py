"""Text-mode stand-ins for the audio layer — `python -m jarvis text`.

Feeds the orchestrator lines of text instead of real wake-word/mic/STT, and
prints instead of speaking, so the *entire* voice loop (standby, follow-up
window, `teach`/`edit_skill`/`revert_skill` dialogs, skill dispatch) can be
driven and inspected without a mic or audio device — for scripted
conversation tests and for developing without hardware.

`TextIO` plays both the `mic=` and `stt=` role: the orchestrator only ever
calls each through its own attribute, so one object can satisfy both without
the two needing to coordinate through anything but `self`.
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Callable, Iterable

import numpy as np


class TextWake:
    """Every wake session "hears" the wake word immediately — there is no
    audio to detect it in. `run()`'s outer loop re-wakes right after standby,
    same as it would on a real repeated wake word."""

    def reset(self) -> None:
        pass

    def triggered(self, frame) -> bool:
        return True


class TextIO:
    """`mic=` + `stt=` combined. Each `record_utterance()` pops the next
    scripted line (or, interactively, reads one from stdin); `transcribe()`
    hands it back unchanged. An empty line reads as an utterance the same
    way real silence does. Exhausting the input (EOF, or the script running
    out) calls `on_exhausted` once — wired to `orchestrator.stop` — so a
    piped/scripted run always terminates instead of looping forever."""

    def __init__(
        self,
        lines: Iterable[str] | None = None,
        *,
        prompt: str = "you> ",
        on_exhausted: Callable[[], None] | None = None,
    ) -> None:
        self._lines: deque[str] | None = deque(lines) if lines is not None else None
        self._pending: str | None = None
        self.prompt = prompt
        self.on_exhausted = on_exhausted
        self.started = False

    # -- as "mic" -------------------------------------------------------------

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def drain(self) -> None:
        pass

    def read(self, timeout: float | None = None):
        # only consumed by `_await_wake`'s loop between triggers; TextWake
        # always fires on the first read, so the content never matters.
        return np.zeros(1280, dtype=np.int16)

    def record_utterance(self, window, silence_s, grace, cancel_flag=None, min_speech=3):
        line = self._next_line()
        if line is None:
            return np.zeros(0, dtype=np.float32)  # reads as silence
        self._pending = line
        return np.ones(1600, dtype=np.float32) * 0.1  # non-empty -> "speech"

    # -- as "stt" ---------------------------------------------------------------

    def transcribe(self, audio, cancel=None) -> str:
        text, self._pending = self._pending, None
        return text or ""

    # -- shared -------------------------------------------------------------

    def _next_line(self) -> str | None:
        # `sys.stdin.readline()` (not `input()`) so a piped/scripted run and
        # an interactive one both get their line echoed by us, consistently —
        # a non-tty stdin gives `input()` no terminal echo of its own, which
        # otherwise makes the transcript unreadable.
        if self._lines is not None:
            if not self._lines:
                self._exhausted()
                return None
            line = self._lines.popleft()
        else:
            raw = sys.stdin.readline()
            if raw == "":  # EOF
                self._exhausted()
                return None
            line = raw.rstrip("\n")
        print(f"{self.prompt}{line}", flush=True)
        return line

    def _exhausted(self) -> None:
        if self.on_exhausted is not None:
            self.on_exhausted()


class TextTTS:
    """Print-only TTS — never touches real audio hardware, regardless of
    whether a Piper voice is installed. Matches the existing `Jarvis: <text>`
    convention `Speaker` already falls back to when no voice is available."""

    def say(self, text: str) -> None:
        print(f"Jarvis: {text}", flush=True)
