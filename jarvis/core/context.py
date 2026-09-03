"""``Context`` — the only thing a skill's ``run()`` is handed.

Keeps skills decoupled from the audio/core stack. M1 exposes ``say`` (speak a
line immediately, e.g. progress chatter), ``data_dir`` (per-skill scratch
directory, created on request), ``config``, and an optional ``llm`` (the Ollama
reasoner, or ``None``). ``schedule``/``http`` land with the M2 sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from jarvis.config import Config
    from jarvis.core.reasoner import Reasoner


@dataclass
class Context:
    say: Callable[[str], None]
    config: "Config"
    _data_dir: Path
    llm: "Reasoner | None" = None

    @property
    def data_dir(self) -> Path:
        """Per-skill scratch directory, created lazily on first access."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir
