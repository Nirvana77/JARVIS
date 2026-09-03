"""Always-on wake-word detection (openwakeword, bundled ``hey_jarvis`` model).

openwakeword 0.4.0 ships ``hey_jarvis_v0.1.onnx`` plus the shared
melspectrogram / embedding ONNX graphs, so this needs no downloads and runs
fully offline. Feed it 80 ms (1280-sample) int16 frames; ``triggered`` is true
once the score crosses ``threshold``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _bundled_model_path(name: str) -> str:
    """Resolve a pretrained model name (or path) to an .onnx file on disk."""
    candidate = Path(name)
    if candidate.is_file():
        return str(candidate)
    import openwakeword

    models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    matches = sorted(models_dir.glob(f"{name}*.onnx"))
    if not matches:
        available = sorted(
            p.stem for p in models_dir.glob("*.onnx")
            if p.stem not in {"melspectrogram", "embedding_model", "silero_vad"}
        )
        raise FileNotFoundError(
            f"wake model {name!r} not found in {models_dir} (have: {available})"
        )
    return str(matches[0])


class WakeWord:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self.model_name = model
        self.threshold = threshold
        self._path = _bundled_model_path(model)
        self._model = None
        self._key: str | None = None

    def load(self) -> "WakeWord":
        if self._model is not None:
            return self
        from openwakeword.model import Model

        self._model = Model(wakeword_model_paths=[self._path])
        self._key = next(iter(self._model.models.keys()))
        log.info("wake word ready: %s (threshold %.2f)", self._key, self.threshold)
        return self

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def score(self, frame: np.ndarray) -> float:
        if self._model is None:
            self.load()
        frame = np.asarray(frame)
        if frame.dtype != np.int16:
            frame = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        return float(self._model.predict(frame).get(self._key, 0.0))

    def triggered(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold
