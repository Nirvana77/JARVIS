"""Speech-to-text via faster-whisper (CTranslate2, int8 CPU).

Replaces the Google Web Speech call in ``libs/command_helper.takeCommand``. The
model is downloaded to ``<data>/models/whisper/`` on first use (or eagerly by
``python -m jarvis models pull``) and then runs fully offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | Path | None = None,
        language: str | None = "en",
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.download_root = str(download_root) if download_root else None
        self.language = language
        self._model = None

    def load(self) -> "Transcriber":
        if self._model is not None:
            return self
        from faster_whisper import WhisperModel

        if self.download_root:
            Path(self.download_root).mkdir(parents=True, exist_ok=True)
        log.info("loading faster-whisper %s (%s/%s)", self.model, self.device, self.compute_type)
        self._model = WhisperModel(
            self.model,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
        )
        return self

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""
        if self._model is None:
            self.load()
        audio = np.asarray(audio, dtype=np.float32)
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
