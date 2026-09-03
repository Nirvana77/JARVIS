"""Text-to-speech via Piper (ONNX, CPU).

Replaces ``libs/voice.py`` (pyttsx3). The voice model
(``<voice>.onnx`` + ``<voice>.onnx.json``) lives under ``<data>/models/piper/``
and is fetched by ``python -m jarvis models pull``. If the model is missing or
playback fails (headless box), ``say`` degrades to printing ``Jarvis: <text>`` —
the same visible behaviour the old code had.
"""

from __future__ import annotations

import logging
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Speaker:
    def __init__(
        self,
        voice: str = "en_GB-alan-medium",
        model_dir: str | Path = "data/models/piper",
        enabled: bool = True,
    ) -> None:
        self.voice = voice
        self.model_dir = Path(model_dir)
        self.enabled = enabled
        self._voice = None
        self._sample_rate = 22050
        self._load_failed = False

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.voice}.onnx"

    def available(self) -> bool:
        return self.model_path.is_file() and self.model_path.with_suffix(".onnx.json").is_file()

    def load(self) -> bool:
        if self._voice is not None:
            return True
        if self._load_failed or not self.available():
            return False
        try:
            from piper import PiperVoice

            self._voice = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.model_path.with_suffix(".onnx.json")),
            )
            self._sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("piper voice %s failed to load: %s", self.voice, exc)
            self._load_failed = True
            return False

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (int16 mono pcm, sample_rate). Empty array if unavailable."""
        if not self.load():
            return np.zeros(0, dtype=np.int16), self._sample_rate
        chunks = [c.audio_int16_array for c in self._voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, dtype=np.int16), self._sample_rate
        return np.concatenate(chunks), self._sample_rate

    def to_wav_bytes(self, text: str) -> bytes:
        pcm, sr = self.synthesize(text)
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        print(f"Jarvis: {text}")  # keep the legacy visible transcript
        if not self.enabled:
            return
        pcm, sr = self.synthesize(text)
        if len(pcm) == 0:
            return  # already printed; nothing to play
        try:
            import sounddevice as sd

            sd.play(pcm, sr)
            sd.wait()
        except Exception as exc:  # noqa: BLE001 - headless / no output device
            log.warning("audio playback failed: %s", exc)
