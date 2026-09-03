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

# words the decoder should lean toward — the command vocabulary + the name
_HOTWORDS = (
    "Jarvis. search, look up, play, open, note, remember, "
    "go to sleep, standby, shut down."
)


def _normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale a quiet clip up so whisper isn't straining to hear it."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if 0.0 < peak < target_peak:
        audio = audio * (target_peak / peak)
    return audio


class Transcriber:
    def __init__(
        self,
        model: str = "medium.en",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | Path | None = None,
        language: str | None = "en",
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.download_root = str(download_root) if download_root else None
        # an ".en" model is single-language; don't also pass language=
        self.language = None if model.endswith(".en") else language
        self._model = None
        #: quality of the last transcription (avg token log-prob; ~0 = confident,
        #: < -1 = shaky) — surfaced in the orchestrator's debug line
        self.last_avg_logprob: float | None = None

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
            self.last_avg_logprob = None
            return ""
        if self._model is None:
            self.load()
        audio = _normalize(np.asarray(audio, dtype=np.float32))

        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            # each utterance stands alone — no cross-command context/hallucination
            condition_on_previous_text=False,
            # trim silence but pad the kept speech so edge words survive
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 400},
            # bias the decoder toward the command words
            hotwords=_HOTWORDS,
        )

        segments = list(segments)
        if segments:
            self.last_avg_logprob = round(
                float(np.mean([s.avg_logprob for s in segments])), 2
            )
        else:
            self.last_avg_logprob = None
        return " ".join(s.text.strip() for s in segments).strip()
