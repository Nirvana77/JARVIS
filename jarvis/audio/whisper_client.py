"""The brain's client for the warm ``jarvis-whisper`` service (M3 decision 8).

A GPU model that takes seconds to load should stay warm across brain restarts,
a transcription that wedges should be survivable by killing one process, and the
brain must still start when the service is down. So STT is not in-process on the
remote path: it is an HTTP call to ``services/whisper/serve.py`` on loopback.

``NullTranscriber`` is the ``--whisper off`` / ``[whisper] url = "off"`` case, so
there is exactly one code path: something is always there, and it either
transcribes or says why it can't.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: What the service speaks, and what the wire carries.
SAMPLE_RATE = 16_000


class TranscriptionUnavailable(RuntimeError):
    """The service is down, misconfigured or off. Carries a reason a person can
    act on — it is spoken to the user as "Cannot hear you: ...", not logged and
    swallowed."""


@dataclass(frozen=True)
class Transcript:
    text: str
    #: The model's own confidence in the words it chose, 0..1, or None.
    confidence: float | None
    language: str | None
    #: How long the service took, its own measurement.
    ms: int
    language_probability: float | None = None
    #: Set when a segment was dropped as "whisper guessing at silence".
    dropped: str | None = None


def _to_pcm_bytes(pcm: np.ndarray | bytes) -> bytes:
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return bytes(pcm)
    array = np.asarray(pcm)
    if array.dtype != np.int16:
        # float32 in -1..1 is what the in-process Transcriber takes; accept it
        # here too rather than making every caller remember which is which.
        array = np.clip(array, -1.0, 1.0)
        array = np.where(array < 0, array * 32768.0, array * 32767.0).astype(np.int16)
    return array.tobytes()


class WhisperClient:
    """``POST /transcribe`` with raw 16 kHz s16le mono PCM."""

    def __init__(self, url: str, timeout_s: float = 20.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        #: the orchestrator prints this next to what it heard (`_print_heard`)
        self.last_avg_logprob: float | None = None

    def transcribe(self, pcm: np.ndarray | bytes, cancel=None) -> str:
        """The ``stt`` role: text, or "" — the orchestrator's contract.
        Raises :class:`TranscriptionUnavailable` when the service is down."""
        return self.transcribe_full(pcm).text

    def transcribe_full(self, pcm: np.ndarray | bytes) -> Transcript:
        import httpx

        body = _to_pcm_bytes(pcm)
        if not body:
            return Transcript("", None, None, 0)
        try:
            response = httpx.post(
                f"{self.url}/transcribe",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout_s,
            )
        except httpx.ConnectError as exc:
            raise TranscriptionUnavailable("the transcription service is not running") from exc
        except httpx.TimeoutException as exc:
            raise TranscriptionUnavailable("the transcription service timed out") from exc
        except httpx.HTTPError as exc:  # noqa: BLE001 — reported, not swallowed
            raise TranscriptionUnavailable(f"the transcription service failed: {exc}") from exc
        if response.status_code != 200:
            raise TranscriptionUnavailable(
                f"the transcription service returned {response.status_code}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise TranscriptionUnavailable("the transcription service sent nonsense") from exc

        result = Transcript(
            text=str(data.get("text") or "").strip(),
            confidence=data.get("confidence"),
            language=data.get("language"),
            ms=int(data.get("ms") or 0),
            language_probability=data.get("languageProbability"),
            dropped=data.get("dropped"),
        )
        # `confidence` is exp(mean logprob); the orchestrator's line wants the
        # logprob itself, the same number the in-process Transcriber reports.
        self.last_avg_logprob = (
            float(np.log(result.confidence)) if result.confidence else None
        )
        return result

    def health(self) -> dict:
        """``GET /healthz``. Raises :class:`TranscriptionUnavailable` if it is
        not answering — this is what ``check_setup.py`` prints a ✓/✗ from."""
        import httpx

        try:
            response = httpx.get(f"{self.url}/healthz", timeout=min(5.0, self.timeout_s))
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionUnavailable(
                "the transcription service is not running"
            ) from exc


class NullTranscriber:
    """``[whisper] url = "off"``. Present so the remote path always has an
    ``stt``; it just says what is missing instead of transcribing."""

    reason = "transcription is turned off"
    last_avg_logprob: float | None = None

    def transcribe(self, pcm, cancel=None) -> str:
        raise TranscriptionUnavailable(self.reason)

    def transcribe_full(self, pcm) -> Transcript:
        raise TranscriptionUnavailable(self.reason)

    def health(self) -> dict:
        return {"ok": False, "model": None, "device": None, "warm": False, "off": True}


def build_transcriber(url: str, timeout_s: float = 20.0):
    """``"off"`` (or empty) -> :class:`NullTranscriber`, anything else a
    :class:`WhisperClient`."""
    if not url or url.strip().lower() == "off":
        return NullTranscriber()
    return WhisperClient(url, timeout_s)
