"""The brain's client for the warm ``jarvis-voder`` service (M3 decision 8).

Same reasoning as :mod:`jarvis.audio.whisper_client`: the model stays warm in
its own process, a synthesis that wedges is survivable by killing that process,
and when it is down the brain keeps working — a spoken answer degrades to a
``text`` message and one line in the log, never a crash mid-conversation.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class SynthesisUnavailable(RuntimeError):
    """The voder is down, misconfigured or off."""


class VoderClient:
    """``POST /speak`` -> raw 16-bit LE mono PCM."""

    def __init__(
        self,
        url: str,
        *,
        voice: str | None = None,
        sample_rate: int = 0,
        timeout_s: float = 20.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.voice = voice
        #: 0 = whatever the voice's own rate is, which is what sounds best
        self.sample_rate = sample_rate
        self.timeout_s = timeout_s
        #: set False after a failure, so the log says it once rather than per
        #: sentence; a successful call sets it back
        self.available = True

    def speak(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        """Returns (int16 pcm, sample_rate). Raises :class:`SynthesisUnavailable`."""
        import httpx

        body = {"text": text, "sample_rate": int(self.sample_rate)}
        chosen = voice or self.voice
        if chosen:
            body["voice"] = chosen
        try:
            response = httpx.post(f"{self.url}/speak", json=body, timeout=self.timeout_s)
        except httpx.ConnectError as exc:
            self.available = False
            raise SynthesisUnavailable("the speech service is not running") from exc
        except httpx.HTTPError as exc:  # noqa: BLE001
            self.available = False
            raise SynthesisUnavailable(f"the speech service failed: {exc}") from exc
        if response.status_code != 200:
            self.available = False
            detail = ""
            try:
                detail = f": {response.json().get('error', '')}"
            except ValueError:
                pass
            raise SynthesisUnavailable(
                f"the speech service returned {response.status_code}{detail}"
            )
        self.available = True
        rate = int(response.headers.get("X-Voder-Sample-Rate") or self.sample_rate or 22050)
        return np.frombuffer(response.content, dtype="<i2"), rate

    def speak_or_none(self, text: str, voice: str | None = None):
        """The forgiving form: ``None`` instead of an exception, with one line
        in the log. The answer still reaches the edge as ``text``."""
        try:
            return self.speak(text, voice)
        except SynthesisUnavailable as exc:
            log.warning("no speech for %r: %s", text[:40], exc)
            return None

    def health(self) -> dict:
        import httpx

        try:
            response = httpx.get(f"{self.url}/healthz", timeout=min(5.0, self.timeout_s))
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise SynthesisUnavailable("the speech service is not running") from exc


class NullVoder:
    """``[voder] url = "off"``: the answer is sent as text only."""

    available = False
    voice = None

    def speak(self, text: str, voice: str | None = None):
        raise SynthesisUnavailable("speech synthesis is turned off")

    def speak_or_none(self, text: str, voice: str | None = None):
        return None

    def health(self) -> dict:
        return {"ok": False, "off": True}


def build_voder(url: str, *, voice: str | None = None, sample_rate: int = 0, timeout_s: float = 20.0):
    if not url or url.strip().lower() == "off":
        return NullVoder()
    return VoderClient(url, voice=voice, sample_rate=sample_rate, timeout_s=timeout_s)
