"""The audio message, decoded, bounded and transcribed (M3 decision 4).

A port of Mike's ``src/audio.js`` plus its intake. Split out of the server
because it is the part with an attacker in it: the bytes arrive base64-encoded
from a socket, and an unbounded audio message is a way to fill a disk.
``decode_segment`` is pure, so the limits can be tested one message at a time
without a server.

Two rules from Mike that are easy to get wrong and expensive to get wrong:

* **An empty transcription produces nothing.** No turn, no ``heard``, and no
  "I didn't catch that". Silence mistaken for speech must cost the user nothing
  at all, and an error message is a cost. (This is a deliberate difference from
  the all-in-one path, where "didn't catch that" fires on *low-confidence text*,
  not on silence.)
* **``heard`` is emitted before routing**, even when the addressing gate then
  drops the utterance. "It heard me and decided I wasn't talking to it" and "it
  didn't hear me" are different problems with different fixes.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from jarvis.audio.whisper_client import TranscriptionUnavailable
from jarvis.remote import protocol as P

log = logging.getLogger(__name__)

#: What the brain accepts per segment, in decoded bytes. 1 MB is ~31 seconds of
#: 16 kHz s16le mono — twice the edge's own 15-second maximum, so a legal
#: segment is never refused and an edge with a broken segmenter is.
DEFAULT_MAX_AUDIO_BYTES = 1_000_000

#: Below this there is no utterance in there, whatever the energy detector
#: thought. Dropped without a word to the user.
MIN_SEGMENT_MS = 120

#: Base64 as the spec writes it, plus its padding. ``b64decode`` without
#: ``validate=True`` silently skips anything else, so without this check a
#: sentence of JSON would come back as a plausible-looking buffer of noise.
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


@dataclass(frozen=True)
class Decoded:
    """What ``decode_segment`` found, or why it stopped.

    ``ignore`` separates "tell the user, something is wrong" from "say nothing,
    there was nothing there".
    """

    ok: bool
    pcm: np.ndarray | None = None
    bytes: int = 0
    samples: int = 0
    duration_ms: int = 0
    error: str = ""
    ignore: bool = False


def base64_bytes(value: str) -> int:
    """Decoded byte count of a base64 string, *without* decoding it — used to
    refuse an oversized segment before allocating a buffer for it."""
    if not value:
        return 0
    padding = 2 if value.endswith("==") else 1 if value.endswith("=") else 0
    return (len(value) // 4) * 3 - padding


def decode_segment(
    msg: dict,
    *,
    max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    sample_rate: int = P.AUDIO_SAMPLE_RATE,
) -> Decoded:
    """Turn one validated ``audio`` message into PCM, or say why not."""
    b64 = msg.get("pcm") if isinstance(msg, dict) else None
    if not isinstance(b64, str) or not b64:
        return Decoded(False, error="empty audio segment", ignore=True)
    if len(b64) % 4 != 0 or not _BASE64_RE.match(b64):
        return Decoded(False, error="audio pcm is not base64")

    size = base64_bytes(b64)
    if size > max_bytes:
        return Decoded(
            False,
            error=f"audio segment is {round(size / 1024)} kB, limit is {round(max_bytes / 1024)} kB",
        )

    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError):
        return Decoded(False, error="audio pcm is not base64")

    # A stream of 16-bit samples has an even length. An odd one means the edge
    # cut a sample in half, and transcribing it would shift every sample after
    # the cut by one byte — noise that sounds like speech to no one.
    if len(raw) % P.AUDIO_BYTES_PER_SAMPLE:
        return Decoded(False, error="audio pcm is not whole 16-bit samples")

    samples = len(raw) // P.AUDIO_BYTES_PER_SAMPLE
    duration_ms = round(samples / sample_rate * 1000)
    if duration_ms < MIN_SEGMENT_MS:
        return Decoded(False, error=f"segment is only {duration_ms} ms", ignore=True)

    return Decoded(
        True,
        pcm=np.frombuffer(raw, dtype="<i2"),
        bytes=len(raw),
        samples=samples,
        duration_ms=duration_ms,
    )


def format_intake_log(
    *,
    duration_ms: int,
    reason: str | None,
    floor_db: int | None,
    peak_db: int | None,
    elapsed_ms: int,
    model_ms: int | None,
    language: str | None,
    text: str,
) -> str:
    """One line per segment, with what the edge thought it was hearing in it.

    Mike's lesson: ``15000ms maximum floor -70 peak -38`` is a diagnosis;
    ``15000ms`` was a mystery.
    """
    parts = [f"heard {duration_ms}ms"]
    if reason:
        parts.append(reason)
    if floor_db is not None:
        parts.append(f"floor {floor_db}")
    if peak_db is not None:
        parts.append(f"peak {peak_db}")
    parts.append(f"in {elapsed_ms}ms")
    if model_ms is not None:
        parts.append(f"(model {model_ms}ms)")
    if language:
        parts.append(language)
    parts.append(f'"{text}"')
    return " ".join(parts)


@dataclass(frozen=True)
class Heard:
    """One transcribed utterance, on its way to the addressing gate."""

    text: str
    confidence: float | None
    duration_ms: int
    reason: str | None
    language: str | None


class AudioIntake:
    """``audio`` message -> transcript, ``heard``, and one log line.

    ``transcribe`` is an async callable taking int16 PCM and returning a
    :class:`~jarvis.audio.whisper_client.Transcript`; the server passes one that
    wraps the blocking HTTP client in ``asyncio.to_thread``. ``send`` is the
    connection's own ``send``, never a broadcast.
    """

    def __init__(
        self,
        transcribe: Callable[[np.ndarray], Awaitable],
        *,
        send: Callable[[dict], Awaitable[None]],
        max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    ) -> None:
        self._transcribe = transcribe
        self._send = send
        self.max_bytes = max_bytes

    async def receive(self, msg: dict) -> Heard | None:
        decoded = decode_segment(msg, max_bytes=self.max_bytes)
        if not decoded.ok:
            if decoded.ignore:
                log.debug("segment ignored: %s", decoded.error)
                return None
            await self._send(P.error(decoded.error))
            return None

        started = time.monotonic()
        try:
            result = await self._transcribe(decoded.pcm)
        except TranscriptionUnavailable as exc:
            # The brain keeps running: text mode still works, and the next
            # segment tries again. The edge is told in a sentence, and the voder
            # speaks it if it's up.
            log.warning("transcription unavailable: %s", exc)
            await self._send(P.error(f"Cannot hear you: {exc}"))
            return None
        elapsed_ms = int((time.monotonic() - started) * 1000)

        text = (getattr(result, "text", "") or "").strip()
        log.info(
            "%s",
            format_intake_log(
                duration_ms=decoded.duration_ms,
                reason=msg.get("reason"),
                floor_db=msg.get("floor_db"),
                peak_db=msg.get("peak_db"),
                elapsed_ms=elapsed_ms,
                model_ms=getattr(result, "ms", None),
                language=getattr(result, "language", None),
                text=text,
            ),
        )
        if not text:
            return None

        confidence = getattr(result, "confidence", None)
        # Before routing, always — even if the gate is about to drop it.
        await self._send(P.heard(text, confidence))
        return Heard(
            text=text,
            confidence=confidence,
            duration_ms=decoded.duration_ms,
            reason=msg.get("reason"),
            language=getattr(result, "language", None),
        )
