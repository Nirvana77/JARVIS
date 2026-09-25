"""The wire protocol between the edge and the brain (M3 decision 3).

One file, imported by both sides, so the two can never drift. Versioned from the
first message: an edge built against an older protocol is told so on connect
rather than failing confusingly ten messages in.

**JSON text frames only, with PCM as base64.** Binary frames are refused. The
~33 % base64 overhead is irrelevant at speech-segment sizes, and one message per
segment keeps the protocol stateless.

The audio format is fixed at 16 kHz, signed 16-bit little-endian, mono. A
segment at any other rate is a named error, not a stream transcribed at the
wrong speed.

A port of Mike's ``src/protocol.js``, trimmed to what JARVIS speaks.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1

#: The one audio format on the wire.
AUDIO_SAMPLE_RATE = 16_000
AUDIO_BYTES_PER_SAMPLE = 2

#: Base64 length wall, checked *before* anything decodes: 2 MB of base64 is
#: ~1.5 MB of PCM, about 47 seconds. Comfortably inside the WebSocket's own
#: 4 MB max_size, so the two limits cannot disagree.
MAX_AUDIO_BASE64 = 2_000_000

#: A line the edge asked the brain to log. Capped, because it ends up in a log
#: a person reads, and rate-limited by the server on top of this.
MAX_CLIENT_LOG = 500

#: A device id becomes a filename (`data/remote/<device_id>.json`), so "it is a
#: string" is not validation.
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SEGMENT_REASONS = ("silence", "maximum", "release", "close")


class C2S:
    """Edge -> brain."""

    HELLO = "hello"          # {protocol, token, device_id} — first, within 10 s
    AUDIO = "audio"          # {pcm, final?, reason?, floor_db?, peak_db?}
    SPEAKING = "speaking"    # {on} — sent the moment the segmenter opens/closes
    INTERRUPT = "interrupt"  # {} — the button's cancel, or a spoken "stop"
    CONTROL = "control"      # {action, args}


class S2C:
    """Brain -> edge."""

    READY = "ready"    # {protocol, mode, speech}
    HEARD = "heard"    # {text, confidence} — sent *before* routing
    STATE = "state"    # {value, mode, mic}
    TEXT = "text"      # {text, from}
    SPEECH = "speech"  # {id, part, text, pcm, sample_rate, final}
    EVENT = "event"    # {kind, data}
    ERROR = "error"    # {message, fatal}


class CONTROL:
    """``control`` actions the edge may send."""

    SET_MODE = "setMode"            # {mode}
    SET_SPEECH = "setSpeech"        # {on, sample_rate?} — "send me `speech`"
    MIC = "mic"                     # {on} — the edge reporting its mic state
    PLAYBACK_DONE = "playbackDone"  # {id} — a spoken answer finished playing
    CLIENT_LOG = "clientLog"        # {level, text}


class CLOSE:
    """Close codes, so the edge can tell "you are not allowed" from "try again"."""

    UNAUTHORIZED = 4001
    BAD_PROTOCOL = 4002   # including no or late `hello`
    BAD_MESSAGE = 4003
    SERVER_SHUTDOWN = 4004


# -- PCM ------------------------------------------------------------------

def encode_pcm(pcm: np.ndarray | bytes) -> str:
    """int16 mono samples -> base64 for the wire."""
    if isinstance(pcm, np.ndarray):
        pcm = np.asarray(pcm, dtype=np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii")


def decode_pcm(b64: str) -> np.ndarray:
    """The inverse. Raises on junk — use ``intake.decode_segment`` for anything
    that came off a socket."""
    return np.frombuffer(base64.b64decode(b64, validate=True), dtype=np.int16)


# -- edge -> brain constructors -------------------------------------------

def hello(token: str, device_id: str) -> dict:
    return {
        "type": C2S.HELLO,
        "protocol": PROTOCOL_VERSION,
        "token": token,
        "device_id": device_id,
    }


def audio(
    pcm_b64: str,
    *,
    final: bool = True,
    reason: str | None = None,
    floor_db: int | None = None,
    peak_db: int | None = None,
) -> dict:
    msg: dict[str, Any] = {"type": C2S.AUDIO, "pcm": pcm_b64, "final": final}
    if reason is not None:
        msg["reason"] = reason
    if floor_db is not None:
        msg["floor_db"] = int(floor_db)
    if peak_db is not None:
        msg["peak_db"] = int(peak_db)
    return msg


def speaking(on: bool) -> dict:
    return {"type": C2S.SPEAKING, "on": bool(on)}


def interrupt() -> dict:
    return {"type": C2S.INTERRUPT}


def control(action: str, **args) -> dict:
    return {"type": C2S.CONTROL, "action": action, "args": args}


# -- brain -> edge constructors -------------------------------------------

def ready(mode: str, speech: bool) -> dict:
    return {"type": S2C.READY, "protocol": PROTOCOL_VERSION, "mode": mode, "speech": bool(speech)}


def heard(text: str, confidence: float | None = None) -> dict:
    return {"type": S2C.HEARD, "text": text, "confidence": confidence}


def state(value: str, mode: str, mic: bool) -> dict:
    return {"type": S2C.STATE, "value": value, "mode": mode, "mic": bool(mic)}


def text(body: str, source: str = "jarvis") -> dict:
    return {"type": S2C.TEXT, "text": body, "from": source}


def speech(
    id: str, part: int, body: str, pcm_b64: str, sample_rate: int, final: bool
) -> dict:
    return {
        "type": S2C.SPEECH,
        "id": id,
        "part": int(part),
        "text": body,
        "pcm": pcm_b64,
        "sample_rate": int(sample_rate),
        "final": bool(final),
    }


def event(kind: str, data: dict | None = None) -> dict:
    return {"type": S2C.EVENT, "kind": kind, "data": data or {}}


def error(message: str, fatal: bool = False) -> dict:
    return {"type": S2C.ERROR, "message": message, "fatal": bool(fatal)}


# -- validation -----------------------------------------------------------

def _is_obj(value) -> bool:
    return isinstance(value, dict)


def _is_str(value) -> bool:
    return isinstance(value, str)


def _is_int(value) -> bool:
    # bool is an int in Python, and "protocol": true is not a protocol number
    return isinstance(value, int) and not isinstance(value, bool)


def validate_c2s(raw: Any) -> tuple[bool, Any]:
    """Validate one frame from the edge.

    Returns ``(True, msg)`` or ``(False, error_string)`` and **never raises** —
    its input is whatever arrived from the internet. The caller decides what a
    rejection costs: before ``hello`` it closes the socket; after it, the edge
    gets an ``error`` reply and the connection stays up, because one bad frame
    shouldn't cost the user their conversation.
    """
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return False, "binary frames are not accepted; send JSON text"
    if _is_str(raw):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return False, "message is not valid JSON"
    if not _is_obj(raw):
        return False, "message must be an object"

    kind = raw.get("type")
    if not _is_str(kind):
        return False, "missing type"

    if kind == C2S.HELLO:
        if not _is_int(raw.get("protocol")):
            return False, "hello needs a protocol number"
        token = raw.get("token")
        if not _is_str(token) or not token:
            return False, "hello needs a token"
        device_id = raw.get("device_id")
        if not _is_str(device_id) or not _DEVICE_ID_RE.match(device_id):
            # It becomes a filename under data/remote/. A client sending
            # "../../x" must not decide where the brain writes.
            return False, "device_id must be 1-64 chars of [A-Za-z0-9._-]"
        return True, raw

    if kind == C2S.AUDIO:
        pcm = raw.get("pcm")
        if not _is_str(pcm):
            return False, "audio needs base64 pcm"
        # The outer wall: a length check on the string, before anything
        # allocates a buffer from it. The intake applies the (smaller) decoded
        # limit. The edge's own maximum segment is 15 s; this allows about
        # three times that, so a legitimate segment is never refused here and
        # an abusive one never gets far.
        if len(pcm) > MAX_AUDIO_BASE64:
            return False, "audio segment too long"
        rate = raw.get("sample_rate")
        if rate is not None and rate != AUDIO_SAMPLE_RATE:
            return False, f"audio must be {AUDIO_SAMPLE_RATE} Hz"
        if raw.get("final") is not None and not isinstance(raw["final"], bool):
            return False, "final must be a boolean"
        reason = raw.get("reason")
        if reason is not None and reason not in _SEGMENT_REASONS:
            return False, "reason must be a segment reason"
        # Diagnostics from the segmenter. Optional, and only ever logged — but a
        # log line is read by a person, so they are bounded.
        for key in ("floor_db", "peak_db"):
            value = raw.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False, f"{key} must be a level in dBFS"
            if not (-100 <= value <= 0):
                return False, f"{key} must be a level in dBFS"
        return True, raw

    if kind == C2S.SPEAKING:
        # The edge's own speech detector, reported the moment it flips, so the
        # brain can tell "the sentence went quiet" from "the sentence is still
        # being said" while it holds a fragment (decision 6). Nothing but a
        # boolean: the audio itself still arrives as a segment.
        if not isinstance(raw.get("on"), bool):
            return False, "speaking needs on: boolean"
        return True, raw

    if kind == C2S.INTERRUPT:
        return True, raw

    if kind == C2S.CONTROL:
        action = raw.get("action")
        if not _is_str(action) or not action:
            return False, "control needs an action"
        if len(action) > 32:
            return False, "control action name too long"
        args = raw.get("args")
        if args is None:
            raw = {**raw, "args": {}}
        elif not _is_obj(args):
            return False, "args must be an object"
        if action == CONTROL.CLIENT_LOG:
            line = raw["args"].get("text")
            if line is not None and _is_str(line):
                raw = {**raw, "args": {**raw["args"], "text": line[:MAX_CLIENT_LOG]}}
        return True, raw

    return False, f'unknown type "{kind}"'
