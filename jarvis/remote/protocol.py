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

#: Where an edge fetches its staged firmware image: a plain ``GET`` on the
#: WebSocket's own port (``firmware.py``).
FIRMWARE_PATH = "/firmware"
#: A notification for the edge, from outside a conversation (``python -m
#: jarvis notify``): ``GET /notify?text=...`` on the same port, same token.
NOTIFY_PATH = "/notify"
#: The edge's power log, fetched and read (``python -m jarvis power``):
#: ``GET /power?day=YYYY-MM-DD&fetch=1``, same port, same token.
POWER_PATH = "/power"
#: ``esp_app_desc_t.version`` is 32 bytes, NUL included.
MAX_FW_VERSION = 31

#: Edge tools (``jarvis/skills/edge.py``): what an edge may declare in ``hello``. Every
#: string ends up in the NLU corpus or a spoken line, so all of it is bounded.
MAX_TOOLS = 16
MAX_TOOL_DESCRIPTION = 200
MAX_TOOL_EXAMPLES = 24
MAX_TOOL_EXAMPLE = 120
MAX_TOOL_PARAMS = 4
#: param types the brain knows how to pull out of an utterance (nlu/slots.py)
TOOL_PARAM_TYPES = ("duration", "number", "text", "name")
#: what a tool's `result` may ask JARVIS to say
MAX_TOOL_SAY = 300
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: Files an edge sends up (``file``): its power log, so far. Base64, because a
#: chunk is cut at a byte offset and may split a UTF-8 character in two.
FILE_KINDS = ("power",)
MAX_FILE_CHUNK = 65_536            # base64 chars in one message
MAX_FILE_OFFSET = 64 * 1024 * 1024
_FILE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|nodate)\.csv$")


class C2S:
    """Edge -> brain."""

    HELLO = "hello"          # {protocol, token, device_id, fw?, tools?} — first, within 10 s
    AUDIO = "audio"          # {pcm, final?, reason?, floor_db?, peak_db?}
    SPEAKING = "speaking"    # {on} — sent the moment the segmenter opens/closes
    INTERRUPT = "interrupt"  # {} — the button's cancel, or a spoken "stop"
    CONTROL = "control"      # {action, args}
    RESULT = "result"        # {id, ok, say?} — the answer to a `call`
    FILE = "file"            # {kind, name, offset, b64, eof?} | {kind, done, files?}


class S2C:
    """Brain -> edge."""

    READY = "ready"    # {protocol, mode, speech}
    HEARD = "heard"    # {text, confidence} — sent *before* routing
    STATE = "state"    # {value, mode, mic}
    TEXT = "text"      # {text, from}
    SPEECH = "speech"  # {id, part, text, pcm, sample_rate, final}
    EVENT = "event"    # {kind, data} — kind "ota": {version, size, sha256, path}
    ERROR = "error"    # {message, fatal}
    CALL = "call"      # {id, tool, args} — run one of the tools the edge declared


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

def hello(token: str, device_id: str, fw: str | None = None) -> dict:
    msg = {
        "type": C2S.HELLO,
        "protocol": PROTOCOL_VERSION,
        "token": token,
        "device_id": device_id,
    }
    if fw is not None:
        # The running firmware version, from an edge that can flash itself.
        msg["fw"] = fw
    return msg


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


def result(id: str, ok: bool, say: str = "") -> dict:
    return {"type": C2S.RESULT, "id": id, "ok": bool(ok), "say": say}


def file_chunk(kind: str, name: str, offset: int, data: bytes | str, eof: bool = False) -> dict:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return {
        "type": C2S.FILE, "kind": kind, "name": name, "offset": int(offset),
        "b64": base64.b64encode(data).decode("ascii"), "eof": bool(eof),
    }


def file_done(kind: str, files: int = 0) -> dict:
    return {"type": C2S.FILE, "kind": kind, "done": True, "files": int(files)}


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


def call(id: str, tool: str, args: dict) -> dict:
    return {"type": S2C.CALL, "id": id, "tool": tool, "args": args}


# -- validation -----------------------------------------------------------

def _is_obj(value) -> bool:
    return isinstance(value, dict)


def _is_str(value) -> bool:
    return isinstance(value, str)


def _is_int(value) -> bool:
    # bool is an int in Python, and "protocol": true is not a protocol number
    return isinstance(value, int) and not isinstance(value, bool)


def validate_tools(raw: Any) -> tuple[bool, Any]:
    """An edge's tool list from ``hello``: ``(True, tools)`` or ``(False,
    error)``. Each tool is ``{name, description, examples, params}``, params
    being ``{name: {"type": one of TOOL_PARAM_TYPES, "required"?: bool}}``."""
    if not isinstance(raw, list):
        return False, "tools must be a list"
    if len(raw) > MAX_TOOLS:
        return False, f"at most {MAX_TOOLS} tools"
    names: set[str] = set()
    tools = []
    for tool in raw:
        if not _is_obj(tool):
            return False, "each tool must be an object"
        name = tool.get("name")
        if not _is_str(name) or not _TOOL_NAME_RE.match(name):
            return False, "tool name must be 1-32 chars of [a-z0-9_], starting with a letter"
        if name in names:
            return False, f"tool {name} declared twice"
        names.add(name)
        description = tool.get("description", "")
        if not _is_str(description) or len(description) > MAX_TOOL_DESCRIPTION:
            return False, f"tool {name}: description must be a string of at most {MAX_TOOL_DESCRIPTION} chars"
        examples = tool.get("examples", [])
        if (
            not isinstance(examples, list)
            or len(examples) > MAX_TOOL_EXAMPLES
            or not all(_is_str(e) and 0 < len(e) <= MAX_TOOL_EXAMPLE for e in examples)
        ):
            return False, (
                f"tool {name}: examples must be at most {MAX_TOOL_EXAMPLES} strings "
                f"of at most {MAX_TOOL_EXAMPLE} chars"
            )
        params = tool.get("params", {})
        if not _is_obj(params) or len(params) > MAX_TOOL_PARAMS:
            return False, f"tool {name}: params must be an object of at most {MAX_TOOL_PARAMS}"
        clean = {}
        for key, spec in params.items():
            if not key.isidentifier() or len(key) > 32:
                return False, f"tool {name}: bad param name"
            if not _is_obj(spec) or spec.get("type") not in TOOL_PARAM_TYPES:
                return False, f"tool {name}: param {key} needs a type of {', '.join(TOOL_PARAM_TYPES)}"
            clean[key] = {"type": spec["type"], "required": spec.get("required") is True}
        tools.append(
            {"name": name, "description": description, "examples": list(examples), "params": clean}
        )
    return True, tools


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
        fw = raw.get("fw")
        if fw is not None and (not _is_str(fw) or len(fw) > MAX_FW_VERSION):
            return False, f"fw must be a version string of at most {MAX_FW_VERSION} chars"
        if raw.get("tools") is not None:
            ok, tools = validate_tools(raw["tools"])
            if not ok:
                return False, tools
            raw = {**raw, "tools": tools}
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

    if kind == C2S.FILE:
        if raw.get("kind") not in FILE_KINDS:
            return False, f"file kind must be one of {', '.join(FILE_KINDS)}"
        if raw.get("done") is True:
            files = raw.get("files", 0)
            if not _is_int(files) or files < 0:
                return False, "files must be a count"
            return True, raw
        name = raw.get("name")
        if not _is_str(name) or not _FILE_NAME_RE.match(name):
            # It becomes a filename under data/remote/power/<device>/.
            return False, "file name must be YYYY-MM-DD.csv or nodate.csv"
        offset = raw.get("offset")
        if not _is_int(offset) or not 0 <= offset <= MAX_FILE_OFFSET:
            return False, "file offset must be a byte offset"
        b64 = raw.get("b64")
        if not _is_str(b64) or len(b64) > MAX_FILE_CHUNK:
            return False, f"file b64 must be a string of at most {MAX_FILE_CHUNK} chars"
        if raw.get("eof") is not None and not isinstance(raw["eof"], bool):
            return False, "eof must be a boolean"
        return True, raw

    if kind == C2S.RESULT:
        call_id = raw.get("id")
        if not _is_str(call_id) or not 0 < len(call_id) <= 40:
            return False, "result needs the call's id"
        if not isinstance(raw.get("ok"), bool):
            return False, "result needs ok: boolean"
        say = raw.get("say", "")
        if not _is_str(say):
            return False, "say must be a string"
        return True, {**raw, "say": say[:MAX_TOOL_SAY]}

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
