"""The wire protocol (M3 decision 3) — one schema, imported by edge and brain.

``validate_c2s`` is the only thing standing between the internet and the rest of
the brain, so most of this file is about what it refuses. It must never raise:
its input is whatever arrived on a socket.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from jarvis.remote import protocol as P


def _hello(**over):
    msg = {"type": "hello", "protocol": P.PROTOCOL_VERSION, "token": "t0ken", "device_id": "livingroom"}
    msg.update(over)
    return msg


def _b64(n_samples: int) -> str:
    return base64.b64encode(np.zeros(n_samples, dtype=np.int16).tobytes()).decode()


# -- round trips -----------------------------------------------------------


def test_every_client_message_round_trips_through_json():
    pcm = _b64(1600)
    messages = [
        P.hello("t0ken", "livingroom"),
        P.audio(pcm, reason="silence", floor_db=-52, peak_db=-18),
        P.speaking(True),
        P.interrupt(),
        P.control(P.CONTROL.SET_MODE, mode="always"),
        P.control(P.CONTROL.PLAYBACK_DONE, id="s1"),
        P.control(P.CONTROL.CLIENT_LOG, level="warn", text="mic reopened"),
    ]
    for msg in messages:
        ok, out = P.validate_c2s(json.dumps(msg))
        assert ok, f"{msg['type']}: {out}"
        assert out["type"] == msg["type"]


def test_every_server_message_has_its_shape():
    assert P.ready("byname", speech=True) == {
        "type": "ready", "protocol": P.PROTOCOL_VERSION, "mode": "byname", "speech": True,
    }
    assert P.heard("search black holes", 0.91) == {
        "type": "heard", "text": "search black holes", "confidence": 0.91,
    }
    assert P.state("listening", "byname", mic=True) == {
        "type": "state", "value": "listening", "mode": "byname", "mic": True,
    }
    assert P.text("of course, sir") == {"type": "text", "text": "of course, sir", "from": "jarvis"}
    part = P.speech("s1", 0, "hello", _b64(160), 22050, final=False)
    assert part["type"] == "speech" and part["id"] == "s1" and part["part"] == 0
    assert part["sample_rate"] == 22050 and part["final"] is False
    assert P.event("stopPlayback") == {"type": "event", "kind": "stopPlayback", "data": {}}
    assert P.error("nope", fatal=True) == {"type": "error", "message": "nope", "fatal": True}
    # every one of them is JSON-serialisable as-is
    for msg in (P.ready("byname", True), P.heard("x"), P.state("idle", "byname", True), part):
        json.loads(json.dumps(msg))


def test_pcm_encodes_and_decodes_unchanged():
    pcm = (np.arange(-2000, 2000, dtype=np.int32) % 3000 - 1500).astype(np.int16)
    out = P.decode_pcm(P.encode_pcm(pcm))
    assert out.dtype == np.int16
    assert np.array_equal(out, pcm)


# -- what validate_c2s refuses --------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"\x00\x01\x02",                      # a binary frame
        bytearray(b"\x00\x01"),
        "not json at all",
        "[1, 2, 3]",                          # JSON, but not an object
        '"a string"',
        "null",
        "{}",                                 # no type
        '{"type": 7}',
        '{"type": "nonsense"}',
        "",
    ],
)
def test_junk_is_rejected_and_never_raises(raw):
    ok, err = P.validate_c2s(raw)
    assert ok is False
    assert isinstance(err, str) and err


def test_hello_needs_a_protocol_a_token_and_a_device():
    assert P.validate_c2s(json.dumps(_hello()))[0] is True
    for bad in (
        _hello(protocol="1"),
        _hello(token=""),
        _hello(device_id=""),
        _hello(device_id="../../etc/passwd"),   # it becomes a filename on the brain
        _hello(device_id="living room!"),
        _hello(device_id="x" * 65),
    ):
        ok, err = P.validate_c2s(json.dumps(bad))
        assert ok is False, bad
        assert isinstance(err, str)
    missing = _hello()
    del missing["token"]
    assert P.validate_c2s(json.dumps(missing))[0] is False


def test_audio_is_bounded_before_anything_is_decoded():
    ok, _ = P.validate_c2s(json.dumps(P.audio(_b64(1600))))
    assert ok is True
    huge = {"type": "audio", "pcm": "A" * (P.MAX_AUDIO_BASE64 + 4)}
    ok, err = P.validate_c2s(json.dumps(huge))
    assert ok is False and "too long" in err
    ok, err = P.validate_c2s(json.dumps({"type": "audio"}))
    assert ok is False


def test_audio_diagnostics_are_bounded():
    good = P.audio(_b64(1600), reason="maximum", floor_db=-70, peak_db=-1)
    assert P.validate_c2s(json.dumps(good))[0] is True
    for bad in (
        {"type": "audio", "pcm": _b64(160), "reason": "because"},
        {"type": "audio", "pcm": _b64(160), "floor_db": 12},
        {"type": "audio", "pcm": _b64(160), "peak_db": -1000},
        {"type": "audio", "pcm": _b64(160), "final": "yes"},
        {"type": "audio", "pcm": _b64(160), "sample_rate": 44100},
    ):
        ok, err = P.validate_c2s(json.dumps(bad))
        assert ok is False, bad
        assert isinstance(err, str)


def test_the_sample_rate_is_fixed_at_16k():
    ok, _ = P.validate_c2s(json.dumps({"type": "audio", "pcm": _b64(160), "sample_rate": P.AUDIO_SAMPLE_RATE}))
    assert ok is True
    ok, err = P.validate_c2s(json.dumps({"type": "audio", "pcm": _b64(160), "sample_rate": 8000}))
    assert ok is False and "16000" in err


def test_speaking_is_only_ever_a_boolean():
    assert P.validate_c2s(json.dumps({"type": "speaking", "on": False}))[0] is True
    assert P.validate_c2s(json.dumps({"type": "speaking", "on": "yes"}))[0] is False
    assert P.validate_c2s(json.dumps({"type": "speaking"}))[0] is False


def test_control_needs_an_action_and_object_args():
    assert P.validate_c2s(json.dumps({"type": "control", "action": "mic", "args": {"on": True}}))[0] is True
    assert P.validate_c2s(json.dumps({"type": "control", "action": "mic"}))[0] is True
    assert P.validate_c2s(json.dumps({"type": "control"}))[0] is False
    assert P.validate_c2s(json.dumps({"type": "control", "action": "mic", "args": [1]}))[0] is False
    assert P.validate_c2s(json.dumps({"type": "control", "action": "x" * 100}))[0] is False


def test_a_client_log_line_is_capped():
    ok, msg = P.validate_c2s(
        json.dumps(P.control(P.CONTROL.CLIENT_LOG, level="info", text="x" * 5000))
    )
    assert ok is True
    # accepted, but never at 5000 characters in the brain's log
    assert len(msg["args"]["text"]) <= P.MAX_CLIENT_LOG


def test_close_codes_say_which_kind_of_wrong():
    assert (P.CLOSE.UNAUTHORIZED, P.CLOSE.BAD_PROTOCOL, P.CLOSE.BAD_MESSAGE, P.CLOSE.SERVER_SHUTDOWN) == (
        4001, 4002, 4003, 4004,
    )
