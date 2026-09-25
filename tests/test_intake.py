"""Server intake (M3 decision 4): the audio message, decoded and bounded, then
transcribed.

This is the part with an attacker in it — the bytes arrive base64-encoded from a
socket — so ``decode_segment`` checks every bound *before* it allocates
anything. Everything here is pure or driven by fakes; no network, no GPU.
"""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest

from jarvis.audio.whisper_client import Transcript, TranscriptionUnavailable
from jarvis.remote import protocol as P
from jarvis.remote.intake import (
    DEFAULT_MAX_AUDIO_BYTES,
    MIN_SEGMENT_MS,
    AudioIntake,
    base64_bytes,
    decode_segment,
    format_intake_log,
)

SR = P.AUDIO_SAMPLE_RATE


def _pcm_b64(ms: int) -> str:
    return P.encode_pcm(np.zeros(int(SR * ms / 1000), dtype=np.int16))


# -- decode_segment --------------------------------------------------------


def test_a_good_segment_decodes_to_pcm():
    out = decode_segment(P.audio(_pcm_b64(1000)))
    assert out.ok is True
    assert out.duration_ms == 1000
    assert out.samples == SR
    assert out.bytes == SR * 2
    assert out.pcm.dtype == np.int16 and len(out.pcm) == SR


def test_an_empty_segment_is_ignored_not_reported():
    out = decode_segment({"type": "audio", "pcm": ""})
    assert out.ok is False and out.ignore is True


def test_a_segment_under_120ms_is_ignored_silently():
    out = decode_segment(P.audio(_pcm_b64(MIN_SEGMENT_MS - 20)))
    assert out.ok is False
    assert out.ignore is True, "silence mistaken for speech must cost the user nothing"
    assert decode_segment(P.audio(_pcm_b64(MIN_SEGMENT_MS + 20))).ok is True


def test_oversized_audio_is_refused_before_it_is_decoded():
    # 31 seconds is inside the limit; 40 is not. Neither is ever decoded here.
    b64 = "A" * (((DEFAULT_MAX_AUDIO_BYTES + 100_000) // 3) * 4)
    out = decode_segment({"type": "audio", "pcm": b64})
    assert out.ok is False and out.ignore is False
    assert "limit" in out.error
    assert out.pcm is None


def test_base64_bytes_matches_the_real_decode():
    for n in (0, 1, 2, 3, 10, 999, 4096):
        b64 = base64.b64encode(b"\x01" * n).decode()
        assert base64_bytes(b64) == n


def test_junk_that_is_not_base64_is_refused():
    # base64.b64decode without validate=True silently skips junk, so a sentence
    # of JSON would come back as a plausible-looking buffer of noise.
    out = decode_segment({"type": "audio", "pcm": '{"hello": "there"}'})
    assert out.ok is False and out.ignore is False
    assert "base64" in out.error


def test_an_odd_number_of_bytes_is_refused():
    odd = base64.b64encode(b"\x01" * 4001).decode()
    out = decode_segment({"type": "audio", "pcm": odd})
    assert out.ok is False
    assert "16-bit" in out.error


def test_the_decoded_limit_is_configurable_and_smaller_than_the_wire_limit():
    assert DEFAULT_MAX_AUDIO_BYTES < P.MAX_AUDIO_BASE64
    out = decode_segment(P.audio(_pcm_b64(2000)), max_bytes=1000)
    assert out.ok is False and "limit" in out.error


# -- the log line ----------------------------------------------------------


def test_the_log_line_names_what_the_edge_thought_it_was_hearing():
    line = format_intake_log(
        duration_ms=2300, reason="silence", floor_db=-52, peak_db=-18,
        elapsed_ms=340, model_ms=300, language="en", text="search black holes",
    )
    assert line == (
        'heard 2300ms silence floor -52 peak -18 in 340ms (model 300ms) en "search black holes"'
    )
    # "15000ms maximum floor -70 peak -38" is a diagnosis; "15000ms" was a mystery
    assert "maximum floor -70 peak -38" in format_intake_log(
        duration_ms=15000, reason="maximum", floor_db=-70, peak_db=-38,
        elapsed_ms=900, model_ms=None, language=None, text="",
    )


# -- AudioIntake -----------------------------------------------------------


class FakeTranscriber:
    """Canned text per call, in order."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[int] = []

    async def __call__(self, pcm):
        self.calls.append(len(pcm))
        out = self.results.pop(0) if self.results else Transcript("", None, "en", 1)
        if isinstance(out, Exception):
            raise out
        return out


def _intake(transcriber, **kw):
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    return AudioIntake(transcriber, send=send, **kw), sent


def test_a_transcribed_segment_produces_heard_before_anything_else():
    stt = FakeTranscriber(Transcript("Jarvis search black holes", 0.9, "en", 300))
    intake, sent = _intake(stt)
    out = asyncio.run(intake.receive(P.audio(_pcm_b64(1500), reason="silence", floor_db=-52, peak_db=-18)))

    assert out is not None
    assert out.text == "Jarvis search black holes"
    assert out.confidence == 0.9
    assert out.duration_ms == 1500
    assert [m["type"] for m in sent] == ["heard"]
    assert sent[0]["text"] == "Jarvis search black holes"


def test_an_empty_transcription_produces_nothing_at_all():
    """No turn, no `heard`, and no "I didn't catch that" — silence mistaken for
    speech must cost the user nothing."""
    stt = FakeTranscriber(Transcript("   ", None, "en", 120))
    intake, sent = _intake(stt)
    assert asyncio.run(intake.receive(P.audio(_pcm_b64(900)))) is None
    assert sent == []


def test_an_ignored_segment_is_never_transcribed():
    stt = FakeTranscriber(Transcript("whatever", None, "en", 1))
    intake, sent = _intake(stt)
    assert asyncio.run(intake.receive(P.audio(_pcm_b64(50)))) is None
    assert stt.calls == [], "a 50 ms segment should not reach the GPU"
    assert sent == []


def test_a_malformed_segment_is_reported_but_not_transcribed():
    stt = FakeTranscriber()
    intake, sent = _intake(stt)
    assert asyncio.run(intake.receive({"type": "audio", "pcm": "not base64!!"})) is None
    assert stt.calls == []
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["fatal"] is False


def test_whisper_being_down_is_a_sentence_the_user_can_act_on():
    stt = FakeTranscriber(TranscriptionUnavailable("the transcription service is not running"))
    intake, sent = _intake(stt)
    assert asyncio.run(intake.receive(P.audio(_pcm_b64(1000)))) is None
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["message"] == "Cannot hear you: the transcription service is not running"
    assert sent[0]["fatal"] is False, "the brain keeps running; text mode still works"


def test_one_log_line_per_segment(caplog):
    import logging

    stt = FakeTranscriber(Transcript("hello", 0.8, "en", 200))
    intake, _ = _intake(stt)
    with caplog.at_level(logging.INFO, logger="jarvis.remote.intake"):
        asyncio.run(intake.receive(P.audio(_pcm_b64(800), reason="silence", floor_db=-60, peak_db=-20)))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("heard ")]
    assert len(lines) == 1
    assert "800ms silence floor -60 peak -20" in lines[0]
    assert '"hello"' in lines[0]


def test_the_intake_never_logs_or_returns_the_audio_itself(caplog):
    import logging

    stt = FakeTranscriber(Transcript("hello", 0.8, "en", 200))
    intake, _ = _intake(stt)
    b64 = _pcm_b64(800)
    with caplog.at_level(logging.DEBUG):
        out = asyncio.run(intake.receive(P.audio(b64)))
    assert not any(b64[:40] in r.getMessage() for r in caplog.records)
    assert not hasattr(out, "pcm")
