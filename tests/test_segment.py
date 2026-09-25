"""The pure segmenter (M3 decision 2) — a port of Mike's `client/src/audio/segment.ts`.

Nothing here opens a device or looks at a clock: the segmenter counts samples,
so a recording gives the same segments on the Pi, on the dev box, and here.
That property (acceptance criterion 8) is what most of these tests check, along
with the noise-floor behaviour that Mike's PRD 6 §8 was written about.
"""

from __future__ import annotations

import math
import wave

import numpy as np
import pytest

from jarvis.audio.segment import (
    DEFAULTS,
    Segmenter,
    SegmenterOptions,
    float_to_pcm16,
    frame_level_db,
    resample_to,
    segment_pcm,
)

SR = 16000
#: a quiet room: rms 0.0005 -> about -66 dBFS, inside the floor's [-70, -25] clamp
ROOM = 0.0005
#: ordinary speech: rms ~0.05 -> about -26 dBFS, a full 40 dB over the room
LOUD = 0.07


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _noise(ms: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    """`ms` of white noise at the given rms (0..1), as int16."""
    n = int(SR * ms / 1000)
    return float_to_pcm16(rng.normal(0.0, rms, n).astype(np.float32).clip(-1, 1))


def _tone(ms: int, amplitude: float = LOUD, freq: float = 220.0) -> np.ndarray:
    """`ms` of a sine — stands in for a voiced sound. Deterministic, so two
    runs of a test compare exactly."""
    n = int(SR * ms / 1000)
    t = np.arange(n, dtype=np.float32) / SR
    return float_to_pcm16(amplitude * np.sin(2 * math.pi * freq * t, dtype=np.float32))


def _room(ms: int, rng: np.random.Generator | None = None) -> np.ndarray:
    return _noise(ms, ROOM, rng or _rng())


def _join(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.int16)


def _key(seg) -> tuple:
    """Everything a segment is, for comparing two runs."""
    return (seg.start_ms, seg.end_ms, seg.speech_ms, seg.reason, seg.pcm.tobytes())


# -- reproducibility -------------------------------------------------------


def test_the_same_audio_gives_the_same_segments_however_it_is_chunked():
    rng = _rng()
    pcm = _join(_room(400, rng), _tone(900), _room(1200, rng), _tone(700), _room(1200, rng))

    whole = segment_pcm(pcm)
    # 1600 = a whole number of frames; 317 and 1 deliberately are not, so the
    # frame-straddling carry (`_tail`) is exercised.
    assert [_key(s) for s in whole] == [_key(s) for s in segment_pcm(pcm, chunk_samples=1600)]
    assert [_key(s) for s in whole] == [_key(s) for s in segment_pcm(pcm, chunk_samples=317)]
    assert len(whole) == 2


def test_a_wav_file_gives_the_same_segments_as_the_buffer(tmp_path):
    rng = _rng(11)
    pcm = _join(_room(500, rng), _tone(800), _room(1500, rng))
    path = tmp_path / "utterance.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR
        from_file = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    assert [_key(s) for s in segment_pcm(from_file)] == [_key(s) for s in segment_pcm(pcm)]
    assert len(segment_pcm(from_file)) == 1


# -- segment shape ---------------------------------------------------------


def test_a_pause_inside_a_sentence_does_not_split_it():
    rng = _rng()
    pcm = _join(
        _room(400, rng), _tone(800), _room(400, rng), _tone(800), _room(1200, rng)
    )
    segments = segment_pcm(pcm)
    assert len(segments) == 1
    assert segments[0].reason == "silence"
    # both halves of the sentence are in it
    assert segments[0].speech_ms > 1400


def test_a_gap_between_sentences_does_split_them():
    rng = _rng()
    pcm = _join(
        _room(400, rng), _tone(800), _room(1000, rng), _tone(800), _room(1200, rng)
    )
    segments = segment_pcm(pcm)
    assert len(segments) == 2
    assert [s.reason for s in segments] == ["silence", "silence"]


def test_a_click_is_not_an_utterance():
    rng = _rng()
    pcm = _join(_room(500, rng), _tone(100), _room(1500, rng))
    assert segment_pcm(pcm) == []


def test_pure_silence_produces_nothing():
    assert segment_pcm(_room(4000)) == []
    assert segment_pcm(np.zeros(SR * 3, dtype=np.int16)) == []


def test_a_segment_keeps_pre_roll_and_tail_around_the_speech():
    rng = _rng()
    pcm = _join(_room(600, rng), _tone(1000), _room(1500, rng))
    (seg,) = segment_pcm(pcm)
    # 1000 ms of speech, plus 300 ms pre-roll and 250 ms tail, minus the frames
    # the detector spends making up its mind — call it "clearly more than the
    # speech, and well short of the whole recording".
    assert 1000 < seg.duration_ms < 1800
    assert seg.duration_ms == pytest.approx(len(seg.pcm) / SR * 1000, abs=25)
    assert 900 <= seg.speech_ms <= 1100
    assert seg.start_ms < 700 and seg.end_ms > 1500


def test_a_segment_carries_its_diagnostics():
    rng = _rng()
    (seg,) = segment_pcm(_join(_room(600, rng), _tone(900), _room(1200, rng)))
    assert seg.reason == "silence"
    assert -70 <= seg.floor_db <= -55           # the quiet room it was recorded in
    assert -32 <= seg.peak_db <= -20            # the tone, about -26 dBFS
    assert seg.peak_db > seg.floor_db + DEFAULTS.margin_db


# -- the noise floor (Mike PRD 6 §8) ---------------------------------------


def test_one_frame_of_zeros_does_not_collapse_the_floor():
    """The failure this whole design exists for: a single dropout frame used to
    drag the floor to its minimum, room tone became speech, and every segment
    ran to the maximum."""
    rng = _rng()
    noisy = 0.01  # a room with a fan in it, about -40 dBFS
    pcm = _join(
        _noise(2000, noisy, rng),
        np.zeros(int(SR * 0.02), dtype=np.int16),  # one 20 ms dropout
        _noise(3000, noisy, rng),
    )
    seg = Segmenter()
    segments = seg.push(pcm)
    assert segments == []
    assert seg.flush("close") is None
    # the floor followed the room, not the dropout
    assert seg.noise_floor_db > DEFAULTS.min_floor_db + 10


def test_a_segment_opened_on_room_tone_closes_itself():
    """A fan starting is loud enough to open a segment against a quiet floor;
    `floor_rise_rate_open` has to close it in seconds, not at max_segment_ms."""
    rng = _rng()
    pcm = _join(_room(500, rng), _noise(8000, 0.01, rng))
    segments = segment_pcm(pcm)
    assert segments, "expected the onset of the room tone to open a segment"
    assert segments[0].reason == "silence"
    assert segments[0].duration_ms < 4000
    # and it does not keep re-opening on the same steady tone
    assert len(segments) <= 2


def test_the_floor_is_clamped_below_so_dither_is_not_speech():
    seg = Segmenter()
    seg.push(np.zeros(SR * 2, dtype=np.int16))
    assert seg.noise_floor_db >= DEFAULTS.min_floor_db


# -- the maximum, and where it cuts ----------------------------------------


def test_a_long_monologue_is_cut_at_a_pause_not_at_the_maximum():
    rng = _rng()
    parts = [_room(400, rng)]
    # 20 s of talking with a short breath every 2 s. 300 ms is under the 700 ms
    # hangover, so none of them ends the segment — but they are over cut_gap_ms,
    # so one of them is where the cut lands.
    for _ in range(10):
        parts.append(_tone(1700))
        parts.append(_room(300, rng))
    parts.append(_room(1500, rng))
    segments = segment_pcm(_join(*parts))

    assert segments[0].reason == "maximum"
    assert segments[0].duration_ms < DEFAULTS.max_segment_ms
    assert segments[0].duration_ms > DEFAULTS.max_segment_ms / 2  # cut in the second half
    # the cut landed in a pause, not mid-word: the segment ends on quiet frames
    tail = segments[0].pcm[-int(SR * 0.06):]
    assert frame_level_db(tail) < segments[0].floor_db + DEFAULTS.margin_db
    # and what followed the pause became the next segment, not lost audio
    assert len(segments) >= 2
    assert segments[1].start_ms >= segments[0].end_ms - DEFAULTS.tail_ms


# -- push-to-talk ----------------------------------------------------------


def test_in_hold_mode_silence_does_not_close_a_segment_but_release_does():
    rng = _rng()
    seg = Segmenter(SegmenterOptions(hold=True))
    assert seg.push(_join(_room(300, rng), _tone(700), _room(2000, rng))) == []
    out = seg.flush("release")
    assert out is not None
    assert out.reason == "release"
    # the 2 s of silence the user left before releasing is trimmed off
    assert out.duration_ms < 1400
    assert 600 <= out.speech_ms <= 800


def test_flush_with_nothing_open_returns_nothing():
    seg = Segmenter()
    seg.push(_room(500))
    assert seg.flush("release") is None
    assert seg.flush("close") is None


def test_speaking_tracks_whether_an_utterance_is_open():
    rng = _rng()
    seg = Segmenter()
    seg.push(_room(400, rng))
    assert seg.speaking is False
    seg.push(_tone(600))
    assert seg.speaking is True
    seg.push(_room(1200, rng))
    assert seg.speaking is False


def test_reset_forgets_the_room_and_the_position():
    rng = _rng()
    seg = Segmenter()
    seg.push(_join(_room(500, rng), _tone(400)))
    assert seg.position_ms > 0
    seg.reset()
    assert seg.speaking is False
    assert seg.position_ms == 0
    assert seg.noise_floor_db == DEFAULTS.max_floor_db


# -- helpers ---------------------------------------------------------------


def test_float_to_pcm16_clamps_instead_of_wrapping():
    out = float_to_pcm16(np.array([1.0, -1.0, 2.0, -2.0, 0.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out.tolist() == [32767, -32768, 32767, -32768, 0]


def test_frame_level_db_floors_digital_silence():
    assert frame_level_db(np.zeros(320, dtype=np.int16)) == -100.0
    assert frame_level_db(np.full(320, 32767, dtype=np.int16)) == pytest.approx(0.0, abs=0.01)


def test_resample_to_16k():
    src = _tone(1000, freq=100.0).astype(np.float32) / 32768
    out = resample_to(src, 48000, 16000)
    assert len(out) == pytest.approx(len(src) / 3, rel=0.01)
    assert resample_to(src, 16000, 16000) is src


def test_options_override_only_what_is_given():
    opts = SegmenterOptions(hangover_ms=1200)
    assert opts.hangover_ms == 1200
    assert opts.margin_db == DEFAULTS.margin_db
    assert Segmenter().opts == DEFAULTS
