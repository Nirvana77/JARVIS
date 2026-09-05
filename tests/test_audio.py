"""Audio layer: import + non-hardware code paths.

Live mic/speaker behaviour is verified manually (``python -m jarvis`` on a box
with audio). These checks just keep the module APIs from drifting.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_wake_word_loads_bundled_hey_jarvis():
    from jarvis.audio.wake import WakeWord

    w = WakeWord(model="hey_jarvis", threshold=0.5).load()
    assert w._key and "jarvis" in w._key.lower()
    # a frame of silence must not trigger
    assert w.score(np.zeros(1280, dtype=np.int16)) < 0.5
    assert not w.triggered(np.zeros(1280, dtype=np.int16))


def test_wake_word_unknown_model_is_a_clear_error():
    from jarvis.audio.wake import WakeWord

    with pytest.raises(FileNotFoundError) as exc:
        WakeWord(model="definitely_not_a_model")
    assert "not found" in str(exc.value)


def test_speaker_degrades_without_a_voice_file(capsys, tmp_path):
    from jarvis.audio.tts import Speaker

    spk = Speaker(voice="en_GB-alan-medium", model_dir=tmp_path)
    assert spk.available() is False
    spk.say("hello there")
    assert "Jarvis: hello there" in capsys.readouterr().out
    pcm, _ = spk.synthesize("hello there")
    assert len(pcm) == 0


def test_transcriber_handles_empty_audio():
    from jarvis.audio.stt import Transcriber

    t = Transcriber(model="base")
    assert t.transcribe(np.zeros(0, dtype=np.float32)) == ""


def test_microphone_constructs_without_opening_a_stream():
    from jarvis.audio.capture import Microphone

    m = Microphone(sample_rate=16000, frame_samples=1280)
    assert m._stream is None
    assert m.frame_samples == 1280


def _mic_fed_with(frames):
    from jarvis.audio.capture import Microphone

    m = Microphone(sample_rate=16000, frame_samples=1280)
    it = iter(frames)
    m.read = lambda timeout=None: next(it)  # type: ignore[method-assign]
    return m


def test_record_utterance_returns_empty_on_silence():
    silence = [np.zeros(1280, dtype=np.int16)] * 200
    audio = _mic_fed_with(silence).record_utterance(max_seconds=2, silence_seconds=0.5)
    assert len(audio) == 0


def test_record_utterance_ignores_a_short_blip():
    # 5 frames ambient, 2 loud frames (a cough), then silence
    frames = [np.zeros(1280, dtype=np.int16)] * 5
    frames += [np.full(1280, 8000, dtype=np.int16)] * 2
    frames += [np.zeros(1280, dtype=np.int16)] * 200
    audio = _mic_fed_with(frames).record_utterance(max_seconds=3, silence_seconds=0.5)
    assert len(audio) == 0  # < 3 speech frames -> treated as noise


def test_record_utterance_captures_real_speech():
    frames = [np.zeros(1280, dtype=np.int16)] * 5
    frames += [np.full(1280, 8000, dtype=np.int16)] * 20   # ~1.6s of "speech"
    frames += [np.zeros(1280, dtype=np.int16)] * 200
    audio = _mic_fed_with(frames).record_utterance(max_seconds=5, silence_seconds=0.5)
    assert len(audio) > 16000  # more than a second of audio
    assert audio.dtype == np.float32


def test_calibration_window_catches_a_brief_pause_even_after_a_loud_start():
    """Regression: talking immediately (no lead-in silence) used to let the
    noise-floor calibration sample nothing but the speaker's own loud opening
    words, inflating the threshold enough that a normal, quieter drop-off in
    volume toward the end of a sentence got misread as silence — cutting the
    command short (and, in practice, feeding Whisper a truncated clip it then
    hallucinates a wrong transcription for). A wider calibration window gives
    a real chance of catching a natural inter-word pause instead."""
    loud = np.full(1280, 8000, dtype=np.int16)     # rms ~0.244
    pause = np.full(1280, 300, dtype=np.int16)     # rms ~0.009 — a brief breath
    quieter_tail = np.full(1280, 900, dtype=np.int16)  # rms ~0.027 — trailing off

    frames = (
        [loud] * 8         # fills the *old* 8-frame calibration window entirely
        + [pause] * 2      # only a *wider* window would ever see this dip
        + [loud] * 10
        + [quieter_tail] * 15   # the sentence trailing off — still real speech
        + [np.zeros(1280, dtype=np.int16)] * 200  # then real silence
    )
    audio = _mic_fed_with(frames).record_utterance(max_seconds=10, silence_seconds=0.5)
    # with the old 8-frame window the floor calibrates against continuous loud
    # speech, the threshold ends up above the quieter tail's level, and the
    # utterance is cut short well before the ~34 speech frames + tail land —
    # assert the wider window keeps enough of the tail that this doesn't happen.
    assert len(audio) >= 1280 * 30


def test_vad_threshold_override_is_used_instead_of_auto_calibration():
    from jarvis.audio.capture import Microphone

    quiet_speech = np.full(1280, 200, dtype=np.int16)  # rms ~0.006
    frames = [quiet_speech] * 10 + [np.zeros(1280, dtype=np.int16)] * 200

    m = Microphone(sample_rate=16000, frame_samples=1280, vad_threshold=0.003)
    it = iter(frames)
    m.read = lambda timeout=None: next(it)  # type: ignore[method-assign]

    audio = m.record_utterance(max_seconds=5, silence_seconds=0.5)
    assert len(audio) > 0  # a low pinned threshold correctly hears quiet speech
