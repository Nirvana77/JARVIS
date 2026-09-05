"""Microphone capture: a ``sounddevice`` stream of fixed-size int16 frames plus
an utterance recorder gated by a simple RMS energy VAD.

The energy gate calibrates a noise floor from the first few frames and treats a
frame as speech when it sits a few dB above that. It is deliberately basic —
silero VAD and barge-in are M4. Everything downstream just needs "the audio
between the wake word and the trailing silence".
"""

from __future__ import annotations

import logging
import queue

import numpy as np

log = logging.getLogger(__name__)


class Microphone:
    #: how many leading frames `record_utterance` (and `python -m jarvis mic`,
    #: so the preview matches reality) sample to calibrate the noise floor —
    #: wide enough (~1.2s @ 80ms frames) to have a real chance of catching a
    #: natural inter-word pause even if the user starts talking immediately.
    CALIBRATION_FRAMES = 15

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        device: int | str | None = None,
        vad_threshold: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        #: absolute RMS threshold `record_utterance` uses instead of
        #: auto-calibrating; 0 = auto-calibrate. See `capture.vad_threshold`
        #: in config.toml — set this from `python -m jarvis mic` if
        #: auto-calibration is cutting off the ends of your sentences.
        self.vad_threshold = vad_threshold
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    # -- stream lifecycle -------------------------------------------------

    def start(self) -> "Microphone":
        import sounddevice as sd

        def _callback(indata, _frames, _time, status):  # pragma: no cover - RT thread
            if status:
                log.debug("sounddevice status: %s", status)
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
            callback=_callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "Microphone":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- frame access --------------------------------------------------------

    def read(self, timeout: float | None = None) -> np.ndarray:
        return self._q.get(timeout=timeout)

    def drain(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    # -- shared VAD maths -------------------------------------------------

    @staticmethod
    def frame_rms(frame: np.ndarray) -> float:
        f = frame.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(f * f)) + 1e-9)

    @staticmethod
    def speech_threshold(noise_floor: float) -> float:
        """RMS above which a frame counts as speech, given a calibrated floor."""
        return max(noise_floor * 2.5, 0.008)

    @staticmethod
    def calibrate_floor(rms_values: list[float]) -> float:
        """Noise floor = quietest early frame, clamped. Robust even if the user
        is already talking (a running average would sit at speech level)."""
        return min(max(min(rms_values), 0.003), 0.02)

    # -- utterance recording -----------------------------------------------

    def record_utterance(
        self,
        max_seconds: float = 8.0,
        silence_seconds: float = 1.0,
        start_grace_seconds: float = 2.0,
        stop_event=None,
        min_speech_frames: int = 3,
    ) -> np.ndarray:
        """Collect frames until trailing silence or ``max_seconds``.

        Returns float32 mono in [-1, 1] (what faster-whisper wants). An empty
        array means nothing but silence was heard, or ``stop_event`` was set
        (the user asked to cancel).
        """
        frame_dur = self.frame_samples / self.sample_rate
        max_frames = int(max_seconds / frame_dur)
        silence_frames = max(1, int(silence_seconds / frame_dur))
        grace_frames = int(start_grace_seconds / frame_dur)

        collected: list[np.ndarray] = []
        early_rms: list[float] = []
        floor = 0.01  # provisional until calibrated
        speech_started = False
        speech_frames = 0
        trailing_silence = 0
        calibrate_frames = self.CALIBRATION_FRAMES
        pinned_threshold = self.vad_threshold or 0.0

        for i in range(max_frames):
            if stop_event is not None and stop_event.is_set():
                return np.zeros(0, dtype=np.float32)
            try:
                frame = self.read(timeout=max(1.0, frame_dur * 4))
            except queue.Empty:
                break
            collected.append(frame)

            rms = self.frame_rms(frame)
            if i < calibrate_frames:
                early_rms.append(rms)
                if i == calibrate_frames - 1:
                    floor = self.calibrate_floor(early_rms)

            is_speech = rms > (pinned_threshold or self.speech_threshold(floor))
            if is_speech:
                speech_started = True
                speech_frames += 1
                trailing_silence = 0
            elif speech_started:
                trailing_silence += 1
                if trailing_silence >= silence_frames:
                    break
            elif i >= grace_frames:
                break  # nobody said anything

        # a few loud frames is a cough / a door / JARVIS's own tail, not a
        # command — require a minimum amount of actual speech
        if not speech_started or speech_frames < min_speech_frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(collected).astype(np.float32) / 32768.0
