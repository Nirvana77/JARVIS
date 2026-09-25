"""PCM playback, split out of ``Speaker.say`` (M3).

The edge has no Piper and no persona — it is handed finished PCM and has to put
it out of a speaker, in order, and stop the moment the button is pressed. That
is this file. ``Speaker`` uses it too, so there is one place that knows how to
open an output device and one place that gets the first-phoneme padding right.

Nothing here imports a model, which is what keeps the edge's install to numpy +
sounddevice + websockets.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from jarvis.audio.pipewire import pipewire_target

log = logging.getLogger(__name__)


class Player:
    """A persistent output stream and a way to cut it short.

    Reopening the device per utterance is what clips the first word, so one
    stream is kept open and reused; ``lead_pad_s`` of silence in front covers
    the device spinning up after a long idle.
    """

    #: written in slices this long, so `stop()` takes effect within one of them
    CHUNK_MS = 50

    def __init__(
        self,
        *,
        device: int | str | None = None,
        pipewire_node: str = "",
        lead_pad_s: float = 0.25,
    ) -> None:
        self.device = device
        self.pipewire_node = pipewire_node
        self.lead_pad_s = lead_pad_s
        self._out = None
        self._sample_rate = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- device ------------------------------------------------------------

    def _output(self, sample_rate: int):
        import sounddevice as sd

        if self._out is not None and self._sample_rate != sample_rate:
            self.close()
        if self._out is None:
            with pipewire_target(self.pipewire_node) as device:
                self._out = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    device=self.device if device is None else device,
                )
            self._out.start()
            self._sample_rate = sample_rate
        return self._out

    def close(self) -> None:
        if self._out is not None:
            try:
                self._out.stop()
                self._out.close()
            finally:
                self._out = None
                self._sample_rate = 0

    # -- playback ----------------------------------------------------------

    def play(self, pcm, sample_rate: int, *, pad: bool = True) -> bool:
        """Play int16 mono PCM. Blocks until it has been written or stopped.

        Returns True if it played to the end, False if it was stopped or the
        device refused — a headless box is a warning, never a crash.
        """
        pcm = np.asarray(pcm, dtype=np.int16)
        if pcm.size == 0:
            return True
        self._stop.clear()
        chunk = max(1, int(sample_rate * self.CHUNK_MS / 1000))
        try:
            with self._lock:
                out = self._output(sample_rate)
                if pad and self.lead_pad_s > 0:
                    out.write(np.zeros(int(sample_rate * self.lead_pad_s), dtype=np.int16))
                for at in range(0, pcm.size, chunk):
                    if self._stop.is_set():
                        return False
                    out.write(pcm[at : at + chunk])
        except Exception as exc:  # noqa: BLE001 — headless / no output device
            log.warning("audio playback failed: %s", exc)
            self.close()
            return False
        return not self._stop.is_set()

    def stop(self) -> None:
        """Cut playback short, now — the push-to-talk button during an answer.

        Safe to call from another thread: it aborts the device's queued audio
        and the write loop notices within one chunk.
        """
        self._stop.set()
        out = self._out
        if out is not None:
            try:
                out.abort()
            except Exception:  # noqa: BLE001 — already closed / never opened
                pass

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()
