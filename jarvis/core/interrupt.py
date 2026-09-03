"""Command-interrupt utilities — a SHELVED EXTRA, layered on the core loop.

This is M4 groundwork landed early as an opt-in add-on; it does **not** change
M1's scope or status. Two ways to abandon a command mid-transcription:

  * **Enter-to-cancel** (``capture.allow_interrupt``, TTY) — cheap and reliable,
    on by default.
  * **voice barge-in** (``capture.barge_in``, **off by default**) — talk over
    the transcription and the new sentence takes over. Needs per-mic VAD tuning
    (``jarvis mic`` / ``capture.barge_in_threshold``). Only armed during
    transcription, not during TTS (self-echo needs AEC — the real M4 version).

The orchestrator owns one :class:`Interrupter` and calls
:meth:`Interrupter.guard_transcription` around each decode. With both features
disabled it is a thin ``await work``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading

import numpy as np

from jarvis.audio.capture import Microphone

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """The user asked to abandon the current listen / transcription (Enter)."""


class Interrupter:
    def __init__(self, config) -> None:
        cap = config.capture
        self.enabled_enter: bool = getattr(cap, "allow_interrupt", True)
        self.enabled_barge: bool = getattr(cap, "barge_in", False)
        self._frame_ms = cap.frame_ms
        self._silence_s = cap.silence_s
        self._window_s = cap.window_timeout_s
        self._min_speech_s = getattr(cap, "barge_in_min_speech_s", 0.6)
        self._pinned_thr = getattr(cap, "barge_in_threshold", 0.0) or 0.0

        self.cancel_flag = threading.Event()   # Enter -> capture worker should stop
        self._barge_stop = threading.Event()   # stop the onset detector
        self._interrupt: asyncio.Event | None = None  # Enter -> wake the race
        self._draining: list[asyncio.Future] = []      # abandoned decodes
        self._stdin_fd: int | None = None

    @property
    def any_enabled(self) -> bool:
        return self.enabled_enter or self.enabled_barge

    @property
    def _onset_frames(self) -> int:
        return max(3, round(self._min_speech_s / (self._frame_ms / 1000)))

    # -- Enter watcher -----------------------------------------------------

    def install(self) -> None:
        """Create the asyncio event and, on a TTY, start watching stdin."""
        self._interrupt = asyncio.Event()
        if not self.enabled_enter:
            return
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            asyncio.get_running_loop().add_reader(fd, self._on_stdin)
            self._stdin_fd = fd
            print("(press Enter to cancel the current command)", flush=True)
        except (ValueError, OSError, NotImplementedError):
            self._stdin_fd = None

    def remove(self) -> None:
        if self._stdin_fd is not None:
            try:
                asyncio.get_running_loop().remove_reader(self._stdin_fd)
            except (ValueError, OSError, RuntimeError):
                pass
            self._stdin_fd = None

    def _on_stdin(self) -> None:
        try:
            data = os.read(self._stdin_fd, 4096)
        except (BlockingIOError, OSError):
            return
        if not data:  # EOF — stop watching
            self.remove()
            return
        print("  (cancelling…)", flush=True)
        self.cancel_flag.set()
        if self._interrupt is not None:
            self._interrupt.set()

    def clear(self) -> None:
        self.cancel_flag.clear()
        if self._interrupt is not None:
            self._interrupt.clear()

    def cancelled_during_capture(self) -> bool:
        return self.cancel_flag.is_set()

    # -- onset detector (voice barge-in) ---------------------------------

    def _onset_worker(self, mic):
        """Blocking: read mic frames until a sustained run of speech
        (``_onset_frames``). Return the collected audio (float32, short lead-in),
        or ``None`` if ``_barge_stop`` is set first."""
        need = self._onset_frames
        pinned = self._pinned_thr
        collected: list = []
        early: list[float] = []
        floor = 0.012
        run = 0
        idx = 0
        peak = 0.0
        dbg = log.isEnabledFor(logging.DEBUG)
        if dbg:
            log.debug("barge-in: listening (need %d speech frames)", need)
        while not self._barge_stop.is_set():
            try:
                frame = mic.read(0.2)
            except queue.Empty:
                continue
            collected.append(frame)
            rms = Microphone.frame_rms(frame)
            peak = max(peak, rms)
            if idx < 8:
                early.append(rms)
                if idx == 7:
                    floor = Microphone.calibrate_floor(early)
            idx += 1
            thr = pinned or Microphone.speech_threshold(floor)
            if dbg and idx % 6 == 0:
                log.debug("barge-in: rms=%.3f thr=%.3f run=%d", rms, thr, run)
            if rms > thr:
                run += 1
                if run >= need:
                    if dbg:
                        log.debug("barge-in: FIRED at frame %d (peak %.3f)", idx, peak)
                    return np.concatenate(collected).astype(np.float32) / 32768.0
            else:
                run = 0
                if len(collected) > 25:  # bounded pre-roll during silence
                    collected = collected[-10:]
        if dbg:
            thr = pinned or Microphone.speech_threshold(floor)
            log.debug(
                "barge-in: stopped without firing after %d frames (peak %.3f, thr %.3f)",
                idx, peak, thr,
            )
        return None

    # -- the guard ------------------------------------------------------

    async def guard_transcription(
        self, work: asyncio.Future, stt_cancel: threading.Event, mic
    ):
        """Await the transcription ``work`` while watching for an interrupt.

        Returns ``("done", text)`` when transcription finishes, or
        ``("barge", new_audio)`` when the user talked over it (``work`` is
        abandoned). Raises :class:`Cancelled` on Enter. With no feature enabled
        this is just ``await work``.
        """
        waiters = {work}
        onset = None
        if self.enabled_barge:
            self._barge_stop.clear()
            onset = asyncio.ensure_future(asyncio.to_thread(self._onset_worker, mic))
            waiters.add(onset)
        intr = None
        if self._interrupt is not None:
            intr = asyncio.ensure_future(self._interrupt.wait())
            waiters.add(intr)

        if len(waiters) == 1:  # nothing to race
            return ("done", (await work or "").strip())

        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

        if intr is not None and intr.done():
            stt_cancel.set()
            work.cancel()
            self._draining.append(work)
            self._barge_stop.set()
            if onset is not None:
                await asyncio.gather(onset, return_exceptions=True)
            raise Cancelled("transcription")
        if intr is not None:
            intr.cancel()

        if (
            onset is not None
            and onset.done()
            and not onset.cancelled()
            and onset.result() is not None
        ):
            onset_audio = onset.result()
            stt_cancel.set()
            work.cancel()
            self._draining.append(work)
            print("  (barge-in — go on)", flush=True)
            rest = await asyncio.to_thread(
                mic.record_utterance,
                self._window_s,
                self._silence_s,
                0.6,  # short grace — they're already speaking
                self.cancel_flag,
                1,  # the onset already counts as speech
            )
            new_audio = (
                np.concatenate([onset_audio, rest])
                if rest is not None and len(rest)
                else onset_audio
            )
            return ("barge", new_audio)

        # nothing barged in -> take the transcription
        self._barge_stop.set()
        if onset is not None:
            await asyncio.gather(onset, return_exceptions=True)
        if not work.done():
            await work
        return ("done", (work.result() or "").strip())

    async def shutdown(self) -> None:
        self.remove()
        self._barge_stop.set()
        for d in self._draining:
            d.cancel()
        if self._draining:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._draining, return_exceptions=True),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                pass
