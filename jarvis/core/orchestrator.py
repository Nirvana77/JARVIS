"""The orchestrator: a single async state machine driving the voice loop.

Replaces ``libs/command_helper.run()`` + the ``standby`` globals. One asyncio
loop owns the state; blocking work (mic, whisper, piper, wake) is pushed to a
thread with ``asyncio.to_thread``. Components are injected so tests can pass
fakes.

Turn:  wait for wake word -> open window -> record -> transcribe -> classify
       + slot-fill -> dispatch -> speak -> close window.

The ``_staged`` slot and ``_merge_gate`` are the seam for M2's seamless
hot-swap (retrain worker -> staged model -> swap only while idle); in M1 the
gate is a no-op.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import random
import sys
import threading
from typing import Literal

from jarvis.nlu import slots as _slots
from jarvis.nlu.classifier import UNKNOWN

log = logging.getLogger(__name__)

State = Literal["idle", "listening", "thinking", "acting", "speaking"]


class _Cancelled(Exception):
    """The user hit Enter to abandon the current listen / transcription."""

# intent tags handled inline by the orchestrator (not skills), and the action
# they map to. Everything else is either action "none" (canned reply) or a skill.
_META_ACTIONS = {
    "greeting": "start",
    "goodbye": "exit",
    "shutdown": "shutdown",
}


class Orchestrator:
    def __init__(
        self,
        *,
        config,
        wake,
        mic,
        stt,
        tts,
        nlu,
        persona,
        registry,
        intent_meta: dict,
        slot_extract=_slots.extract,
    ) -> None:
        self.config = config
        self.wake = wake
        self.mic = mic
        self.stt = stt
        self.tts = tts
        self.nlu = nlu
        self.persona = persona
        self.registry = registry
        self.intent_meta = intent_meta
        self._slot_extract = slot_extract

        self.state: State = "idle"
        self.standby = True
        self.running = False
        self._staged = None  # M2: a gate-passed replacement NLU

        # cancel / interrupt support (press Enter in the terminal)
        self.allow_interrupt = getattr(config.capture, "allow_interrupt", True)
        self._cancel_flag = threading.Event()   # seen by the capture worker thread
        self._interrupt: asyncio.Event | None = None  # wakes the transcribe race
        self._stt_lock = asyncio.Lock()         # serialise whisper calls
        self._draining: list[asyncio.Future] = []  # abandoned-but-still-running work
        self._stdin_fd: int | None = None

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    # -- cancel / interrupt ---------------------------------------------------

    def _clear_cancel(self) -> None:
        self._cancel_flag.clear()
        if self._interrupt is not None:
            self._interrupt.clear()

    def _on_stdin(self) -> None:
        """stdin became readable — the user pressed Enter. Flag a cancel."""
        try:
            data = os.read(self._stdin_fd, 4096)
        except (BlockingIOError, OSError):
            return
        if not data:  # EOF — stop watching
            if self._stdin_fd is not None:
                asyncio.get_running_loop().remove_reader(self._stdin_fd)
                self._stdin_fd = None
            return
        if self.state in ("listening", "thinking"):
            print("  (cancelling…)", flush=True)
        self._cancel_flag.set()
        if self._interrupt is not None:
            self._interrupt.set()

    async def _cancellable(self, awaitable, what: str):
        """Await ``awaitable`` but raise :class:`_Cancelled` if Enter is pressed.

        The underlying work (a whisper call holding ``_stt_lock``) is *not*
        killed — it runs to completion in the background so the model isn't
        touched concurrently; we just stop waiting for it and drop the result.
        """
        assert self._interrupt is not None
        work = asyncio.ensure_future(awaitable)
        intr = asyncio.ensure_future(self._interrupt.wait())
        try:
            await asyncio.wait({work, intr}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            intr.cancel()
        if work.done():
            return work.result()
        self._draining.append(work)
        raise _Cancelled(what)

    # -- speaking ---------------------------------------------------------

    async def _speak(self, text: str) -> None:
        if not text:
            return
        await asyncio.to_thread(self.tts.say, text)
        # Half-duplex: while JARVIS was speaking, the mic recorded its own
        # voice. Let the tail settle, then drop those frames so the wake word
        # and the VAD don't trigger on them. (Real echo cancellation / barge-in
        # is M4.)
        await asyncio.sleep(0.2)
        self._drain_mic()

    def _drain_mic(self) -> None:
        drain = getattr(self.mic, "drain", None)
        if callable(drain):
            drain()

    # -- wake -----------------------------------------------------------------

    async def _await_wake(self) -> bool:
        """Block until the wake word fires. False if the loop was stopped."""
        await asyncio.to_thread(self.wake.reset)
        self._drain_mic()
        while self.running:
            try:
                frame = await asyncio.to_thread(self.mic.read, 1.0)
            except queue.Empty:
                continue
            if await asyncio.to_thread(self.wake.triggered, frame):
                return True
        return False

    # -- one wake session --------------------------------------------------

    async def _transcribe(self, audio) -> str:
        async with self._stt_lock:  # never two whisper calls at once
            return await asyncio.to_thread(self.stt.transcribe, audio)

    async def _listen(self, window: float, grace: float | None = None) -> str:
        """Capture one utterance and transcribe it. Raises :class:`_Cancelled`
        if the user hits Enter during either step."""
        self._clear_cancel()
        audio = await asyncio.to_thread(
            self.mic.record_utterance,
            window,
            self.config.capture.silence_s,
            window if grace is None else grace,
            self._cancel_flag,
        )
        if self._cancel_flag.is_set():
            raise _Cancelled("listening")
        if len(audio) == 0:
            return ""
        self.state = "thinking"
        text = await self._cancellable(self._transcribe(audio), "transcription")
        return (text or "").strip()

    async def _session(self) -> None:
        """One wake: the first command plus any follow-ups spoken within
        ``capture.follow_up_s``, then a spoken transition to standby.

        The follow-up window is why JARVIS does not drop to standby the instant
        a command finishes — you can keep talking without saying the wake word
        again, and only a quiet ``follow_up_s`` ends the session.
        """
        acked = False
        window = self.config.capture.window_timeout_s
        grace = 2.0  # first capture bails fast if the user says nothing

        while self.running:
            self.state = "listening"
            try:
                text = await self._listen(window, grace)
            except _Cancelled as exc:
                print(f"  (cancelled {exc} — still listening)", flush=True)
                acked = True
                window = grace = self.config.capture.follow_up_s
                continue

            if not text:
                if not acked:
                    # only the wake word so far — acknowledge, then wait longer
                    await self._speak(self.persona.line("wake_ack"))
                    acked = True
                    grace = window
                    continue
                break  # follow-up window lapsed quietly

            acked = True
            self.state = "thinking"
            label = await self._classify_and_report(text)
            await self.handle(label, text)

            if not self.running or self.standby:
                return  # shutdown / explicit goodbye already handled the exit

            window = grace = self.config.capture.follow_up_s

        if self.running:
            self.standby = True
            await self._speak(self.persona.line("standby"))

    async def _classify_and_report(self, text: str) -> str:
        """Classify ``text`` and print what was heard / what it resolved to.

        Printed on every turn (not gated on -v) so it's obvious why JARVIS did
        what it did when recognition is shaky.
        """
        asr = getattr(self.stt, "last_avg_logprob", None)
        asr_note = f"  (asr {asr:+.2f})" if asr is not None else ""
        explain = getattr(self.nlu, "explain", None)
        if callable(explain):
            p = await asyncio.to_thread(explain, text)
            top = " · ".join(f"{lbl} {prob:.2f}" for lbl, prob in p.ranking[:3])
            print(f'  heard   : "{text}"{asr_note}')
            print(f"  intent  : {p.label}  (conf {p.confidence:.2f}, sim {p.similarity:.2f})")
            print(f"  ranked  : {top}")
            log.info("heard %r -> %s (%.2f)", text, p.label, p.confidence)
            return p.label
        label, confidence = await asyncio.to_thread(self.nlu.predict, text)
        print(f'  heard   : "{text}"   -> {label} ({confidence:.2f})')
        return label

    # -- dispatch (also the unit-test entry point) --------------------------

    async def handle(self, label: str, text: str) -> None:
        action = _META_ACTIONS.get(label, self._action_for(label))

        if label == UNKNOWN:
            await self._speak(self.persona.line("unknown"))
            return

        if action == "start":
            if self.standby:
                self.standby = False
                await self._speak(self.persona.line("greeting"))
            else:
                await self._speak(self.persona.line("already_awake"))
            return
        if action == "exit":
            self.standby = True
            await self._speak(self.persona.line("standby"))
            return
        if action == "shutdown":
            await self._speak(self.persona.line("shutdown"))
            self.running = False
            return

        if self.standby:
            log.info("ignoring %r while in standby", label)
            return

        if action == "none":
            meta = self.intent_meta.get(label)
            reply = random.choice(meta.responses) if meta and meta.responses else ""
            await self._speak(self.persona.phrase(reply) if reply else "")
            return

        # a skill
        self.state = "acting"
        params = self._slot_extract(label, text)
        try:
            line = await asyncio.to_thread(self.registry.dispatch, label, params)
        except KeyError:
            await self._speak(self.persona.line("skill_missing"))
            return
        except Exception as exc:  # noqa: BLE001 - a skill blew up
            log.exception("skill %s failed: %s", label, exc)
            await self._speak(self.persona.line("error"))
            return
        self.state = "speaking"
        await self._speak(self.persona.phrase(line))

    def _action_for(self, label: str) -> str:
        meta = self.intent_meta.get(label)
        return meta.action if meta is not None else label

    # -- M2 seam ----------------------------------------------------------

    def _merge_gate(self) -> None:
        """M2: swap ``self.nlu``/``self.registry`` for a staged replacement,
        but only while idle so an in-flight turn is never disrupted."""
        if self._staged is not None and self.state == "idle":
            old, self.nlu = self.nlu, self._staged
            self._staged = None
            close = getattr(old, "close", None)
            if callable(close):
                close()

    # -- main loop --------------------------------------------------------

    def _install_interrupt(self) -> None:
        self._interrupt = asyncio.Event()
        if not self.allow_interrupt:
            return
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            asyncio.get_running_loop().add_reader(fd, self._on_stdin)
            self._stdin_fd = fd
            print("(press Enter to cancel the current command)", flush=True)
        except (ValueError, OSError, NotImplementedError):
            self._stdin_fd = None  # no usable stdin / loop doesn't support it

    def _remove_interrupt(self) -> None:
        if self._stdin_fd is not None:
            try:
                asyncio.get_running_loop().remove_reader(self._stdin_fd)
            except (ValueError, OSError, RuntimeError):
                pass
            self._stdin_fd = None

    async def run(self) -> None:
        self.running = True
        self.mic.start()
        self._install_interrupt()
        try:
            while self.running:
                self.state = "idle"
                self._merge_gate()
                if not await self._await_wake():
                    break
                self.standby = False
                try:
                    await self._session()
                finally:
                    self.state = "idle"
                    self.standby = True
        finally:
            self.running = False
            self._remove_interrupt()
            if self._draining:  # let abandoned whisper calls finish
                await asyncio.gather(*self._draining, return_exceptions=True)
            self.mic.stop()
            close = getattr(self.tts, "close", None)
            if callable(close):
                close()

    def stop(self) -> None:
        self.running = False
