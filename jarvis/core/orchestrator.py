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
import queue
import random
import threading
from typing import Literal

from jarvis.core.interrupt import Cancelled as _Cancelled
from jarvis.core.interrupt import Interrupter
from jarvis.nlu import slots as _slots
from jarvis.nlu.classifier import UNKNOWN

log = logging.getLogger(__name__)

State = Literal["idle", "listening", "thinking", "acting", "speaking"]

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

        # Optional mid-command interrupt (Enter, and off-by-default voice
        # barge-in). Self-contained utility — see jarvis/core/interrupt.py.
        self._interrupter = Interrupter(config)

    @property
    def busy(self) -> bool:
        return self.state != "idle"

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

    async def _transcribe(self, audio, cancel: threading.Event) -> str:
        # No lock: faster-whisper / CTranslate2 tolerate concurrent calls, and an
        # abandoned decode must not block the new one. `cancel` stops it between
        # decoded segments (a single segment can't be halted mid-decode).
        return await asyncio.to_thread(self.stt.transcribe, audio, cancel)

    async def _capture(self, window: float, grace: float, min_speech: int = 3):
        """Record one utterance. Returns the float32 audio, or raises
        :class:`_Cancelled` if Enter was pressed."""
        self._interrupter.clear()
        self.state = "listening"
        audio = await asyncio.to_thread(
            self.mic.record_utterance,
            window,
            self.config.capture.silence_s,
            grace,
            self._interrupter.cancel_flag,
            min_speech,
        )
        if self._interrupter.cancelled_during_capture():
            raise _Cancelled("listening")
        return audio

    async def _next_utterance(self, window: float, grace: float) -> str:
        """Capture a command and transcribe it. An interrupt (Enter, or opt-in
        voice barge-in) is handled by ``self._interrupter``. Returns the
        transcript, or "" for silence."""
        audio = await self._capture(window, grace)
        if audio is None or len(audio) == 0:
            return ""

        while True:
            self.state = "thinking"
            stt_cancel = threading.Event()
            work = asyncio.ensure_future(self._transcribe(audio, stt_cancel))
            kind, value = await self._interrupter.guard_transcription(
                work, stt_cancel, self.mic
            )
            if kind == "barge":
                self.state = "listening"
                audio = value
                continue
            return value

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
            try:
                text = await self._next_utterance(window, grace)
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

    async def run(self) -> None:
        self.running = True
        self.mic.start()
        self._interrupter.install()
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
            await self._interrupter.shutdown()
            self.mic.stop()
            close = getattr(self.tts, "close", None)
            if callable(close):
                close()

    def stop(self) -> None:
        self.running = False
