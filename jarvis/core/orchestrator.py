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
from typing import Literal

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

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    # -- speaking ---------------------------------------------------------

    async def _speak(self, text: str) -> None:
        if text:
            await asyncio.to_thread(self.tts.say, text)

    # -- wake -----------------------------------------------------------------

    async def _await_wake(self) -> bool:
        """Block until the wake word fires. False if the loop was stopped."""
        await asyncio.to_thread(self.wake.reset)
        while self.running:
            try:
                frame = await asyncio.to_thread(self.mic.read, 1.0)
            except queue.Empty:
                continue
            if await asyncio.to_thread(self.wake.triggered, frame):
                return True
        return False

    # -- one command --------------------------------------------------------

    async def _listen(self) -> str:
        audio = await asyncio.to_thread(
            self.mic.record_utterance,
            self.config.capture.window_timeout_s,
            self.config.capture.silence_s,
        )
        if len(audio) == 0:
            return ""
        text = await asyncio.to_thread(self.stt.transcribe, audio)
        return (text or "").strip()

    async def _turn(self) -> None:
        """One listening window: capture, understand, act. Assumes just woken."""
        self.state = "listening"
        text = await self._listen()
        if not text:
            # user said only the wake word — acknowledge and give them a beat
            await self._speak(self.persona.line("wake_ack"))
            self.state = "listening"
            text = await self._listen()
        if not text:
            return

        log.info("heard: %s", text)
        self.state = "thinking"
        label, confidence = await asyncio.to_thread(self.nlu.predict, text)
        log.info("intent: %s (%.2f)", label, confidence)
        await self.handle(label, text)

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
        try:
            while self.running:
                self.state = "idle"
                self._merge_gate()
                if not await self._await_wake():
                    break
                self.standby = False
                try:
                    await self._turn()
                finally:
                    self.state = "idle"
                    self.standby = True
        finally:
            self.running = False
            self.mic.stop()

    def stop(self) -> None:
        self.running = False
