"""The orchestrator: a single async state machine driving the voice loop.

Replaces ``libs/command_helper.run()`` + the ``standby`` globals. One asyncio
loop owns the state; blocking work (mic, whisper, piper, wake) is pushed to a
thread with ``asyncio.to_thread``. Components are injected so tests can pass
fakes.

Turn:  wait for wake word -> open window -> record -> transcribe -> classify
       + slot-fill -> dispatch -> speak -> close window.

The ``_staged`` slot and ``_merge_gate`` are M2's seamless hot-swap (retrain
worker -> staged model -> swap only while idle). M2.5 moves everything after
the teach/edit/revert dialog into background learning jobs; their questions
and notices are spoken only at safe points (``_safe_point``) between turns.

M4: a transcript the NLU gives up on gets one more move before "didn't catch
that" — the local reasoner's guess at what was said, confirmed by voice and
then classified like any other command (``_unclear``).

M4.5: a sentence that is several commands is run as several (``_compound``,
no model needed); a turn that is still unclear after M4 is planned or answered
by the reasoner (``_think``); and every turn is remembered under the device
it came from (``_turn``, ``jarvis/core/memory.py``).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import queue
import random
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jarvis.core import mishear as _mishear
from jarvis.core import reasoning as _reasoning
from jarvis.core.interrupt import Cancelled as _Cancelled
from jarvis.core.interrupt import Interrupter
from jarvis.core.memory import LOCAL, Memory, is_device_id
from jarvis.factory import flows as _flows
from jarvis.factory.flows import (
    EditSkillFlow,
    FlowOutcome,
    LearningRequest,
    RemoveSkillFlow,
    RevertSkillFlow,
    TeachFlow,
    ask_yes_no,
    ask_yes_no_or_none,
)
from jarvis.factory.jobs import LearningJob, run_detached
from jarvis.nlu import compound as _compound
from jarvis.nlu import slots as _slots
from jarvis.nlu.classifier import UNKNOWN, Classifier
from jarvis.nlu.corpus import build_corpus
from jarvis.nlu.retrain_worker import RetrainWorker
from jarvis.skills.registry import LEARNED_PACKAGE

#: a retrain normally takes seconds; past this the worker is presumed stuck
#: and the job is set aside out loud rather than waiting in silence forever
RETRAIN_TIMEOUT_S = 300.0

log = logging.getLogger(__name__)

State = Literal["idle", "listening", "thinking", "acting", "speaking"]

# intent tags handled inline by the orchestrator (not skills), and the action
# they map to. Everything else is either action "none" (canned reply) or a skill.
_META_ACTIONS = {
    "greeting": "start",
    "goodbye": "exit",
    "shutdown": "shutdown",
    "teach": "teach",
    "edit_skill": "edit_skill",
    "revert_skill": "revert_skill",
    "remove_skill": "remove_skill",
    "recall_memory": "recall_memory",
    "forget_memory": "forget_memory",
}

#: A wrong guess at one of these costs a dialog, a skill's code or a device's
#: memory, so they need `nlu.meta_action_threshold`, not just "not unknown".
_GUARDED_ACTIONS = ("teach", "edit_skill", "revert_skill", "remove_skill", "forget_memory")

#: Skills whose argument is whatever was said: never one step of a chain
#: ("note that buy milk and call mum" is one note).
_DICTATION = frozenset({"note"})

#: how many facts "what do you remember?" reads out, newest last
_RECALL_SPOKEN = 5

#: `_reason_about_unclear`: the reasoner was asked and did not answer
_FAILED = object()

# NLU self-check probes run against a freshly-trained model before it's ever
# staged — a handful of the seed intents plus the new/changed skill's own
# examples. Keeps a bad retrain (e.g. one skill's examples drowning out the
# rest) from ever reaching a live conversation.
_SELF_CHECK_SEED_PROBES = (
    ("search black holes", "search"),
    ("open github", "open_app"),
    ("go to sleep", "goodbye"),
)

#: a background question answered unclearly this many times counts as "no"
_MAX_UNCLEAR_ANSWERS = 3


@dataclass
class _Decision:
    """A background job's yes/no question, waiting for a safe point."""

    prompt: str
    future: asyncio.Future
    unclear: int = 0


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
        reasoner=None,
        memory: Memory | None = None,
        claude_client=None,
        sandbox=None,
        train_and_load=None,
        allow_shutdown: bool = True,
        knowledge=None,
    ) -> None:
        self.config = config
        #: False under `serve`: the brain is a server, and "shut down" said to
        #: an edge only stands it by — nobody is at the AI PC to restart it.
        self.allow_shutdown = allow_shutdown
        self.wake = wake
        self.mic = mic
        self.stt = stt
        self.tts = tts
        self.nlu = nlu
        self.persona = persona
        self.registry = registry
        self.intent_meta = intent_meta
        self._slot_extract = slot_extract
        # M4: the optional local LLM (Ollama). Only ever asked for a guess at
        # a misheard command, which is then confirmed and re-classified —
        # never for understanding or dispatch. None / unavailable: skipped.
        self.reasoner = reasoner
        # M4.5: per-device memory — the recent conversation and lasting
        # facts. Without one handed in (tests), facts stay in RAM.
        self.memory = memory if memory is not None else Memory(
            None,
            turns=config.memory.turns,
            idle_forget_s=config.memory.idle_forget_s,
            max_facts=config.memory.max_facts,
        )
        #: what has been said back in the turn in flight (None between turns)
        self._turn_said: list[str] | None = None
        # M2: the skill factory. Both are best-effort — a missing Anthropic key
        # or sandbox degrades `teach`/`edit_skill` to a spoken "can't do that
        # right now", never a hard crash (same pattern as the reasoner).
        self.claude_client = claude_client
        self.sandbox = sandbox
        # M2.5: `async (examples) -> ("ok", (TrainResult, classifier)) |
        # ("error", message)`. Default: a worker-process retrain + load;
        # tests inject a fake (no process, no fastembed).
        self._train_and_load = train_and_load or self._default_train_and_load

        self.state: State = "idle"
        self.standby = True
        self.running = False
        self._staged = None  # M2: a gate-passed replacement (nlu, registry)
        self._pending_announcement: str | None = None
        # M2.5: background learning. `_jobs` is name -> task in queue order;
        # jobs run one at a time (each waits for `_last_job`).
        self._jobs: dict[str, asyncio.Task] = {}
        self._last_job: asyncio.Task | None = None
        self._notices: list[str] = []
        self._decisions: list[_Decision] = []
        self._in_session = False
        self._idle_task: asyncio.Task | None = None
        # M5: the knowledge base (or None when it is off). Skills reach it
        # through the registry; the orchestrator only runs its re-scan task.
        self.knowledge = knowledge
        self._knowledge_task: asyncio.Task | None = None

        # Optional mid-command interrupt (Enter, and off-by-default voice
        # barge-in). Self-contained utility — see jarvis/core/interrupt.py.
        self._interrupter = Interrupter(config)

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    @property
    def interrupter(self) -> Interrupter:
        """M3: the remote link cancels the current listen through this, so the
        edge's ``interrupt`` takes exactly the path Enter takes."""
        return self._interrupter

    # -- speaking ---------------------------------------------------------

    async def _speak(self, text: str) -> None:
        if not text:
            return
        if self._turn_said is not None:
            self._turn_said.append(text)
        await asyncio.to_thread(self.tts.say, text)
        # Half-duplex: while JARVIS was speaking, the mic recorded its own
        # voice. Let the tail settle, then drop those frames so the wake word
        # and the VAD don't trigger on them. (Real echo cancellation / barge-in
        # is M6.)
        await asyncio.sleep(0.2)
        self._drain_mic()

    async def _enter_standby(self) -> None:
        """Drop to standby, and tell the speaker *before* saying so.

        A remote edge on a battery (the watch) powers down on standby. The
        spoken line is persona text — and "Standing by, sir." is also a wake
        acknowledgement — so it gets a signal it can trust instead, early
        enough to skip playing the line. Speakers without the hook (the local
        one, the fakes) are unaffected.
        """
        self.standby = True
        notify = getattr(self.tts, "entering_standby", None)
        if callable(notify):
            await asyncio.to_thread(notify)
        await self._speak(self.persona.line("standby"))

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

    async def _ask(self, prompt: str) -> str:
        """M2: speak a follow-up question and capture the reply, reusing the
        normal capture/transcribe path (no wake word needed — used by the
        teach/edit_skill/revert_skill dialogs)."""
        await self._speak(prompt)
        text = await self._next_utterance(
            self.config.capture.window_timeout_s, self.config.capture.follow_up_s
        )
        # Unlike the always-on line for a normal command, this is verbose-only
        # (-v/--debug) — these free-text dialog answers aren't classified, so
        # there's no intent/confidence to show, just the raw transcription.
        if log.isEnabledFor(logging.INFO):
            self._print_heard(text)
        return text

    async def _ask_yes_no(self, prompt: str) -> bool:
        return await ask_yes_no(self._ask, prompt)

    async def _confirmed(self, prompt: str) -> bool:
        """A clear, whole-word yes — nothing else. For anything that would
        act on a guess or destroy something: "I'm unsure" has "sure" in it
        and "incorrect" has "correct", and neither is consent."""
        return await ask_yes_no_or_none(self._ask, prompt) is True

    def _interrupted(self) -> bool:
        """Enter, or the edge's button, since the last listen began (each
        capture clears it). A turn that takes several steps checks between
        them, because nothing else would."""
        return self._interrupter.cancel_flag.is_set()

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
            label, confidence = await self._classify_and_report(text)
            await self._turn(label, text, confidence)

            if not self.running or self.standby:
                return  # shutdown / explicit goodbye already handled the exit

            # Safe point: no turn is in flight between here and the next
            # capture. A staged retrain is promoted now, not just at the outer
            # wake-session boundary (PRD/milestone-2-skill-factory.md decision
            # 1), and background learning jobs get their notices spoken and
            # their questions asked (PRD/milestone-2.5-background-learning.md).
            await self._safe_point()

            window = grace = self.config.capture.follow_up_s

        if self.running:
            # before the drop to standby: a finished job's question shouldn't
            # have to wait for the next wake word
            await self._safe_point()
            await self._enter_standby()

    def _print_heard(self, text: str) -> None:
        asr = getattr(self.stt, "last_avg_logprob", None)
        asr_note = f"  (asr {asr:+.2f})" if asr is not None else ""
        print(f'  heard   : "{text}"{asr_note}')

    async def _classify_and_report(self, text: str) -> tuple[str, float]:
        """Classify ``text`` and print what was heard / what it resolved to.

        Printed on every turn (not gated on -v) so it's obvious why JARVIS did
        what it did when recognition is shaky.
        """
        explain = getattr(self.nlu, "explain", None)
        if callable(explain):
            p = await asyncio.to_thread(explain, text)
            top = " · ".join(f"{lbl} {prob:.2f}" for lbl, prob in p.ranking[:3])
            self._print_heard(text)
            print(f"  intent  : {p.label}  (conf {p.confidence:.2f}, sim {p.similarity:.2f})")
            print(f"  ranked  : {top}")
            log.info("heard %r -> %s (%.2f)", text, p.label, p.confidence)
            return p.label, p.confidence
        label, confidence = await asyncio.to_thread(self.nlu.predict, text)
        print(f'  heard   : "{text}"   -> {label} ({confidence:.2f})')
        return label, confidence

    # -- dispatch (also the unit-test entry point) --------------------------

    def _device(self) -> str:
        """Which device this turn came from: the edge's id under `serve`
        (``RemoteLink.device_id``), else ``local``."""
        device = getattr(self.mic, "device_id", None)
        return device if is_device_id(device) else LOCAL

    async def _turn(self, label: str, text: str, confidence: float) -> None:
        """One turn of a session: handled, then remembered — what was heard
        and what was said back — under the device it came from.

        A cancel (Enter, the edge's button) during a question asked from
        inside the turn ends the turn, not the loop."""
        device = self._device()
        self.memory.focus(device)
        self._turn_said = []
        try:
            await self.handle(label, text, confidence)
        except _Cancelled as exc:
            print(f"  (cancelled {exc} — still listening)", flush=True)
        finally:
            said, self._turn_said = self._turn_said, None
            if said is not None:  # None: the turn was "forget everything"
                self.memory.device(device).add_turn(text, " ".join(said))

    async def handle(self, label: str, text: str, confidence: float = 1.0) -> None:
        steps = await self._compound(text)
        if steps:
            await self._run_steps(steps)
            return
        await self._handle_one(label, text, confidence)

    async def _handle_one(self, label: str, text: str, confidence: float = 1.0) -> None:
        action = _META_ACTIONS.get(label, self._action_for(label))

        if label == UNKNOWN:
            await self._unclear(text)
            return

        if action == "start":
            if self.standby:
                self.standby = False
                await self._speak(self.persona.line("greeting"))
            else:
                await self._speak(self.persona.line("already_awake"))
            return
        if action == "exit":
            await self._enter_standby()
            return
        if action == "shutdown":
            if not self.allow_shutdown:
                log.info("shutdown asked for by voice on a remote brain: standing by instead")
                await self._enter_standby()
                return
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

        if self._too_unsure_for_meta(label, confidence):
            # A wrong guess here launches a whole multi-turn dialog (or, for
            # revert_skill/remove_skill, rewrites or tears down a skill's live
            # code) — costlier than a wrong guess on an ordinary skill, so a
            # merely-above-"unknown" confidence isn't enough to commit to it.
            log.info(
                "treating low-confidence %s (%.2f < %.2f) as unknown",
                label, confidence, self.config.nlu.meta_action_threshold,
            )
            await self._unclear(text)
            return

        if action in ("teach", "edit_skill"):
            self.state = "acting"
            if self.claude_client is None or not self.claude_client.available:
                await self._speak("I can't learn or change skills right now, sir — no factory available.")
                return
            flow_cls = TeachFlow if action == "teach" else EditSkillFlow
            await self._run_flow(
                flow_cls(
                    ask=self._ask, say=self._speak, registry=self._latest_registry(),
                    busy_names=frozenset(self._jobs),
                )
            )
            return

        if action == "revert_skill":
            # Doesn't touch Claude — it restores already-approved code — so it
            # doesn't need the factory to be available.
            self.state = "acting"
            await self._run_flow(
                RevertSkillFlow(
                    ask=self._ask, say=self._speak, registry=self._latest_registry(),
                    config=self.config, busy_names=frozenset(self._jobs),
                )
            )
            return

        if action == "remove_skill":
            # Also Claude-free — it only quarantines an already-approved skill
            # and retrains without it.
            self.state = "acting"
            await self._run_flow(
                RemoveSkillFlow(
                    ask=self._ask, say=self._speak, registry=self._latest_registry(),
                    busy_names=frozenset(self._jobs),
                )
            )
            return

        if action == "recall_memory":
            await self._recall()
            return
        if action == "forget_memory":
            await self._forget()
            return

        # a skill
        self.state = "acting"
        params = await self._fill_missing(label, self._params_for(label, text))
        if params is None:
            return
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

    # -- M4: misheard-command reasoning ---------------------------------------

    async def _unclear(self, text: str) -> None:
        """The NLU gave up on ``text``. Before saying so:

        1. M4 — is it a mishearing of something JARVIS knows? One guess, one
           confirmation; a "no" gets the plain line, not a second guess.
        2. M4.5 — otherwise, what does the speaker want? The reasoner may
           re-say it as commands (confirmed, then run in order) or answer it.

        Nothing the reasoner said is acted on before a spoken yes, and every
        command goes through the normal path."""
        confirm = self.persona.line("did_you_mean", "Did you mean '{guess}'?")

        corrected = await self._reason_about_unclear(text)
        if self._interrupted():
            print("  (cancelled thinking — still listening)", flush=True)
            return
        if corrected is _FAILED:
            # the reasoner did not answer in time: not a second, longer wait
            await self._speak(self.persona.line("unknown"))
            return
        if corrected is not None:
            guess, label, confidence = corrected
            if await self._confirmed(confirm.replace("{guess}", guess)):
                await self.handle(label, guess, confidence)
            else:
                await self._speak(self.persona.line("unknown"))
            return

        thought = await self._think(text)
        if self._interrupted():
            print("  (cancelled thinking — still listening)", flush=True)
            return
        if isinstance(thought, str):
            # already in the persona's voice: its character was in the prompt
            await self._speak(thought)
            return
        if thought:
            said = ", then ".join(step for step, _label, _confidence in thought)
            if await self._confirmed(confirm.replace("{guess}", said)):
                await self._run_steps(thought)
                return
        await self._speak(self.persona.line("unknown"))

    async def _reason_about_unclear(self, text: str):
        """The reasoner's single best guess at the command that was actually
        said, as ``(corrected text, label, confidence)`` — the label and
        confidence being the real classifier's verdict on the corrected text,
        not anything the reasoner said. ``None`` when there is no reasoner,
        it finds nothing plausible, or its guess is not a command JARVIS
        would act on anyway; ``_FAILED`` when it was asked and did not answer
        (down, or past ``guess_timeout_s``)."""
        reasoner = self.reasoner
        if (
            reasoner is None
            or not reasoner.available
            or not self.config.reasoner.correct_misheard
            or not text.strip()
        ):
            return None
        self.state = "thinking"
        known = _mishear.vocabulary(self.registry.manifests(), self.intent_meta)
        # Its own thread rather than `asyncio.to_thread`: it is gone when the
        # guess is, so the thread count is back at its baseline after the
        # turn, and a slow Ollama never holds up shutdown. (The loop's
        # executor grows a worker whenever a call is followed at once by
        # another, which is exactly what guess-then-speak is.)
        try:
            found = await run_detached(self._guess, text, known, name="mishear-guess")
        except Exception as exc:  # noqa: BLE001 - degradation is the point
            log.warning("mishearing guess failed, treating as unknown: %s", exc)
            return _FAILED
        if found is None:
            return None
        guess, label, confidence = found
        print(f'  guess   : "{guess}"   -> {label} ({confidence:.2f})')
        # Not worth a question unless the answer leads somewhere: the guess
        # has to be something the classifier itself recognises.
        if label == UNKNOWN or self._too_unsure_for_meta(label, confidence):
            log.info("guess %r is no clearer to the NLU (%s %.2f)", guess, label, confidence)
            return None
        return found

    def _guess(self, text: str, known: list[str]) -> tuple[str, str, float] | None:
        """Blocking half of :meth:`_reason_about_unclear`: ask the reasoner,
        keep only a reply that could be a mishearing of ``text``, and have
        the classifier say what that corrected phrase is."""
        reply = self.reasoner.generate(
            _mishear.SYSTEM,
            _mishear.build_prompt(text, known),
            temperature=0.0,
            timeout=self.config.reasoner.guess_timeout_s,
        )
        guess = _mishear.parse_guess(reply, text)
        if guess is None:
            log.info("no plausible mishearing for %r (reasoner said %r)", text, reply)
            return None
        if _mishear.soundalike(guess, text) < _mishear.MIN_SOUNDALIKE:
            log.info("guess %r sounds too little like %r to be a mishearing", guess, text)
            return None
        label, confidence = self.nlu.predict(guess)
        return guess, label, confidence

    def _too_unsure_for_meta(self, label: str, confidence: float) -> bool:
        action = _META_ACTIONS.get(label, self._action_for(label))
        return (
            action in _GUARDED_ACTIONS
            and confidence < self.config.nlu.meta_action_threshold
        )

    # -- M4.5: planning and answering -------------------------------------------

    async def _think(self, text: str) -> str | list[tuple[str, str, float]] | None:
        """What the reasoner makes of a turn that is neither a command nor a
        mishearing of one: an **answer** to speak (a ``str``), or the request
        re-said as **commands** — ``(text, label, confidence)`` each, the
        label and confidence being the real classifier's — or ``None``.

        A plan is all or nothing: one step the NLU does not recognise, or a
        chain with anything but skills in it, and there is no plan."""
        reasoner, options = self.reasoner, self.config.reasoner
        if (
            reasoner is None
            or not reasoner.available
            or not (options.answer_questions or options.plan_commands)
            or not text.strip()
        ):
            return None
        self.state = "thinking"
        memory = self.memory.device()
        character = getattr(self.persona, "character", None)
        system = _reasoning.system(character() if callable(character) else "")
        prompt = _reasoning.build_prompt(
            text,
            _mishear.vocabulary(self.registry.manifests(), self.intent_meta),
            memory.facts(),
            memory.turns(),
            _dt.datetime.now(),
        )
        try:  # its own thread, for the reasons `_reason_about_unclear` gives
            found = await run_detached(self._reason, system, prompt, name="reason")
        except Exception as exc:  # noqa: BLE001 - degradation is the point
            log.warning("reasoning failed, treating as unknown: %s", exc)
            return None
        if found is None:
            return None

        if isinstance(found, str):
            if not options.answer_questions:
                return None
            print("  answer  : (the reasoner's own)")
            return found

        print("  plan    : " + " · ".join(f'"{t}" -> {lbl} ({c:.2f})' for t, lbl, c in found))
        if not options.plan_commands:
            return None
        for step, label, confidence in found:
            if label == UNKNOWN or self._too_unsure_for_meta(label, confidence):
                log.info("plan dropped: %r is no clearer to the NLU (%s %.2f)", step, label, confidence)
                return None
        if len(found) > 1 and not all(self._chainable(label) for _s, label, _c in found):
            log.info("plan dropped: a chain may only contain skills")
            return None
        if len(found) == 1 and _mishear.same_words(found[0][0], text):
            return None  # it only repeated what was heard
        return found

    def _reason(self, system: str, prompt: str) -> str | list[tuple[str, str, float]] | None:
        """Blocking half of :meth:`_think`: ask, parse, and have the
        classifier say what each planned command is."""
        reply = self.reasoner.generate(
            system,
            prompt,
            temperature=0.2,
            timeout=self.config.reasoner.reason_timeout_s,
            format="json",
        )
        thought = _reasoning.parse(reply)
        if thought is None:
            log.info("the reasoner had nothing for it (said %r)", reply)
            return None
        if thought.answer:
            return thought.answer
        return [(step, *self.nlu.predict(step)) for step in thought.commands]

    # -- M4.5: several commands in one turn --------------------------------------

    def _chainable(self, label: str) -> bool:
        """May ``label`` be one step of several? Skills only — no dialog, no
        session action, no dictation."""
        if label in _META_ACTIONS or label in _DICTATION:
            return False
        try:
            return label in self.registry.names()
        except Exception:  # noqa: BLE001 — a fake without a roster
            return False

    async def _compound(self, text: str) -> list[tuple[str, str, float]] | None:
        """``text`` as several commands — ``(clause, label, confidence)`` each
        — when it is a sentence joined by "and" / "then" / a comma and every
        clause is, on its own, a different skill the classifier is sure of.
        These are the speaker's own words classified by the real NLU: nothing
        is guessed, so nothing is asked."""
        if self.standby or not self.config.nlu.compound:
            return None
        clauses = _compound.split(text)
        if not clauses:
            return None
        # its own thread, not the loop's executor: classify-then-dispatch is
        # the back-to-back pair that makes the executor grow a worker
        verdicts = await run_detached(self._classify_clauses, clauses, name="compound")
        previous = None
        for _clause, label, confidence, similarity in verdicts:
            if (
                label == previous
                or not self._chainable(label)
                or confidence < _compound.MIN_CONFIDENCE
                or similarity < _compound.MIN_SIMILARITY
            ):
                return None
            previous = label
        return [(clause, label, confidence) for clause, label, confidence, _sim in verdicts]

    def _classify_clauses(self, clauses: list[str]) -> list[tuple[str, str, float, float]]:
        explain = getattr(self.nlu, "explain", None)
        out = []
        for clause in clauses:
            if callable(explain):
                p = explain(clause)
                out.append((clause, p.label, p.confidence, p.similarity))
            else:
                label, confidence = self.nlu.predict(clause)
                out.append((clause, label, confidence, 1.0))
        return out

    async def _run_steps(self, steps: list[tuple[str, str, float]]) -> None:
        """Run commands in order, each through the normal single-command
        path, so each fills its own slots and speaks its own line."""
        print("  steps   : " + " · ".join(f'"{t}" -> {lbl} ({c:.2f})' for t, lbl, c in steps))
        for n, (step, label, confidence) in enumerate(steps):
            if n and self._interrupted():
                print("  (cancelled — the rest is dropped)", flush=True)
                return
            await self._handle_one(label, step, confidence)

    # -- M4.5: what this device was asked to remember -----------------------------

    async def _recall(self) -> None:
        facts = self.memory.device().facts()
        if not facts:
            await self._speak(self.persona.line(
                "nothing_remembered", "You haven't asked me to remember anything."))
            return
        shown = facts[-_RECALL_SPOKEN:]
        line = "You asked me to remember: " + "; ".join(f.rstrip(" .") for f in shown) + "."
        if len(facts) > len(shown):
            line += f" And {len(facts) - len(shown)} more before that."
        # spoken as it is, not rephrased: these are the speaker's own words
        await self._speak(line)

    def _forget_in_knowledge(self, refs: list[str]) -> None:
        for ref in refs:
            self.knowledge.store.remove_source(ref)

    async def _forget(self) -> None:
        memory = self.memory.device()
        refs = memory.knowledge_refs()
        if not memory.facts() and not memory.turns() and not refs:
            await self._speak(self.persona.line(
                "nothing_remembered", "You haven't asked me to remember anything."))
            return
        prompt = self.persona.line("forget_confirm", "Forget everything you've asked me to remember?")
        if await self._confirmed(prompt):
            try:
                # M5: what this device said to `remember` is in the knowledge
                # base too — out of there first, so a failure leaves both
                if refs and self.knowledge is not None:
                    await asyncio.to_thread(self._forget_in_knowledge, refs)
                memory.forget()
            except Exception as exc:  # noqa: BLE001 — still stored: do not say otherwise
                log.error("could not forget for %s: %s", memory.device, exc)
                await self._speak(self.persona.line("error"))
                return
            self._turn_said = None  # and this exchange is not remembered either
            await self._speak(self.persona.line("forgotten", "Forgotten."))
        else:
            await self._speak(self.persona.line("forget_kept", "I'll keep it."))

    def _params_for(self, label: str, text: str) -> dict:
        """An edge tool declares typed params (``jarvis/skills/edge.py``) and
        gets them by type; every other skill by its own slot rules."""
        try:
            manifest = self.registry.manifest(label)
        except Exception:  # noqa: BLE001 — not a registered skill / a fake
            manifest = None
        if manifest is not None and manifest.origin == "edge":
            return _slots.extract_typed(manifest.params, text)
        return self._slot_extract(label, text)

    async def _fill_missing(self, label: str, params: dict) -> dict | None:
        """An edge tool said without a required param ("set a timer"): ask
        for it ("How long, sir?") and take it from the answer ("five
        minutes"). ``None`` when the answer didn't have it either — said so."""
        try:
            manifest = self.registry.manifest(label)
        except Exception:  # noqa: BLE001 — not a registered skill / a fake
            return params
        if manifest.origin != "edge":
            return params
        from jarvis.skills.edge import missing

        params = dict(params)
        for name, spec, question in missing(manifest, params):
            answer = await self._ask(question)
            found = _slots.extract_typed({name: spec}, answer or "")
            if name not in found and spec.get("type") in ("text", "name") and (answer or "").strip(" .!?"):
                found = {name: answer.strip(" .!?")}  # asked for the text itself: all of it
            if name not in found:
                await self._speak(self.persona.phrase("I didn't catch that, sir."))
                return None
            params.update(found)
        return params

    async def refresh_skills(self) -> None:
        """An edge declared new tools (``RemoteServer.on_tools_changed``):
        retrain on a rebuilt registry and stage both for the merge gate, the
        same way a learned skill lands. Nothing to ask, nothing to self-check:
        the tools are the edge's, and a bad example list costs only them."""
        registry = self._latest_registry().rebuilt()
        examples = build_corpus(manifests=registry.manifests())
        kind, payload = await self._train_and_load(examples)
        if kind != "ok":
            log.error("retrain for the edge's tools failed: %s", payload)
            return
        _train_result, classifier = payload
        if self._staged is not None:
            close = getattr(self._staged[0], "close", None)
            if callable(close):
                close()
        self._staged = (classifier, registry)
        log.info("edge tools learned; staged for the next safe point")

    def _action_for(self, label: str) -> str:
        meta = self.intent_meta.get(label)
        return meta.action if meta is not None else label

    def _generate(self, spec, existing_source, feedback=None):
        """Bound to `self.claude_client` so `build()`/the flows never import
        `anthropic` — passed as the flows' `generate` callable."""
        return self.claude_client.generate_skill(spec, existing_source, feedback)

    # -- M2/M2.5: skill factory, run as background learning jobs -------------

    async def _run_flow(self, flow) -> None:
        """Run a teach/edit/revert *dialog* in the foreground, then hand what
        it gathered to a background job and return — the session carries on
        while the skill is built (PRD/milestone-2.5-background-learning.md)."""
        request: LearningRequest | None = await flow.run()
        if request is None:
            log.info("%s dialog ended without a request", type(flow).__name__)
            return
        await self._start_learning(request)

    async def _start_learning(self, request: LearningRequest) -> None:
        ahead = list(self._jobs)[-1] if self._jobs else None
        task = asyncio.ensure_future(self._learn(request, self._last_job))
        # registered before any await, so a dialog started right after this
        # already sees the name as busy
        self._jobs[request.name] = task
        self._last_job = task
        if ahead is None:
            await self._speak("I'll work on that in the background, sir.")
        else:
            await self._speak(f"I'll get to that after '{ahead}', sir.")

    def learning_names(self) -> set[str]:
        """Skills with a queued or running background job."""
        return set(self._jobs)

    def pending_questions(self) -> list[str]:
        """Background questions waiting for the next safe point."""
        return [d.prompt for d in self._decisions if not d.future.done()]

    async def _learn(self, request: LearningRequest, previous: asyncio.Task | None) -> None:
        """One background job: wait for the one ahead of it, then build →
        validate → sandbox (`LearningJob`) → retrain → self-check → keep
        decision → stage. Never raises into the loop."""
        name = request.name
        outcome: FlowOutcome | None = None
        try:
            if previous is not None:
                # `wait`, not `await previous`: cancelling this job must not
                # cancel the one ahead of it
                await asyncio.wait({previous})
            job = LearningJob(
                request,
                notify=self._notify,
                decide=self._decide,
                registry=self._latest_registry(),
                generate=self._generate,
                sandbox=self.sandbox,
                max_attempts=self.config.factory.max_generate_attempts,
            )
            outcome = await job.run()
            if outcome.accepted:
                await self._retrain_and_stage(outcome, request.versioning)
            else:
                log.info("%s job not accepted: %s", name, outcome.reason)
        except asyncio.CancelledError:
            self._unstage(outcome.name if outcome and outcome.name else name)
            raise
        except Exception:  # noqa: BLE001 — a job must never take the loop down
            log.exception("learning job for %s crashed", name)
            self._unstage(outcome.name if outcome and outcome.name else name)
            self._queue_notice(f"Something went wrong while I was learning '{name}', sir.")
        finally:
            if self._jobs.get(name) is asyncio.current_task():
                del self._jobs[name]

    async def _retrain_and_stage(self, outcome: FlowOutcome, versioning: str) -> None:
        """Retrain against the *latest* registry (staged-but-unmerged skills
        included, so back-to-back jobs don't erase each other), self-check,
        ask to keep it (not for revert), then promote + stage for the merge
        gate — see PRD/milestone-2-skill-factory.md decisions 2–3 and
        PRD/milestone-2.5-background-learning.md decisions 5–7."""
        name = outcome.name
        removing = versioning == "remove"
        registry = self._latest_registry()
        other_manifests = [m for m in registry.manifests() if m.name != name]
        if removing:
            # "relearn the module" without the removed skill: retrain on every
            # *remaining* skill's examples, nothing added.
            examples = build_corpus(manifests=other_manifests)
        else:
            examples = build_corpus(manifests=other_manifests + [outcome.manifest])

        kind, payload = await self._train_and_load(examples)
        if kind != "ok":
            log.error("retrain failed for %s: %s", name, payload)
            self._unstage(name)
            msg = (
                f"I couldn't retrain without '{name}', sir — I've left it in place."
                if removing
                else f"'{name}' didn't train cleanly, sir. I've set it aside."
            )
            self._queue_notice(msg)
            return

        train_result, classifier = payload
        try:
            if not self._self_check(classifier, outcome.manifest, present=not removing):
                self._discard_unused_version(train_result)
                self._unstage(name)
                msg = (
                    f"Something looked off after removing '{name}', sir; I've left it as it was."
                    if removing
                    else f"I set '{name}' aside, sir; it didn't check out in practice."
                )
                self._queue_notice(msg)
                return

            if versioning not in ("revert", "remove"):
                description = outcome.manifest.description.rstrip(".")
                description = description[:1].lower() + description[1:]
                keep = await self._decide(
                    f"I've finished '{name}', sir. I can now {description}. Shall I keep it?"
                )
                if not keep:
                    self._discard_unused_version(train_result)
                    self._unstage(name)
                    self._queue_notice(f"I've set '{name}' aside, sir.")
                    return
        except BaseException:
            # cancelled (shutdown) while waiting for the keep answer
            self._discard_unused_version(train_result)
            self._unstage(name)
            raise

        new_registry = self._promote_files(name, outcome.module_source, versioning, outcome)
        if self._staged is not None:
            # an earlier job's staged model is superseded — this one was
            # trained on a corpus that already includes that skill
            close = getattr(self._staged[0], "close", None)
            if callable(close):
                close()
        self._staged = (classifier, new_registry)
        verb = {"new": "learned", "edit": "updated", "revert": "reverted", "remove": "removed"}[versioning]
        self._queue_announcement(f"I've {verb} '{name}', sir. My capabilities are updated.")

    async def _default_train_and_load(self, examples):
        worker = RetrainWorker()
        await asyncio.to_thread(
            worker.start, examples, self.config.nlu.embedding_model, self.config.nlu_model_dir
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + RETRAIN_TIMEOUT_S
        try:
            while True:
                result = await asyncio.to_thread(worker.poll, 0.5)
                if result is not None:
                    break
                if loop.time() > deadline:
                    worker.terminate()
                    result = ("error", f"retrain timed out after {RETRAIN_TIMEOUT_S:.0f}s")
                    break
                if worker.exited():
                    result = await asyncio.to_thread(worker.poll, 0.5)
                    if result is None:
                        result = ("error", "retrain worker exited without a result")
                    break
        except asyncio.CancelledError:
            worker.terminate()
            raise
        kind, payload = result
        if kind != "ok":
            return result
        classifier = await asyncio.to_thread(
            Classifier.load,
            self.config.nlu_model_dir,
            self.config.nlu.embedding_model,
            self.config.nlu.threshold,
            self.config.nlu.similarity_floor,
        )
        return "ok", (payload, classifier)

    def _latest_registry(self):
        """The registry the *next* turn will use: a staged-but-unmerged one if
        there is one, else the live one."""
        return self._staged[1] if self._staged is not None else self.registry

    def _unstage(self, name: str) -> None:
        (_flows.staging_dir() / f"{name}.py").unlink(missing_ok=True)

    # -- M2.5: background notices + questions, spoken at safe points --------

    def _queue_announcement(self, text: str) -> None:
        if self._pending_announcement:
            text = f"{self._pending_announcement} {text}"
        self._pending_announcement = text

    def _queue_notice(self, text: str) -> None:
        self._notices.append(text)

    async def _notify(self, text: str) -> None:
        """`LearningJob`'s `notify` — never speaks directly."""
        self._queue_notice(text)

    async def _decide(self, prompt: str) -> bool:
        """`LearningJob`'s `decide` — queue a yes/no question and wait until a
        safe point has asked it and got a clear answer."""
        decision = _Decision(prompt, asyncio.get_running_loop().create_future())
        self._decisions.append(decision)
        try:
            return await decision.future
        finally:
            if decision in self._decisions:
                self._decisions.remove(decision)

    async def _speak_notices(self) -> None:
        while self._notices:
            await self._speak(self._notices.pop(0))

    async def _safe_point(self) -> None:
        """No turn in flight and the user is present (between turns, or just
        before the drop to standby): merge a staged model, speak queued
        notices, and ask pending background questions."""
        self.state = "idle"
        await self._merge_gate()
        await self._speak_notices()
        for decision in list(self._decisions):
            if decision.future.done():
                continue
            try:
                answer = await ask_yes_no_or_none(self._ask, decision.prompt)
            except _Cancelled:
                answer = None
            self.state = "idle"
            if answer is None:
                decision.unclear += 1
                if decision.unclear < _MAX_UNCLEAR_ANSWERS:
                    continue  # still pending; asked again at the next safe point
                await self._speak("I'll take that as a no, sir.")
                answer = False
            if not decision.future.done():
                decision.future.set_result(answer)
            # let the job run its synchronous tail (promote + stage, or
            # discard + notice) so "try me" holds on the very next turn
            for _ in range(3):
                await asyncio.sleep(0)
        await self._merge_gate()
        await self._speak_notices()

    async def _idle_tick(self) -> None:
        """Outside a wake session (standby): merge and speak notices, but never
        ask a question unprompted — those wait for the next session."""
        if self._in_session:
            return
        await self._merge_gate()
        await self._speak_notices()

    async def _idle_loop(self) -> None:
        while self.running:
            await asyncio.sleep(0.5)
            try:
                await self._idle_tick()
            except Exception:  # noqa: BLE001
                log.exception("idle tick failed")

    async def cancel_learning(self) -> None:
        """Cancel every queued/running job (shutdown). Each job cleans up its
        own staging file and unstaged model version on the way out."""
        tasks = list(self._jobs.values())
        for task in tasks:
            task.cancel()
        for decision in self._decisions:
            decision.future.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._jobs.clear()
        self._decisions.clear()
        self._last_job = None

    def _self_check(self, classifier: Classifier, manifest, *, present: bool = True) -> bool:
        """A freshly-trained model must still classify the seed intents sanely
        before it's ever staged — PRD: "new model must load and still classify
        the seed examples sanely."

        For `present=True` (teach/edit/revert) the new/changed skill's own
        examples must also resolve to it. For `present=False` (remove) it's
        the mirror image — those same examples must *not* resolve to the
        just-removed skill any more (proving the retrain actually dropped
        it)."""
        for text, expected in _SELF_CHECK_SEED_PROBES:
            label, _ = classifier.predict(text)
            if label != expected:
                log.warning("self-check regression: %r -> %s (expected %s)", text, label, expected)
                return False
        for text in manifest.examples:
            label, _ = classifier.predict(text)
            if present and label != manifest.name:
                log.warning(
                    "self-check: %r -> %s (expected %s)", text, label, manifest.name
                )
                return False
            if not present and label == manifest.name:
                log.warning(
                    "self-check: %r still -> %s after removal", text, manifest.name
                )
                return False
        return True

    def _discard_unused_version(self, train_result) -> None:
        """A trained-but-never-staged model version — self-check failed, or
        the user declined the confirm. Delete it so disk state matches the
        live (unchanged) `self.nlu`."""
        shutil.rmtree(train_result.path, ignore_errors=True)

    def _promote_files(
        self, name: str, module_source: str | None, versioning: str, outcome: FlowOutcome
    ):
        """Write the confirmed module to `skills/learned/`, with version /
        quarantine bookkeeping per PRD directory-layout decision 4, and
        return a freshly rebuilt registry. The actual `self.registry` swap
        happens later, through `self._staged` + the merge gate.

        `versioning="remove"` has no module to write — it just quarantines the
        existing file and returns the rebuilt (now smaller) registry."""
        registry = self._latest_registry()
        learned_path = _flows.learned_source_path(name)
        learned_path.parent.mkdir(parents=True, exist_ok=True)

        if versioning == "remove":
            # No module to write — quarantine the live file (kept for
            # inspection, never re-imported), drop its now-meaningless rollback
            # history, and evict the stale import so a later re-teach of the
            # same name starts clean.
            quarantine_dir = self.config.skill_quarantine_dir
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            if learned_path.is_file():
                stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                shutil.move(str(learned_path), str(quarantine_dir / f"{name}.{stamp}.py"))
            shutil.rmtree(self.config.skill_versions_dir(name), ignore_errors=True)
            sys.modules.pop(f"{LEARNED_PACKAGE}.{name}", None)
            return registry.rebuilt()

        if versioning == "edit":
            old_manifest = registry.manifest(name)
            vdir = self.config.skill_versions_dir(name)
            vdir.mkdir(parents=True, exist_ok=True)
            if learned_path.is_file():
                shutil.copy2(learned_path, vdir / f"v{old_manifest.version}.py")
            self._prune_versions(vdir)
        elif versioning == "revert":
            quarantine_dir = self.config.skill_quarantine_dir
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            if learned_path.is_file():
                stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                shutil.move(str(learned_path), str(quarantine_dir / f"{name}.{stamp}.py"))
            vdir = self.config.skill_versions_dir(name)
            (vdir / f"v{outcome.reverted_from_version}.py").unlink(missing_ok=True)

        learned_path.write_text(module_source, encoding="utf-8")
        self._unstage(name)
        return registry.rebuilt()

    @staticmethod
    def _prune_versions(vdir: Path, keep: int = 3) -> None:
        versions = sorted(
            (int(p.stem[1:]) for p in vdir.glob("v*.py") if p.stem[1:].isdigit()),
            reverse=True,
        )
        for stale in versions[keep:]:
            (vdir / f"v{stale}.py").unlink(missing_ok=True)

    # -- M2 seam: seamless hot-swap -----------------------------------------

    async def _merge_gate(self) -> None:
        """Swap ``self.nlu``/``self.registry`` for a staged replacement, but
        only while idle so an in-flight turn is never disrupted. Speaks any
        queued announcement once the swap lands."""
        if self._staged is not None and self.state == "idle":
            (new_nlu, new_registry), self._staged = self._staged, None
            old_nlu = self.nlu
            self.nlu, self.registry = new_nlu, new_registry
            close = getattr(old_nlu, "close", None)
            if callable(close):
                close()
            if self._pending_announcement:
                announcement, self._pending_announcement = self._pending_announcement, None
                await self._speak(announcement)

    # -- main loop --------------------------------------------------------

    async def run(self) -> None:
        self.running = True
        self.mic.start()
        self._interrupter.install()
        self._idle_task = asyncio.ensure_future(self._idle_loop())
        if self.knowledge is not None:
            self._knowledge_task = asyncio.ensure_future(self.knowledge.watch())
        try:
            while self.running:
                self.state = "idle"
                await self._merge_gate()
                if not await self._await_wake():
                    break
                self.standby = False
                self._in_session = True
                try:
                    await self._session()
                finally:
                    self._in_session = False
                    self.state = "idle"
                    self.standby = True
        finally:
            self.running = False
            # the re-scan waits for its worker thread before it is done
            for task in (self._idle_task, self._knowledge_task):
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
            await self.cancel_learning()
            await self._interrupter.shutdown()
            self.mic.stop()
            close = getattr(self.tts, "close", None)
            if callable(close):
                close()

    def stop(self) -> None:
        self.running = False
