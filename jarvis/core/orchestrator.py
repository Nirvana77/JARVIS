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

M7: every turn is logged (``_record``, ``jarvis/learning/``); a confirmed turn
teaches the classifier the words that were actually heard
(``_learn_phrasing``), retrained in batches at idle behind a regression gate
(``_retrain_phrasings``); "no, I meant …" undoes that (``_correction``).
A request nothing fits is built as a new skill (``_learn_capability``) and a
learned skill that raises is repaired (``_start_repair``), both without a
"Shall I keep it?".
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as _dt
import logging
import queue
import random
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jarvis.core import forgetting as _forgetting
from jarvis.core import mishear as _mishear
from jarvis.core import reasoning as _reasoning
from jarvis.core.interrupt import Cancelled as _Cancelled
from jarvis.core.interrupt import Interrupter
from jarvis.core.memory import LOCAL, Memory, is_device_id, same_text
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
from jarvis.factory.spec import SkillSpec
from jarvis.learning import Learning
from jarvis.nlu import compound as _compound
from jarvis.nlu import slots as _slots
from jarvis.nlu.classifier import UNKNOWN, Classifier
from jarvis.nlu.corpus import build_corpus
from jarvis.nlu.retrain_worker import RetrainWorker
from jarvis.skills.builtin import note as _note_skill
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
    "forget_fact": "forget_fact",
    # M7
    "correction": "correction",
    "learned_today": "learned_today",
}

#: A wrong guess at one of these costs a dialog, a skill's code or a device's
#: memory, so they need `nlu.meta_action_threshold`, not just "not unknown".
#: (`forget_fact` is not here: it always names what it would forget and asks,
#: and needs its own "forget …" lead-in instead — see `_forget_fact`.)
_GUARDED_ACTIONS = (
    "teach", "edit_skill", "revert_skill", "remove_skill", "forget_memory",
    # M7: undoes what the last turn taught
    "correction",
)

#: Skills whose argument is whatever was said: never one step of a chain
#: ("note that buy milk and call mum" is one note).
_DICTATION = frozenset({"note", "remember"})

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

#: M7: two build requests this similar (MiniLM cosine) are the same request
_SAME_REQUEST = 0.85

#: M7: logged turns read back as regression probes for a phrasing retrain
_MAX_REGRESSION_PROBES = 200

#: M7: "no, I meant set a timer" -> "set a timer"
_MEANT = re.compile(
    r"\b(?:i\s+)?(?:meant|mean|said|wanted|asked\s+for|asked\s+you\s+to)\s+(?:to\s+)?(.+)$",
    re.IGNORECASE,
)


#: The edge a background job was asked from (``tts.connected_device_id``), set in the
#: job's own task: its notices, questions and announcement go to that device.
#: None for a job asked on the local mic — or anywhere, when nothing tells.
_ORIGIN: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_origin", default=None)


@dataclass
class _Decision:
    """A background job's yes/no question, waiting for a safe point."""

    prompt: str
    future: asyncio.Future
    unclear: int = 0
    origin: str | None = None
    #: asked unprompted in standby already: not again until the next session
    offered: bool = False


#: The factory's numeric param types (``claude_client._CONTRACT``) in the
#: vocabulary of ``slots.extract_typed``. Strings and booleans are not guessed:
#: text extraction takes what follows "to"/"that", which for "add eggs to the
#: shopping list" is the wrong half, and a wrong value is worse than the
#: skill's own default.
_LEARNED_TYPES = {"integer": "number", "number": "number"}


def _learned_params(params: dict, text: str) -> dict:
    """A learned skill's declared params, by type (known issue #7: before
    this, "flip three coins" flipped one). What is not there is left out, so
    the skill's own default applies."""
    spec = {
        name: {"type": _LEARNED_TYPES[kind]}
        for name, decl in params.items()
        if isinstance(decl, dict) and (kind := decl.get("type")) in _LEARNED_TYPES
    }
    found = _slots.extract_typed(spec, text)
    for name, value in list(found.items()):
        if params[name].get("type") == "integer":
            found[name] = int(round(value))
    return found


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
        learning: Learning | None = None,
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
        #: the `note` skill's own log, which "forget about …" also edits
        self._notes_file = config.skill_data_dir(_note_skill.MANIFEST.name) / _note_skill.NOTES_FILE
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
        self._announcement_origin: str | None = None
        # M2.5: background learning. `_jobs` is name -> task in queue order;
        # jobs run one at a time (each waits for `_last_job`).
        self._jobs: dict[str, asyncio.Task] = {}
        self._last_job: asyncio.Task | None = None
        self._notices: list[tuple[str, str | None]] = []  # (text, origin)
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

        # The persona's rewrite picks style lines by similarity: let it use the
        # classifier's MiniLM (whichever is live) rather than load a second one
        lend = getattr(persona, "use_embed", None)
        if callable(lend):
            lend(lambda text: self.nlu.embed(text))

        # M7: learning from every turn. Without one handed in (tests), the
        # stores are in RAM.
        self.learning = learning if learning is not None else Learning.in_memory(config)
        self._clock = time.monotonic
        #: the log line of the turn in flight (None between turns)
        self._record: dict | None = None
        #: the classifier's full verdict on the turn in flight, when it gave one
        self._last_prediction = None
        #: the last skill that ran, for "no, I meant …"
        self._last_skill: dict | None = None
        #: an unsure turn that ran: learned unless the next turn corrects it
        self._provisional: dict | None = None
        self._phrasing_task: asyncio.Task | None = None
        self._phrasings_added_at = self._clock()

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

    async def _phrase(self, text: str, skill: str | None = None) -> str:
        """The line to speak for ``text``. A skill whose lines are written in
        the active persona's voice (``SkillManifest.voice``) is spoken as
        written. Anything else gets the persona's rewrite — an Ollama call
        (0.4–0.9 s on the cluster's qwen3:8b), so off the event loop (known
        issue #16), and checked for facts it dropped (``jarvis/core/voice.py``)."""
        if skill is not None and self._voiced(skill):
            return text
        return await asyncio.to_thread(self.persona.phrase, text)

    def _voiced(self, skill: str) -> bool:
        try:
            voice = self.registry.manifest(skill).voice
        except Exception:  # noqa: BLE001 — not a registered skill / a fake
            return False
        return bool(voice) and voice == getattr(self.persona, "name", None)

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

    async def _await_wake(self) -> str:
        """Block until the wake word fires (``"wake"``) or a background
        question is due for the connected edge (``"offer"``). ``""`` if the
        loop was stopped."""
        await asyncio.to_thread(self.wake.reset)
        self._drain_mic()
        while self.running:
            if self._offer_due():
                return "offer"
            try:
                frame = await asyncio.to_thread(self.mic.read, 1.0)
            except queue.Empty:
                continue
            if await asyncio.to_thread(self.wake.triggered, frame):
                return "wake"
        return ""

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

        # nobody said "no, I meant …" before the session ended
        self._commit_provisional()
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
            self._last_prediction = (text, p)
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
        if _META_ACTIONS.get(label) != "correction":
            self._commit_provisional()
        self._record = self._new_record(device, label, text, confidence)
        self._turn_said = []
        try:
            await self.handle(label, text, confidence)
        except _Cancelled as exc:
            print(f"  (cancelled {exc} — still listening)", flush=True)
            self._note(outcome="cancelled")
        finally:
            said, self._turn_said = self._turn_said, None
            record, self._record = self._record, None
            if said is not None:  # None: the turn was "forget everything"
                self.memory.device(device).add_turn(text, " ".join(said))
                record["said"] = " ".join(said)
                record.setdefault("path", self._default_path(label))
                record.setdefault("outcome", "ok")
                if self.learning.settings.log:
                    self.learning.log.append(record)

    # -- M7: the interaction log -------------------------------------------------

    def _new_record(self, device: str, label: str, text: str, confidence: float) -> dict:
        record = {
            "id": self.learning.log.new_id(),
            "device": device,
            "heard": text,
            "label": label,
            "confidence": round(float(confidence), 3),
        }
        if self._last_prediction is not None and self._last_prediction[0] == text:
            ranking = self._last_prediction[1].ranking
            if len(ranking) > 1:
                record["runner_up"] = [ranking[1][0], round(float(ranking[1][1]), 3)]
        version = getattr(self.nlu, "version", None)
        if isinstance(version, int):
            record["model"] = version
        self._last_prediction = None
        return record

    def _note(self, **fields) -> None:
        """Fill in the turn's log line (a no-op outside a turn)."""
        if self._record is not None:
            self._record.update(fields)

    def _note_path(self, path: str) -> None:
        """The path a turn took, unless an outer step already named it: a
        mishearing that runs a skill stays a ``mishear``."""
        if self._record is not None:
            self._record.setdefault("path", path)

    def _default_path(self, label: str) -> str:
        if label == UNKNOWN:
            return "unknown"
        action = _META_ACTIONS.get(label, self._action_for(label))
        return "reply" if action == "none" else action

    async def handle(self, label: str, text: str, confidence: float = 1.0) -> None:
        steps = await self._compound(text)
        if steps:
            self._note_path("compound")
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
            # the persona's own words for it, written in voice ("reply_thanks"),
            # else one of intents.json's replies through the checked rewrite
            voiced = self.persona.line(f"reply_{label}", "")
            if voiced:
                await self._speak(voiced)
                return
            meta = self.intent_meta.get(label)
            reply = random.choice(meta.responses) if meta and meta.responses else ""
            await self._speak(await self._phrase(reply) if reply else "")
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
            await self._recall(text)
            return
        if action == "forget_memory":
            await self._forget()
            return
        if action == "forget_fact":
            await self._forget_fact(text)
            return
        if action == "correction":
            await self._correction(text)
            return
        if action == "learned_today":
            await self._learned_today()
            return

        if label in _slots.STORING and not _slots.has_lead_in(label, text):
            # "Forget about the park", heard as `remember`: with no "remember
            # that …" to strip, the whole sentence would be stored as a fact.
            log.info("%s without its lead-in (%r): treating it as unclear", label, text)
            await self._unclear(text)
            return

        # a skill
        self.state = "acting"
        self._note_path("direct")
        if self.learning.state.is_disabled(label):
            # M7: a learned skill that kept failing after its repairs
            self._note(skill=label, outcome="error", error="disabled")
            await self._speak(self.persona.line(
                "out_of_order", "That skill is switched off until you look at it, sir."))
            return
        params = await self._fill_missing(label, self._params_for(label, text))
        if params is None:
            return
        self._note(skill=label, params=_loggable(params))
        try:
            line = await asyncio.to_thread(self.registry.dispatch, label, params)
        except Exception as exc:  # noqa: BLE001 - a skill blew up
            if isinstance(exc, KeyError) and not self._registered(label):
                await self._speak(self.persona.line("skill_missing"))
                self._note(outcome="error", error="missing")
                return
            # a KeyError from inside a skill is the skill's own bug, not a
            # missing skill: it is repaired like any other failure
            log.exception("skill %s failed: %s", label, exc)
            self._note(outcome="error", error=f"{type(exc).__name__}: {exc}"[:500])
            self._remember_skill_turn(label)
            await self._speak(self.persona.line("error"))
            await self._after_failure(label, text, params, exc)
            return
        self._note(outcome="ok")
        self._remember_skill_turn(label)
        if (
            self._record is not None
            and self._record.get("path") == "direct"
            and self.config.nlu.threshold <= confidence < self.learning.settings.learn_below
            and self._learnable(label)
            # known issue #18: "forget how to flip a coin" ran the coin with
            # remove_skill second — never make that kind of near miss permanent
            and (self._record.get("runner_up") or [None])[0] not in _META_ACTIONS
        ):
            # it ran, but the classifier was unsure: worth learning the words
            # unless the next turn says it was the wrong thing
            self._provisional = {
                "text": self._record["heard"], "label": label, "turn": self._record["id"],
            }
        self.state = "speaking"
        await self._speak(await self._phrase(line, label))

    def _registered(self, label: str) -> bool:
        try:
            return label in self.registry.names()
        except Exception:  # noqa: BLE001 — a fake without a roster
            return False

    def _remember_skill_turn(self, label: str) -> None:
        """The skill that just ran, for a correction in the next turn."""
        if self._record is None:
            return
        self._last_skill = {
            "id": self._record["id"],
            "heard": self._record["heard"],
            "label": label,
            "device": self._record["device"],
            "at": self._clock(),
            "learned": self._record.get("learned"),
        }

    async def _after_failure(self, label: str, heard: str, params: dict, exc: BaseException) -> None:
        """M7 part F: a skill raised. Builtins are written down for a person;
        a learned skill is repaired in the background (``_start_repair``)."""
        try:
            manifest = self._latest_registry().manifest(label)
        except Exception:  # noqa: BLE001 — not a registered skill / a fake
            return
        error = "".join(traceback.format_exception(exc))[-2000:]
        if manifest.origin == "builtin":
            try:
                await asyncio.to_thread(
                    self.learning.state.builtin_failure, label, heard, _loggable(params), error
                )
            except OSError as exc2:
                log.error("could not record the failure of %s: %s", label, exc2)
            return
        if manifest.origin == "learned":
            await self._start_repair(manifest, heard, params, error)

    async def _start_repair(self, manifest, heard: str, params: dict, error: str) -> None:
        """M7 part F: rewrite a learned skill that raised, in the background,
        and keep the result only if the call that failed now works. Past
        ``max_repairs_per_skill_per_day`` the skill is switched off instead,
        and JARVIS says so rather than failing the same way again."""
        s = self.learning.settings
        name = manifest.name
        if (
            not (s.enabled and s.auto_repair)
            or self.claude_client is None
            or not self.claude_client.available
            or self.sandbox is None
            or name in self._jobs
        ):
            return
        if self.learning.state.repairs_today(name) >= s.max_repairs_per_skill_per_day:
            last_line = error.strip().splitlines()[-1] if error.strip() else "it failed"
            self.learning.state.disable(name, last_line)
            self.learning.state.add_event("disabled", name)
            log.warning("%s keeps failing after its repairs: switched off (%s)", name, last_line)
            await self._speak(self.persona.line(
                "skill_disabled", "'{name}' keeps failing, sir. I've switched it off."
            ).replace("{name}", name))
            return
        try:
            source = await asyncio.to_thread(
                _flows.learned_source_path(name).read_text, encoding="utf-8"
            )
        except OSError as exc:
            log.error("cannot repair %s: no source (%s)", name, exc)
            return
        self.learning.state.count_repair(name)
        replay = _loggable(params)
        description = (
            f"Repair this skill. Asked {heard!r}, it was called as "
            f"run(ctx, **{replay!r}) and raised:\n{error.strip()}\n"
            "Fix the cause so that this exact call returns a line to speak; "
            "keep everything else it does the same."
        )
        request = LearningRequest(
            versioning="edit",
            name=name,
            spec=SkillSpec(
                name=name,
                description=description,
                examples=list(manifest.examples),
                based_on_version=manifest.version,
            ),
            existing_source=source,
            allow_name=name,
            autonomous=True,
            replay_params=replay,
            utterance=heard,
        )
        log.info("repairing %s in the background", name)
        await self._start_learning(request, announce=False)

    # -- M7 part E: learning what nothing fits -------------------------------------------

    async def _learn_capability(self, text: str, thought) -> None:
        """The reasoner says this is a request for something JARVIS cannot do
        yet: have the factory build it, in the background, and keep it without
        asking. Capped per day, never the same request twice in a day, and a
        permission beyond pure/notify still waits for a yes (unless
        ``[learning] auto_permissions``)."""
        s = self.learning.settings
        self._note_path("build")
        if (
            not (s.enabled and s.auto_build)
            or self.claude_client is None
            or not self.claude_client.available
            or self.sandbox is None
        ):
            self._note(outcome="declined")
            await self._speak(self.persona.line("unknown"))
            return
        description = thought.learn
        for earlier in self.learning.state.requests_today():
            if await self._same_request(text, description, earlier):
                self._note(outcome="declined", skill=earlier["name"])
                key = {"building": "already_learning", "failed": "learn_failed_today"}.get(
                    earlier["status"], "unknown"
                )
                await self._speak(self.persona.line(key))
                return
        if self.learning.state.builds_today() >= s.max_builds_per_day:
            self._note(outcome="declined")
            await self._speak(self.persona.line(
                "learn_limit", "I've done all the learning I may today, sir."))
            return
        name = self._new_skill_name(description)
        examples: list[str] = []
        for example in (text, *thought.examples, description):
            if example and not any(same_text(example, e) for e in examples):
                examples.append(example)
        self.learning.state.count_build()
        self.learning.state.add_request(text, description, name=name, status="building")
        self._note(skill=name)
        log.info("learning %s by itself: %r", name, description)
        await self._speak(self.persona.line(
            "learning_it", "I can't do that yet, sir. I'll learn it."))
        await self._start_learning(
            LearningRequest(
                versioning="new",
                name=name,
                spec=SkillSpec(name=name, description=description, examples=examples),
                autonomous=True,
                utterance=text,
            ),
            announce=False,
        )

    async def _same_request(self, text: str, description: str, earlier: dict) -> bool:
        if same_text(text, earlier.get("text", "")) or same_text(
            description, earlier.get("description", "")
        ):
            return True
        embed = getattr(self.nlu, "embed", None)
        if not callable(embed):
            return False
        try:
            a, b = await asyncio.to_thread(lambda: (embed(text), embed(earlier.get("text", ""))))
        except Exception:  # noqa: BLE001 — a fake, or the model is busy
            return False
        return float(a @ b) >= _SAME_REQUEST

    def _new_skill_name(self, description: str) -> str:
        base = "_".join(_flows._slugify(description).split("_")[:4]).strip("_") or "new_skill"
        if base[0].isdigit():
            base = f"_{base}"
        taken = set(self._latest_registry().names()) | set(self._jobs)
        name, n = base, 2
        while name in taken:
            name, n = f"{base}_{n}", n + 1
        return name

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
            # the reasoner did not answer in time: not a second, longer wait —
            # but the classifier's own near miss costs nothing to offer
            if not await self._offer_near_miss(text, confirm):
                self._note_path("unknown")
                await self._speak(self.persona.line("unknown"))
            return
        if corrected is not None:
            guess, label, confidence = corrected
            self._note_path("mishear")
            if await self._confirmed(confirm.replace("{guess}", guess)):
                # M7: the words that were heard mean what was confirmed
                self._learn_phrasing(text, label, "mishear")
                await self.handle(label, guess, confidence)
            else:
                self._note(outcome="declined")
                await self._speak(self.persona.line("unknown"))
            return

        thought = await self._think(text)
        if self._interrupted():
            print("  (cancelled thinking — still listening)", flush=True)
            return
        if isinstance(thought, _reasoning.Thought):
            await self._learn_capability(text, thought)
            return
        if isinstance(thought, str):
            # already in the persona's voice: its character was in the prompt
            self._note_path("answer")
            await self._speak(thought)
            return
        if thought:
            self._note_path("plan")
            said = ", then ".join(step for step, _label, _confidence in thought)
            if await self._confirmed(confirm.replace("{guess}", said)):
                if len(thought) == 1:
                    # one command: the heard words mean it. Several mean no one label.
                    self._learn_phrasing(text, thought[0][1], "plan")
                await self._run_steps(thought)
                return
            self._note(outcome="declined")
            await self._speak(self.persona.line("unknown"))
            return
        if await self._offer_near_miss(text, confirm):
            return
        self._note_path("unknown")
        await self._speak(self.persona.line("unknown"))

    async def _offer_near_miss(self, text: str, confirm: str) -> bool:
        """M7: with no reasoner (or none that helped), the classifier's own
        best guess, when it fell just short of ``nlu.threshold``: "Did you mean
        'flip a coin'?" A yes runs that skill on the words that were said and
        learns them. ``True`` when the question was asked."""
        margin = self.learning.settings.confirm_margin
        explain = getattr(self.nlu, "explain", None)
        if margin <= 0 or not callable(explain) or not text.strip():
            return False
        p = await asyncio.to_thread(explain, text)
        if not p.ranking or p.similarity < self.config.nlu.similarity_floor:
            return False
        label, confidence = p.ranking[0]
        if (
            confidence < self.config.nlu.threshold - margin
            or not self._chainable(label)
            or label in _DICTATION
        ):
            return False
        try:
            examples = self._latest_registry().manifest(label).examples
        except Exception:  # noqa: BLE001
            examples = []
        offered = examples[0] if examples else label.replace("_", " ")
        self._note_path("confirmed")
        if not await self._confirmed(confirm.replace("{guess}", offered.rstrip(" .?!"))):
            self._note(outcome="declined")
            await self._speak(self.persona.line("unknown"))
            return True
        self._learn_phrasing(text, label, "confirmed")
        await self._handle_one(label, text, confidence)
        return True

    # -- M7: learning phrasings ------------------------------------------------------

    def _learnable(self, label: str) -> bool:
        """May the words for ``label`` be learned from use? Skills only: never
        a session or meta action — above all not a guarded one, which a
        learned phrasing would make easier to set off by mistake — and not
        dictation, whose words are the content, not the command."""
        if label == UNKNOWN or label in _META_ACTIONS or label in _DICTATION:
            return False
        if label in _slots.STORING:
            return False
        try:
            return label in self._latest_registry().names()
        except Exception:  # noqa: BLE001
            return False

    def _learn_phrasing(self, text: str, label: str, why: str):
        """``text`` means ``label``: kept for the next retrain. The phrasing,
        or ``None`` when learning is off, the label may not be learned, or it
        is known (or was undone) already."""
        if not self.learning.settings.enabled or not self._learnable(label):
            return None
        turn = self._record["id"] if self._record is not None else None
        try:
            phrasing = self.learning.phrasings.add(text, label, why=why, turn=turn)
        except OSError as exc:
            log.error("could not keep the phrasing %r: %s", text, exc)
            return None
        if phrasing is None:
            return None
        self._note(learned=phrasing.id)
        if self._last_skill is not None and self._last_skill["id"] == turn:
            self._last_skill["learned"] = phrasing.id
        self._phrasings_added_at = self._clock()
        log.info("learned: %r means %s (%s)", text, label, why)
        print(f'  learned : "{text}" -> {label} ({why})')
        try:
            self.learning.state.add_event("phrasing", f"'{phrasing.text}' means {self._meaning(label)}")
        except OSError:
            pass
        return phrasing

    def _commit_provisional(self) -> None:
        """The unsure turn was not corrected: learn its words now."""
        provisional, self._provisional = self._provisional, None
        if provisional is None:
            return
        record, self._record = self._record, None  # not this turn's line
        try:
            phrasing = self._learn_phrasing(provisional["text"], provisional["label"], "unsure")
        finally:
            self._record = record
        if phrasing is not None and self._last_skill is not None:
            if self._last_skill["id"] == provisional["turn"]:
                self._last_skill["learned"] = phrasing.id

    def _corpus(self, manifests):
        """Every retrain's corpus: seeds, the skills' examples and the learned
        phrasings. A retrain that left the phrasings out would unlearn them."""
        try:
            learned = self.learning.phrasings.examples()
        except OSError as exc:
            log.error("could not read the learned phrasings: %s", exc)
            learned = []
        return build_corpus(manifests=manifests, learned=learned)

    def _phrasings_due(self) -> bool:
        """Time to retrain on what was learned: enough phrasings waiting, or
        one waiting long enough — and nothing else retraining or staged."""
        s = self.learning.settings
        if (
            not s.enabled
            or self._phrasing_task is not None
            or self._jobs
            or self._staged is not None
        ):
            return False
        try:
            pending = len(self.learning.phrasings.pending())
        except OSError:
            return False
        if not pending:
            return False
        if pending >= s.retrain_after:
            return True
        return self._clock() - self._phrasings_added_at >= s.retrain_idle_s

    def _maybe_retrain_phrasings(self) -> None:
        if self._phrasings_due():
            self._phrasing_task = asyncio.ensure_future(self._retrain_phrasings())

    async def _retrain_phrasings(self) -> None:
        """Retrain with the learned phrasings, in the worker; keep the model
        only if it does no worse than the live one on the regression probes;
        stage it for the merge gate. Never raises into the loop."""
        try:
            batch = self.learning.phrasings.pending()
            registry = self._latest_registry()
            examples = self._corpus(registry.manifests())
            print(f"  (retraining with {len(batch)} learned phrasing(s))", flush=True)
            kind, payload = await self._train_and_load(examples)
            if kind != "ok":
                log.error("phrasing retrain failed: %s", payload)
                self._phrasings_added_at = self._clock()  # try again later
                return
            train_result, classifier = payload
            kept = await asyncio.to_thread(self._regression_gate, self.nlu, classifier)
            if not kept or self._staged is not None or self._jobs:
                self._discard_unused_version(train_result)
                _close(classifier)
                if not kept:
                    gone = self.learning.phrasings.quarantine([p.id for p in batch])
                    log.warning(
                        "phrasing retrain got worse; set aside: %s",
                        ", ".join(f"{p.text!r} -> {p.label}" for p in gone),
                    )
                    self.learning.state.add_event(
                        "quarantine", f"{len(gone)} phrasing(s) made recognition worse"
                    )
                else:
                    self._phrasings_added_at = self._clock()  # a job got in first
                return
            self.learning.phrasings.mark_trained([p.id for p in batch])
            self._staged = (classifier, registry)
            if self.learning.settings.announce == "idle":
                self._queue_announcement(self._learned_line(batch))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — learning must never take the loop down
            log.exception("phrasing retrain crashed")
            self._phrasings_added_at = self._clock()
        finally:
            self._phrasing_task = None

    def _learned_line(self, batch) -> str:
        if len(batch) == 1:
            p = batch[0]
            line = self.persona.line(
                "learned_phrasing", "I've learned that '{text}' means {meaning}, sir."
            )
            return line.replace("{text}", p.text).replace("{meaning}", self._meaning(p.label))
        line = self.persona.line(
            "learned_phrasings", "I've learned {n} new ways of putting things, sir."
        )
        return line.replace("{n}", str(len(batch)))

    def _meaning(self, label: str) -> str:
        try:
            examples = self._latest_registry().manifest(label).examples
        except Exception:  # noqa: BLE001
            examples = []
        meaning = (examples[0] if examples else label.replace("_", " ")).rstrip(" .?!")
        return meaning[:1].lower() + meaning[1:]

    def _regression_gate(self, old, new) -> bool:
        """Blocking: does ``new`` get at least as many of the probes right as
        ``old``? The probes are the seed self-check phrases plus recent turns
        that ran a skill directly, surely and without error — what JARVIS
        already understands must not be unlearned by what it just learned."""
        probes = list(_SELF_CHECK_SEED_PROBES)
        seen = {text for text, _ in probes}
        try:
            records = self.learning.log.records()
        except OSError:
            records = []
        for r in reversed(records):
            if len(probes) >= _MAX_REGRESSION_PROBES:
                break
            if (
                r.get("path") == "direct"
                and r.get("outcome") == "ok"
                and r.get("skill")
                and float(r.get("confidence", 0)) >= self.learning.settings.learn_below
                and r.get("heard") not in seen
            ):
                probes.append((r["heard"], r["skill"]))
                seen.add(r["heard"])
        old_ok = sum(old.predict(text)[0] == label for text, label in probes)
        new_ok = sum(new.predict(text)[0] == label for text, label in probes)
        if new_ok < old_ok:
            log.warning("regression gate: %d/%d right before, %d after", old_ok, len(probes), new_ok)
            return False
        return True

    # -- M7: corrections ---------------------------------------------------------------

    async def _correction(self, text: str) -> None:
        """"No, I meant …" just after a skill ran: undo what that turn taught,
        learn its words as what was meant, and do that instead."""
        last = self._last_skill
        window = self.learning.settings.correction_window_s
        self._note_path("correction")
        if (
            last is None
            or last["device"] != self._device()
            or self._clock() - last["at"] > window
        ):
            self._note(outcome="declined")
            await self._speak(self.persona.line(
                "nothing_to_correct", "There's nothing for me to correct, sir."))
            return
        self._last_skill = None
        self._note(corrects=last["id"])
        if self._provisional is not None and self._provisional["turn"] == last["id"]:
            self._provisional = None
        if last.get("learned"):
            try:
                self.learning.phrasings.remove(last["learned"], reason="corrected")
            except OSError as exc:
                log.error("could not undo phrasing %s: %s", last["learned"], exc)

        found = _MEANT.search(text)
        meant = found.group(1).strip(" .!?,") if found else ""
        if not meant:
            meant = (await self._ask(self.persona.line(
                "what_did_you_mean", "What did you mean, sir?"))).strip(" .!?,")
        if not meant:
            await self._speak(self.persona.line("unknown"))
            return
        label, confidence = await asyncio.to_thread(self.nlu.predict, meant)
        print(f'  meant   : "{meant}" -> {label} ({confidence:.2f})')
        if label == UNKNOWN or label == last["label"] or not self._learnable(label):
            await self._speak(self.persona.line("unknown"))
            return
        try:
            self.learning.state.confusion(last["label"], label)
            self.learning.state.add_event(
                "correction",
                f"'{last['heard']}' means {self._meaning(label)}, not {self._meaning(last['label'])}",
            )
        except OSError:
            pass
        learned = None
        if not _mishear.same_words(last["heard"], meant):
            learned = self._learn_phrasing(last["heard"], label, "correction")
        await self._handle_one(label, meant, confidence)
        if self._last_skill is not None and self._last_skill["id"] == (self._record or {}).get("id"):
            # a correction of this correction is about the words first said
            self._last_skill["heard"] = last["heard"]
            self._last_skill["learned"] = learned.id if learned else None

    async def _learned_today(self) -> None:
        """"What have you learned today?" — from the learning's own events."""
        try:
            events = self.learning.state.events(since=_dt.date.today())
        except OSError:
            events = []
        if not events:
            await self._speak(self.persona.line("nothing_learned", "Nothing new today, sir."))
            return
        by_kind: dict[str, list[str]] = {}
        for e in events:
            by_kind.setdefault(e["kind"], []).append(e["text"])
        parts = []
        if by_kind.get("phrasing"):
            shown = by_kind["phrasing"][-3:]
            parts.append("that " + "; that ".join(shown))
            if len(by_kind["phrasing"]) > len(shown):
                parts.append(f"{len(by_kind['phrasing']) - len(shown)} more phrasings")
        if by_kind.get("skill"):
            parts.append("new skills: " + ", ".join(by_kind["skill"]))
        if by_kind.get("repair"):
            parts.append("repairs to " + ", ".join(by_kind["repair"]))
        if by_kind.get("correction"):
            parts.append(f"{len(by_kind['correction'])} correction(s) from you")
        if not parts:
            await self._speak(self.persona.line("nothing_learned", "Nothing new today, sir."))
            return
        # the speaker's own words: spoken as they are, not rephrased
        await self._speak("Today I learned " + "; ".join(parts) + ".")

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

        if isinstance(found, _reasoning.Thought):
            print(f'  learn   : "{found.learn}"')
            return found

        if isinstance(found, str):
            if not options.answer_questions:
                return None
            print("  answer  : (the reasoner's own)")
            return found

        print("  plan    : " + " · ".join(f'"{t}" -> {lbl} ({c:.2f})' for t, lbl, c in found))
        if not options.plan_commands:
            return None
        if len(found) == 1 and found[0][1] == UNKNOWN:
            # M7: a small model asked for something nothing does often re-says
            # it as a "command" instead of answering "learn" (qwen3:8b, "roll a
            # twenty sided die"). One step no command knows is a capability.
            print(f'  learn   : "{found[0][0]}" (a plan nothing knows)')
            return _reasoning.Thought(learn=found[0][0])
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
        if thought.learn:
            return thought  # M7: a capability to learn, not a command
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

    async def _recall(self, text: str = "") -> None:
        """"What do you remember?" — and "what should I remember today?",
        "what are the to-dos for today?": with "today" in it, what was
        remembered today; everything when nothing was."""
        entries = self.memory.device().entries()
        if not entries:
            await self._speak(self.persona.line(
                "nothing_remembered", "You haven't asked me to remember anything."))
            return
        facts = [t for t, _when in entries]
        lead = "You asked me to remember: "
        if re.search(r"\btoday\b", text, re.IGNORECASE):
            today = _dt.date.today().isoformat()
            todays = [t for t, when in entries if when.startswith(today)]
            if todays:
                facts, lead = todays, "Today you asked me to remember: "
            else:
                lead = "Nothing from today. Before that you asked me to remember: "
        shown = facts[-_RECALL_SPOKEN:]
        line = lead + "; ".join(f.rstrip(" .") for f in shown) + "."
        if len(facts) > len(shown):
            line += f" And {len(facts) - len(shown)} more before that."
        # spoken as it is, not rephrased: these are the speaker's own words
        await self._speak(line)

    async def _forget_fact(self, text: str) -> None:
        """"Forget about the park": this device's facts that match, offered in
        one question, and on a clear yes taken out of everything that holds
        them — the device's memory, the knowledge base (its own facts only,
        by id) and, for a note, both notes files."""
        if not _slots.has_lead_in("forget_fact", text):
            # "stop", heard as forget_fact: there is nothing to look for
            await self._unclear(text)
            return
        query = _slots.extract("forget_fact", text).get("query", "")
        memory = self.memory.device()
        embed = getattr(self.nlu, "embed", None)
        found = await asyncio.to_thread(
            _forgetting.select, query, memory.entries(),
            today=_dt.date.today().isoformat(), embed=embed if callable(embed) else None,
        )
        if not found:
            await self._speak(self.persona.line(
                "nothing_to_forget", "I have nothing like that to forget."))
            return
        prompt = self.persona.line("forget_fact_confirm", "Forget '{facts}'?")
        if not await self._confirmed(prompt.replace("{facts}", "; ".join(found))):
            await self._speak(self.persona.line("forget_kept", "I'll keep it."))
            return
        try:
            refs = await asyncio.to_thread(self._forget_found, memory, found)
            memory.forget_facts(found, refs)
        except Exception as exc:  # noqa: BLE001 — part of it may remain: say so
            log.error("could not forget %r for %s: %s", found, memory.device, exc)
            await self._speak(self.persona.line("error"))
            return
        await self._speak(self.persona.line("forgotten", "Forgotten."))

    def _forget_found(self, memory, found: list[str]) -> list[str]:
        """Blocking half of :meth:`_forget_fact`: out of the knowledge base and
        the notes files. Returns the knowledge refs that went."""
        gone: list[str] = []
        if self.knowledge is not None:
            texts = self.knowledge.store.fact_texts(memory.knowledge_refs())
            for ref, fact in texts.items():
                if any(same_text(fact, f) for f in found):
                    self.knowledge.store.remove_source(ref)
                    gone.append(ref)
            mirror = self.knowledge.docs_dir / _note_skill.MIRROR_FILE
            if mirror.is_file() and _drop_paragraphs(mirror, found):
                self.knowledge.index_file(mirror)
        if self._notes_file.is_file():
            _drop_note_lines(self._notes_file, found)
        return gone

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
                # M7: what this device said is in the interaction log too
                if self.learning.settings.log:
                    self.learning.log.forget(memory.device)
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
        if manifest is not None and manifest.origin == "learned" and manifest.params:
            return _learned_params(manifest.params, text)
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
                await self._speak(await self._phrase("I didn't catch that, sir."))
                return None
            params.update(found)
        return params

    async def refresh_skills(self) -> None:
        """An edge declared new tools (``RemoteServer.on_tools_changed``):
        retrain on a rebuilt registry and stage both for the merge gate, the
        same way a learned skill lands. Nothing to ask, nothing to self-check:
        the tools are the edge's, and a bad example list costs only them."""
        registry = self._latest_registry().rebuilt()
        examples = self._corpus(registry.manifests())
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

    async def _start_learning(self, request: LearningRequest, announce: bool = True) -> None:
        ahead = list(self._jobs)[-1] if self._jobs else None
        task = asyncio.ensure_future(self._learn(request, self._last_job, self._listening_device()))
        # registered before any await, so a dialog started right after this
        # already sees the name as busy
        self._jobs[request.name] = task
        self._last_job = task
        if not announce:
            return  # M7: the caller has already said what it is doing
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

    async def _learn(
        self, request: LearningRequest, previous: asyncio.Task | None, origin: str | None = None
    ) -> None:
        """One background job: wait for the one ahead of it, then build →
        validate → sandbox (`LearningJob`) → retrain → self-check → keep
        decision → stage. Never raises into the loop. What it says goes to
        ``origin``, the edge it was asked from."""
        _ORIGIN.set(origin)  # this task's own context: only this job's lines
        name = request.name
        outcome: FlowOutcome | None = None
        try:
            if previous is not None:
                # `wait`, not `await previous`: cancelling this job must not
                # cancel the one ahead of it
                await asyncio.wait({previous})
            decide = self._decide
            if request.autonomous and self.learning.settings.auto_permissions:
                decide = _granted
            job = LearningJob(
                request,
                notify=self._notify,
                decide=decide,
                registry=self._latest_registry(),
                generate=self._generate,
                sandbox=self.sandbox,
                max_attempts=self.config.factory.max_generate_attempts,
            )
            outcome = await job.run()
            if outcome.accepted:
                await self._retrain_and_stage(outcome, request.versioning, request)
            else:
                log.info("%s job not accepted: %s", name, outcome.reason)
                self._autonomous_done(request, False)
        except asyncio.CancelledError:
            self._unstage(outcome.name if outcome and outcome.name else name)
            raise
        except Exception:  # noqa: BLE001 — a job must never take the loop down
            log.exception("learning job for %s crashed", name)
            self._unstage(outcome.name if outcome and outcome.name else name)
            self._autonomous_done(request, False)
            self._queue_notice(f"Something went wrong while I was learning '{name}', sir.")
        finally:
            if self._jobs.get(name) is asyncio.current_task():
                del self._jobs[name]

    def _autonomous_done(self, request: LearningRequest, built: bool) -> None:
        """M7: how a build JARVIS started by itself ended, for the same-day
        request check."""
        if not request.autonomous or request.versioning != "new":
            return
        try:
            self.learning.state.set_request_status(request.name, "built" if built else "failed")
        except OSError as exc:
            log.error("could not record how learning %s ended: %s", request.name, exc)

    async def _retrain_and_stage(
        self, outcome: FlowOutcome, versioning: str, request: LearningRequest | None = None
    ) -> None:
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
            examples = self._corpus(other_manifests)
        else:
            examples = self._corpus(other_manifests + [outcome.manifest])

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
            if request is not None:
                self._autonomous_done(request, False)
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
                if request is not None:
                    self._autonomous_done(request, False)
                return

            autonomous = request is not None and request.autonomous
            if versioning not in ("revert", "remove") and not autonomous:
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
        if versioning != "remove":
            # a skill switched off for failing is back once it is taught, edited or reverted
            self.learning.state.enable(name)
        if request is not None and request.autonomous:
            self._autonomous_done(request, True)
            if versioning == "new":
                description = outcome.manifest.description.rstrip(".")
                description = description[:1].lower() + description[1:]
                self.learning.state.add_event("skill", name.replace("_", " "))
                self._queue_announcement(
                    f"I've learned to {description}, sir. Ask me again and I'll do it."
                )
            else:
                self.learning.state.add_event("repair", name.replace("_", " "))
                self._queue_announcement(f"I've repaired '{name}', sir.")
            return
        verb = {"new": "learned", "edit": "updated", "revert": "reverted", "remove": "removed"}[versioning]
        if versioning == "new":
            self.learning.state.add_event("skill", name.replace("_", " "))
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

    def _listening_device(self) -> str | None:
        """The edge the speaker is connected to now
        (``RemoteLink.connected_device_id``); None for the local speaker, or
        with no edge connected. (Not :meth:`_device`, which names whose turn
        it is, for memory, and falls back to ``local``.)"""
        return getattr(self.tts, "connected_device_id", None)

    def _here(self, origin: str | None) -> bool:
        """Can a line for ``origin`` be said now? Only to the device that
        asked; anything without an origin, to whoever is listening."""
        return origin is None or origin == self._listening_device()

    def _queue_announcement(self, text: str) -> None:
        origin = _ORIGIN.get()
        if self._pending_announcement and self._announcement_origin == origin:
            text = f"{self._pending_announcement} {text}"
        elif self._pending_announcement:
            self._notices.append((self._pending_announcement, self._announcement_origin))
        self._pending_announcement = text
        self._announcement_origin = origin

    def _queue_notice(self, text: str) -> None:
        self._notices.append((text, _ORIGIN.get()))

    async def _notify(self, text: str) -> None:
        """`LearningJob`'s `notify` — never speaks directly."""
        self._queue_notice(text)

    async def _decide(self, prompt: str) -> bool:
        """`LearningJob`'s `decide` — queue a yes/no question and wait until a
        safe point has asked it and got a clear answer."""
        decision = _Decision(
            prompt, asyncio.get_running_loop().create_future(), origin=_ORIGIN.get()
        )
        self._decisions.append(decision)
        try:
            return await decision.future
        finally:
            if decision in self._decisions:
                self._decisions.remove(decision)

    async def _speak_notices(self) -> None:
        """The queued notices for whoever is listening, oldest first; those
        for a device that is not connected wait for it."""
        while True:
            entry = next((n for n in self._notices if self._here(n[1])), None)
            if entry is None:
                return
            self._notices.remove(entry)
            await self._speak(entry[0])

    async def _safe_point(self, unprompted: bool = False) -> None:
        """No turn in flight and the user is present (between turns, or just
        before the drop to standby): merge a staged model, speak queued
        notices, and ask pending background questions.

        ``unprompted``: in standby, for the edge that asked (:meth:`_offer`).
        Each question is asked once that way, and no answer is not an unclear
        one — the watch may be in a pocket; the next session asks again."""
        self.state = "idle"
        await self._merge_gate()
        await self._speak_notices()
        for decision in list(self._decisions):
            if decision.future.done() or not self._here(decision.origin):
                continue
            if unprompted:
                if decision.offered:
                    continue
                decision.offered = True
            try:
                answer = await ask_yes_no_or_none(self._ask, decision.prompt)
            except _Cancelled:
                answer = None
            self.state = "idle"
            if answer is None and unprompted:
                continue
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
        self._maybe_retrain_phrasings()

    def _offer_due(self) -> bool:
        """A background question for the connected edge, not yet offered."""
        device = self._listening_device()
        return device is not None and any(
            d.origin == device and not d.offered and not d.future.done()
            for d in self._decisions
        )

    async def _offer(self) -> None:
        """Standby, and the edge a job was asked from is connected with its
        question waiting ("I've finished it — shall I keep it?"): ask now
        rather than at its next wake, then stand by again. A job asked on the
        local mic still waits for the wake word: nobody in the room is talked
        at out of the blue."""
        await self._safe_point(unprompted=True)
        if self.running:
            await self._enter_standby()

    async def _idle_tick(self) -> None:
        """Outside a wake session (standby): merge and speak notices, but never
        ask a question unprompted — those wait for the next session."""
        if self._in_session:
            return
        await self._merge_gate()
        await self._speak_notices()
        self._maybe_retrain_phrasings()

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
                # for the device that asked; another, or none, gets it later
                self._notices.insert(0, (announcement, self._announcement_origin))
                await self._speak_notices()

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
                woke = await self._await_wake()
                if not woke:
                    break
                self.standby = False
                self._in_session = True
                try:
                    if woke == "offer":
                        await self._offer()
                    else:
                        await self._session()
                finally:
                    self._in_session = False
                    self.state = "idle"
                    self.standby = True
        finally:
            self.running = False
            # the re-scan waits for its worker thread before it is done
            self._commit_provisional()
            for task in (self._idle_task, self._knowledge_task, self._phrasing_task):
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


# -- the notes files, for "forget about …" ---------------------------------------

_NOTED = re.compile(r"\s*\(noted [^)]*\)\s*$")


def _drop_paragraphs(path: Path, found: list[str]) -> bool:
    """Take the dictated notes that are one of ``found`` out of the knowledge
    base's mirror (one paragraph each). True if anything went."""
    paragraphs = path.read_text(encoding="utf-8").split("\n\n")
    kept = [p for p in paragraphs
            if not any(same_text(_NOTED.sub("", p), f) for f in found)]
    if len(kept) == len(paragraphs):
        return False
    path.write_text("\n\n".join(kept), encoding="utf-8")
    return True


def _drop_note_lines(path: Path, found: list[str]) -> None:
    """The same, from the `note` skill's own log ("<timestamp>  <text>")."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [l for l in lines
            if not any(same_text(l.split("  ", 1)[-1], f) for f in found)]
    if len(kept) != len(lines):
        path.write_text("".join(kept), encoding="utf-8")


def _close(classifier) -> None:
    close = getattr(classifier, "close", None)
    if callable(close):
        close()


def _loggable(params: dict) -> dict:
    """Params as the log can hold them: JSON's own types, else their text."""
    out = {}
    for key, value in (params or {}).items():
        out[key] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    return out


async def _granted(prompt: str) -> bool:
    """``[learning] auto_permissions``: a job's permission question, answered
    yes without asking."""
    log.info("granted without asking (auto_permissions): %s", prompt)
    return True
