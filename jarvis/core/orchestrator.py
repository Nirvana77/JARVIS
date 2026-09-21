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
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import queue
import random
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jarvis.core.interrupt import Cancelled as _Cancelled
from jarvis.core.interrupt import Interrupter
from jarvis.factory import flows as _flows
from jarvis.factory.flows import (
    EditSkillFlow,
    FlowOutcome,
    LearningRequest,
    RevertSkillFlow,
    TeachFlow,
    ask_yes_no,
    ask_yes_no_or_none,
)
from jarvis.factory.jobs import LearningJob
from jarvis.nlu import slots as _slots
from jarvis.nlu.classifier import UNKNOWN, Classifier
from jarvis.nlu.corpus import build_corpus
from jarvis.nlu.retrain_worker import RetrainWorker

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
}

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
        claude_client=None,
        sandbox=None,
        train_and_load=None,
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
            await self.handle(label, text, confidence)

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
            self.standby = True
            await self._speak(self.persona.line("standby"))

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

    async def handle(self, label: str, text: str, confidence: float = 1.0) -> None:
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

        if action in ("teach", "edit_skill", "revert_skill") and confidence < self.config.nlu.meta_action_threshold:
            # A wrong guess here launches a whole multi-turn dialog (or, for
            # revert_skill, rewrites a skill's live code) — costlier than a
            # wrong guess on an ordinary skill, so a merely-above-"unknown"
            # confidence isn't enough to commit to it.
            log.info(
                "treating low-confidence %s (%.2f < %.2f) as unknown",
                label, confidence, self.config.nlu.meta_action_threshold,
            )
            await self._speak(self.persona.line("unknown"))
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

    def _generate(self, spec, existing_source):
        """Bound to `self.claude_client` so `build()`/the flows never import
        `anthropic` — passed as the flows' `generate` callable."""
        return self.claude_client.generate_skill(spec, existing_source)

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
        registry = self._latest_registry()
        other_manifests = [m for m in registry.manifests() if m.name != name]
        examples = build_corpus(manifests=other_manifests + [outcome.manifest])

        kind, payload = await self._train_and_load(examples)
        if kind != "ok":
            log.error("retrain failed for %s: %s", name, payload)
            self._unstage(name)
            self._queue_notice(f"'{name}' didn't train cleanly, sir. I've set it aside.")
            return

        train_result, classifier = payload
        try:
            if not self._self_check(classifier, outcome.manifest):
                self._discard_unused_version(train_result)
                self._unstage(name)
                self._queue_notice(f"I set '{name}' aside, sir; it didn't check out in practice.")
                return

            if versioning != "revert":
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
        verb = {"new": "learned", "edit": "updated", "revert": "reverted"}[versioning]
        self._queue_announcement(f"I've {verb} '{name}', sir. My capabilities are updated.")

    async def _default_train_and_load(self, examples):
        worker = RetrainWorker()
        await asyncio.to_thread(
            worker.start, examples, self.config.nlu.embedding_model, self.config.nlu_model_dir
        )
        try:
            while True:
                result = await asyncio.to_thread(worker.poll, 0.5)
                if result is not None:
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

    def _self_check(self, classifier: Classifier, manifest) -> bool:
        """A freshly-trained model must still classify the seed intents *and*
        the new/changed skill's own examples correctly before it's ever
        staged — PRD: "new model must load and still classify the seed
        examples sanely.\""""
        for text, expected in _SELF_CHECK_SEED_PROBES:
            label, _ = classifier.predict(text)
            if label != expected:
                log.warning("self-check regression: %r -> %s (expected %s)", text, label, expected)
                return False
        for text in manifest.examples:
            label, _ = classifier.predict(text)
            if label != manifest.name:
                log.warning(
                    "self-check: %r -> %s (expected %s)", text, label, manifest.name
                )
                return False
        return True

    def _discard_unused_version(self, train_result) -> None:
        """A trained-but-never-staged model version — self-check failed, or
        the user declined the confirm. Delete it so disk state matches the
        live (unchanged) `self.nlu`."""
        shutil.rmtree(train_result.path, ignore_errors=True)

    def _promote_files(self, name: str, module_source: str, versioning: str, outcome: FlowOutcome):
        """Write the confirmed module to `skills/learned/`, with version /
        quarantine bookkeeping per PRD directory-layout decision 4, and
        return a freshly rebuilt registry. The actual `self.registry` swap
        happens later, through `self._staged` + the merge gate."""
        registry = self._latest_registry()
        learned_path = _flows.learned_source_path(name)
        learned_path.parent.mkdir(parents=True, exist_ok=True)

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
            if self._idle_task is not None:
                self._idle_task.cancel()
                try:
                    await self._idle_task
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
