"""M2.5 — background skill learning, at the orchestrator level.

`PRD/milestone-2.5-background-learning.md`, "Verification". After the
teach/edit/revert dialog, the build → sandbox → retrain → keep-confirm
pipeline runs as a background job. The session keeps serving commands. The
job's questions and notices are spoken only at safe points.

Everything is faked at the component boundary: a gated `generate` (so "still
building" is deterministic), a fake sandbox, a fake `train_and_load` (no
worker process, no fastembed), and a registry whose `rebuilt()` re-reads a
tmp `learned/` dir. `staging_dir`/`learned_source_path` and `data_dir` are
redirected into `tmp_path`, so nothing touches the real repo.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from types import SimpleNamespace

import pytest

from jarvis.config import load_config
from jarvis.core.orchestrator import Orchestrator
from jarvis.factory import flows as flows_mod
from jarvis.factory.claude_client import GeneratedSkill
from jarvis.factory.sandbox import SandboxResult
from jarvis.factory.validate import validate
from jarvis.nlu.classifier import Prediction
from jarvis.nlu.corpus import intent_meta
from jarvis.skills.contract import SkillManifest

_MODULE_TEMPLATE = '''\
from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="{name}",
    description="{description}",
    examples={examples!r},
    params={{}},
    permissions={perms},
)


def run(ctx, **params) -> str:
    return "{reply}"
'''

GOOD_TEST = "def test_ok():\n    assert True\n"

SEED_PROBES = {
    "search black holes": "search",
    "open github": "open_app",
    "go to sleep": "goodbye",
}


def module(name, *, perms='{"pure"}', reply="Heads.", examples=None):
    words = name.replace("_", " ")
    return _MODULE_TEMPLATE.format(
        name=name,
        description=f"Do the {words} thing.",
        examples=examples or [words, f"please {words}"],
        perms=perms,
        reply=reply,
    )


# -- fakes ------------------------------------------------------------------

class FakeNLU:
    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.closed = False

    def predict(self, text):
        return self.mapping.get(text, ("unknown", 0.1))

    def explain(self, text):
        label, conf = self.predict(text)
        return Prediction(label, conf, [(label, conf)], 1.0)

    def close(self):
        self.closed = True


class LearnRegistry:
    """Builtins + whatever validates in the (tmp) learned dir. `rebuilt()`
    re-reads that dir, like the real `Registry.rebuilt()` re-discovers."""

    def __init__(self, learned_dir, builtins=()):
        self.learned_dir = learned_dir
        self.builtins = list(builtins)
        self._manifests = list(builtins)
        for path in sorted(learned_dir.glob("*.py")):
            m = validate(path.read_text(encoding="utf-8"), frozenset())
            self._manifests.append(dataclasses.replace(m, origin="learned"))
        self.calls = []

    def dispatch(self, label, params):
        self.calls.append((label, params))
        if label not in self.names():
            raise KeyError(label)
        return f"did {label}"

    def names(self):
        return [m.name for m in self._manifests]

    def manifests(self):
        return list(self._manifests)

    def manifest(self, name):
        return next(m for m in self._manifests if m.name == name)

    def __contains__(self, name):
        return name in self.names()

    def rebuilt(self):
        return LearnRegistry(self.learned_dir, self.builtins)


class FakeClaude:
    """`generate_skill` blocks (in the job's worker thread) until `gate` is
    set, so a test can hold a job in "still building"."""

    available = True

    def __init__(self, *, perms='{"pure"}', block=False, raises=None):
        self.gate = threading.Event()
        if not block:
            self.gate.set()
        self.perms = perms
        self.raises = raises
        self.calls = []

    def generate_skill(self, spec, existing_source):
        self.calls.append(spec.name)
        self.gate.wait(10)
        if self.raises:
            raise self.raises
        return GeneratedSkill(
            name=spec.name,
            module_source=module(spec.name, perms=self.perms, examples=spec.examples),
            test_source=GOOD_TEST,
        )


class FakeSandbox:
    def __init__(self):
        self.calls = []

    def run_tests(self, module_path, test_path, permissions):
        self.calls.append("run_tests")
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)

    def dry_run(self, module_path, params, permissions):
        self.calls.append("dry_run")
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)


class FakeTrainer:
    """Stands in for the worker-process retrain + `Classifier.load`. The
    "trained" classifier knows every example it was trained on."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.label_sets = []
        self.version = 1

    async def __call__(self, examples):
        self.label_sets.append({e.label for e in examples})
        self.version += 1
        path = self.model_dir / f"v{self.version}"
        path.mkdir(parents=True)
        mapping = {e.text: (e.label, 0.95) for e in examples}
        mapping.update({t: (l, 0.95) for t, l in SEED_PROBES.items()})
        return "ok", (SimpleNamespace(path=path, version=self.version), FakeNLU(mapping))


class Voice:
    """mic + stt + tts in one. `lines` answers each capture in order (a
    callable is run in the capture thread first — used to time events inside
    a listen); running out reads as silence."""

    def __init__(self, lines=()):
        self.lines = list(lines)
        self.spoken = []
        self._pending = ""
        self.started = False

    def feed(self, *lines):
        self.lines.extend(lines)

    # mic
    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def drain(self):
        pass

    def read(self, timeout=None):
        import numpy as np

        return np.zeros(1280, dtype=np.int16)

    def record_utterance(self, *a, **k):
        import numpy as np

        item = self.lines.pop(0) if self.lines else ""
        if callable(item):
            item = item()
        self._pending = item
        return np.ones(1600, dtype=np.float32) * 0.1

    # stt
    last_avg_logprob = -0.2

    def transcribe(self, audio, cancel=None):
        text, self._pending = self._pending, ""
        return text

    # tts
    def say(self, text):
        self.spoken.append(text)


class Persona:
    def line(self, event, default=""):
        return f"<{event}>"

    def phrase(self, text):
        return text


class OneShotWake:
    def __init__(self):
        self.fired = False
        self.stop = None

    def reset(self):
        pass

    def triggered(self, frame):
        if not self.fired:
            self.fired = True
            return True
        if self.stop:
            self.stop()
        return False


# -- fixture ----------------------------------------------------------------

@pytest.fixture
def rig(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    learned = tmp_path / "learned"
    staging.mkdir()
    learned.mkdir()
    monkeypatch.setattr(flows_mod, "staging_dir", lambda: staging)
    monkeypatch.setattr(flows_mod, "learned_source_path", lambda name: learned / f"{name}.py")

    config = dataclasses.replace(load_config(), data_dir=tmp_path / "data")
    voice = Voice()
    builtins = [
        SkillManifest(name="search", description="Search the web.", examples=["search black holes"]),
        SkillManifest(name="open_app", description="Open an app.", examples=["open github"]),
    ]
    registry = LearnRegistry(learned, builtins)
    trainer = FakeTrainer(config.nlu_model_dir)
    claude = FakeClaude()
    sandbox = FakeSandbox()
    nlu = FakeNLU({
        "search black holes": ("search", 0.9),
        "learn how to flip a coin": ("teach", 0.95),
        "shut down": ("shutdown", 0.95),
    })
    o = Orchestrator(
        config=config, wake=OneShotWake(), mic=voice, stt=voice, tts=voice,
        nlu=nlu, persona=Persona(), registry=registry, intent_meta=intent_meta(),
        claude_client=claude, sandbox=sandbox, train_and_load=trainer,
    )
    o.wake.stop = o.stop
    o.standby = False
    return SimpleNamespace(
        o=o, voice=voice, claude=claude, sandbox=sandbox, trainer=trainer,
        staging=staging, learned=learned, config=config, tmp=tmp_path,
    )


async def until(pred, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


async def teach(rig, name="coin flip", description="flip a coin", example="toss a coin"):
    rig.voice.feed(name, description, example)
    await rig.o.handle("teach", "learn how to flip a coin", 0.95)


def spoken(rig):
    return " | ".join(rig.voice.spoken)


def keep_questions(rig):
    return [s for s in rig.voice.spoken if "keep it" in s.lower()]


# -- 1. control returns after the dialog ------------------------------------

def test_teach_returns_after_the_dialog_and_other_commands_still_run(rig):
    rig.claude = rig.o.claude_client = FakeClaude(block=True)

    async def scenario():
        await teach(rig)
        # the dialog is over, the build is still blocked in "Claude"
        assert rig.o.learning_names() == {"coin_flip"}
        assert "background" in spoken(rig).lower()
        await until(lambda: rig.claude.calls)
        assert not rig.claude.gate.is_set()

        await rig.o.handle("search", "search black holes", 0.9)
        assert rig.o.registry.calls == [("search", {"query": "black holes"})]

        rig.claude.gate.set()
        await rig.o.cancel_learning()

    asyncio.run(scenario())


# -- 2. keep question only at a safe point; yes -> merged -------------------

def test_keep_question_waits_for_a_safe_point_then_yes_merges(rig):
    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        # queued, not spoken: nothing was asked mid-turn
        assert keep_questions(rig) == []

        # another turn in flight doesn't ask it either
        await rig.o.handle("search", "search black holes", 0.9)
        assert keep_questions(rig) == []

        rig.voice.feed("yes")
        await rig.o._safe_point()
        assert len(keep_questions(rig)) == 1
        assert "coin_flip" in rig.o.registry
        assert "learned 'coin_flip'" in spoken(rig)
        assert (rig.learned / "coin_flip.py").is_file()
        assert not (rig.staging / "coin_flip.py").exists()

        # "try me" on the very next turn
        await rig.o.handle("coin_flip", "flip a coin", 0.95)
        assert ("coin_flip", {}) in rig.o.registry.calls
        await rig.o.cancel_learning()

    asyncio.run(scenario())


# -- 3. decline discards everything ----------------------------------------

def test_declined_keep_discards_staging_and_model_version(rig):
    old_nlu, old_registry = rig.o.nlu, rig.o.registry

    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        rig.voice.feed("no")
        await rig.o._safe_point()
        await until(lambda: not rig.o.learning_names())

    asyncio.run(scenario())
    assert rig.o.nlu is old_nlu
    assert rig.o.registry is old_registry
    assert not (rig.staging / "coin_flip.py").exists()
    assert not (rig.learned / "coin_flip.py").exists()
    assert not (rig.config.nlu_model_dir / "v2").exists()


# -- 4. permissions are granted before the sandbox runs ---------------------

def test_permission_is_asked_before_sandbox_and_then_keep(rig):
    rig.claude = rig.o.claude_client = FakeClaude(perms='{"net"}')

    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        assert "access" in rig.o.pending_questions()[0].lower()
        assert rig.sandbox.calls == []

        rig.voice.feed("yes")
        await rig.o._safe_point()
        await until(lambda: rig.o.pending_questions())
        assert rig.sandbox.calls == ["run_tests", "dry_run"]
        assert "keep it" in rig.o.pending_questions()[0].lower()

        rig.voice.feed("yes")
        await rig.o._safe_point()
        assert "coin_flip" in rig.o.registry
        await rig.o.cancel_learning()

    asyncio.run(scenario())


def test_declined_permission_never_sandboxes(rig):
    rig.claude = rig.o.claude_client = FakeClaude(perms='{"net"}')

    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        rig.voice.feed("no")
        await rig.o._safe_point()
        await until(lambda: not rig.o.learning_names())

    asyncio.run(scenario())
    assert rig.sandbox.calls == []
    assert rig.trainer.label_sets == []
    assert "coin_flip" not in rig.o.registry


# -- 5. an unclear answer stays pending, 3 strikes -> no --------------------

def test_unclear_answer_leaves_the_question_pending(rig):
    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())

        rig.voice.feed("what time is it")
        await rig.o._safe_point()
        assert len(rig.o.pending_questions()) == 1
        assert "coin_flip" not in rig.o.registry

        rig.voice.feed("yes")
        await rig.o._safe_point()
        assert "coin_flip" in rig.o.registry
        await rig.o.cancel_learning()

    asyncio.run(scenario())


def test_three_unclear_answers_resolve_as_no(rig):
    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        for _ in range(3):
            rig.voice.feed("hmm")
            await rig.o._safe_point()
        await until(lambda: not rig.o.learning_names())
        await rig.o._safe_point()  # speak the resulting notice

    asyncio.run(scenario())
    assert len(keep_questions(rig)) == 3
    assert "coin_flip" not in rig.o.registry
    assert "set 'coin_flip' aside" in spoken(rig)


# -- 6. a question is asked before the drop to standby ----------------------

def test_pending_question_is_asked_before_standby(rig):
    rig.claude = rig.o.claude_client = FakeClaude(block=True)
    o = rig.o

    def release_during_the_follow_up_listen():
        rig.claude.gate.set()
        # wait (in the capture thread) until the job has queued its question
        for _ in range(500):
            if o.pending_questions():
                break
            threading.Event().wait(0.01)
        return ""  # ...then the user stays silent

    rig.voice.feed(
        "learn how to flip a coin", "coin flip", "flip a coin", "toss a coin",
        release_during_the_follow_up_listen,
        "yes",  # the keep question, asked at the session's end
    )
    asyncio.run(asyncio.wait_for(o.run(), timeout=10))

    lines = rig.voice.spoken
    keep_at = next(i for i, s in enumerate(lines) if "keep it" in s.lower())
    standby_at = max(i for i, s in enumerate(lines) if s == "<standby>")
    assert keep_at < standby_at
    assert "coin_flip" in o.registry


# -- 7. jobs are serialized and later jobs see earlier ones -----------------

def test_back_to_back_teaches_are_serialized_and_keep_both(rig):
    async def scenario():
        await teach(rig, name="coin flip", description="flip a coin", example="toss a coin")
        await teach(rig, name="dice roll", description="roll a die", example="roll the dice")
        assert rig.o.learning_names() == {"coin_flip", "dice_roll"}
        assert "after 'coin_flip'" in spoken(rig)

        await until(lambda: rig.o.pending_questions())
        assert rig.claude.calls == ["coin_flip"]  # the second hasn't started

        rig.voice.feed("yes")
        await rig.o._safe_point()
        await until(lambda: rig.o.pending_questions())
        assert "dice_roll" in rig.o.pending_questions()[0]

        rig.voice.feed("yes")
        await rig.o._safe_point()
        await rig.o.cancel_learning()

    asyncio.run(scenario())
    assert rig.claude.calls == ["coin_flip", "dice_roll"]
    assert "coin_flip" in rig.trainer.label_sets[1]  # 2nd retrain saw the 1st
    assert {"coin_flip", "dice_roll"} <= set(rig.o.registry.names())
    assert rig.o.nlu.predict("toss a coin")[0] == "coin_flip"
    assert rig.o.nlu.predict("roll the dice")[0] == "dice_roll"


# -- 8. a duplicate teach is refused ----------------------------------------

def test_teaching_a_name_already_in_flight_is_refused(rig):
    rig.claude = rig.o.claude_client = FakeClaude(block=True)

    async def scenario():
        await teach(rig)
        rig.voice.feed("coin flip", "")  # the same name twice, then give up
        await rig.o.handle("teach", "learn how to flip a coin", 0.95)
        assert rig.o.learning_names() == {"coin_flip"}
        assert "already working on 'coin_flip'" in spoken(rig)
        rig.claude.gate.set()
        await rig.o.cancel_learning()

    asyncio.run(scenario())


# -- 9. a failure is a notice at a safe point, never mid-turn ---------------

def test_build_failure_is_announced_at_the_next_safe_point(rig):
    rig.claude = rig.o.claude_client = FakeClaude(raises=RuntimeError("API down"))

    async def scenario():
        await teach(rig)
        await until(lambda: not rig.o.learning_names())
        assert "couldn't" not in spoken(rig)
        await rig.o.handle("search", "search black holes", 0.9)
        assert "couldn't" not in spoken(rig)
        await rig.o._safe_point()
        assert "couldn't put 'coin_flip' together" in spoken(rig)

    asyncio.run(scenario())


def test_standby_idle_tick_speaks_notices_but_never_asks(rig):
    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        rig.o.standby = True
        rig.o._queue_notice("A notice, sir.")
        await rig.o._idle_tick()
        assert "A notice, sir." in rig.voice.spoken
        assert keep_questions(rig) == []
        assert rig.o.pending_questions()
        await rig.o.cancel_learning()

    asyncio.run(scenario())


# -- 10. shutdown cancels a job cleanly -------------------------------------

def test_shutdown_cancels_a_running_job_and_cleans_up(rig):
    rig.claude = rig.o.claude_client = FakeClaude(block=True)
    rig.voice.feed(
        "learn how to flip a coin", "coin flip", "flip a coin", "toss a coin",
        lambda: "shut down",
    )
    try:
        asyncio.run(asyncio.wait_for(rig.o.run(), timeout=5))
    finally:
        rig.claude.gate.set()
    assert rig.o.running is False
    assert rig.o.learning_names() == set()
    assert list(rig.staging.glob("*.py")) == []


def test_cancel_during_a_pending_question_discards_the_trained_version(rig):
    async def scenario():
        await teach(rig)
        await until(lambda: rig.o.pending_questions())
        await rig.o.cancel_learning()

    asyncio.run(scenario())
    assert rig.o.pending_questions() == []
    assert not (rig.config.nlu_model_dir / "v2").exists()
    assert list(rig.staging.glob("*.py")) == []


# -- 11. revert runs in the background with no keep question ----------------

def test_revert_runs_in_the_background_without_a_keep_question(rig):
    v1 = module("coin_flip", reply="Heads v1.")
    v2 = module("coin_flip", reply="Heads v2.")
    (rig.learned / "coin_flip.py").write_text(v2, encoding="utf-8")
    vdir = rig.config.skill_versions_dir("coin_flip")
    vdir.mkdir(parents=True)
    (vdir / "v1.py").write_text(v1, encoding="utf-8")
    rig.o.registry = rig.o.registry.rebuilt()

    async def scenario():
        rig.voice.feed("coin flip")
        await rig.o.handle("revert_skill", "revert the coin flip skill", 0.95)
        await until(lambda: not rig.o.learning_names())
        await rig.o._safe_point()

    asyncio.run(scenario())
    assert keep_questions(rig) == []
    assert "reverted 'coin_flip'" in spoken(rig)
    assert (rig.learned / "coin_flip.py").read_text(encoding="utf-8") == v1
    assert list(rig.config.skill_quarantine_dir.glob("coin_flip.*.py"))
