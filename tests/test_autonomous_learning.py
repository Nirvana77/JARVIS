"""M7 parts E and F: JARVIS builds a skill for a request nothing fits, and
repairs a learned skill that breaks, both without being asked.

The owner chose "fully autonomous": no "Shall I keep it?". What stays is
a permission beyond pure/notify still waits for a spoken yes (unless
``[learning] auto_permissions``), a daily cap on builds and on repairs, and
a skill that keeps failing is switched off.

The rig is ``test_background_learning``'s: fake Claude, sandbox and trainer,
staging and learned dirs in ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import dataclasses

from jarvis.config import LearningConfig
from jarvis.core import reasoning
from jarvis.learning import Learning
from jarvis.factory.sandbox import SandboxResult
from tests.test_background_learning import (  # noqa: F401 — `rig` is a fixture
    FakeClaude,
    keep_questions,
    module,
    rig,
    spoken,
    until,
)
from tests.test_misheard import FakeReasoner

LEARN_DIE = '{"learn": "roll a die", "examples": ["throw a die", "roll a d6"]}'


def setup(rig, reasoner=None, factory=None, **learning):
    config = rig.o.config
    config = dataclasses.replace(
        config,
        learning=dataclasses.replace(LearningConfig(), **learning),
        # the guess is M4's job; these tests are about what `_think` says
        reasoner=dataclasses.replace(config.reasoner, correct_misheard=False),
    )
    if factory:
        config = dataclasses.replace(config, factory=dataclasses.replace(config.factory, **factory))
    rig.o.config = config
    rig.o.learning = Learning.in_memory(config)
    rig.o.reasoner = reasoner
    return rig.o


async def settle(rig):
    """Let the background job run to the end and the merge gate swap it in."""
    await until(lambda: not rig.o._jobs)
    await rig.o._safe_point()


# -- the reasoner's new answer -----------------------------------------------------

def test_the_reasoner_may_ask_to_learn():
    thought = reasoning.parse(LEARN_DIE)
    assert thought.learn == "roll a die"
    assert thought.examples == ("throw a die", "roll a d6")
    assert not thought.commands and not thought.answer


def test_a_learn_reply_without_a_description_is_nothing():
    assert reasoning.parse('{"learn": "  "}') is None


def test_the_prompt_offers_learning():
    assert '"learn"' in reasoning.system("")


# -- E: building -----------------------------------------------------------------------

def test_a_new_capability_is_built_and_kept_without_asking(rig):
    o = setup(rig, FakeReasoner(LEARN_DIE))

    async def go():
        await o.handle("unknown", "roll me a die", 0.1)
        assert "<learning_it>" in rig.voice.spoken
        assert o.learning_names() == {"roll_a_die"}
        await settle(rig)
    asyncio.run(go())

    assert keep_questions(rig) == []
    assert "roll_a_die" in o.registry.names()
    assert (rig.learned / "roll_a_die.py").is_file()
    assert "roll a die" in spoken(rig)               # announced
    state = o.learning.state
    assert state.builds_today() == 1
    assert [(r["name"], r["status"]) for r in state.requests_today()] == [("roll_a_die", "built")]
    assert [e["kind"] for e in state.events()] == ["skill"]


def test_the_heard_words_are_one_of_its_examples(rig):
    o = setup(rig, FakeReasoner(LEARN_DIE))
    seen = []
    claude = rig.o.claude_client
    original = claude.generate_skill

    def spy(spec, existing, feedback=None):
        seen.append(spec)
        return original(spec, existing, feedback)

    claude.generate_skill = spy

    async def go():
        await o.handle("unknown", "roll me a die", 0.1)
        await settle(rig)
    asyncio.run(go())
    assert "roll me a die" in seen[0].examples
    assert "throw a die" in seen[0].examples


def test_a_permission_still_waits_for_a_yes(rig):
    rig.o.claude_client = FakeClaude(perms='{"pure", "net"}')
    o = setup(rig, FakeReasoner(LEARN_DIE))

    async def go():
        await o.handle("unknown", "roll me a die", 0.1)
        await until(lambda: o.pending_questions())
        assert "net access" in o.pending_questions()[0]
        await o.cancel_learning()
    asyncio.run(go())


def test_auto_permissions_grants_them(rig):
    rig.o.claude_client = FakeClaude(perms='{"pure", "net"}')
    o = setup(rig, FakeReasoner(LEARN_DIE), auto_permissions=True)

    async def go():
        await o.handle("unknown", "roll me a die", 0.1)
        await settle(rig)
    asyncio.run(go())
    assert o.pending_questions() == []
    assert "roll_a_die" in o.registry.names()


def test_the_daily_cap_stops_a_build(rig):
    o = setup(rig, FakeReasoner(LEARN_DIE), max_builds_per_day=0)
    asyncio.run(o.handle("unknown", "roll me a die", 0.1))
    assert rig.voice.spoken[-1] == "<learn_limit>"
    assert rig.claude.calls == [] and not o._jobs


def test_the_same_request_is_not_built_twice(rig):
    rig.o.claude_client = rig.claude = FakeClaude(block=True)
    o = setup(rig, FakeReasoner(LEARN_DIE))

    async def go():
        await o.handle("unknown", "roll me a die", 0.1)
        await o.handle("unknown", "roll me a die", 0.1)
        assert rig.voice.spoken[-1] == "<already_learning>"
        assert list(o._jobs) == ["roll_a_die"]
        await o.cancel_learning()
        rig.claude.gate.set()
    asyncio.run(go())


def test_a_request_that_failed_today_is_not_tried_again(rig):
    rig.o.claude_client = FakeClaude(raises=RuntimeError("no"))
    o = setup(rig, FakeReasoner(LEARN_DIE), factory={"max_generate_attempts": 1})

    async def go():
        await o.handle("unknown", "roll me a die", 0.1)
        await settle(rig)
        await o.handle("unknown", "roll me a die", 0.1)
    asyncio.run(go())
    assert rig.voice.spoken[-1] == "<learn_failed_today>"
    assert o.learning.state.requests_today()[0]["status"] == "failed"


def test_a_plan_of_one_step_nothing_knows_is_a_capability(rig):
    """Seen with qwen3:8b on the cluster (2026-10-03): asked about "roll a
    twenty sided die" it sent a plan of that very sentence rather than
    "learn". A single step the classifier does not know is a capability."""
    o = setup(rig, FakeReasoner('{"commands": ["roll a twenty sided die"]}'))

    async def go():
        await o.handle("unknown", "roll a twenty sided die", 0.1)
        assert "<learning_it>" in rig.voice.spoken
        assert list(o._jobs) == ["roll_a_twenty_sided"]
        await settle(rig)
    asyncio.run(go())


def test_a_plan_with_an_unknown_step_among_known_ones_is_still_dropped(rig):
    o = setup(rig, FakeReasoner('{"commands": ["search black holes", "roll a die"]}'))
    asyncio.run(o.handle("unknown", "search black holes and roll a die", 0.1))
    assert rig.voice.spoken[-1] == "<unknown>" and not o._jobs


def test_noise_builds_nothing(rig):
    o = setup(rig, FakeReasoner('{"none": true}'))
    asyncio.run(o.handle("unknown", "uh the", 0.1))
    assert rig.voice.spoken[-1] == "<unknown>" and not o._jobs


def test_without_a_reasoner_nothing_is_built(rig):
    o = setup(rig, None)
    asyncio.run(o.handle("unknown", "roll me a die", 0.1))
    assert rig.voice.spoken[-1] == "<unknown>" and not o._jobs


def test_auto_build_off_builds_nothing(rig):
    o = setup(rig, FakeReasoner(LEARN_DIE), auto_build=False)
    asyncio.run(o.handle("unknown", "roll me a die", 0.1))
    assert rig.voice.spoken[-1] == "<unknown>" and not o._jobs


def test_learning_off_builds_nothing(rig):
    o = setup(rig, FakeReasoner(LEARN_DIE), enabled=False)
    asyncio.run(o.handle("unknown", "roll me a die", 0.1))
    assert rig.voice.spoken[-1] == "<unknown>" and not o._jobs


def test_a_new_name_never_collides(rig):
    rig.o.claude_client = rig.claude = FakeClaude(block=True)
    o = setup(rig, FakeReasoner('{"learn": "search"}'))

    async def go():
        await o.handle("unknown", "look something up differently", 0.1)
        assert list(o._jobs) == ["search_2"]
        await o.cancel_learning()
        rig.claude.gate.set()
    asyncio.run(go())


# -- F: repairing ----------------------------------------------------------------------

class RecordingSandbox:
    """Records every dry run's params; ``fails_with`` makes a dry run with
    exactly those params fail, as the unrepaired call would."""

    def __init__(self, fails_with=None):
        self.dry_runs = []
        self.fails_with = fails_with

    def run_tests(self, module_path, test_path, permissions):
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)

    def dry_run(self, module_path, params, permissions):
        self.dry_runs.append(dict(params))
        if self.fails_with is not None and params == self.fails_with:
            return SandboxResult(ok=False, stdout="", stderr="ValueError: still broken", returncode=1)
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)


def install(rig, name="flip_a_coin"):
    """A learned skill that raises when it runs."""
    (rig.learned / f"{name}.py").write_text(module(name), encoding="utf-8")
    o = rig.o
    o.registry = o.registry.rebuilt()
    registry = o.registry

    def dispatch(label, params):
        registry.calls.append((label, params))
        raise ValueError("bad count")

    registry.dispatch = dispatch
    return o


def test_a_failing_learned_skill_is_repaired_in_the_background(rig):
    sandbox = rig.o.sandbox = RecordingSandbox()
    o = install(rig)
    o = setup(rig, None)

    async def go():
        await o.handle("flip_a_coin", "flip three coins", 0.9)
        assert "<error>" in rig.voice.spoken
        assert o.learning_names() == {"flip_a_coin"}
        await settle(rig)
    asyncio.run(go())

    assert rig.claude.calls == ["flip_a_coin"]
    assert len(sandbox.dry_runs) == 2  # the sample call, then the call that failed
    assert keep_questions(rig) == []
    assert "repaired" in spoken(rig).lower()
    assert o.learning.state.repairs_today("flip_a_coin") == 1
    assert [e["kind"] for e in o.learning.state.events()] == ["repair"]


def test_the_repair_is_told_what_failed(rig):
    rig.o.sandbox = RecordingSandbox()
    o = install(rig)
    o = setup(rig, None)
    specs = []
    original = rig.claude.generate_skill
    rig.claude.generate_skill = lambda spec, existing, feedback=None: (
        specs.append((spec, existing)) or original(spec, existing, feedback)
    )

    async def go():
        await o.handle("flip_a_coin", "flip three coins", 0.9)
        await settle(rig)
    asyncio.run(go())
    spec, existing = specs[0]
    assert "ValueError: bad count" in spec.description
    assert "flip three coins" in spec.description
    assert existing and "MANIFEST" in existing


def test_a_repair_that_does_not_fix_the_failing_call_is_not_kept(rig):
    sandbox = rig.o.sandbox = RecordingSandbox(fails_with={"count": 3})
    o = install(rig)
    o = setup(rig, None, factory={"max_generate_attempts": 1})
    o._params_for = lambda label, text: {"count": 3}

    async def go():
        await o.handle("flip_a_coin", "flip three coins", 0.9)
        await settle(rig)
    asyncio.run(go())
    assert {"count": 3} in sandbox.dry_runs
    assert "repaired" not in spoken(rig).lower()
    assert o._staged is None


def test_a_skill_that_keeps_failing_is_switched_off(rig):
    rig.o.sandbox = RecordingSandbox()
    o = install(rig)
    o = setup(rig, None, max_repairs_per_skill_per_day=0)

    async def go():
        await o.handle("flip_a_coin", "flip a coin", 0.9)
        assert rig.voice.spoken[-1] == "<skill_disabled>"
        assert not o._jobs
        calls = len(o.registry.calls)
        await o.handle("flip_a_coin", "flip a coin", 0.9)
        assert rig.voice.spoken[-1] == "<out_of_order>"
        assert len(o.registry.calls) == calls        # not even tried
    asyncio.run(go())
    assert o.learning.state.is_disabled("flip_a_coin")


def test_a_failing_builtin_is_written_down_not_repaired(rig):
    o = setup(rig, None)

    def dispatch(label, params):
        raise RuntimeError("weather service moved")

    o.registry.dispatch = dispatch
    asyncio.run(o.handle("search", "search black holes", 0.9))
    assert not o._jobs and rig.claude.calls == []
    (failure,) = o.learning.state.builtin_failures()
    assert failure["skill"] == "search" and "weather service moved" in failure["error"]


def test_repair_off_repairs_nothing(rig):
    rig.o.sandbox = RecordingSandbox()
    o = install(rig)
    o = setup(rig, None, auto_repair=False)
    asyncio.run(o.handle("flip_a_coin", "flip a coin", 0.9))
    assert not o._jobs and rig.claude.calls == []
