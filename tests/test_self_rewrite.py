"""M8 part A in the orchestrator: a builtin that fails is rewritten in the
background, swapped in live as an override, watched for its first uses
(probation), reverted on its own if it fails again, and "undo that" puts it
back. The owner chose: skills and builtins only; deploy, tell after.

The rig is test_background_learning's (fake Claude, trainer, registry);
overrides are written to tmp_path.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from jarvis.config import LearningConfig
from jarvis.factory import flows as flows_mod
from jarvis.factory.sandbox import SandboxResult
from jarvis.learning import Learning
from tests.test_background_learning import (  # noqa: F401 — `rig` is a fixture
    keep_questions,
    rig,
    spoken,
    until,
)


class GateSandbox:
    def __init__(self):
        self.repo_tests = []
        self.dry_runs = []

    def run_tests(self, module_path, test_path, permissions):
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)

    def dry_run(self, module_path, params, permissions):
        self.dry_runs.append(dict(params))
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)

    def run_repo_tests(self, module_path, name, test_files, permissions):
        self.repo_tests.append((name, tuple(test_files)))
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)


@pytest.fixture
def self_rig(rig, tmp_path, monkeypatch):
    overrides = tmp_path / "overrides"
    monkeypatch.setattr(flows_mod, "override_source_path", lambda name: overrides / f"{name}.py")
    o = rig.o
    config = dataclasses.replace(o.config, learning=dataclasses.replace(LearningConfig(), probation_calls=3))
    o.config = config
    o.learning = Learning.in_memory(config)
    o.sandbox = rig.sandbox = GateSandbox()
    rig.overrides = overrides
    state = {"raise": True, "calls": 0}
    registry = o.registry

    def dispatch(label, params):
        registry.calls.append((label, params))
        state["calls"] += 1
        if state["raise"]:
            raise RuntimeError("wikipedia moved")
        return "According to Wikipedia, sir: fine."

    def install(reg):
        reg.dispatch = dispatch
        original = reg.rebuilt

        def rebuilt():
            new = original()
            install(new)
            return new

        reg.rebuilt = rebuilt

    install(registry)
    rig.state = state
    return rig


async def fail_and_rewrite(rig):
    await rig.o.handle("search", "search black holes", 0.9)
    await until(lambda: not rig.o._jobs)
    await rig.o._safe_point()


def test_a_failing_builtin_is_rewritten_and_swapped_in(self_rig):
    rig = self_rig

    async def go():
        await rig.o.handle("search", "search black holes", 0.9)
        assert rig.o.learning_names() == {"search"}
        await until(lambda: not rig.o._jobs)
        await rig.o._safe_point()
    asyncio.run(go())

    assert (rig.overrides / "search.py").is_file()
    assert keep_questions(rig) == []
    assert "rewritten 'search'" in spoken(rig) and "undo that" in spoken(rig).lower()
    assert rig.sandbox.repo_tests and rig.sandbox.repo_tests[0][0] == "search"
    assert [e["kind"] for e in rig.o.learning.state.events()] == ["rewrite"]
    assert rig.o.learning.state.builtin_failures()  # still written down for a person


def test_the_rewrite_replays_the_builtins_recent_good_calls(self_rig):
    rig = self_rig
    rig.o.learning.log.append({"device": "local", "heard": "search python", "path": "direct",
                               "skill": "search", "params": {"query": "python"}, "outcome": "ok"})
    asyncio.run(fail_and_rewrite(rig))
    assert {"query": "python"} in rig.sandbox.dry_runs


def test_a_rewrite_that_fails_on_probation_is_reverted_on_its_own(self_rig):
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        assert (rig.overrides / "search.py").is_file()
        await rig.o.handle("search", "search black holes", 0.9)   # still raises
        await rig.o._safe_point()
    asyncio.run(go())
    assert not (rig.overrides / "search.py").exists()
    assert "put back" in spoken(rig)
    assert "reverted" in [e["kind"] for e in rig.o.learning.state.events()]
    assert not rig.o._jobs  # no second rewrite of a rewrite that just failed


def test_a_rewrite_that_works_through_probation_stays(self_rig):
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        rig.state["raise"] = False
        for _ in range(3):
            await rig.o.handle("search", "search black holes", 0.9)
    asyncio.run(go())
    assert rig.o._on_probation("search") is None
    assert (rig.overrides / "search.py").is_file()


def test_undo_that_puts_the_previous_version_back(self_rig):
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        rig.state["raise"] = False
        await rig.o.handle("undo_change", "undo that", 0.9)
        await rig.o._safe_point()
    asyncio.run(go())
    assert not (rig.overrides / "search.py").exists()
    assert rig.voice.spoken[-1].startswith("Undone, sir")


def test_undo_with_nothing_to_undo_says_so(self_rig):
    asyncio.run(self_rig.o.handle("undo_change", "undo that", 0.9))
    assert self_rig.voice.spoken[-1] == "<nothing_to_undo>"


def test_a_rewrite_of_a_rewrite_reverts_to_the_first_rewrite(self_rig):
    rig = self_rig
    rig.overrides.mkdir(parents=True)
    (rig.overrides / "search.py").write_text("# the first rewrite\n", encoding="utf-8")

    async def go():
        await fail_and_rewrite(rig)
        await rig.o.handle("undo_change", "undo that", 0.9)
    asyncio.run(go())
    assert (rig.overrides / "search.py").read_text(encoding="utf-8") == "# the first rewrite\n"


def test_past_the_daily_cap_a_builtin_is_not_rewritten_again(self_rig):
    rig = self_rig
    rig.o.config = dataclasses.replace(
        rig.o.config, learning=dataclasses.replace(rig.o.config.learning, max_repairs_per_skill_per_day=0)
    )
    rig.o.learning = Learning.in_memory(rig.o.config)
    asyncio.run(rig.o.handle("search", "search black holes", 0.9))
    assert not rig.o._jobs and rig.claude.calls == []
    assert not rig.o.learning.state.is_disabled("search")  # a builtin is never switched off


def test_what_was_rewritten_today_is_told(self_rig):
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        await rig.o.handle("learned_today", "what have you learned today", 0.9)
    asyncio.run(go())
    assert "rewrote search" in rig.voice.spoken[-1]


# -- every kept rewrite, and every revert, goes to jarvis/self --------------------------------

class FakePublisher:
    available = True

    def __init__(self):
        self.published = []

    def publish(self, path, content, message):
        self.published.append((path, content, message))
        return "c0ffee"


def test_a_kept_rewrite_is_published_to_its_branch(self_rig):
    rig = self_rig
    rig.o.publisher = FakePublisher()

    async def go():
        await fail_and_rewrite(rig)
        await rig.o._publishing_done()
    asyncio.run(go())
    ((path, content, message),) = rig.o.publisher.published
    assert path == "jarvis/skills/builtin/search.py"
    assert content == (rig.overrides / "search.py").read_text(encoding="utf-8")
    assert "search" in message and "wikipedia moved" in message


def test_a_revert_is_published_too(self_rig):
    rig = self_rig
    rig.o.publisher = FakePublisher()

    async def go():
        await fail_and_rewrite(rig)
        await rig.o.handle("undo_change", "undo that", 0.9)
        await rig.o._publishing_done()
    asyncio.run(go())
    paths = [p for p, _c, _m in rig.o.publisher.published]
    assert paths == ["jarvis/skills/builtin/search.py"] * 2
    restored = rig.o.publisher.published[-1][1]
    assert restored == flows_mod.packaged_source_path("search").read_text(encoding="utf-8")


# -- found in the dry run (2026-10-03) ----------------------------------------------------

def test_asking_again_after_the_skill_was_rewritten_uses_the_rewrite(self_rig):
    """"What time is it?" failed, the clock was rewritten, "What time is it?"
    again counted as asked-again and ruled the fresh rewrite out."""
    rig = self_rig

    async def go():
        await rig.o._turn("search", "what is a black hole?", 0.9)       # fails
        await until(lambda: not rig.o._jobs)
        await rig.o._safe_point()                                        # rewrite in
        rig.state["raise"] = False
        calls = rig.state["calls"]
        await rig.o._turn("search", "what is a black hole?", 0.9)       # asked again
        assert rig.state["calls"] == calls + 1
    asyncio.run(go())
    assert rig.o.learning.log.records()[-1]["path"] != "retry"


def test_an_unsure_undo_asks_before_it_puts_anything_back(self_rig):
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        rig.voice.feed("yes")
        await rig.o.handle("undo_change", "Undo that.", 0.59)
    asyncio.run(go())
    assert any(s == "<undo_confirm>" for s in rig.voice.spoken)
    assert not (rig.overrides / "search.py").exists()


def test_an_unsure_undo_answered_no_keeps_the_rewrite(self_rig):
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        rig.voice.feed("no")
        await rig.o.handle("undo_change", "Undo that.", 0.59)
    asyncio.run(go())
    assert (rig.overrides / "search.py").is_file()


def test_after_undo_the_skill_is_not_rewritten_again_today(self_rig):
    """The owner rejected that rewrite: its next failure must not start
    another one on top of their undo."""
    rig = self_rig

    async def go():
        await fail_and_rewrite(rig)
        await rig.o.handle("undo_change", "undo that", 0.9)
        await rig.o._safe_point()
        claude_calls = len(rig.claude.calls)
        await rig.o.handle("search", "search black holes", 0.9)   # the old version fails
        assert not rig.o._jobs and len(rig.claude.calls) == claude_calls
    asyncio.run(go())


def test_a_name_is_told_once_however_often_it_was_rewritten(self_rig):
    rig = self_rig
    rig.o.learning.state.add_event("rewrite", "clock")
    rig.o.learning.state.add_event("rewrite", "clock")
    asyncio.run(rig.o.handle("learned_today", "what have you learned today", 0.9))
    assert "rewrote clock." in rig.voice.spoken[-1] and "clock, clock" not in rig.voice.spoken[-1]
