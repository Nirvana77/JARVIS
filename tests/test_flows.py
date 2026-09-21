"""`TeachFlow`/`EditSkillFlow`/`RevertSkillFlow` — the voice dialogs only.

M2.5: a flow ends when the questions only the user can answer are answered,
and returns a `LearningRequest`. Build/validate/sandbox is the background
`LearningJob` (`test_jobs.py`); retrain/confirm/promote is the orchestrator
(`test_background_learning.py`)."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.factory.flows import (
    EditSkillFlow,
    LearningRequest,
    RevertSkillFlow,
    TeachFlow,
    ask_yes_no_or_none,
)

_MODULE_TEMPLATE = '''\
from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="{name}",
    description="Flip a coin and report heads or tails.",
    examples=["flip a coin", "flip a coin for me"],
    params={{}},
    permissions={{"pure"}},
)


def run(ctx, **params) -> str:
    return "Heads."
'''


def _good_module(name: str = "coin_flip") -> str:
    return _MODULE_TEMPLATE.format(name=name)


GOOD_MODULE = _good_module()
GOOD_TEST = "def test_ok():\n    assert True\n"


class Script:
    """Feeds canned replies to `ask()`, recording every prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def ask(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class Recorder:
    def __init__(self):
        self.said = []

    async def say(self, text):
        self.said.append(text)


class FakeRegistry:
    def __init__(self, manifests=()):
        self._manifests = list(manifests)

    def names(self):
        return [m.name for m in self._manifests]

    def manifests(self):
        return list(self._manifests)

    def manifest(self, name):
        return next(m for m in self._manifests if m.name == name)

    def __contains__(self, name):
        return name in self.names()


@pytest.fixture(autouse=True)
def _clean_staging():
    import shutil

    from jarvis.factory.flows import staging_dir

    yield
    for p in staging_dir().glob("*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def test_teach_flow_returns_a_request_after_the_dialog():
    script = Script(["coin flip", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        seed_description="flip a coin and tell me heads or tails",
    )
    request = asyncio.run(flow.run())
    assert isinstance(request, LearningRequest)
    assert request.versioning == "new"
    assert request.name == "coin_flip"
    assert request.spec.examples == ["flip a coin and tell me heads or tails", "flip a coin for me"]
    assert request.existing_source is None
    assert request.allow_name is None


def test_teach_flow_does_not_build_anything():
    """The dialog must not generate or sandbox — that's the background job."""
    from jarvis.factory.flows import staging_dir

    script = Script(["coin flip", "yes", "flip a coin for me"])
    flow = TeachFlow(ask=script.ask, say=Recorder().say, registry=FakeRegistry(), seed_description="flip a coin")
    asyncio.run(flow.run())
    assert list(staging_dir().glob("*.py")) == []


def test_teach_flow_rejects_a_name_that_already_exists():
    from jarvis.skills.contract import SkillManifest

    existing = SkillManifest(name="coin_flip", description="x", examples=["x"])
    script = Script(["coin flip", "timer", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]),
        seed_description="flip a coin",
    )
    request = asyncio.run(flow.run())
    assert request.name == "timer"
    assert any("already have" in s for s in recorder.said)


def test_teach_flow_rejects_a_name_already_being_learned():
    script = Script(["coin flip", "timer", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        seed_description="flip a coin", busy_names=frozenset({"coin_flip"}),
    )
    request = asyncio.run(flow.run())
    assert request.name == "timer"
    assert any("already working on 'coin_flip'" in s for s in recorder.said)


def test_teach_flow_with_no_usable_name_returns_none():
    script = Script(["", ""])
    flow = TeachFlow(ask=script.ask, say=Recorder().say, registry=FakeRegistry())
    assert asyncio.run(flow.run()) is None


def test_edit_skill_flow_reads_existing_source(tmp_path, monkeypatch):
    from jarvis.factory import flows as flows_mod
    from jarvis.skills.contract import SkillManifest

    learned_dir = tmp_path / "learned"
    learned_dir.mkdir()
    (learned_dir / "coin_flip.py").write_text(GOOD_MODULE, encoding="utf-8")
    monkeypatch.setattr(flows_mod, "learned_source_path", lambda name: learned_dir / f"{name}.py")

    existing = SkillManifest(name="coin_flip", description="Flip a coin.", examples=["flip a coin"], version=1, origin="learned")
    script = Script(["coin flip", "also say tails sometimes"])
    flow = EditSkillFlow(ask=script.ask, say=Recorder().say, registry=FakeRegistry([existing]))
    request = asyncio.run(flow.run())
    assert request.versioning == "edit"
    assert request.name == "coin_flip"
    assert request.existing_source == GOOD_MODULE
    assert request.allow_name == "coin_flip"
    assert request.spec.description == "also say tails sometimes"
    assert request.spec.based_on_version == 1


def test_edit_skill_flow_refuses_a_skill_already_being_learned():
    from jarvis.skills.contract import SkillManifest

    existing = SkillManifest(name="coin_flip", description="x", examples=["x"], origin="learned")
    script = Script(["coin flip", "also say tails"])
    recorder = Recorder()
    flow = EditSkillFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]),
        busy_names=frozenset({"coin_flip"}),
    )
    assert asyncio.run(flow.run()) is None
    assert any("already working on 'coin_flip'" in s for s in recorder.said)


def test_revert_skill_flow_with_no_history_returns_none():
    from jarvis.skills.contract import SkillManifest

    existing = SkillManifest(name="coin_flip", description="x", examples=["x"], origin="learned")
    script = Script(["coin flip"])
    recorder = Recorder()

    class Cfg:
        def skill_versions_dir(self, name):
            import pathlib
            return pathlib.Path("/nonexistent/path/for/test")

    flow = RevertSkillFlow(ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]), config=Cfg())
    assert asyncio.run(flow.run()) is None
    assert any("earlier version" in s for s in recorder.said)


def test_revert_skill_flow_picks_the_highest_version(tmp_path):
    from jarvis.skills.contract import SkillManifest

    existing = SkillManifest(name="coin_flip", description="x", examples=["x"], version=3, origin="learned")
    vdir = tmp_path / "coin_flip"
    vdir.mkdir()
    (vdir / "v1.py").write_text(GOOD_MODULE, encoding="utf-8")
    (vdir / "v2.py").write_text(GOOD_MODULE, encoding="utf-8")

    class Cfg:
        def skill_versions_dir(self, name):
            return vdir

    script = Script(["coin flip"])
    flow = RevertSkillFlow(ask=script.ask, say=Recorder().say, registry=FakeRegistry([existing]), config=Cfg())
    request = asyncio.run(flow.run())
    assert request.versioning == "revert"
    assert request.name == "coin_flip"
    assert request.reverted_from_version == 2
    assert request.module_source == GOOD_MODULE
    assert request.manifest.name == "coin_flip"


# -- ask_yes_no_or_none (M2.5: background questions) -------------------------

@pytest.mark.parametrize(
    "reply, expected",
    [
        ("yes", True),
        ("yes please", True),
        ("sure", True),
        ("no", False),
        ("no thanks", False),
        ("never mind", False),
        ("", None),
        ("what time is it", None),
        ("search black holes", None),
    ],
)
def test_ask_yes_no_or_none(reply, expected):
    script = Script([reply])
    assert asyncio.run(ask_yes_no_or_none(script.ask, "Shall I keep it, sir?")) is expected
