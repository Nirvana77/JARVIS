"""`TeachFlow`/`EditSkillFlow`/`RevertSkillFlow` — the voice dialogs, with a
fake `generate` and a fake sandbox so no network or real subprocess is
involved. These only exercise the dialog + build/validate/sandbox wiring;
retrain/confirm/promote are the orchestrator's job (see `test_orchestrator.py`
for the merge-gate side)."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.factory.claude_client import GeneratedSkill
from jarvis.factory.flows import EditSkillFlow, RevertSkillFlow, TeachFlow
from jarvis.factory.sandbox import SandboxResult

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


class FakeSandbox:
    def __init__(self, tests_ok=True, dry_run_ok=True):
        self.tests_ok = tests_ok
        self.dry_run_ok = dry_run_ok
        self.calls = []

    def run_tests(self, module_path, test_path, permissions):
        self.calls.append(("run_tests", module_path.name, permissions))
        return SandboxResult(ok=self.tests_ok, stdout="", stderr="boom", returncode=0 if self.tests_ok else 1)

    def dry_run(self, module_path, params, permissions):
        self.calls.append(("dry_run", module_path.name, permissions))
        return SandboxResult(ok=self.dry_run_ok, stdout="", stderr="boom", returncode=0 if self.dry_run_ok else 1)


def fake_generate(spec, existing_source):
    return GeneratedSkill(name=spec.name, module_source=_good_module(spec.name), test_source=GOOD_TEST)


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


def test_teach_flow_happy_path_is_accepted():
    script = Script(["coin flip", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        generate=fake_generate, sandbox=FakeSandbox(),
        seed_description="flip a coin and tell me heads or tails",
    )
    outcome = asyncio.run(flow.run())
    assert outcome.accepted
    assert outcome.name == "coin_flip"
    assert outcome.manifest.permissions == {"pure"}


def test_teach_flow_rejects_a_name_that_already_exists():
    from jarvis.skills.contract import SkillManifest

    existing = SkillManifest(name="coin_flip", description="x", examples=["x"])
    script = Script(["coin flip", "timer", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]),
        generate=fake_generate, sandbox=FakeSandbox(),
        seed_description="flip a coin",
    )
    outcome = asyncio.run(flow.run())
    assert outcome.accepted
    assert outcome.name == "timer"
    assert any("already have" in s for s in recorder.said)


def test_teach_flow_aborts_when_sandbox_tests_fail():
    script = Script(["coin flip", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        generate=fake_generate, sandbox=FakeSandbox(tests_ok=False),
        seed_description="flip a coin",
    )
    outcome = asyncio.run(flow.run())
    assert not outcome.accepted
    assert outcome.reason == "sandbox tests failed"


def test_teach_flow_aborts_when_dry_run_fails():
    script = Script(["coin flip", "yes", "flip a coin for me"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        generate=fake_generate, sandbox=FakeSandbox(dry_run_ok=False),
        seed_description="flip a coin",
    )
    outcome = asyncio.run(flow.run())
    assert not outcome.accepted
    assert outcome.reason == "sandbox dry-run failed"


def test_teach_flow_asks_for_permission_grant_beyond_pure_and_notify():
    net_module = GOOD_MODULE.replace('permissions={"pure"}', 'permissions={"net"}')

    def generate_net(spec, existing_source):
        return GeneratedSkill(name=spec.name, module_source=net_module, test_source=GOOD_TEST)

    script = Script(["coin flip", "yes", "flip a coin for me", "yes"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        generate=generate_net, sandbox=FakeSandbox(),
        seed_description="flip a coin",
    )
    outcome = asyncio.run(flow.run())
    assert outcome.accepted
    assert any("access" in p.lower() for p in script.prompts)


def test_teach_flow_declined_permission_grant_is_not_accepted():
    net_module = GOOD_MODULE.replace('permissions={"pure"}', 'permissions={"net"}')

    def generate_net(spec, existing_source):
        return GeneratedSkill(name=spec.name, module_source=net_module, test_source=GOOD_TEST)

    script = Script(["coin flip", "yes", "flip a coin for me", "no"])
    recorder = Recorder()
    flow = TeachFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry(),
        generate=generate_net, sandbox=FakeSandbox(),
        seed_description="flip a coin",
    )
    outcome = asyncio.run(flow.run())
    assert not outcome.accepted
    assert outcome.reason == "permission declined"


def test_edit_skill_flow_reads_existing_source(tmp_path, monkeypatch):
    from jarvis.factory import flows as flows_mod
    from jarvis.skills.contract import SkillManifest

    learned_dir = tmp_path / "learned"
    learned_dir.mkdir()
    (learned_dir / "coin_flip.py").write_text(GOOD_MODULE, encoding="utf-8")
    monkeypatch.setattr(flows_mod, "learned_source_path", lambda name: learned_dir / f"{name}.py")

    existing = SkillManifest(name="coin_flip", description="Flip a coin.", examples=["flip a coin"], version=1, origin="learned")
    seen = {}

    def capture_generate(spec, existing_source):
        seen["existing_source"] = existing_source
        return GeneratedSkill(name=spec.name, module_source=GOOD_MODULE, test_source=GOOD_TEST)

    script = Script(["coin flip", "also say tails sometimes"])
    recorder = Recorder()
    flow = EditSkillFlow(
        ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]),
        generate=capture_generate, sandbox=FakeSandbox(),
    )
    outcome = asyncio.run(flow.run())
    assert outcome.accepted
    assert seen["existing_source"] == GOOD_MODULE


def test_revert_skill_flow_with_no_history_is_declined():
    from jarvis.skills.contract import SkillManifest

    existing = SkillManifest(name="coin_flip", description="x", examples=["x"], origin="learned")
    script = Script(["coin flip"])
    recorder = Recorder()

    class Cfg:
        def skill_versions_dir(self, name):
            import pathlib
            return pathlib.Path("/nonexistent/path/for/test")

    flow = RevertSkillFlow(ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]), config=Cfg())
    outcome = asyncio.run(flow.run())
    assert not outcome.accepted
    assert outcome.reason == "no earlier version"


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
    recorder = Recorder()
    flow = RevertSkillFlow(ask=script.ask, say=recorder.say, registry=FakeRegistry([existing]), config=Cfg())
    outcome = asyncio.run(flow.run())
    assert outcome.accepted
    assert outcome.reverted_from_version == 2
