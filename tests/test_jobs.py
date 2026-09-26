"""`LearningJob` — M2.5's background half of teach/edit_skill/revert_skill:
build → validate → (permission decision) → sandbox. It never speaks or asks
directly; it only calls the injected `notify` / `decide`, so these tests
resolve both without an orchestrator, a network or a real subprocess."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.factory.build import BuildError
from jarvis.factory.claude_client import GeneratedSkill
from jarvis.factory.flows import LearningRequest, staging_dir
from jarvis.factory.jobs import LearningJob
from jarvis.factory.sandbox import SandboxResult
from jarvis.factory.spec import SkillSpec

_MODULE_TEMPLATE = '''\
from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="{name}",
    description="Flip a coin and report heads or tails.",
    examples=["flip a coin", "flip a coin for me"],
    params={{}},
    permissions={perms},
)


def run(ctx, **params) -> str:
    return "Heads."
'''

GOOD_TEST = "def test_ok():\n    assert True\n"


def module(name="coin_flip", perms='{"pure"}'):
    return _MODULE_TEMPLATE.format(name=name, perms=perms)


def generator(source):
    def generate(spec, existing_source, feedback=None):
        return GeneratedSkill(name=spec.name, module_source=source, test_source=GOOD_TEST)

    return generate


def sequence(*sources):
    """A `generate` that returns each `sources[i]` in turn, one per call — for
    exercising the retry loop's "attempt N recovers" path. Recording the
    `feedback` it was called with lets a test assert it actually saw the
    previous attempt's error."""
    calls = []

    def generate(spec, existing_source, feedback=None):
        calls.append(feedback)
        source = sources[min(len(calls) - 1, len(sources) - 1)]
        return GeneratedSkill(name=spec.name, module_source=source, test_source=GOOD_TEST)

    generate.calls = calls
    return generate


class FakeRegistry:
    def __init__(self, names=()):
        self._names = list(names)

    def names(self):
        return list(self._names)

    def __contains__(self, name):
        return name in self._names


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


class FlakySandbox:
    """`run_tests` fails the first `fail_times` calls, then succeeds — for
    exercising the retry loop's "a later attempt recovers" path."""

    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.calls = []

    def run_tests(self, module_path, test_path, permissions):
        self.calls.append(("run_tests", module_path.name, permissions))
        tries = sum(1 for c in self.calls if c[0] == "run_tests")
        ok = tries > self.fail_times
        return SandboxResult(ok=ok, stdout="", stderr="boom", returncode=0 if ok else 1)

    def dry_run(self, module_path, params, permissions):
        self.calls.append(("dry_run", module_path.name, permissions))
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)


class Channel:
    """Records `notify` lines and answers `decide` from a canned list."""

    def __init__(self, answers=()):
        self.notices = []
        self.questions = []
        self.answers = list(answers)

    async def notify(self, text):
        self.notices.append(text)

    async def decide(self, prompt):
        self.questions.append(prompt)
        return self.answers.pop(0)


def teach_request(name="coin_flip"):
    return LearningRequest(
        versioning="new",
        name=name,
        spec=SkillSpec(name=name, description="flip a coin", examples=["flip a coin"]),
    )


def make_job(request, *, channel, generate, sandbox, registry=None, max_attempts=None):
    kwargs = {} if max_attempts is None else {"max_attempts": max_attempts}
    return LearningJob(
        request,
        notify=channel.notify,
        decide=channel.decide,
        registry=registry or FakeRegistry(),
        generate=generate,
        sandbox=sandbox,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_staging():
    yield
    for p in staging_dir().glob("*.py"):
        p.unlink(missing_ok=True)


def test_happy_path_is_accepted_and_staged():
    channel, sandbox = Channel(), FakeSandbox()
    job = make_job(teach_request(), channel=channel, generate=generator(module()), sandbox=sandbox)
    outcome = asyncio.run(job.run())
    assert outcome.accepted
    assert outcome.name == "coin_flip"
    assert outcome.manifest.permissions == {"pure"}
    assert (staging_dir() / "coin_flip.py").is_file()
    assert [c[0] for c in sandbox.calls] == ["run_tests", "dry_run"]
    # pure/notify only -> no permission question
    assert channel.questions == []
    # the generated test file never outlives the job
    assert not (staging_dir() / "_test_coin_flip.py").exists()


def test_build_error_is_notified_not_raised():
    def broken(spec, existing_source, feedback=None):
        raise BuildError("no code block")

    channel = Channel()
    outcome = asyncio.run(make_job(teach_request(), channel=channel, generate=broken, sandbox=FakeSandbox()).run())
    assert not outcome.accepted
    assert any("coin_flip" in n for n in channel.notices)


def test_validation_failure_is_notified():
    channel = Channel()
    job = make_job(teach_request(), channel=channel, generate=generator("def nope(:\n"), sandbox=FakeSandbox())
    outcome = asyncio.run(job.run())
    assert not outcome.accepted
    assert channel.notices


def test_sandbox_test_failure_sets_it_aside():
    channel = Channel()
    job = make_job(teach_request(), channel=channel, generate=generator(module()), sandbox=FakeSandbox(tests_ok=False))
    outcome = asyncio.run(job.run())
    assert not outcome.accepted
    assert outcome.reason == "sandbox tests failed"
    assert not (staging_dir() / "coin_flip.py").exists()
    assert channel.notices


def test_dry_run_failure_sets_it_aside():
    channel = Channel()
    job = make_job(teach_request(), channel=channel, generate=generator(module()), sandbox=FakeSandbox(dry_run_ok=False))
    outcome = asyncio.run(job.run())
    assert not outcome.accepted
    assert outcome.reason == "sandbox dry-run failed"
    assert not (staging_dir() / "coin_flip.py").exists()


def test_extra_permission_is_decided_before_the_sandbox_runs():
    sandbox = FakeSandbox()
    order = []

    class OrderedChannel(Channel):
        async def decide(self, prompt):
            order.append(("decide", len(sandbox.calls)))
            return await super().decide(prompt)

    channel = OrderedChannel(answers=[True])
    job = make_job(teach_request(), channel=channel, generate=generator(module(perms='{"net"}')), sandbox=sandbox)
    outcome = asyncio.run(job.run())
    assert outcome.accepted
    assert order == [("decide", 0)]  # asked while the sandbox had run nothing
    assert "access" in channel.questions[0].lower()
    assert "coin_flip" in channel.questions[0]


def test_declined_permission_never_runs_generated_code():
    sandbox = FakeSandbox()
    channel = Channel(answers=[False])
    job = make_job(teach_request(), channel=channel, generate=generator(module(perms='{"net"}')), sandbox=sandbox)
    outcome = asyncio.run(job.run())
    assert not outcome.accepted
    assert outcome.reason == "permission declined"
    assert sandbox.calls == []
    assert not (staging_dir() / "coin_flip.py").exists()


def test_name_collision_is_checked_against_the_registry_the_job_was_given():
    channel = Channel()
    job = make_job(
        teach_request(), channel=channel, generate=generator(module()),
        sandbox=FakeSandbox(), registry=FakeRegistry(["coin_flip"]),
    )
    outcome = asyncio.run(job.run())
    assert not outcome.accepted


def test_edit_passes_existing_source_and_allows_its_own_name():
    seen = {}

    def generate(spec, existing_source, feedback=None):
        seen["existing"] = existing_source
        return GeneratedSkill(name=spec.name, module_source=module(), test_source=GOOD_TEST)

    request = LearningRequest(
        versioning="edit",
        name="coin_flip",
        spec=SkillSpec(name="coin_flip", description="say tails too", examples=["flip a coin"], based_on_version=1),
        existing_source="OLD SOURCE",
        allow_name="coin_flip",
    )
    job = make_job(request, channel=Channel(), generate=generate, sandbox=FakeSandbox(), registry=FakeRegistry(["coin_flip"]))
    outcome = asyncio.run(job.run())
    assert outcome.accepted
    assert seen["existing"] == "OLD SOURCE"


def test_revert_passes_through_without_building_or_sandboxing():
    from jarvis.factory.validate import validate

    source = module()
    request = LearningRequest(
        versioning="revert",
        name="coin_flip",
        module_source=source,
        manifest=validate(source, frozenset(), allow_name="coin_flip"),
        reverted_from_version=2,
    )

    def must_not_generate(spec, existing_source, feedback=None):
        raise AssertionError("revert must not call Claude")

    sandbox = FakeSandbox()
    outcome = asyncio.run(make_job(request, channel=Channel(), generate=must_not_generate, sandbox=sandbox).run())
    assert outcome.accepted
    assert outcome.module_source == source
    assert outcome.reverted_from_version == 2
    assert sandbox.calls == []


# -- remove: no Claude, no sandbox, same shortcut as revert ------------------

def test_remove_passes_through_without_building_or_sandboxing():
    from jarvis.skills.contract import SkillManifest

    manifest = SkillManifest(name="coin_flip", description="x", examples=["flip a coin"], origin="learned")
    request = LearningRequest(versioning="remove", name="coin_flip", manifest=manifest)

    def must_not_generate(spec, existing_source, feedback=None):
        raise AssertionError("remove must not call Claude")

    sandbox = FakeSandbox()
    outcome = asyncio.run(make_job(request, channel=Channel(), generate=must_not_generate, sandbox=sandbox).run())
    assert outcome.accepted
    assert outcome.name == "coin_flip"
    assert outcome.manifest is manifest
    assert outcome.module_source is None
    assert sandbox.calls == []


# -- retry with feedback ------------------------------------------------------

def test_default_max_attempts_is_five():
    from jarvis.factory.jobs import DEFAULT_MAX_ATTEMPTS

    assert DEFAULT_MAX_ATTEMPTS == 5

    calls = []

    def always_fails(spec, existing_source, feedback=None):
        calls.append(feedback)
        return GeneratedSkill(name=spec.name, module_source="def nope(:\n", test_source=GOOD_TEST)

    channel = Channel()
    outcome = asyncio.run(
        make_job(teach_request(), channel=channel, generate=always_fails, sandbox=FakeSandbox()).run()
    )
    assert not outcome.accepted
    assert len(calls) == DEFAULT_MAX_ATTEMPTS
    # only the final give-up is spoken, not one notice per failed attempt
    assert len(channel.notices) == 1


def test_max_attempts_is_configurable():
    calls = []

    def always_fails(spec, existing_source, feedback=None):
        calls.append(feedback)
        return GeneratedSkill(name=spec.name, module_source="def nope(:\n", test_source=GOOD_TEST)

    channel = Channel()
    outcome = asyncio.run(
        make_job(
            teach_request(), channel=channel, generate=always_fails, sandbox=FakeSandbox(), max_attempts=2,
        ).run()
    )
    assert not outcome.accepted
    assert len(calls) == 2


def test_validation_failure_is_retried_with_feedback_and_recovers():
    channel = Channel()
    generate = sequence("def nope(:\n", module())
    job = make_job(teach_request(), channel=channel, generate=generate, sandbox=FakeSandbox())
    outcome = asyncio.run(job.run())
    assert outcome.accepted
    assert len(generate.calls) == 2
    assert generate.calls[0] is None  # first attempt: no prior failure yet
    assert generate.calls[1]  # second attempt got fed the first attempt's error
    assert channel.notices == []  # silent recovery — no "didn't check out" line


def test_sandbox_test_failure_is_retried_with_feedback_and_recovers():
    channel = Channel()
    sandbox = FlakySandbox(fail_times=1)
    job = make_job(teach_request(), channel=channel, generate=generator(module()), sandbox=sandbox)
    outcome = asyncio.run(job.run())
    assert outcome.accepted
    assert len([c for c in sandbox.calls if c[0] == "run_tests"]) == 2
    assert channel.notices == []


def test_exhausted_retries_reports_the_original_reason_once():
    channel = Channel()
    job = make_job(
        teach_request(), channel=channel, generate=generator(module()),
        sandbox=FakeSandbox(tests_ok=False), max_attempts=3,
    )
    outcome = asyncio.run(job.run())
    assert not outcome.accepted
    assert outcome.reason == "sandbox tests failed"
    assert len(channel.notices) == 1
    assert "coin_flip" in channel.notices[0]


def test_permission_is_only_asked_once_across_a_retry():
    channel = Channel(answers=[True])
    sandbox = FlakySandbox(fail_times=1)
    job = make_job(teach_request(), channel=channel, generate=generator(module(perms='{"net"}')), sandbox=sandbox)
    outcome = asyncio.run(job.run())
    assert outcome.accepted
    # not re-asked on the retry that needs the same permission again
    assert len(channel.questions) == 1


def test_declined_permission_still_ends_the_flow_immediately():
    """A permission decline is the user's call, not something a retry can fix
    — it must end the job outright, same as with no retry loop at all."""
    channel = Channel(answers=[False])
    sandbox = FlakySandbox(fail_times=1)
    job = make_job(teach_request(), channel=channel, generate=generator(module(perms='{"net"}')), sandbox=sandbox)
    outcome = asyncio.run(job.run())
    assert not outcome.accepted
    assert outcome.reason == "permission declined"
    assert len(channel.questions) == 1
    assert sandbox.calls == []
