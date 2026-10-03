"""M8 part A: JARVIS rewrites a builtin that failed, behind a stricter gate
than a learned skill: the repo's own tests for that builtin, and the
builtin's recent good calls from the interaction log, replayed. Permissions
the packaged builtin already has were granted by the owner, so they are not
asked again."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import sys

import pytest

import jarvis.skills.overrides as overrides_pkg
from jarvis.factory import flows as flows_mod
from jarvis.factory.claude_client import GeneratedSkill
from jarvis.factory.flows import LearningRequest
from jarvis.factory.jobs import LearningJob
from jarvis.factory.sandbox import SandboxResult, SubprocessSandbox
from jarvis.factory.spec import SkillSpec

NOTE_SOURCE = flows_mod.packaged_source_path("note").read_text(encoding="utf-8")
BROKEN_NOTE = NOTE_SOURCE.replace('return "Noted, sir."', 'return None')


class Recording:
    """A sandbox that records what it was asked and passes, unless told."""

    def __init__(self, repo_tests_ok=True, fails_with=None):
        self.repo_tests = []
        self.dry_runs = []
        self.repo_tests_ok = repo_tests_ok
        self.fails_with = fails_with

    def run_tests(self, module_path, test_path, permissions):
        return SandboxResult(ok=True, stdout="", stderr="", returncode=0)

    def dry_run(self, module_path, params, permissions):
        self.dry_runs.append(dict(params))
        ok = self.fails_with is None or params != self.fails_with
        return SandboxResult(ok=ok, stdout="", stderr="" if ok else "boom", returncode=0 if ok else 1)

    def run_repo_tests(self, module_path, name, test_files, permissions):
        self.repo_tests.append((name, tuple(test_files)))
        return SandboxResult(ok=self.repo_tests_ok, stdout="1 failed" if not self.repo_tests_ok else "",
                             stderr="", returncode=0 if self.repo_tests_ok else 1)


def _request(**extra):
    return LearningRequest(
        versioning="edit", name="note",
        spec=SkillSpec(name="note", description="repair it", examples=["note that milk"]),
        existing_source=NOTE_SOURCE, allow_name="note", autonomous=True, override=True,
        pre_granted=frozenset({"fs_write"}), **extra,
    )


def _job(request, sandbox, decided):
    async def decide(prompt):
        decided.append(prompt)
        return False

    async def notify(text):
        pass

    def generate(spec, existing, feedback=None):
        return GeneratedSkill(name="note", module_source=NOTE_SOURCE, test_source="def test_ok():\n    pass\n")

    class Reg:
        def names(self):
            return ["note", "search"]

    return LearningJob(request, notify=notify, decide=decide, registry=Reg(),
                       generate=generate, sandbox=sandbox, max_attempts=1)


@pytest.fixture(autouse=True)
def staging(tmp_path, monkeypatch):
    d = tmp_path / "staging"
    d.mkdir()
    monkeypatch.setattr(flows_mod, "staging_dir", lambda: d)


def test_permissions_the_builtin_already_has_are_not_asked_again():
    decided = []
    outcome = asyncio.run(_job(_request(), Recording(), decided).run())
    assert outcome.accepted and decided == []


def test_the_repo_tests_for_the_builtin_are_part_of_the_gate():
    box = Recording()
    outcome = asyncio.run(_job(_request(repo_tests=("tests/test_skills.py",)), box, []).run())
    assert outcome.accepted
    assert box.repo_tests == [("note", ("tests/test_skills.py",))]


def test_a_rewrite_that_fails_the_repo_tests_is_not_kept():
    box = Recording(repo_tests_ok=False)
    outcome = asyncio.run(_job(_request(repo_tests=("tests/test_skills.py",)), box, []).run())
    assert not outcome.accepted


def test_recent_good_calls_are_replayed_and_must_still_work():
    box = Recording(fails_with={"text": "buy milk"})
    req = _request(regression_params=({"text": "the gate code"}, {"text": "buy milk"}))
    outcome = asyncio.run(_job(req, box, []).run())
    assert {"text": "the gate code"} in box.dry_runs and {"text": "buy milk"} in box.dry_runs
    assert not outcome.accepted


# -- the sandbox runs the repo's tests against the override ---------------------------------

def test_the_sandbox_runs_the_repo_tests_with_the_override_in_place(tmp_path):
    box = SubprocessSandbox(timeout_s=120.0, mem_mb=2048, cpu_s=120)
    repo = pathlib.Path(__file__).resolve().parent.parent
    good = tmp_path / "note.py"
    good.write_text(NOTE_SOURCE, encoding="utf-8")
    result = box.run_repo_tests(good, "note", [str(repo / "tests/test_skills.py")], frozenset({"fs_write"}))
    assert result.ok, result.stdout[-2000:] + result.stderr[-2000:]
    bad = tmp_path / "note_bad" / "note.py"
    bad.parent.mkdir()
    bad.write_text(BROKEN_NOTE, encoding="utf-8")
    result = box.run_repo_tests(bad, "note", [str(repo / "tests/test_skills.py")], frozenset({"fs_write"}))
    assert not result.ok


# -- the orchestrator starts it ------------------------------------------------------------------

def test_tests_for_a_builtin_are_found_by_what_imports_it():
    from jarvis.core.orchestrator import tests_for_builtin

    found = {pathlib.Path(p).name for p in tests_for_builtin("note")}
    assert "test_skills.py" in found
    assert all(pathlib.Path(p).is_file() for p in tests_for_builtin("note"))
