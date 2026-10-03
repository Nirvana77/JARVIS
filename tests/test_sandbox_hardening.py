"""M8 groundwork: the sandbox that will run JARVIS's rewrites of its own
builtins, with the repo's own tests, must not hand those rewrites the brain's
secrets — and its stand-in ``ctx`` must have what skills are told to use.

Found 2026-10-03 while planning M8: the sandbox inherited the brain's whole
environment (``ANTHROPIC_API_KEY``, ``HF_TOKEN``, the edge tokens), and on the
owner's machine ``unshare`` is unavailable, so there is no network
isolation either. And M7's factory prompt tells Claude to use ``ctx.now()``,
which the dry run's stand-in did not have.
"""

from __future__ import annotations

import pathlib

import pytest

from jarvis.factory import sandbox as sandbox_mod
from jarvis.factory.sandbox import SubprocessSandbox
from jarvis.factory.validate import validate

SPY = '''\
import os
from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(name="spy", description="Look around.", examples=["spy"],
                         permissions={"pure"})


def run(ctx, **params) -> str:
    leaked = sorted(k for k in os.environ if any(s in k for s in ("KEY", "TOKEN", "SECRET")))
    assert not leaked, leaked
    assert ctx.now().tzinfo is not None
    assert ctx.knowledge is None and ctx.edges is None and ctx.memory is None
    return "nothing to see, sir."
'''


@pytest.fixture
def box():
    return SubprocessSandbox(timeout_s=15.0, mem_mb=512, cpu_s=10)


def test_sandboxed_code_sees_no_secrets_and_a_full_ctx(box, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("JARVIS_EDGE_TOKENS", "watch:s3cret")
    module = tmp_path / "spy.py"
    module.write_text(SPY, encoding="utf-8")
    result = box.dry_run(module, {}, frozenset({"pure"}))
    assert result.ok, result.stderr


def test_the_sandbox_environment_is_a_short_allowlist(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    env = sandbox_mod.sandbox_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert env["JARVIS_SANDBOX"] == "1"
    assert set(env) <= {"PATH", "HOME", "LANG", "LC_ALL", "TZ", "JARVIS_SANDBOX", "TMPDIR"}


def test_config_in_the_sandbox_does_not_read_dotenv(tmp_path, monkeypatch):
    from jarvis import config as config_mod

    called = []
    monkeypatch.setattr(config_mod, "load_dotenv", lambda *a, **k: called.append(1))
    monkeypatch.setenv("JARVIS_SANDBOX", "1")
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    config_mod.load_config(tmp_path / "config.toml")
    assert called == []


@pytest.mark.parametrize("path", sorted(
    p for p in pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "jarvis/skills/builtin").glob("*.py") if not p.name.startswith("_")
), ids=lambda p: p.stem)
def test_every_builtin_passes_the_validator(path):
    """A rewrite of a builtin must pass validation, so the builtin as written
    must too (``permissions=frozenset({...})`` counts as a literal)."""
    manifest = validate(path.read_text(encoding="utf-8"), frozenset(), allow_name=path.stem)
    assert manifest.name == path.stem


def test_frozenset_of_anything_but_a_literal_set_is_still_refused():
    source = (
        "from jarvis.skills.contract import SkillManifest\n"
        "PERMS = {'net'}\n"
        "MANIFEST = SkillManifest(name='x', description='x', permissions=frozenset(PERMS))\n"
        "def run(ctx, **params):\n    return 'x'\n"
    )
    with pytest.raises(Exception):
        validate(source, frozenset())
