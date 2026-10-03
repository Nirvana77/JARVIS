"""M8 part A: JARVIS's own version of a builtin lives in
``<data>/skills/overrides/<name>.py`` and is loaded in place of the packaged
one, by both brains, the way learned skills are shared (known issue #19)."""

from __future__ import annotations

import dataclasses
import sys

import pytest

import jarvis.skills.overrides as overrides_pkg
from jarvis.factory import flows
from jarvis.skills.registry import BUILTIN_PACKAGE, Registry

NOTE_V2 = (
    "from jarvis.skills.contract import SkillManifest\n"
    "MANIFEST = SkillManifest(name='note', description='Take a note.', "
    "examples=['note that milk'], permissions=frozenset({'fs_write'}), voice='jarvis')\n"
    "def run(ctx, **params):\n    return 'Noted, sir. (the rewrite)'\n"
)


@pytest.fixture
def cfg(config, tmp_path):
    original = list(overrides_pkg.__path__)
    yield dataclasses.replace(config, data_dir=tmp_path)
    overrides_pkg.__path__ = original
    for name in [m for m in sys.modules if m.startswith("jarvis.skills.overrides.")]:
        sys.modules.pop(name, None)


def _write(cfg, name, source):
    cfg.skill_overrides_dir.mkdir(parents=True, exist_ok=True)
    (cfg.skill_overrides_dir / f"{name}.py").write_text(source, encoding="utf-8")


def test_the_config_names_the_overrides_folder(cfg, tmp_path):
    assert cfg.skill_overrides_dir == tmp_path / "skills" / "overrides"


def test_an_override_replaces_the_packaged_builtin(cfg):
    _write(cfg, "note", NOTE_V2)
    reg = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    assert reg.overridden == {"note"}
    assert reg.manifest("note").origin == "builtin"
    assert reg.dispatch("note", {"text": "milk"}) == "Noted, sir. (the rewrite)"


def test_without_an_override_the_packaged_one_runs(cfg):
    reg = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    assert reg.overridden == set()
    assert "rewrite" not in reg.manifest("note").description


def test_an_override_for_something_that_is_not_a_builtin_is_ignored(cfg):
    _write(cfg, "roll_a_die", NOTE_V2.replace("'note'", "'roll_a_die'"))
    reg = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    assert "roll_a_die" not in reg.names() and reg.overridden == set()


def test_a_broken_override_leaves_the_packaged_one_in_place(cfg):
    _write(cfg, "note", "def run(:\n")
    reg = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    assert reg.overridden == set()
    assert "note" in reg.names()


def test_an_override_with_another_name_inside_is_ignored(cfg):
    _write(cfg, "note", NOTE_V2.replace("name='note'", "name='search'"))
    reg = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    assert reg.overridden == set()


def test_paths_follow_discovery(cfg):
    Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    assert flows.override_source_path("note") == cfg.skill_overrides_dir / "note.py"
    assert flows.packaged_source_path("note").name == "note.py"
    assert flows.packaged_source_path("note").parent.name == "builtin"
