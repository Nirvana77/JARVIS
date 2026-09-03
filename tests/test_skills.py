"""Builtin skills and the registry."""

from __future__ import annotations

import pytest

from jarvis.core.context import Context
from jarvis.skills.builtin import note, open_app, play, search
from jarvis.skills.contract import PERMISSIONS, SkillManifest, SkillNotFound
from jarvis.skills.registry import Registry

BUILTINS = {"search": search, "open_app": open_app, "play": play, "note": note}


def _ctx(config, tmp_path):
    return Context(say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None)


# -- manifests / registry -------------------------------------------------

def test_registry_discovers_the_four_builtins(config):
    reg = Registry.discover(config)
    assert reg.names() == ["note", "open_app", "play", "search"]


@pytest.mark.parametrize("name,module", BUILTINS.items())
def test_manifest_is_well_formed(name, module):
    m = module.MANIFEST
    assert isinstance(m, SkillManifest)
    assert m.name == name                      # name matches the module file
    assert m.examples and m.description
    assert set(m.permissions) <= PERMISSIONS
    assert isinstance(m.params, dict)


def test_dispatch_routes_to_the_named_skill(config, monkeypatch):
    reg = Registry.discover(config)
    monkeypatch.setattr(
        "wikipedia.summary", lambda *a, **k: "A black hole is a region of spacetime."
    )
    line = reg.dispatch("search", {"query": "black holes"})
    assert "black hole" in line.lower()


def test_dispatch_unknown_skill_raises_skillnotfound(config):
    reg = Registry.discover(config)
    with pytest.raises(SkillNotFound):
        reg.dispatch("teleport", {"destination": "mars"})


def test_dispatch_drops_params_the_skill_does_not_declare(config, monkeypatch):
    reg = Registry.discover(config)
    seen = {}
    monkeypatch.setattr(play.webbrowser, "open", lambda url: seen.setdefault("url", url))
    line = reg.dispatch("play", {"query": "lofi", "bogus": "x"})  # bogus must be dropped
    assert "lofi" in line
    assert "youtube.com" in seen["url"]


# -- individual skills --------------------------------------------------------

def test_search_disambiguation(config, tmp_path, monkeypatch):
    import wikipedia

    def _raise(*a, **k):
        raise wikipedia.exceptions.DisambiguationError("Mercury", ["Mercury (planet)", "Mercury (element)"])

    monkeypatch.setattr(wikipedia, "summary", _raise)
    line = search.run(_ctx(config, tmp_path), query="mercury")
    assert "Mercury (planet)" in line


def test_search_requires_a_query(config, tmp_path):
    assert "look up" in search.run(_ctx(config, tmp_path), query="").lower()


def test_open_app_builds_a_url(config, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(open_app.webbrowser, "open", lambda url: seen.setdefault("url", url))
    line = open_app.run(_ctx(config, tmp_path), app="GitHub")
    assert seen["url"] == "https://www.github.com"
    assert line == "Opening github."


def test_note_appends_and_confirms(config, tmp_path):
    ctx = _ctx(config, tmp_path)
    assert note.run(ctx, text="the wifi code is 1234") == "Noted."
    assert note.run(ctx, text="buy milk") == "Noted."
    body = (tmp_path / "notes.txt").read_text(encoding="utf-8").splitlines()
    assert len(body) == 2
    assert body[0].endswith("the wifi code is 1234")


def test_note_without_text_prompts(config, tmp_path):
    assert "note" in note.run(_ctx(config, tmp_path), text="").lower()
