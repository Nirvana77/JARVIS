"""Builtin skills and the registry."""

from __future__ import annotations

import pytest

from jarvis.core.context import Context
from jarvis.skills.builtin import note, open_app, play, search
from jarvis.skills.contract import PERMISSIONS, SkillManifest, SkillNotFound
from jarvis.skills.registry import BUILTIN_PACKAGE, Registry

BUILTINS = {"search": search, "open_app": open_app, "play": play, "note": note}


def _ctx(config, tmp_path):
    return Context(say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None)


# -- manifests / registry -------------------------------------------------

def test_registry_discovers_the_four_builtins(config):
    # `packages=(BUILTIN_PACKAGE,)` — deliberately not the default, which also
    # scans `jarvis.skills.learned`. That directory is real, per-install state
    # (M2's `teach`/`edit_skill` write to it), so a test asserting the exact
    # builtin roster must not be coupled to whatever a developer has actually
    # taught their local JARVIS.
    reg = Registry.discover(config, packages=(BUILTIN_PACKAGE,))
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
    monkeypatch.setattr(search, "_search_title", lambda s, q: "Black hole")
    monkeypatch.setattr(
        search, "_summary",
        lambda s, t: {"type": "standard", "extract": "A black hole is a region of spacetime."},
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
    monkeypatch.setattr(search, "_search_title", lambda s, q: "Mercury")
    monkeypatch.setattr(
        search, "_summary", lambda s, t: {"type": "disambiguation", "extract": "Mercury may refer to..."}
    )
    line = search.run(_ctx(config, tmp_path), query="mercury")
    assert "several things" in line.lower()


def test_search_not_found(config, tmp_path, monkeypatch):
    monkeypatch.setattr(search, "_search_title", lambda s, q: None)
    line = search.run(_ctx(config, tmp_path), query="qwertyzxcv nonsense")
    assert "couldn't find" in line.lower()


def test_search_network_error_is_graceful(config, tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(search, "_search_title", _boom)
    assert search.run(_ctx(config, tmp_path), query="anything") == "I had trouble reaching Wikipedia."


def test_search_requires_a_query(config, tmp_path):
    assert "look up" in search.run(_ctx(config, tmp_path), query="").lower()


def test_search_trims_wiki_markup_and_extra_sentences(config, tmp_path, monkeypatch):
    raw = (
        "PG Tips is a brand of tea. It is sold in the UK. Third sentence here.\n\n"
        "== Brand name ==\n\nIn the 1930s, Brooke Bond launched it."
    )
    monkeypatch.setattr(search, "_search_title", lambda s, q: "PG Tips")
    monkeypatch.setattr(search, "_summary", lambda s, t: {"type": "standard", "extract": raw})
    line = search.run(_ctx(config, tmp_path), query="pg tips")
    assert "==" not in line
    assert "Brand name" not in line
    assert line == "According to Wikipedia: PG Tips is a brand of tea. It is sold in the UK."


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


# -- M2: skills/learned/ discovery ---------------------------------------------

def test_registry_discovers_a_learned_skill(config, tmp_path):
    import sys

    import jarvis.skills.learned as learned_pkg

    (tmp_path / "coin_flip.py").write_text(
        "from jarvis.skills.contract import SkillManifest\n"
        "MANIFEST = SkillManifest(name='coin_flip', description='Flip a coin.', "
        "examples=['flip a coin'], origin='learned')\n"
        "def run(ctx, **params):\n    return 'Heads.'\n",
        encoding="utf-8",
    )
    original_path = list(learned_pkg.__path__)
    learned_pkg.__path__ = [str(tmp_path)]
    try:
        reg = Registry.discover(config)
        assert "coin_flip" in reg.names()
        assert reg.manifest("coin_flip").origin == "learned"
    finally:
        learned_pkg.__path__ = original_path
        sys.modules.pop("jarvis.skills.learned.coin_flip", None)


def test_a_learned_skills_origin_is_forced_even_if_the_code_omits_it(config, tmp_path):
    """Regression: Claude has no reason to declare `origin` correctly (the
    factory prompt never mentions it, and `validate()` can't enforce a value
    that only matters after promotion) — a generated MANIFEST that omits
    `origin` defaults to "builtin" per SkillManifest's own default, which
    silently broke edit_skill/revert_skill's `origin="learned"` filter until
    Registry.discover() started normalizing it by package location."""
    import sys

    import jarvis.skills.learned as learned_pkg

    (tmp_path / "coin_flip.py").write_text(
        "from jarvis.skills.contract import SkillManifest\n"
        "MANIFEST = SkillManifest(name='coin_flip', description='Flip a coin.', "
        "examples=['flip a coin'])\n"  # no `origin=` — defaults to "builtin"
        "def run(ctx, **params):\n    return 'Heads.'\n",
        encoding="utf-8",
    )
    original_path = list(learned_pkg.__path__)
    learned_pkg.__path__ = [str(tmp_path)]
    try:
        reg = Registry.discover(config)
        assert reg.manifest("coin_flip").origin == "learned"
    finally:
        learned_pkg.__path__ = original_path
        sys.modules.pop("jarvis.skills.learned.coin_flip", None)


def test_a_builtins_origin_is_forced_to_builtin_even_if_misdeclared(config, tmp_path):
    import sys

    import jarvis.skills.builtin as builtin_pkg

    (tmp_path / "fake_builtin.py").write_text(
        "from jarvis.skills.contract import SkillManifest\n"
        "MANIFEST = SkillManifest(name='fake_builtin', description='x', "
        "examples=['x'], origin='learned')\n"  # deliberately wrong for this package
        "def run(ctx, **params):\n    return 'x'\n",
        encoding="utf-8",
    )
    original_path = list(builtin_pkg.__path__)
    builtin_pkg.__path__ = [str(tmp_path)]
    try:
        reg = Registry.discover(config, packages=(BUILTIN_PACKAGE,))
        assert reg.manifest("fake_builtin").origin == "builtin"
    finally:
        builtin_pkg.__path__ = original_path
        sys.modules.pop("jarvis.skills.builtin.fake_builtin", None)
