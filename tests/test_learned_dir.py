"""Known issue #19: learned skills live in ``<data>/skills/learned/``, which
the dev brain and the pod share, instead of each brain's own checkout (the
dev brain) or PVC (the pod)."""

from __future__ import annotations

import dataclasses
import sys

import pytest

import jarvis.skills.learned as learned_pkg
from jarvis.factory import flows
from jarvis.skills.registry import Registry, migrate_learned_skills

COIN = (
    "from jarvis.skills.contract import SkillManifest\n"
    "MANIFEST = SkillManifest(name='{name}', description='Flip a coin.', "
    "examples=['flip a coin'])\n"
    "def run(ctx, **params):\n    return '{reply}'\n"
)


@pytest.fixture
def restore_learned_path():
    original = list(learned_pkg.__path__)
    yield
    learned_pkg.__path__ = original
    for name in [m for m in sys.modules if m.startswith("jarvis.skills.learned.")]:
        sys.modules.pop(name, None)


def test_the_config_names_the_shared_folder(config, tmp_path):
    cfg = dataclasses.replace(config, data_dir=tmp_path)
    assert cfg.learned_skills_dir == tmp_path / "skills" / "learned"


def test_discovery_reads_learned_skills_from_data(config, tmp_path, restore_learned_path):
    cfg = dataclasses.replace(config, data_dir=tmp_path)
    cfg.learned_skills_dir.mkdir(parents=True)
    (cfg.learned_skills_dir / "coin_flip.py").write_text(
        COIN.format(name="coin_flip", reply="Heads."), encoding="utf-8"
    )
    reg = Registry.discover(cfg)
    assert "coin_flip" in reg.names()
    assert reg.manifest("coin_flip").origin == "learned"


def test_promotion_writes_into_the_shared_folder(config, tmp_path, restore_learned_path):
    cfg = dataclasses.replace(config, data_dir=tmp_path)
    Registry.discover(cfg)
    assert flows.learned_source_path("coin_flip") == cfg.learned_skills_dir / "coin_flip.py"
    assert not cfg.learned_skills_dir.exists()  # discovery writes nothing


def test_skills_in_the_old_place_move_over(tmp_path):
    old, new = tmp_path / "checkout", tmp_path / "data"
    old.mkdir()
    new.mkdir()
    (old / "__init__.py").write_text("", encoding="utf-8")
    (old / "coin_flip.py").write_text(COIN.format(name="coin_flip", reply="Old."), encoding="utf-8")
    (old / "roll_a_die.py").write_text(COIN.format(name="roll_a_die", reply="6."), encoding="utf-8")
    (new / "roll_a_die.py").write_text(COIN.format(name="roll_a_die", reply="Mine."), encoding="utf-8")

    moved = migrate_learned_skills(old, new)

    assert moved == ["coin_flip"]
    assert (new / "coin_flip.py").read_text(encoding="utf-8").count("Old.") == 1
    assert "Mine." in (new / "roll_a_die.py").read_text(encoding="utf-8")  # never overwritten
    assert not (new / "__init__.py").exists()
    assert (old / "coin_flip.py").exists()  # copied, not moved: the old brain may still run


def test_migrating_twice_or_from_nowhere_is_harmless(tmp_path):
    new = tmp_path / "data"
    assert migrate_learned_skills(tmp_path / "missing", new) == []
    assert migrate_learned_skills(new, new) == []
