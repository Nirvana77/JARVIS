"""Known issue #7: a learned skill's declared params are filled.

`Orchestrator._params_for` used typed extraction only for edge tools; every
other skill went through the builtin slot rules, which know four labels and
return `{}` for anything else — so "flip three coins" flipped one coin.
The factory declares params as `integer` / `number` / `string` / `boolean`.
"""

from __future__ import annotations

import pytest

from jarvis.nlu import slots
from jarvis.skills.contract import SkillManifest

COIN = SkillManifest(
    name="flip_a_coin",
    description="Flip one or more coins.",
    examples=["flip a coin", "flip three coins"],
    params={"count": {"type": "integer", "required": False}},
    origin="learned",
)
REMIND = SkillManifest(
    name="shopping_list",
    description="Add an item to the shopping list.",
    examples=["add milk to the shopping list"],
    params={"item": {"type": "string", "required": True}, "urgent": {"type": "boolean"}},
    origin="learned",
)
TIP = SkillManifest(
    name="tip",
    description="Work out a tip.",
    examples=["tip on 40 dollars"],
    params={"amount": {"type": "number"}},
    origin="learned",
)


class _Registry:
    def __init__(self, *manifests):
        self._m = {m.name: m for m in manifests}

    def manifest(self, name):
        return self._m[name]


def _orch(*manifests):
    from jarvis.core.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.registry = _Registry(*manifests)
    orch._slot_extract = slots.extract
    return orch


@pytest.mark.parametrize(
    "text, expected",
    [
        ("flip three coins", {"count": 3}),
        ("flip 10 coins", {"count": 10}),
        ("flip a coin", {}),              # "a" is not a count: the skill's default applies
    ],
)
def test_an_integer_param_is_filled(text, expected):
    params = _orch(COIN)._params_for("flip_a_coin", text)
    assert params == expected
    assert all(type(v) is int for v in params.values())


def test_a_number_param_keeps_its_fraction():
    assert _orch(TIP)._params_for("tip", "what is the tip on 42.5") == {"amount": 42.5}


def test_strings_and_booleans_are_not_guessed():
    """Text extraction takes what follows "to"/"that": "add eggs to the
    shopping list" would make the item "the shopping list". A wrong string is
    worse than none, so only numbers are filled."""
    assert _orch(REMIND)._params_for("shopping_list", "add eggs to the shopping list") == {}


def test_builtins_keep_their_own_slot_rules():
    search = SkillManifest(
        name="search", description="x", examples=["x"],
        params={"query": {"type": "string"}}, origin="builtin",
    )
    assert _orch(search)._params_for("search", "search for cats") == {"query": "cats"}
