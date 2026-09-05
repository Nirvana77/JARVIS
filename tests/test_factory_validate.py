"""Static AST validation of generated skill modules — the load-bearing
defense (PRD: independent of whatever isolation the sandbox manages)."""

from __future__ import annotations

import pytest

from jarvis.factory.validate import ValidationError, validate

_GOOD = '''\
from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="coin_flip",
    description="Flip a coin and report heads or tails.",
    examples=["flip a coin", "heads or tails"],
    params={},
    permissions={"pure"},
)


def run(ctx, **params) -> str:
    return "Heads."
'''


def test_well_formed_module_validates():
    manifest = validate(_GOOD)
    assert manifest.name == "coin_flip"
    assert manifest.permissions == {"pure"}


def test_socket_import_without_net_permission_is_rejected():
    source = _GOOD.replace(
        "from __future__ import annotations",
        "from __future__ import annotations\nimport socket",
    ).replace("return \"Heads.\"", "socket.socket()\n    return \"Heads.\"")
    with pytest.raises(ValidationError, match="net"):
        validate(source)


def test_socket_import_with_net_permission_is_allowed():
    source = _GOOD.replace(
        "from __future__ import annotations",
        "from __future__ import annotations\nimport socket",
    ).replace(
        "permissions={\"pure\"}", "permissions={\"net\"}"
    ).replace("return \"Heads.\"", "socket.socket()\n    return \"Heads.\"")
    manifest = validate(source)
    assert manifest.permissions == {"net"}


def test_shell_call_without_permission_is_rejected():
    source = _GOOD.replace('return "Heads."', 'import subprocess\n    subprocess.run(["ls"])\n    return "Heads."')
    with pytest.raises(ValidationError, match="shell"):
        validate(source)


def test_eval_without_shell_permission_is_rejected():
    source = _GOOD.replace('return "Heads."', 'eval("1+1")\n    return "Heads."')
    with pytest.raises(ValidationError, match="shell"):
        validate(source)


def test_open_without_fs_permission_is_rejected():
    source = _GOOD.replace('return "Heads."', 'open("/tmp/x", "w")\n    return "Heads."')
    with pytest.raises(ValidationError, match="fs_read.*fs_write|fs_write.*fs_read"):
        validate(source)


def test_missing_manifest_is_rejected():
    source = "def run(ctx, **params):\n    return 'hi'\n"
    with pytest.raises(ValidationError, match="MANIFEST"):
        validate(source)


def test_missing_run_function_is_rejected():
    source = _GOOD.split("def run")[0]
    with pytest.raises(ValidationError, match="run"):
        validate(source)


def test_manifest_with_non_literal_value_is_rejected():
    source = _GOOD.replace('name="coin_flip"', "name=str('coin_flip')")
    with pytest.raises(ValidationError, match="literal"):
        validate(source)


def test_name_collision_is_rejected_unless_allowed():
    with pytest.raises(ValidationError, match="already exists"):
        validate(_GOOD, frozenset({"coin_flip"}))
    # editing the same skill in place is fine
    manifest = validate(_GOOD, frozenset({"coin_flip"}), allow_name="coin_flip")
    assert manifest.name == "coin_flip"


def test_syntax_error_is_rejected():
    with pytest.raises(ValidationError, match="syntax"):
        validate("def run(:\n")
