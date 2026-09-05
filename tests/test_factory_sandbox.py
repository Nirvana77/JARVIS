"""`SubprocessSandbox` — real subprocess execution (no network mocking)."""

from __future__ import annotations

import pytest

from jarvis.factory.sandbox import SubprocessSandbox

GOOD_MODULE = '''\
from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="coin_flip",
    description="Flip a coin.",
    examples=["flip a coin"],
    params={},
    permissions={"pure"},
)


def run(ctx, **params) -> str:
    return "Heads."
'''

GOOD_TEST = '''\
import coin_flip


class _Ctx:
    llm = None
    def say(self, text):
        pass


def test_run_returns_heads():
    assert coin_flip.run(_Ctx()) == "Heads."
'''

FAILING_MODULE = GOOD_MODULE.replace("coin_flip", "broken", 1).replace(
    'return "Heads."', 'raise RuntimeError("boom")'
)
FAILING_TEST = GOOD_TEST.replace("coin_flip", "broken").replace(
    'assert coin_flip.run(_Ctx()) == "Heads."', "broken.run(_Ctx())"
)

NET_MODULE = '''\
from __future__ import annotations

import socket

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="sneaky",
    description="Opens a socket.",
    examples=["do the thing"],
    params={},
    permissions={"net"},
)


def run(ctx, **params) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("93.184.216.34", 80))
    s.close()
    return "connected"
'''
NET_TEST = '''\
import sneaky


class _Ctx:
    llm = None
    def say(self, text):
        pass


def test_connects():
    assert sneaky.run(_Ctx()) == "connected"
'''


@pytest.fixture(scope="module")
def sandbox():
    return SubprocessSandbox(timeout_s=15.0, mem_mb=512, cpu_s=10)


def test_passing_tests_report_ok(tmp_path, sandbox):
    module_path = tmp_path / "coin_flip.py"
    test_path = tmp_path / "test_coin_flip.py"
    module_path.write_text(GOOD_MODULE, encoding="utf-8")
    test_path.write_text(GOOD_TEST, encoding="utf-8")

    result = sandbox.run_tests(module_path, test_path, frozenset({"pure"}))
    assert result.ok, result.stderr


def test_failing_tests_report_not_ok(tmp_path, sandbox):
    module_path = tmp_path / "broken.py"
    test_path = tmp_path / "test_broken.py"
    module_path.write_text(FAILING_MODULE, encoding="utf-8")
    test_path.write_text(FAILING_TEST, encoding="utf-8")

    result = sandbox.run_tests(module_path, test_path, frozenset({"pure"}))
    assert not result.ok


def test_dry_run_calls_run_and_checks_return_type(tmp_path, sandbox):
    module_path = tmp_path / "coin_flip.py"
    module_path.write_text(GOOD_MODULE, encoding="utf-8")

    result = sandbox.dry_run(module_path, {}, frozenset({"pure"}))
    assert result.ok, result.stderr
    assert "Heads." in result.stdout


def test_net_isolation_blocks_outbound_connection_when_available(tmp_path, sandbox):
    if not sandbox.net_isolated:
        pytest.skip("unshare --user --net not available in this environment")

    module_path = tmp_path / "sneaky.py"
    test_path = tmp_path / "test_sneaky.py"
    module_path.write_text(NET_MODULE, encoding="utf-8")
    test_path.write_text(NET_TEST, encoding="utf-8")

    # declared "net" permission, but the sandbox call site is told the skill
    # only has "pure" — simulates validate.py being bypassed — network access
    # must fail at the kernel level regardless of what the manifest claims.
    result = sandbox.run_tests(module_path, test_path, frozenset({"pure"}))
    assert not result.ok
