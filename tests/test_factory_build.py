"""`build()` — pure orchestration around an injected `generate`, no network."""

from __future__ import annotations

import pytest

from jarvis.factory.build import BuildError, build
from jarvis.factory.claude_client import GeneratedSkill
from jarvis.factory.spec import SkillSpec

SPEC = SkillSpec(name="coin_flip", description="Flip a coin.", examples=["flip a coin"])


def test_build_returns_the_generated_skill():
    def fake_generate(spec, existing_source):
        assert spec is SPEC
        assert existing_source is None
        return GeneratedSkill(name=spec.name, module_source="MODULE", test_source="TEST")

    result = build(SPEC, generate=fake_generate)
    assert result.module_source == "MODULE"
    assert result.test_source == "TEST"


def test_build_passes_existing_source_through_for_edits():
    seen = {}

    def fake_generate(spec, existing_source):
        seen["existing"] = existing_source
        return GeneratedSkill(name=spec.name, module_source="M", test_source="T")

    build(SPEC, generate=fake_generate, existing_source="old code")
    assert seen["existing"] == "old code"


def test_build_wraps_generate_exceptions():
    def boom(spec, existing_source):
        raise RuntimeError("network down")

    with pytest.raises(BuildError, match="network down"):
        build(SPEC, generate=boom)


def test_build_rejects_empty_module_source():
    def fake_generate(spec, existing_source):
        return GeneratedSkill(name=spec.name, module_source="   ", test_source="T")

    with pytest.raises(BuildError, match="empty"):
        build(SPEC, generate=fake_generate)


def test_build_rejects_empty_test_source():
    def fake_generate(spec, existing_source):
        return GeneratedSkill(name=spec.name, module_source="M", test_source="")

    with pytest.raises(BuildError, match="empty"):
        build(SPEC, generate=fake_generate)
