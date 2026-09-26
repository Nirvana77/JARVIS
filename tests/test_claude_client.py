"""`ClaudeClient._build_prompt` — pure string assembly, no network."""

from __future__ import annotations

from jarvis.factory.claude_client import ClaudeClient
from jarvis.factory.spec import SkillSpec

SPEC = SkillSpec(name="coin_flip", description="Flip a coin.", examples=["flip a coin"])


def test_prompt_has_no_feedback_section_by_default():
    prompt = ClaudeClient._build_prompt(SPEC, None)
    assert "previous attempt" not in prompt.lower()


def test_prompt_includes_feedback_when_given():
    prompt = ClaudeClient._build_prompt(SPEC, None, "sandbox tests failed: boom")
    assert "previous attempt" in prompt.lower()
    assert "sandbox tests failed: boom" in prompt


def test_prompt_still_mentions_existing_source_alongside_feedback():
    prompt = ClaudeClient._build_prompt(SPEC, "OLD SOURCE", "validation failed: bad name")
    assert "OLD SOURCE" in prompt
    assert "validation failed: bad name" in prompt
