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


def test_prompt_tells_the_test_how_to_import_the_skill():
    """Known issue #1: the test runs beside `<name>.py` in a staging dir, so
    only a top-level `import <name>` resolves. Left to guess, Claude wrote
    `from jarvis.skills import coin_flip` — an ImportError on every attempt."""
    prompt = ClaudeClient._build_prompt(SPEC, None)
    assert "coin_flip.py" in prompt
    assert "import coin_flip" in prompt
    assert "from jarvis.skills import coin_flip" in prompt  # named as the wrong way


class _FakeMessages:
    def __init__(self, text, stop_reason):
        self.text, self.stop_reason, self.kwargs = text, stop_reason, None

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.kwargs = kwargs
        block = SimpleNamespace(type="text", text=self.text)
        return SimpleNamespace(content=[block], stop_reason=self.stop_reason)


def _client_with(messages):
    from types import SimpleNamespace

    client = ClaudeClient(api_key=None)
    client.available = True
    client._client = SimpleNamespace(messages=messages)
    return client


def test_a_truncated_reply_says_it_was_cut_off():
    """Known issue #1: at max_tokens=8000 (shared with thinking) a larger skill
    was cut off mid-test and reported as "didn't contain the two expected code
    blocks" — so the retry feedback never said *why*."""
    import pytest

    from jarvis.factory.claude_client import ClaudeClientError

    messages = _FakeMessages("```python skill\nMANIFEST = 1\n```\n```python test\ndef te", "max_tokens")
    with pytest.raises(ClaudeClientError, match="cut off"):
        _client_with(messages).generate_skill(SPEC)


def test_generation_leaves_room_for_thinking_and_both_modules():
    messages = _FakeMessages("```python skill\nA\n```\n```python test\nB\n```", "end_turn")
    generated = _client_with(messages).generate_skill(SPEC)
    assert generated.module_source == "A\n"
    assert messages.kwargs["max_tokens"] >= 16000
